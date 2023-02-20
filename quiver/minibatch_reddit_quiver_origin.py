
'''
sample : quiver_sampler
dataset: reddit
getmicrobatch : no
'''

import os

import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler

from timeit import default_timer
import quiver
import argparse
from microGNN.utils.model import SAGE


def parse_args(default_run_config):
    argparser = argparse.ArgumentParser("GCN Training")

    add_common_arguments(argparser, default_run_config)

    argparser.add_argument('--fanout', nargs='+',
                           type=int, default=default_run_config['fanout'])
    argparser.add_argument('--lr', type=float,
                           default=default_run_config['lr'])
    argparser.add_argument('--dropout', type=float,
                           default=default_run_config['dropout'])

    return vars(argparser.parse_args())


def get_run_config():
    run_config = {}
    run_config.update(get_default_common_config(run_mode=RunMode.NORMAL))
    run_config['sample_type'] = 'khop2'
    run_config['fanout'] = [25, 10]
    run_config['lr'] = 0.01
    run_config['dropout'] = 0.5

    run_config.update(parse_args(run_config))

    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])

    print_run_config(run_config)

    return run_config


def run(rank, world_size, data_split, edge_index, x, quiver_sampler: quiver.pyg.GraphSageSampler, y, num_features, num_classes):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)

    train_mask, val_mask, test_mask = data_split
    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]
    
    train_loader = torch.utils.data.DataLoader(
        train_idx, batch_size=1024, shuffle=True, drop_last=True)

    if rank == 0:
        subgraph_loader = NeighborSampler(edge_index, node_idx=None,
                                          sizes=[-1], batch_size=2048,
                                          shuffle=False, num_workers=6)

    torch.manual_seed(12345)
    model = SAGE(num_features, 256, num_classes).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    y = y.to(rank)

    for epoch in range(1, 6):
        model.train()
        epoch_start = default_timer()
        for seeds in train_loader:
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            adjs = [adj.to(rank) for adj in adjs]
            optimizer.zero_grad()
            out = model(x[n_id].to(rank), adjs)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()

        dist.barrier()

        if rank == 0:
            print(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {default_timer() - epoch_start}')

        if rank == 0 and epoch % 5 == 0:  # We evaluate on a single GPU for now
            model.eval()
            with torch.no_grad():
                out = model.module.inference(x, rank, subgraph_loader)
            res = out.argmax(dim=-1) == y
            acc1 = int(res[train_mask].sum()) / int(train_mask.sum())
            acc2 = int(res[val_mask].sum()) / int(val_mask.sum())
            acc3 = int(res[test_mask].sum()) / int(test_mask.sum())
            print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')

        dist.barrier()

    dist.destroy_process_group()


def run_train(worker_id, run_config, x, quiver_sampler, dataset):
    data = dataset[0]
    edge_index = data.edge_index
    num_worker = run_config['num_train_worker']
    if num_worker > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = num_worker
        torch.distributed.init_process_group(backend="nccl",
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=worker_id)

    torch.cuda.set_device(worker_id)

    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(train_idx.size(0) // world_size)[worker_id]

    train_loader = torch.utils.data.DataLoader(
        train_idx, batch_size=1024, shuffle=True, drop_last=True)

    if worker_id == 0:
        subgraph_loader = NeighborSampler(edge_index, node_idx=None,
                                          sizes=[-1], batch_size=2048,
                                          shuffle=False, num_workers=6)
    num_features, num_classes = dataset.num_features, dataset.num_classes

    torch.manual_seed(12345)
    model = SAGE(num_features, run_config['num_hidden'], num_classes).to(
        worker_id)
    model = DistributedDataParallel(model, device_ids=[worker_id])
    optimizer = torch.optim.Adam(model.parameters(), lr=run_config['lr'])
    rank = worker_id
    y = data.y.to(rank)

    for epoch in range(1, 6):
        model.train()
        epoch_start = default_timer()
        for seeds in train_loader:
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            adjs = [adj.to(rank) for adj in adjs]
            optimizer.zero_grad()
            out = model(x[n_id].to(rank), adjs)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()

        dist.barrier()

        if rank == 0:
            print(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {default_timer() - epoch_start}')

        if rank == 0 and epoch % 5 == 0:  # We evaluate on a single GPU for now
            model.eval()
            with torch.no_grad():
                out = model.module.inference(x, rank, subgraph_loader)
            res = out.argmax(dim=-1) == y
            acc1 = int(res[train_mask].sum()) / int(train_mask.sum())
            acc2 = int(res[val_mask].sum()) / int(val_mask.sum())
            acc3 = int(res[test_mask].sum()) / int(test_mask.sum())
            print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')

        dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    dataset = Reddit('/data/Reddit')
    run_config = get_run_config()
    num_train_workers = run_config['num_train_worker']

    data = dataset[0]

    csr_topo = quiver.CSRTopo(data.edge_index)

    quiver_sampler = quiver.pyg.GraphSageSampler(
        csr_topo, sizes=[25, 10], device=0, mode='GPU')  # 这里是0, 但是spawn之后会变成fake,然后再lazy init 赋值
    # cache feature 到rank0
    # quiver_feature = quiver.Feature(rank=0, device_list=list(range(
    #     world_size)), device_cache_size="2G", cache_policy="device_replicate", csr_topo=csr_topo)
    quiver_feature = quiver.Feature(rank=0, device_list=list(range(
        num_train_workers)), device_cache_size="2G", cache_policy="device_replicate", csr_topo=csr_topo)

    quiver_feature.from_cpu_tensor(data.x)

    # origin
    # mp.spawn(
    #     run,
    #     args=(world_size, data_split, data.edge_index, quiver_feature,
    #           quiver_sampler, data.y, dataset.num_features, dataset.num_classes),
    #     nprocs=world_size,
    #     join=True
    # )
    # gnnlab version
    workers = []
    mp.set_start_method('spawn')
    for worker_id in range(num_train_workers):
        p = mp.Process(target=run_train, args=(
            worker_id, run_config, quiver_feature, quiver_sampler, dataset))
        p.start()
        workers.append(p)
    for p in workers:
        p.join()
