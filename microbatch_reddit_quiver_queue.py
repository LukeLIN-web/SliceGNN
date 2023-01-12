'''
sample : quiver 
dataset: reddit
getmicrobatch : yes
'''


import argparse
import os

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler

import quiver
from timeit import default_timer
from utils.get_micro_batch import get_micro_batch
from utils.model import SAGE
from utils.common_config import *


def parse_args(default_run_config):
    argparser = argparse.ArgumentParser("GCN Training")

    add_common_arguments(argparser, default_run_config)

    argparser.add_argument('--fanout', nargs='+',
                           type=int, default=default_run_config['fanout'])
    argparser.add_argument('--lr', type=float,
                           default=default_run_config['lr'])
    argparser.add_argument('--dropout', type=float,
                           default=default_run_config['dropout'])
    argparser.add_argument('--gpu_num', type=int, default=2)
    argparser.add_argument('--micro_pergpu', type=int, default=1)
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


def run_sample(worker_id, run_config,  dataset):
    num_worker = run_config['num_sample_worker']
    torch.cuda.set_device(worker_id)
    print('[Sample Worker {:d}/{:d}] Started with PID {:d}'.format(
        worker_id, num_worker, os.getpid()))
    data = dataset[0]

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)

    train_loader = torch.utils.data.DataLoader(
        train_idx, batch_size=1024*run_config[num_train_workers], shuffle=False, drop_last=True)

    torch.manual_seed(12345)

    for epoch in range(1, run_config['num_epoch']):
        epoch_start = default_timer()
        for seeds in train_loader:
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            micro_batchs = get_micro_batch(adjs,
                                           n_id,
                                           batch_size, run_config['num_train_worker']*run_config['micro_pergpu'])
            micro_batchs = [
                micro_batchs[i * run_config['micro_pergpu']:(i+1) * run_config['micro_pergpu']] for i in range(world_size)]
            nodeid = [n_id]


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

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)

    train_loader = torch.utils.data.DataLoader(
        train_idx, batch_size=1024*world_size, shuffle=False, drop_last=True)

    if worker_id == 1:
        subgraph_loader = NeighborSampler(data.edge_index, node_idx=None,
                                          sizes=[-1], batch_size=2048,
                                          shuffle=False, num_workers=6)

    torch.manual_seed(12345)
    model = SAGE(dataset.num_features, 256, dataset.num_classes).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    y = data.y.to(rank)

    for epoch in range(1, 16):
        model.train()
        epoch_start = default_timer()
        for seeds in train_loader:
            optimizer.zero_grad()
            if rank == 0:
                n_id, batch_size, adjs = quiver_sampler.sample(seeds)
                micro_batchs = get_micro_batch(adjs,
                                               n_id,
                                               batch_size, world_size*run_config['micro_pergpu'])
                micro_batchs = [
                    micro_batchs[i * run_config['micro_pergpu']:(i+1) * run_config['micro_pergpu']] for i in range(world_size)]
                # because we don't know n_id length, so we use list
                nodeid = [n_id]
            else:
                micro_batchs = []
                nodeid = [None]
            dist.broadcast_object_list(
                nodeid, src=0, device=torch.device(rank))
            outputlist = [None]
            # TODO: microbatch to tensor so that we can use scatter
            dist.scatter_object_list(outputlist, micro_batchs, src=0)
            n_id = nodeid[0]
            target_node = n_id[:len(
                seeds)][rank * (len(seeds)//world_size): (rank+1)*(len(seeds)//world_size)]
            for i in range(len(outputlist[0])):
                micro_batch = outputlist[0][i]
                micro_batch_adjs = [adj.to(rank)
                                    for adj in micro_batch.adjs]  # load topo
                out = model(x[n_id][micro_batch.n_id],
                            micro_batch_adjs)  # forward
                loss = F.nll_loss(
                    out, y[target_node][i * (micro_batch.size):(i+1) * (micro_batch.size)])
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
            acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
            acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
            acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
            print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')

        dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    run_config = get_run_config()
    num_train_workers = run_config['num_train_worker']
    num_sample_worker = run_config['num_sample_worker']

    dataset = Reddit('/data/Reddit')
    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)

    quiver_sampler = quiver.pyg.GraphSageSampler(
        csr_topo, sizes=[25, 10], device=0, mode='GPU')  # 这里是0, 但是spawn之后会变成fake,然后再lazy init 赋值

    quiver_feature = quiver.Feature(rank=0, device_list=list(range(
        num_train_workers)), device_cache_size="2G", cache_policy="device_replicate", csr_topo=csr_topo)
    quiver_feature.from_cpu_tensor(data.x)

    workers = []
    mp.set_start_method('spawn')

    for worker_id in range(num_sample_worker):
        p = mp.Process(target=run_sample, args=(
            worker_id, run_config,    dataset))
        p.start()
        workers.append(p)

    for worker_id in range(num_train_workers):
        p = mp.Process(target=run_train, args=(
            worker_id, run_config, quiver_feature, quiver_sampler, dataset))
        p.start()
        workers.append(p)
    for p in workers:
        p.join()
