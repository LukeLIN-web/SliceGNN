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

    process_common_config(run_config)
    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])

    print_run_config(run_config)

    return run_config


def run_sample(worker_id, run_config, dataset, quiver_sampler, micro_queues):
    num_worker = run_config['num_sample_worker']
    per_gpu = run_config['micro_pergpu']
    ctx = run_config['sample_workers'][worker_id]

    print('[Sample Worker {:d}/{:d}] Started with PID {:d}({:s})'.format(
        worker_id, num_worker, os.getpid(), torch.cuda.get_device_name(ctx)))
    data = dataset[0]
    torch.cuda.set_device(ctx)
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)

    train_loader = torch.utils.data.DataLoader(
        train_idx, batch_size=1024*run_config['num_train_worker'], shuffle=False, drop_last=True)

    torch.manual_seed(12345)

    for epoch in range(1, run_config['num_epoch']):
        epoch_start = default_timer()
        for seeds in train_loader:
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            micro_batchs = get_micro_batch(adjs,
                                           n_id,
                                           batch_size, run_config['num_train_worker']*per_gpu)
            for i in range(run_config['num_train_worker']):
                micro_queues[i].put(
                    (n_id, micro_batchs[i * per_gpu:(i+1) * per_gpu]))
        epoch_end = default_timer()


def run_train(worker_id, run_config, x,  dataset, queue):
    ctx = run_config['train_workers'][worker_id]
    num_worker = run_config['num_train_worker']

    if num_worker > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        torch.distributed.init_process_group(backend="nccl",
                                             init_method=dist_init_method,
                                             world_size=num_worker,
                                             rank=worker_id)
    train_device = torch.device(ctx)
    torch.cuda.set_device(train_device)

    print('[Train Worker {:d}/{:d}] Started with PID {:d}({:s})'.format(
        worker_id, num_worker, os.getpid(), torch.cuda.get_device_name(ctx)))

    data = dataset[0]
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)

    train_loader = torch.utils.data.DataLoader(
        train_idx, batch_size=1024*num_worker, shuffle=False, drop_last=True)

    if worker_id == 0:
        subgraph_loader = NeighborSampler(data.edge_index, node_idx=None,
                                          sizes=[-1], batch_size=2048,
                                          shuffle=False, num_workers=6)
    num_features, num_classes = dataset.num_features, dataset.num_classes
    torch.manual_seed(12345)
    model = SAGE(num_features, run_config['num_hidden'], num_classes).to(
        train_device)
    if num_worker > 1:
        model = DistributedDataParallel(
            model, device_ids=[train_device], output_device=train_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=run_config['lr'])

    y = data.y.to(train_device)

    for epoch in range(1, run_config['num_epoch']):
        model.train()
        epoch_start = default_timer()
        for seeds in train_loader:
            optimizer.zero_grad()
            (n_id, micro_batchs) = queue.get()
            target_node = n_id[:len(
                seeds)][worker_id * (len(seeds)//num_worker): (worker_id+1)*(len(seeds)//num_worker)]
            for i in range(len(micro_batchs)):
                micro_batch = micro_batchs[i]
                micro_batch_adjs = [adj.to(train_device)
                                    for adj in micro_batch.adjs]  # load topo
                out = model(x[n_id][micro_batch.n_id],
                            micro_batch_adjs)  # forward
                loss = F.nll_loss(
                    out, y[target_node][i * (micro_batch.size):(i+1) * (micro_batch.size)])
                loss.backward()
            optimizer.step()

        if worker_id == 0:
            print(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {default_timer() - epoch_start}')

        if worker_id == 0 and epoch % 5 == 0:  # We evaluate on a single GPU for now
            model.eval()
            with torch.no_grad():
                out = model.inference(x, worker_id, subgraph_loader)
            res = out.argmax(dim=-1) == y
            acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
            acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
            acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
            print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')


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
    microbatchs_qs = [mp.Queue(10) for i in range(num_train_workers)]
    for worker_id in range(num_sample_worker):
        p = mp.Process(target=run_sample, args=(
            worker_id, run_config, dataset, quiver_sampler, microbatchs_qs))
        p.start()
        workers.append(p)
    for worker_id in range(num_train_workers):
        p = mp.Process(target=run_train, args=(
            worker_id, run_config, quiver_feature, dataset, microbatchs_qs[worker_id]))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()
