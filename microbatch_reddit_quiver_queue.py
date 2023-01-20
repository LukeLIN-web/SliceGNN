'''
sample : quiver 
dataset: reddit
getmicrobatch : yes
'''
import os

import hydra
from omegaconf import OmegaConf

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
from utils.common_config import gpu


def run_sample(worker_id, params, dataset, quiver_sampler, micro_queues):
    sample_workers = [gpu(params.num_train_worker + i)
                      for i in range(params.num_sample_worker)]

    num_sample_worker = params.num_sample_worker
    per_gpu = params.micro_pergpu
    ctx = sample_workers[worker_id]
    num_train_worker = params.num_train_worker
    print('[Sample Worker {:d}/{:d}] Started with PID {:d}({:s})'.format(
        worker_id, num_sample_worker, os.getpid(), torch.cuda.get_device_name(ctx)))
    data = dataset[0]
    torch.cuda.set_device(ctx)
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)

    train_loader = torch.utils.data.DataLoader(
        train_idx, batch_size=1024*num_train_worker, shuffle=False, drop_last=True)

    torch.manual_seed(12345)

    for epoch in range(1, params.num_epoch+1):
        epoch_start = default_timer()
        for seeds in train_loader:
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            micro_batchs = get_micro_batch(adjs,
                                           n_id,
                                           batch_size, num_train_worker*per_gpu)
            for i in range(num_train_worker):
                micro_queues[i].put(
                    (n_id, micro_batchs[i * per_gpu:(i+1) * per_gpu]))
        epoch_end = default_timer()
        print(
            f'Epoch: {epoch:03d}, Sample Epoch Time: {epoch_end - epoch_start}')


def run_train(worker_id, params, x,  dataset, queue):
    train_workers = [
        gpu(i) for i in range(params.num_train_worker)]
    ctx = train_workers[worker_id]
    num_worker = params.num_train_worker

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
    batch_size = params.batch_size * num_worker
    length = train_idx.size(dim=0) // (1024*num_worker)

    if worker_id == 0:
        subgraph_loader = NeighborSampler(data.edge_index, node_idx=None,
                                          sizes=[-1], batch_size=2048,
                                          shuffle=False, num_workers=6)
    num_features, num_classes = dataset.num_features, dataset.num_classes
    torch.manual_seed(12345)
    model = SAGE(num_features, params.architecture.hidden_channels, num_classes).to(
        train_device)
    if num_worker > 1:
        model = DistributedDataParallel(
            model, device_ids=[train_device], output_device=train_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    y = data.y.to(train_device)

    for epoch in range(1, params.num_epoch + 1):
        model.train()
        epoch_start = default_timer()
        for i in range(length):
            optimizer.zero_grad()
            (n_id, micro_batchs) = queue.get()
            target_node = n_id[:batch_size][worker_id *
                                            params.batch_size: (worker_id+1)*params.batch_size]
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
        epoch_end = default_timer()

        if worker_id == 0:
            print(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Epoch Time: {epoch_end - epoch_start}')

        if worker_id == 0 and epoch % 5 == 0:  # We evaluate on a single GPU for now
            model.eval()
            with torch.no_grad():
                if num_worker > 1:
                    out = model.module.inference(x, worker_id, subgraph_loader)
                else:
                    out = model.inference(x, worker_id, subgraph_loader)
            res = out.argmax(dim=-1) == y
            acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
            acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
            acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
            print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')


@hydra.main(config_path='conf', config_name='config', version_base='1.1')
def main(conf):
    model_name, dataset_name = conf.model.name, conf.dataset.name
    params = conf.model.params[dataset_name]

    num_train_workers = params.num_train_worker
    num_sample_worker = params.num_sample_worker
    print(OmegaConf.to_yaml(conf))

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
    microbatchs_qs = [mp.Queue(30) for i in range(num_train_workers)]
    for worker_id in range(num_sample_worker):
        p = mp.Process(target=run_sample, args=(
            worker_id, params, dataset, quiver_sampler, microbatchs_qs))
        p.start()
        workers.append(p)

    for worker_id in range(num_train_workers):
        p = mp.Process(target=run_train, args=(
            worker_id, params, quiver_feature, dataset, microbatchs_qs[worker_id]))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()


if __name__ == '__main__':
    main()
