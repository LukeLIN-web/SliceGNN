import argparse
import os

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.loader import NeighborSampler

import quiver
from timeit import default_timer
from get_micro_batch import *
from model import SAGE
from utils import get_dataset
import numpy as np

def run(rank, world_size, data, x, quiver_sampler: quiver.pyg.GraphSageSampler,num_classes, args: argparse.ArgumentParser,micro_pergpu:int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)
    torch.manual_seed(12345)

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)

    train_loader = torch.utils.data.DataLoader(
        train_idx, batch_size=1024*world_size, shuffle=False, drop_last=True)

    if rank == 0:
        subgraph_loader = NeighborSampler(data.edge_index, node_idx=None,
                                          sizes=[-1], batch_size=2048,
                                          shuffle=False, num_workers=6)

    model = SAGE(data.num_features, 256, num_classes).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    y = data.y.to(rank)
    MODEL_SAMPLE = "sample"
    MODEL_TRAIN = "train"
    PROPAGATE = "propagate"
    metrics = {MODEL_SAMPLE: [], MODEL_TRAIN: [], PROPAGATE: []}

    for epoch in range(1, args.warmup + args.num_epochs + 1):
        for seeds in train_loader:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if rank == 0:
                n_id, batch_size, adjs = quiver_sampler.sample(seeds)
                micro_batchs = get_micro_batch(adjs,
                                               n_id,
                                               batch_size, world_size*micro_pergpu)
                micro_batchs = [
                    micro_batchs[i * micro_pergpu:(i+1) * micro_pergpu] for i in range(world_size)]
                nodeid = [n_id]
            else:
                micro_batchs = []
                nodeid = [None]
            end.record()
            torch.cuda.synchronize()
            if epoch >= args.warmup:
                metrics[MODEL_SAMPLE].append(start.elapsed_time(end))
            start.record()
            dist.broadcast_object_list(
                nodeid, src=0, device=torch.device(rank))
            n_id = nodeid[0]
            outputlist = [None]
            dist.scatter_object_list(outputlist, micro_batchs, src=0)
            end.record()
            torch.cuda.synchronize()
            if epoch >= args.warmup:
                metrics[PROPAGATE].append(start.elapsed_time(end))
            start.record()
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
            end.record()
            torch.cuda.synchronize()
            if epoch >= args.warmup:
                metrics[MODEL_TRAIN].append(start.elapsed_time(end))
            optimizer.step()
            optimizer.zero_grad()

        dist.barrier()

    if rank == 0:
        print(f"\nMetrics for GPU {rank}")
        print(f"Skipped {args.warmup} CUDA warmpup epochs. ")
        for step, elapsed_milliseconds in metrics.items():
            A = np.array(elapsed_milliseconds)
            print(' {N} iterations, {step}, mean={mean} ms, median={median} ms, p90={p90} ms, p99={p99} ms'.format(
                N=len(A), step=step, mean=np.mean(A),
                median=np.percentile(A, 50), p90=np.percentile(A, 90),
                p99=np.percentile(A, 99)))

    dist.destroy_process_group()


def main(args: argparse.ArgumentParser) -> None:
    world_size = 2
    dataset_name = "Reddit"
    data, num_classes = get_dataset(dataset_name, args.root,
                                    args.use_sparse_tensor, args.bf16)
    csr_topo = quiver.CSRTopo(data.edge_index)

    quiver_sampler = quiver.pyg.GraphSageSampler(
        csr_topo, sizes=[25, 10], device=0, mode='GPU')

    quiver_feature = quiver.Feature(rank=0, device_list=list(range(
        world_size)), device_cache_size="2G", cache_policy="device_replicate", csr_topo=csr_topo)
    quiver_feature.from_cpu_tensor(data.x)

    print('Let\'s use', world_size, 'GPUs!')
    print('BENCHMARK STARTS')

# 就传入, 函数参数多一些没关系.
    # for world_size in args.gpu_num:
    for micro_pergpu in args.micro_pergpu:
        mp.spawn(
            run,
            args=(world_size, data, quiver_feature,
                quiver_sampler, num_classes, args,micro_pergpu),
            nprocs=world_size,
            join=True
        )


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Gloo training benchmark')
    argparser.add_argument('--datasets', nargs='+',
                           default=['Reddit'], type=str)
    argparser.add_argument('--root', default='../../data', type=str,
                           help='relative path to look for the datasets')
    argparser.add_argument('--warmup', default=1, type=int)
    argparser.add_argument('--num-epochs', default=1, type=int)
    argparser.add_argument('--gpu_num', nargs='+',
                           default=[2, 3], type=int)
    argparser.add_argument('--micro_pergpu', nargs='+',
                           default=[1,2], type=int)
    argparser.add_argument('--bf16', action='store_true')
    argparser.add_argument(
        '--use-sparse-tensor', action='store_true')
    args = argparser.parse_args()

    main(args)
