
'''
sample : quiver 
dataset: reddit
getmicrobatch : no

'''
import argparse
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
from get_micro_batch import *
from model import SAGE


def run(rank, world_size, data, x, quiver_sampler: quiver.pyg.GraphSageSampler, dataset, args):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)

    train_loader = torch.utils.data.DataLoader(
        train_idx, batch_size=1024, shuffle=False, drop_last=True)

    if rank == 0:
        subgraph_loader = NeighborSampler(data.edge_index, node_idx=None,
                                          sizes=[-1], batch_size=2048,
                                          shuffle=False, num_workers=6)

    torch.manual_seed(12345)
    model = SAGE(data.num_features, 256, dataset.num_classes).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    y = data.y.to(rank)

    for epoch in range(1, 16):
        model.train()
        epoch_start = default_timer()
        for seeds in train_loader:
            optimizer.zero_grad()
            microseeds = seeds[rank * seeds.size(0) // world_size: (
                rank + 1) * seeds.size(0) // world_size]
            n_id, batch_size, adjs = quiver_sampler.sample(microseeds)
            target_node = n_id[:len(seeds)]
            micro_batch_adjs = [adj.to(rank)
                                for adj in adjs]  # load topo
            out = model(x[n_id], micro_batch_adjs)  # forward
            loss = F.nll_loss(out, y[target_node][:batch_size])
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
    dataset = Reddit('/data/Reddit')
    parser = argparse.ArgumentParser(description='Quiver')
    parser.add_argument('--gpu_num', type=int, default=3)
    parser.add_argument('--micro_pergpu', type=int, default=1)
    args = parser.parse_args()
    world_size = args.gpu_num  # torch.cuda.device_count()
    data = dataset[0]

    csr_topo = quiver.CSRTopo(data.edge_index)

    quiver_sampler = quiver.pyg.GraphSageSampler(
        csr_topo, sizes=[25, 10], device=0, mode='GPU')  # 这里是0, 但是spawn之后会变成fake,然后再lazy init 赋值

    quiver_feature = quiver.Feature(rank=0, device_list=list(range(
        world_size)), device_cache_size="2G", cache_policy="device_replicate", csr_topo=csr_topo)
    quiver_feature.from_cpu_tensor(data.x)

    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(
        run,
        args=(world_size, data, quiver_feature,
              quiver_sampler, dataset, args),
        nprocs=world_size,
        join=True
    )
