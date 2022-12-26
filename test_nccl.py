import os
from statistics import mean

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler

import quiver
from timeit import default_timer
from utils.get_micro_batch import get_micro_batch
from utils.model import SAGE 

def run(rank, world_size, data, x, quiver_sampler: quiver.pyg.GraphSageSampler, dataset):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

    if rank == 0:
        train_loader = torch.utils.data.DataLoader(
            train_idx, batch_size=1024, shuffle=False, drop_last=True)
        subgraph_loader = NeighborSampler(data.edge_index, node_idx=None,
                                          sizes=[-1], batch_size=2048,
                                          shuffle=False, num_workers=6)

    torch.manual_seed(12345)
    model = SAGE(dataset.num_features, 256, dataset.num_classes).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])

    y = data.y.to(rank)
    micro_batch_num = 2

    for epoch in range(1, 3):
        model.train()
        if rank == 0:
            seeds = next(iter(train_loader))
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            micro_batchs = get_micro_batch(adjs,
                                           n_id,
                                           batch_size, micro_batch_num)
            nodeid = [n_id, batch_size]
        # for i in range(0, len(seeds), 1024):
        if True:
            if rank != 0:
                micro_batchs = [None, None]
                nodeid = [None, None]
            dist.broadcast_object_list(
                nodeid, src=0, device=torch.device(rank))
            dist.broadcast_object_list(
                micro_batchs, src=0, device=torch.device(rank))
            micro_batch_n_id, micro_batch_size, micro_batch_adjs = micro_batchs[rank]
            if rank == 0:
                assert torch.equal(micro_batch_n_id, micro_batchs[rank][0])
                assert micro_batch_size == batch_size//micro_batch_num
                assert nodeid[1] == 1024
            micro_batch_adjs = [adj.to(rank)
                                for adj in micro_batch_adjs]  # load topo
            n_id = nodeid[0]
            mini_batch_batchsize = nodeid[1]
            out = model(x[n_id][micro_batch_n_id], micro_batch_adjs)  # forward
            subgraphout = model(
                x[n_id][micro_batchs[rank][0]], micro_batchs[rank][2])
            assert torch.abs((out - subgraphout).mean()) < 0.01
            target_node = n_id[:mini_batch_batchsize][rank *
                                                      micro_batch_size: (rank+1)*micro_batch_size]
            loss = F.nll_loss(
                out, y[target_node])
            loss.backward()
        dist.barrier()

        if rank == 0:
            assert loss < 300

        dist.barrier()

    dist.destroy_process_group()


def test_nccl():
    dataset = Reddit('/data/Reddit')
    world_size = 2  # torch.cuda.device_count()

    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)

    quiver_sampler = quiver.pyg.GraphSageSampler(
        csr_topo, sizes=[25, 10], device=0, mode='GPU')  # 这里是0, 但是spawn之后会变成fake,然后再lazy init 赋值

    mp.spawn(
        run,
        args=(world_size, data, data.x,
              quiver_sampler, dataset),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    test_nccl()