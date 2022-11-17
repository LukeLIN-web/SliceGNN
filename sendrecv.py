import os
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
from utils.get_micro_batch import *

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'


def broadcast_obj(rank, world_size, data, x, quiver_sampler: quiver.pyg.GraphSageSampler, dataset):
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

    train_loader = torch.utils.data.DataLoader(
        train_idx, batch_size=1024, shuffle=True, drop_last=True)

    torch.manual_seed(12345)
    model = SAGE(dataset.num_features, 256, dataset.num_classes).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])

    for epoch in range(1, 6):
        # model.train()
        # epoch_start = default_timer()
        for seeds in train_loader:
            # Note: Process group initialization omitted on each rank.
            tensor_size = 2
            t_ones = torch.ones(tensor_size)
            t_fives = torch.ones(tensor_size) * 5
            output_tensor = torch.zeros(tensor_size)
            if dist.get_rank() == 0:
                # Assumes world_size of 2.
                # Only tensors, all of which must be the same size.
                scatter_list = [t_ones, t_fives]
            else:
                scatter_list = None
            dist.scatter(output_tensor, scatter_list, src=0)
            # Rank i gets scatter_list[i]. For example, on rank 1:
            print(output_tensor)
            


def test_broadcast_obj():
    dataset = Reddit('/data/Reddit')
    world_size = torch.cuda.device_count()
    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)

    quiver_sampler = quiver.pyg.GraphSageSampler(
        csr_topo, sizes=[25, 10], device=0, mode='GPU')  # 这里是0, 但是spawn之后会变成fake,然后再lazy init 赋值

    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(
        broadcast_obj,
        args=(world_size, data, None,
              quiver_sampler, dataset),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    test_broadcast_obj()