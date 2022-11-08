import os
from statistics import mean

import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler

######################
# Import From Quiver
######################
import quiver
from timeit import default_timer
from get_micro_batch import *
import argparse


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2):
        super(SAGE, self).__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    @torch.no_grad()
    def inference(self, x_all, device, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x)

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


def run(rank, world_size, data_split, edge_index, x, quiver_sampler: quiver.pyg.GraphSageSampler, y, num_features, num_classes):
    LOCAL_ADDR = 'localhost'
    MASTER_ADDR = 'localhost'
    MASTER_PORT = 12355
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] =  MASTER_PORT
    tcpstore = dist.TCPStore(MASTER_ADDR,  MASTER_PORT, world_size,
                             MASTER_ADDR == LOCAL_ADDR)
    dist.init_process_group('nccl', store= tcpstore, rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)

    train_mask, val_mask, test_mask = data_split
    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

    train_loader = torch.utils.data.DataLoader(
        train_idx, batch_size=1024, shuffle=True, drop_last=True)

    torch.manual_seed(12345)
    model = SAGE(num_features, 256, num_classes).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])

    for epoch in range(1, 6):
        model.train()
        for seeds in train_loader:
            a = 2
            
        dist.barrier()


    dist.destroy_process_group()


if __name__ == '__main__':
    dataset = Reddit('/data/Reddit')
    world_size = torch.cuda.device_count()
    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--num_micro_batch', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()

    ##############################
    # Create Sampler And Feature
    ##############################
    quiver_sampler = quiver.pyg.GraphSageSampler(
        csr_topo, sizes=[25, 10], device=0, mode='GPU')  # 这里是0, 但是spawn之后会变成fake,然后再lazy init 赋值
    # cache feature 到rank0
    quiver_feature = quiver.Feature(rank=0, device_list=list(range(
        world_size)), device_cache_size="2G", cache_policy="device_replicate", csr_topo=csr_topo)
    quiver_feature.from_cpu_tensor(data.x)

    print('Let\'s use', world_size, 'GPUs!')
    data_split = (data.train_mask, data.val_mask, data.test_mask)
    mp.spawn(
        run,
        args=(world_size, data_split, data.edge_index, quiver_feature,
              quiver_sampler, data.y, dataset.num_features, dataset.num_classes, args),
        nprocs=world_size,
        join=True
    )
