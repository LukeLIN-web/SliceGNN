import os
from statistics import mean

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv

from timeit import default_timer
import argparse
from get_micro_batch import *


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2):
        super().__init__()
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
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


def run(rank, world_size, dataset, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    data = dataset[0]
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(train_idx.size(
        0) // world_size)[rank]  # train idx shape is ?

    train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                                   sizes=[25, 10], batch_size=1024,
                                   shuffle=True, num_workers=14, drop_last=True)
    # 第一层每个node 25个neibor, 第二层每个node 访问10个.
    if rank == 0:
        subgraph_loader = NeighborSampler(data.edge_index, node_idx=None,
                                          sizes=[-1], batch_size=16,
                                          shuffle=False, num_workers=0)

    torch.manual_seed(12345)
    model = SAGE(dataset.num_features, 256, dataset.num_classes).to(rank)
    # guartee  final  allreduce avg gradient
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x, y = data.x.to(rank), data.y.to(rank)

    for epoch in range(1, 4):
        model.train()
        loadtimes,  gputimes = [], []
        get_micro_batch_times = []
        start = default_timer()
        for batch_size, n_id, adjs in train_loader:
            dataloadtime = default_timer()
            micro_batchs = get_micro_batch(adjs,
                                           n_id,
                                           batch_size, args.num_micro_batch)
            get_micro_batch_time = default_timer()
            optimizer.zero_grad()
            for i, micro_batch in enumerate(micro_batchs):
                adjs = [adj.to(rank) for adj in micro_batch.adjs]  # load topo
                out = model(x[n_id][micro_batch.nid], adjs)  # forward
                loss = F.nll_loss(
                    out, y[n_id[:batch_size]][i * micro_batch.batch_size: (i+1)*micro_batch.batch_size])
                loss.backward()
            optimizer.step()
            stop = default_timer()
            get_micro_batch_times.append(get_micro_batch_time - dataloadtime)
            loadtimes.append(dataloadtime - start)
            gputimes.append(stop - dataloadtime)
            start = default_timer()

        dist.barrier()

        if rank == 0:
            avg_get_micro_batch_times = round(
                mean(get_micro_batch_times[5:-5]), 3)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            avggputime = round(mean(gputimes[5:-5]), 3)
            avgloadtime = round(mean(loadtimes[5:-5]), 3)
            print(f'avg_get_micro_batch_times={avg_get_micro_batch_times}, batchloadtime={avgloadtime}, '
                  f' gputime = {avggputime} ')
            print(f'batch size={batch_size}, batchloadtime={avgloadtime}, '
                  f' gputime = {avggputime} ')

        if rank == 0 and epoch % 1 == 0:  # We evaluate on a single GPU for now
            model.eval()
            with torch.no_grad():
                out = model.module.inference(x, rank, subgraph_loader)
            res = out.argmax(dim=-1) == data.y
            acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
            acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
            acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
            print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')

        dist.barrier()

    dist.destroy_process_group()


# gpu =2 , acc = 0.9590 没有降低
if __name__ == '__main__':
    datapath = "/root/share/data/Reddit"
    dataset = Reddit(datapath)

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--num_micro_batch', type=int, default=4)
    args = parser.parse_args()

    # world_size = torch.cuda.device_count()
    world_size = args.gpus  # small case for debug
    print('Let\'s use', args.gpus, 'GPUs!')
    mp.spawn(run, args=(world_size, dataset, args),
             nprocs=world_size, join=True)