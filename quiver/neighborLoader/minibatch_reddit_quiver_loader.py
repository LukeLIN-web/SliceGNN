import os

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader

import copy
from timeit import default_timer
import quiver
import argparse


# tested Neighborsampler  , very slower than origin quiver sampler. 

class SAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all: Tensor, device: torch.device,
                  subgraph_loader: NeighborLoader) -> Tensor:

        # pbar = tqdm(total=len(subgraph_loader) * len(self.convs))
        # pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.node_id.to(x_all.device)]
                x = conv(x, batch.edge_index)
                x = x[:batch.batch_size]
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x.cpu())
                # pbar.update(1)
            x_all = torch.cat(xs, dim=0)

        # pbar.close()
        return x_all


def run(rank, world_size, data, x, num_features: int, num_classes: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)
    data = data.to(rank, 'y')  # Move to device for faster feature fetch.

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

    kwargs = dict(batch_size=1024, num_workers=4, persistent_workers=True)
    train_loader = NeighborLoader(data, input_nodes=train_idx,
                                  num_neighbors=[25, 10], shuffle=True,
                                  drop_last=True, **kwargs)

    if rank == 0:  # Create single-hop evaluation neighbor loader:
        subgraph_loader = NeighborLoader(copy.copy(data), num_neighbors=[-1],
                                         shuffle=False, **kwargs)
        # No need to maintain these features during evaluation:
        del subgraph_loader.data.x, subgraph_loader.data.y
        # Add global node index information:
        subgraph_loader.data.node_id = torch.arange(data.num_nodes)

    torch.manual_seed(12345)
    model = SAGE(num_features, 256, num_classes).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 21):
        model.train()
        epoch_start = default_timer()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)[:batch.batch_size]
            loss = F.cross_entropy(out, batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()

        dist.barrier()

        if rank == 0:
            print(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {default_timer() - epoch_start:.4f}')

        if rank == 0 and epoch % 5 == 0:  # We evaluate on a single GPU for now
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


if __name__ == '__main__':
    dataset = Reddit('/data/Reddit')
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--gpus', type=int, default=2)
    args = parser.parse_args()
    world_size = args.gpus

    data = dataset[0]

    csr_topo = quiver.CSRTopo(data.edge_index)

    # cache feature 到rank0
    quiver_feature = quiver.Feature(rank=0, device_list=list(range(
        world_size)), device_cache_size="2G", cache_policy="device_replicate", csr_topo=csr_topo)
    quiver_feature.from_cpu_tensor(data.x)

    print('Let\'s use', world_size, 'GPUs!')

    mp.spawn(
        run,
        args=(world_size,  data, quiver_feature,
              dataset.num_features, dataset.num_classes),
        nprocs=world_size,
        join=True
    )