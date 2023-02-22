from torch import Tensor
from typing import List
from torch_geometric.loader import NeighborSampler
import torch
from microGNN.utils.get_micro_batch import *
from microGNN.utils.calu_similarity import *
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import quiver
# three hop
edge_index = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9],
                           [1, 6, 0, 2, 6, 1, 3, 7, 2, 4, 8, 3, 5, 9, 4, 9, 0, 1, 7, 2, 6, 8, 3, 7, 9, 4, 5, 8]], dtype=torch.long)
x = Tensor([[1, 2], [2, 3], [3, 3], [4, 3], [5, 3],
            [6, 3], [7, 3], [8, 3], [9, 3], [10, 3]])
num_features, hidden_size, num_classes = 2, 16, 1


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
        assert len(adjs[0]) == 3
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
        return x


def test_overlap():
    train_loader = NeighborSampler(edge_index,
                                   sizes=[-1], batch_size=6,
                                   shuffle=False, num_workers=6, drop_last=True)
    model = SAGE(num_features, hidden_size, num_classes)
    # one hop
    train_loader = NeighborSampler(edge_index,
                                   sizes=[-1], batch_size=6,
                                   shuffle=False, num_workers=6, drop_last=True)
    for batch_size, n_id, adjs in train_loader:
        if isinstance(adjs[0], Tensor):
            # when hop = 1 , adjs is a EdgeIndex, we need convert it to list.
            adjs = [adjs]
        num_micro_batch = 2
        micro_batchs = get_micro_batch(adjs,
                                       n_id,
                                       batch_size, num_micro_batch)
        for i in range(num_micro_batch - 1):
            print(micro_batchs[i].n_id)
            print(micro_batchs[i+1].n_id)
            print(n_id[micro_batchs[i].n_id])
            print(n_id[micro_batchs[i+1].n_id])
            similarity1 = Jaccard(
                n_id[micro_batchs[i].n_id], n_id[micro_batchs[i+1].n_id])
            similarity2 = Ochiai(
                n_id[micro_batchs[i].n_id], n_id[micro_batchs[i+1].n_id])
            print(similarity1, similarity2)


def test_nodeid():
    hop = [1, 1]
    train_idx = torch.arange(0, 10, dtype=torch.int64)
    train_loader = torch.utils.data.DataLoader(
        train_idx, batch_size=2, shuffle=False, drop_last=True)
    csr_topo = quiver.CSRTopo(edge_index)  # Quiver
    quiver_sampler = quiver.pyg.GraphSageSampler(
        csr_topo, sizes=hop, device=0, mode='CPU')  # Quiver

    for seeds in train_loader:  # Quiver
        n_id, batch_size, adjs = quiver_sampler.sample(seeds)  # Quiver
        print(adjs)
        num_micro_batch = 2
        micro_batchs = get_micro_batch(adjs,
                                       n_id,
                                       batch_size, num_micro_batch)
        for i in range(num_micro_batch):
            print(micro_batchs[i])
        
        # exit()

def test_get_micro_batch():
    # three hop
    hop = [-1, -1, -1]
    train_loader = NeighborSampler(edge_index,
                                   sizes=hop, batch_size=4,
                                   shuffle=False, num_workers=0, drop_last=True)
    model = SAGE(num_features, hidden_size, num_classes, num_layers=3)
    for batch_size, n_id, adjs in train_loader:
        out = model(x[n_id], adjs)
        num_micro_batch = 4
        micro_batchs = get_micro_batch(adjs,
                                       n_id,
                                       batch_size, num_micro_batch)
        subgraphout = []
        for micro_batch in micro_batchs:
            subgraphout.append(
                model(x[n_id][micro_batch[0]], micro_batch[2]))
        subgraphout = torch.cat(subgraphout, 0)
        assert torch.abs((out - subgraphout).mean()) < 0.01

    # two hop
    hop = [-1, -1]
    train_loader = NeighborSampler(edge_index,
                                   sizes=hop, batch_size=4,
                                   shuffle=False, num_workers=6, drop_last=True)
    model = SAGE(num_features, hidden_size, num_classes)
    for batch_size, n_id, adjs in train_loader:
        out = model(x[n_id], adjs)
        num_micro_batch = 4
        micro_batchs = get_micro_batch(adjs,
                                       n_id,
                                       batch_size, num_micro_batch)
        subgraphout = []
        for micro_batch in micro_batchs:
            subgraphout.append(
                model(x[n_id][micro_batch[0]], micro_batch[2]))
        subgraphout = torch.cat(subgraphout, 0)
        assert torch.abs((out - subgraphout).mean()) < 0.01

    # one hop
    train_loader = NeighborSampler(edge_index,
                                   sizes=[-1], batch_size=6,
                                   shuffle=False, num_workers=6, drop_last=True)
    for batch_size, n_id, adjs in train_loader:
        if isinstance(adjs[0], Tensor):
            # when hop = 1 , adjs is a EdgeIndex, we need convert it to list.
            adjs = [adjs]
        num_micro_batch = 2
        micro_batchs = get_micro_batch(adjs,
                                       n_id,
                                       batch_size, num_micro_batch)
        out = model(x[n_id], adjs)
        subgraphout = []
        for micro_batch in micro_batchs:
            subgraphout.append(
                model(x[n_id][micro_batch[0]], micro_batch[2]))
        subgraphout = torch.cat(subgraphout, 0)
        assert torch.abs((out - subgraphout).mean()) < 0.01


if __name__ == '__main__':
    # test_get_micro_batch()
    # test_overlap()
    test_nodeid()
