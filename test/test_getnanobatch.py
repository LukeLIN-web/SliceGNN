from torch import Tensor
from typing import List
from torch_geometric.loader import NeighborSampler
import torch
from microGNN.utils.get_nano_batch import *
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

edge_index = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9],
                           [1, 6, 0, 2, 6, 1, 3, 7, 2, 4, 8, 3, 5, 9, 4, 9, 0, 1, 7, 2, 6, 8, 3, 7, 9, 4, 5, 8]], dtype=torch.long)  # fmt: skip
x = Tensor(
    [[1, 2], [2, 3], [3, 3], [4, 3], [5, 3], [6, 3], [7, 3], [8, 3], [9, 3], [10, 3]]
)
num_features, hidden_size, num_classes = 2, 16, 1


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
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
            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
        return x


def test_slice_adj():
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [2, 2, 4, 4, 6, 6],
        ]
    )

    subset, edge_index, edge_mask = slice_adj(6, edge_index, relabel_nodes=True)
    assert subset.tolist() == [6, 4, 5]
    assert edge_index.tolist() == [[1, 2], [0, 0]]
    assert edge_mask.tolist() == [False, False, False, False, True, True]

    edge_index = torch.tensor(
        [
            [1, 2, 4, 5],
            [0, 1, 5, 6],
        ]
    )

    subset, edge_index, edge_mask = slice_adj([0, 6], edge_index, relabel_nodes=True)

    assert subset.tolist() == [0, 6, 1, 5]
    assert edge_index.tolist() == [[2, 3], [0, 1]]
    assert edge_mask.tolist() == [True, False, False, True]


def test_get_nano_batch():
    # three hop
    hop = [-1, -1, -1]
    train_loader = NeighborSampler(
        edge_index,
        sizes=hop,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    model = SAGE(num_features, hidden_size, num_classes, num_layers=3)
    for batch_size, n_id, adjs in train_loader:
        out = model(x[n_id], adjs)
        num_micro_batch = 4
        micro_batchs = get_nano_batch(adjs, n_id, batch_size, num_micro_batch)
        subgraphout = []
        for micro_batch in micro_batchs:
            subgraphout.append(model(x[n_id][micro_batch[0]], micro_batch[2]))
        subgraphout = torch.cat(subgraphout, 0)
        assert torch.abs((out - subgraphout).mean()) < 0.01

    # two hop
    hop = [-1, -1]
    train_loader = NeighborSampler(
        edge_index,
        sizes=hop,
        batch_size=4,
        shuffle=False,
        num_workers=6,
        drop_last=True,
    )
    model = SAGE(num_features, hidden_size, num_classes)
    for batch_size, n_id, adjs in train_loader:
        out = model(x[n_id], adjs)
        num_micro_batch = 4
        micro_batchs = get_nano_batch(adjs, n_id, batch_size, num_micro_batch)
        subgraphout = []
        for micro_batch in micro_batchs:
            subgraphout.append(model(x[n_id][micro_batch[0]], micro_batch[2]))
        subgraphout = torch.cat(subgraphout, 0)
        assert torch.abs((out - subgraphout).mean()) < 0.01

    # one hop
    train_loader = NeighborSampler(
        edge_index,
        sizes=[-1],
        batch_size=6,
        shuffle=False,
        num_workers=6,
        drop_last=True,
    )
    for batch_size, n_id, adjs in train_loader:
        if isinstance(adjs[0], Tensor):
            # when hop = 1 , adjs is a EdgeIndex, we need convert it to list.
            adjs = [adjs]
        num_micro_batch = 2
        micro_batchs = get_nano_batch(adjs, n_id, batch_size, num_micro_batch)
        out = model(x[n_id], adjs)
        subgraphout = []
        for micro_batch in micro_batchs:
            subgraphout.append(model(x[n_id][micro_batch[0]], micro_batch[2]))
        subgraphout = torch.cat(subgraphout, 0)
        assert torch.abs((out - subgraphout).mean()) < 0.01


if __name__ == "__main__":
    # test_get_nano_batch()
    test_slice_adj()
