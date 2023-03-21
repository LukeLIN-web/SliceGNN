import torch
from torch import Tensor
from torch_geometric.loader import NeighborSampler

from microGNN.models import SAGE
from microGNN.utils import get_nano_batch, slice_adj

num_features, hidden_size, num_classes = 2, 16, 1


def test_slice_adj():
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5],
        [2, 2, 4, 4, 6, 6],
    ])

    subset, edge_index, edge_mask = slice_adj(6,
                                              edge_index,
                                              relabel_nodes=True)
    assert subset.tolist() == [6, 4, 5]
    assert edge_index.tolist() == [[1, 2], [0, 0]]
    assert edge_mask.tolist() == [False, False, False, False, True, True]

    edge_index = torch.tensor([
        [1, 2, 4, 5],
        [0, 1, 5, 6],
    ])

    subset, edge_index, edge_mask = slice_adj([0, 6],
                                              edge_index,
                                              relabel_nodes=True)

    assert subset.tolist() == [0, 6, 1, 5]
    assert edge_index.tolist() == [[2, 3], [0, 1]]
    assert edge_mask.tolist() == [True, False, False, True]


def test_mapping():
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 6, 7], [1, 6, 0, 2, 7, 1, 0, 2]], dtype=torch.long)
    hop = [-1, -1]
    train_loader = NeighborSampler(
        edge_index,
        sizes=hop,
        batch_size=2,
        shuffle=False,
        drop_last=True,
    )
    batch_size, n_id, adjs = next(iter(train_loader))
    nano_batchs = get_nano_batch(adjs,
                                 n_id,
                                 batch_size,
                                 num_nano_batch=2,
                                 relabel_nodes=True)
    assert nano_batchs[0].n_id.tolist() == [0, 1, 2, 3]
    assert nano_batchs[1].n_id.tolist() == [1, 0, 3, 2, 4]
    assert n_id[nano_batchs[0].n_id].tolist() == [0, 1, 6, 2]
    assert n_id[nano_batchs[1].n_id].tolist() == [1, 0, 2, 6, 7]
    assert torch.equal(nano_batchs[1].adjs[0].edge_index,
                       torch.tensor([[0, 3, 1, 2, 0, 4], [1, 1, 0, 0, 2, 2]]))
    # TODO: [1,1] is not the target nodes, it is potential problem


def test_nano_batch():
    # yapf: disable
    edge_index = torch.tensor([
        [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9], # noqa
        [1, 6, 0, 2, 6, 1, 3, 7, 2, 4, 8, 3, 5, 9, 4, 9, 0, 1, 7, 2, 6, 8, 3, 7, 9, 4, 5, 8]],dtype=torch.long)  # noqa
    x = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], # noqa
                    [8, 3], [9, 3], [10, 3]],dtype=torch.float) # noqa
    # yapf: enable
    hop = [-1, -1, -1]  # three hop
    train_loader = NeighborSampler(
        edge_index,
        sizes=hop,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    model = SAGE(num_features, hidden_size, num_classes, num_layers=3)
    model.eval()
    for batch_size, n_id, adjs in train_loader:
        out = model(x[n_id], adjs)
        num_micro_batch = 4
        nano_batchs = get_nano_batch(adjs, n_id, batch_size, num_micro_batch)
        subgraphout = []
        for nb in nano_batchs:
            subgraphout.append(model(x[n_id][nb[0]], nb[2]))
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
    model.eval()
    for batch_size, n_id, adjs in train_loader:
        out = model(x[n_id], adjs)
        num_micro_batch = 4
        nano_batchs = get_nano_batch(adjs, n_id, batch_size, num_micro_batch)
        subgraphout = []
        for nb in nano_batchs:
            subgraphout.append(model(x[n_id][nb[0]], nb[2]))
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
        nano_batchs = get_nano_batch(adjs, n_id, batch_size, num_micro_batch)
        out = model(x[n_id], adjs)
        subgraphout = []
        for nb in nano_batchs:
            subgraphout.append(model(x[n_id][nb[0]], nb[2]))
        subgraphout = torch.cat(subgraphout, 0)
        assert torch.abs((out - subgraphout).mean()) < 0.01


if __name__ == "__main__":
    # test_get_nano_batch()
    test_mapping()
    # test_slice_adj()
