import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, NeighborSampler

from microGNN.models import SAGE, newSAGE
from microGNN.utils import get_loader_nano_batch, get_nano_batch, slice_adj

hop = [-1, -1]
num_layers = 2
in_channels = 8
hidden_channels = 4
out_channels = 2
node_num = 8
features = [[i for j in range(in_channels)] for i in range(node_num)]
edge_index = torch.tensor([[2, 3, 3, 4, 5, 6, 7], [0, 0, 1, 1, 2, 3, 4]],
                          dtype=torch.long)
data = Data(x=torch.tensor(features, dtype=torch.float), edge_index=edge_index)


def test_loader_mapping():
    loader = NeighborLoader(data, hop, batch_size=2)
    batch = next(iter(loader))
    nano_batchs = get_loader_nano_batch(batch, num_nano_batch=2, hop=2)
    assert nano_batchs[0].n_id.tolist() == [0, 2, 3, 5, 6]
    assert nano_batchs[1].n_id.tolist() == [1, 3, 4, 6, 7]
    assert torch.equal(nano_batchs[0].edge_index,
                       torch.tensor([[1, 2, 3, 4], [0, 0, 1, 2]]))
    assert torch.equal(nano_batchs[1].edge_index,
                       torch.tensor([[1, 2, 3, 4], [0, 0, 1, 2]]))


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
    assert nano_batchs[0].n_id.tolist() == [0, 2, 3, 5, 6]
    assert nano_batchs[1].n_id.tolist() == [1, 3, 4, 6, 7]
    assert n_id[nano_batchs[0].n_id].tolist() == [0, 2, 3, 5, 6]
    assert n_id[nano_batchs[1].n_id].tolist() == [1, 3, 4, 6, 7]
    assert torch.equal(nano_batchs[1].adjs[0].edge_index,
                       torch.tensor([[1, 2, 3, 4], [0, 0, 1, 2]]))
    # TODO: [1,1] is not the target nodes, it is potential problem
    loader = NeighborLoader(data, hop, batch_size=2)
    batch = next(iter(loader))
    nano_batchs = get_loader_nano_batch(batch, num_nano_batch=2, hop=2)
    assert nano_batchs[0].n_id.tolist() == [0, 2, 3, 5, 6]
    assert nano_batchs[1].n_id.tolist() == [1, 3, 4, 6, 7]
    assert batch.n_id[nano_batchs[0].n_id].tolist() == [0, 2, 3, 5, 6]
    assert batch.n_id[nano_batchs[1].n_id].tolist() == [1, 3, 4, 6, 7]
    assert torch.equal(nano_batchs[1].edge_index,
                       torch.tensor([[1, 2, 3, 4], [0, 0, 1, 2]]))


def test_loader_forward():
    # hop = [-1, -1, -1]  # three hop
    # train_loader = NeighborLoader(data, hop, batch_size=2)
    # model = newSAGE(in_channels, hidden_channels, out_channels, num_layers=3)
    # model.eval()
    # for batch in train_loader:
    #     out = model(batch.x, batch.edge_index)[:batch.batch_size]
    #     nano_batchs = get_loader_nano_batch(batch, num_nano_batch=2, hop=2)
    #     subgraphout = []
    #     for nb in nano_batchs:
    #         subgraphout.append(model(batch.x[batch.n_id][nb.n_id], nb.edge_index)[:nb.batch_size] )
    #     subgraphout = torch.cat(subgraphout, 0)
    #     assert torch.abs((out - subgraphout).mean()) < 0.01

    # two hop
    hop = [-1, -1]
    train_loader = NeighborLoader(data, hop, batch_size=2, drop_last=True)
    model = newSAGE(in_channels, hidden_channels, out_channels)
    model.eval()
    for batch in train_loader:
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        nano_batchs = get_loader_nano_batch(batch, num_nano_batch=2, hop=2)
        subgraphout = []
        for nb in nano_batchs:
            print(nb.n_id)
            print(batch.n_id[nb.n_id])
            print(nb.edge_index)
            subgraphout.append(
                model(batch.x[nb.n_id], nb.edge_index)[:nb.batch_size])
        subgraphout = torch.cat(subgraphout, 0)
        assert torch.abs((out - subgraphout).mean()) < 0.01

    # one hop
    train_loader = NeighborLoader(data, hop, batch_size=2, drop_last=True)
    for batch in train_loader:
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        nano_batchs = get_loader_nano_batch(batch, num_nano_batch=2, hop=2)
        subgraphout = []
        for nb in nano_batchs:
            subgraphout.append(
                model(batch.x[nb.n_id], nb.edge_index)[:nb.batch_size])
        subgraphout = torch.cat(subgraphout, 0)
        assert torch.abs((out - subgraphout).mean()) < 0.01


def test_forward():
    # yapf: disable
    edge_index = torch.tensor([
        [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9], # noqa
        [1, 6, 0, 2, 6, 1, 3, 7, 2, 4, 8, 3, 5, 9, 4, 9, 0, 1, 7, 2, 6, 8, 3, 7, 9, 4, 5, 8]],dtype=torch.long)  # noqa
    x = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], # noqa
                    [8, 3], [9, 3], [10, 3]],dtype=torch.float) # noqa
    num_features =2
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
    model = SAGE(num_features, hidden_channels, out_channels, num_layers=3)
    model.eval()
    for batch_size, n_id, adjs in train_loader:
        out = model(x[n_id], adjs)
        num_micro_batch = 4
        nano_batchs = get_nano_batch(adjs, n_id, batch_size, num_micro_batch)
        subgraphout = []
        for nb in nano_batchs:
            subgraphout.append(model(x[n_id][nb.n_id], nb.adjs))
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
    model = SAGE(num_features, hidden_channels, out_channels)
    model.eval()
    for batch_size, n_id, adjs in train_loader:
        out = model(x[n_id], adjs)
        num_micro_batch = 4
        nano_batchs = get_nano_batch(adjs, n_id, batch_size, num_micro_batch)
        subgraphout = []
        for nb in nano_batchs:
            subgraphout.append(model(x[n_id][nb.n_id], nb.adjs))
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
            subgraphout.append(model(x[n_id][nb.n_id], nb.adjs))
        subgraphout = torch.cat(subgraphout, 0)
        assert torch.abs((out - subgraphout).mean()) < 0.01


if __name__ == "__main__":
    # test_mapping()
    # test_slice_adj()
    test_loader_mapping()
