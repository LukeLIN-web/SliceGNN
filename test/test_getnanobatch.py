import torch
from torch import Tensor
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn.conv import SAGEConv

from microGNN.models import SAGE
from microGNN.utils import get_nano_batch, get_nano_batch_histories, slice_adj
from microGNN.utils.common_class import Adj, Nanobatch

hop = [-1, -1]
num_layers = 2
in_channels = 8
hidden_channels = 4
out_channels = 2
node_num = 8
features = [[i for j in range(in_channels)] for i in range(node_num)]
mb_n_id = torch.arange(node_num)


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


def test_get_nano_batch_histories():
    n_id = torch.arange(node_num)
    edge1 = torch.tensor([[2, 3, 3, 4], [0, 0, 1, 1]])
    adjs1 = Adj(edge1, None, (5, 2))
    edge2 = torch.tensor([[2, 3, 3, 4, 5, 6, 7], [0, 0, 1, 1, 2, 3, 4]])
    adjs2 = Adj(edge2, None, (8, 5))
    adjs = [adjs2, adjs1]
    num_nano_batch = 2
    batch_size = 2

    mod = batch_size % num_nano_batch
    if mod != 0:
        batch_size -= mod
    assert batch_size % num_nano_batch == 0, "batch_size must be divisible by num_nano_batch"
    assert isinstance(adjs, list), "adjs must be a list"
    adjs.reverse()
    nano_batch_size = batch_size // num_nano_batch
    nano_batchs = []
    cached_id = [[] for i in range(num_layers)]
    cached_nodes = torch.full((num_layers - 1, node_num),
                              False,
                              dtype=torch.bool)
    for i in range(num_nano_batch):
        sub_nid = n_id[i * nano_batch_size:(i + 1) * nano_batch_size]
        subadjs = []
        for j, adj in enumerate(adjs):
            target_size = len(sub_nid)
            sub_nid, sub_adjs, edge_mask = slice_adj(
                sub_nid,
                adj.edge_index,
                relabel_nodes=True,
            )
            if j != num_layers - 1:
                for id in sub_nid:
                    if cached_nodes[j][id] == False:
                        cached_nodes[j][id] = True
                    elif cached_nodes[j][id] == True:
                        cached_id[j].append(id)
            subadjs.append(Adj(sub_adjs, None, (len(sub_nid), target_size)))
        subadjs.reverse()  # O(n) 大的在前面
        nano_batchs.append(Nanobatch(sub_nid, nano_batch_size, subadjs))
    assert cached_id[0] == [3]


def test_cache_id():
    edge1 = torch.tensor([[2, 3, 3, 4], [0, 0, 1, 1]])
    adjs1 = Adj(edge1, None, (5, 2))
    edge2 = torch.tensor([[2, 3, 3, 4, 5, 6, 7], [0, 0, 1, 1, 2, 3, 4]])
    adjs2 = Adj(edge2, None, (8, 5))
    adjs = [adjs2, adjs1]
    convs = torch.nn.ModuleList()
    convs.append(
        SAGEConv(in_channels, hidden_channels, root_weight=False, bias=False))
    convs.append(
        SAGEConv(hidden_channels, out_channels, root_weight=False, bias=False))

    nano_batchs, cached_id = get_nano_batch_histories(adjs,
                                                      mb_n_id,
                                                      batch_size=2,
                                                      num_nano_batch=2)
    assert len(cached_id) == 1
    assert cached_id[0] == torch.tensor(3)


def test_mapping():
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 6, 7], [1, 6, 0, 2, 7, 1, 0, 2]], dtype=torch.long)
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
    test_slice_adj()
