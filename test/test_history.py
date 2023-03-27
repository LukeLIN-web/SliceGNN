import torch
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn.conv import SAGEConv

from microGNN import History
from microGNN.models import SAGE, ScaleSAGE
from microGNN.prune import prune_computation_graph
from microGNN.utils import get_nano_batch
from microGNN.utils.common_class import Adj, Nanobatch


def test_same_out():
    # yapf: disable
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 6, 7],
                               [1, 6, 0, 2, 1, 7, 0, 2]], dtype=torch.long) # noqa
    # yapf: enable
    node_num = 8
    in_channels = 8
    num_hidden = 4
    out_channels = 2
    features = [[i for j in range(in_channels)] for i in range(node_num)]
    x = torch.tensor(features, dtype=torch.float)
    hop = [-1, -1]
    num_layers = len(hop)
    train_loader = NeighborSampler(
        edge_index,
        sizes=hop,
        batch_size=2,
        shuffle=False,
        drop_last=True,
    )

    torch.manual_seed(0)
    batch_size, n_id, adjs = next(iter(train_loader))
    model = ScaleSAGE(in_channels=in_channels,
                      hidden_channels=num_hidden,
                      out_channels=out_channels,
                      num_layers=num_layers)
    model.eval()
    nano_batchs = get_nano_batch(adjs,
                                 n_id,
                                 batch_size,
                                 num_nano_batch=2,
                                 relabel_nodes=True)
    histories = torch.nn.ModuleList(
        [History(len(n_id), num_hidden, 'cpu') for _ in range(num_layers - 1)])
    nb = nano_batchs[0]
    out = model(x[n_id][nb.n_id], nb, histories)
    model2 = SAGE(in_channels=in_channels,
                  hidden_channels=num_hidden,
                  out_channels=out_channels,
                  num_layers=num_layers)
    model2.load_state_dict(model.state_dict())
    model2.eval()
    for key, value1 in model.state_dict().items():
        value2 = model.state_dict()[key]
        assert torch.equal(value1, value2)
    out2 = model2(x[n_id][nb.n_id], nb.adjs)
    assert torch.abs((out - out2).mean()) < 0.01


def test_history_function():
    x = torch.tensor([[0,0],[1, 1],[2, 2],[3, 3],[4, 4],[5, 5],[6, 6],[7,7]],dtype=torch.float) # yapf: disable
    mb_n_id = torch.arange(8)
    edge1 = torch.tensor([[2, 3, 3, 4], [0, 0, 1, 1]])
    adjs1 = Adj(edge1, None, (5, 2))
    edge2 = torch.tensor([[2, 3, 3, 4, 5, 6, 7], [0, 0, 1, 1, 2, 3, 4]])
    adjs2 = Adj(edge2, None, (8, 5))
    adjs = [adjs2, adjs1]

    torch.manual_seed(23)
    num_layers = 2
    hidden_channels = 2
    in_channels = 2
    out_channels = 2
    convs = torch.nn.ModuleList()
    convs.append(
        SAGEConv(in_channels, hidden_channels, root_weight=False, bias=False))
    convs.append(
        SAGEConv(hidden_channels, out_channels, root_weight=False, bias=False))

    nano_batchs = get_nano_batch(adjs,
                                 mb_n_id,
                                 batch_size=2,
                                 num_nano_batch=2,
                                 relabel_nodes=True)
    histories = torch.nn.ModuleList([
        History(len(mb_n_id), hidden_channels, 'cpu')
        for _ in range(num_layers - 1)
    ])
    histories[0].cached_nodes = torch.tensor(
        [False, False, False, True, False, False, False, False])
    histories[0].emb[3] = torch.tensor([3.3, 3.4])  # should be pull
    nb = nano_batchs[1]
    pruned_adjs = prune_computation_graph(nb, histories)
    x = x[mb_n_id][nb.n_id]
    for i, (edge_index, _, size) in enumerate(pruned_adjs):
        h = convs[i](x, edge_index)  # compute the non cached nodes embedding
        non_empty_indices = (h != 0).nonzero()
        x[non_empty_indices] = h[non_empty_indices]
        if i != num_layers - 1:  # last layer is not saved
            history = histories[i]
            batch_size = nb.adjs[i].size[1]  # require 前size[0]个节点是 layer nodes
            history.pull(x, nb.n_id[:batch_size])
            assert torch.equal(x[1], torch.tensor([3.3, 3.4]))
            history.push(x[:batch_size], nb.n_id[:batch_size]
                         )  # Push all, including the ones just pulled.
            assert torch.equal(
                history.cached_nodes,
                torch.tensor(
                    [False, True, False, True, True, False, False, False]))


def test_pull_and_push():
    x = torch.tensor([[0,0],[1, 1],[2, 2],[3, 3],[4, 4],[5, 5],[6, 6],[7,7]],dtype=torch.float) # yapf: disable
    mb_n_id = torch.arange(8)
    edge1 = torch.tensor([[2, 3, 3, 4], [0, 0, 1, 1]])
    adjs1 = Adj(edge1, None, (5, 2))
    edge2 = torch.tensor([[2, 3, 3, 4, 5, 6, 7], [0, 0, 1, 1, 2, 3, 4]])
    adjs2 = Adj(edge2, None, (8, 5))
    adjs = [adjs2, adjs1]

    torch.manual_seed(23)
    num_layers = 2
    hidden_channels = 2
    in_channels = 2
    out_channels = 2
    convs = torch.nn.ModuleList()
    convs.append(
        SAGEConv(in_channels, hidden_channels, root_weight=False, bias=False))
    convs.append(
        SAGEConv(hidden_channels, out_channels, root_weight=False, bias=False))

    nano_batchs = get_nano_batch(adjs,
                                 mb_n_id,
                                 batch_size=2,
                                 num_nano_batch=2,
                                 relabel_nodes=True)
    histories = torch.nn.ModuleList([
        History(len(mb_n_id), hidden_channels, 'cpu')
        for _ in range(num_layers - 1)
    ])
    histories[0].cached_nodes = torch.tensor(
        [False, False, False, True, False, False, False, False])
    histories[0].emb[3] = torch.tensor([3.3, 3.4])  # should be pull
    nb = nano_batchs[1]
    pruned_adjs = prune_computation_graph(nb, histories)
    x = x[mb_n_id][nb.n_id]
    for i, (edge_index, _, size) in enumerate(pruned_adjs):
        h = convs[i](x, edge_index)  # compute the non cached nodes embedding
        non_empty_indices = (h != 0).nonzero()
        x[non_empty_indices] = h[non_empty_indices]
        if i != num_layers - 1:  # last layer is not saved
            history = histories[i]
            batch_size = nb.adjs[i].size[1]  # require 前size[0]个节点是 layer nodes
            for j, id in enumerate(nb.n_id[:batch_size]):
                if history.cached_nodes[id] == True:
                    assert j == 1
                    assert id == 3
                    x[j] = history.emb[id]
            history.push(x[:batch_size], nb.n_id[:batch_size]
                         )  # Push all, including the ones just pulled.
            assert torch.equal(
                history.cached_nodes,
                torch.tensor(
                    [False, True, False, True, True, False, False, False]))


def test_prune_computatition_graph():
    histories = torch.nn.ModuleList([History(5, 2, 'cpu') for _ in range(1)])
    histories[0].cached_nodes = torch.tensor([False, False, True, True, False])
    nb = Nanobatch(torch.arange(5), 5, [
        Adj(torch.tensor([[1, 2, 3, 4], [0, 0, 1, 2]]), None, (5, 3)),
        Adj(torch.tensor([[1, 2], [0, 0]]), None, (3, 1))
    ])
    pruned_adjs = prune_computation_graph(nb, histories)
    assert pruned_adjs[0].edge_index.tolist() == [[1, 2, 3], [0, 0, 1]]
    assert pruned_adjs[1].edge_index.tolist() == [[1, 2], [0, 0]]

    mb_n_id = torch.arange(8)
    edge1 = torch.tensor([[2, 3, 3, 4], [0, 0, 1, 1]])
    adjs1 = Adj(edge1, None, (5, 2))
    edge2 = torch.tensor([[2, 3, 3, 4, 5, 6, 7], [0, 0, 1, 1, 2, 3, 4]])
    adjs2 = Adj(edge2, None, (8, 5))
    adjs = [adjs2, adjs1]
    nano_batchs = get_nano_batch(adjs,
                                 mb_n_id,
                                 batch_size=2,
                                 num_nano_batch=2,
                                 relabel_nodes=True)
    histories = torch.nn.ModuleList(
        [History(len(mb_n_id), 2, 'cpu') for _ in range(1)])
    nb = nano_batchs[1]
    assert nb.n_id.tolist() == [1, 3, 4, 6, 7]
    histories[0].cached_nodes = torch.tensor(
        [False, False, True, True, False, False, False, False])
    pruned_adjs = prune_computation_graph(nb, histories)
    assert pruned_adjs[0].edge_index.tolist() == [[1, 2, 4], [0, 0, 2]]
    assert pruned_adjs[1].edge_index.tolist() == [[1, 2], [0, 0]]


def test_save_embedding():
    # yapf: disable
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 6, 7],
                               [1, 6, 0, 2, 1, 7, 0, 2]], dtype=torch.long) # noqa
    # yapf: enable
    x = torch.tensor([[0,0],[1, 1],[2, 2],[3, 3],[4, 4],[5, 5],[6, 6],[7,7]],dtype=torch.float) # yapf: disable
    hop = [-1, -1]
    num_layers = len(hop)
    train_loader = NeighborSampler(
        edge_index,
        sizes=hop,
        batch_size=2,
        shuffle=False,
        drop_last=True,
    )
    num_hidden = 2
    torch.manual_seed(0)
    batch_size, n_id, adjs = next(iter(train_loader))
    model = ScaleSAGE(in_channels=2,
                      hidden_channels=num_hidden,
                      out_channels=2,
                      num_layers=num_layers)
    model.eval()
    nano_batchs = get_nano_batch(adjs,
                                 n_id,
                                 batch_size,
                                 num_nano_batch=2,
                                 relabel_nodes=True)
    histories = torch.nn.ModuleList(
        [History(len(n_id), num_hidden, 'cpu') for _ in range(num_layers - 1)])
    nb = nano_batchs[0]
    model(x[n_id][nb.n_id], nb, histories)
    assert torch.equal(histories[0].emb[3],
                       torch.tensor([0.0, 0.0]))  # node 2 don't save
    assert torch.equal(histories[0].cached_nodes,
                       torch.tensor([True, True, True, False, False]))
    histories[0].reset_parameters()

    nb = nano_batchs[1]
    model(x[n_id][nb.n_id], nb, histories)
    assert torch.equal(histories[0].emb[2],
                       torch.tensor([0.0, 0.0]))  # node 6 don't save
    assert torch.equal(histories[0].cached_nodes,
                       torch.tensor([True, True, False, True, False]))


if __name__ == "__main__":
    # test_prune_computatition_graph()
    # test_save_embedding()
    # test_load_embedding()
    test_pull_and_push()
