import torch
from torch_geometric.nn.conv import SAGEConv

from microGNN import History
from microGNN.prune import prune_computation_graph
from microGNN.utils import get_nano_batch, get_nano_batch_histories
from microGNN.utils.common_class import Adj

hop = [-1, -1]
num_layers = 2
in_channels = 8
hidden_channels = 4
out_channels = 2
node_num = 8
features = [[i for j in range(in_channels)] for i in range(node_num)]


def test_forward():
    mb_n_id = torch.arange(8)
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
                                                      num_nano_batch=2,
                                                      relabel_nodes=True)
    histories = torch.nn.ModuleList([
        History(cacheid, node_num, hidden_channels, 'cpu')
        for cacheid in cached_id
    ])
    histories[0].cached_nodes = torch.tensor(
        [False, False, False, True, False, False, False, False])
    x = torch.tensor(features, dtype=torch.float)
    nb = nano_batchs[0]
    x = x[mb_n_id][nb.n_id]
    pruned_adjs = prune_computation_graph(nb.n_id, nb.adjs, histories)

    for i, adj in enumerate(pruned_adjs):
        batch_size = nb.adjs[i].size[1]
        x_target = x[:batch_size]  # require 前size[0]个节点是 layer nodes
        x = convs[i]((x, x_target),
                     adj.edge_index)  # compute the non cached nodes embedding
        if i != num_layers - 1:  # last layer is not saved
            assert torch.equal(x[2], torch.zeros(4))

    nb = nano_batchs[1]
    pruned_adjs = prune_computation_graph(nb.n_id, nb.adjs, histories)
    x = torch.tensor(features, dtype=torch.float)
    x = x[mb_n_id][nb.n_id]
    for i, adj in enumerate(pruned_adjs):
        batch_size = nb.adjs[i].size[1]
        x_target = x[:batch_size]  # require 前size[0]个节点是 layer nodes
        x = convs[i]((x, x_target),
                     adj.edge_index)  # compute the non cached nodes embedding
        if i != num_layers - 1:  # last layer is not saved
            assert torch.equal(x[1], torch.zeros(4))


def test_prune_computatition_graph():
    mb_n_id = torch.arange(8)
    edge1 = torch.tensor([[2, 3, 3, 4], [0, 0, 1, 1]])
    adjs1 = Adj(edge1, None, (5, 2))
    edge2 = torch.tensor([[2, 3, 3, 4, 5, 6, 7], [0, 0, 1, 1, 2, 3, 4]])
    adjs2 = Adj(edge2, None, (8, 5))
    adjs = [adjs2, adjs1]

    nano_batchs, cached_id = get_nano_batch_histories(adjs,
                                                      mb_n_id,
                                                      batch_size=2,
                                                      num_nano_batch=2,
                                                      relabel_nodes=True)
    histories = torch.nn.ModuleList([
        History(cacheid, node_num, hidden_channels, 'cpu')
        for cacheid in cached_id
    ])
    histories[0].cached_nodes = torch.tensor(
        [False, False, False, True, False, False, False, False])

    nb = nano_batchs[0]
    pruned_adjs = prune_computation_graph(nb.n_id, nb.adjs, histories)
    assert pruned_adjs[0].edge_index.tolist() == [[1, 2, 3], [0, 0, 1]]
    assert pruned_adjs[1].edge_index.tolist() == [[1, 2], [0, 0]]

    nb = nano_batchs[1]
    pruned_adjs = prune_computation_graph(nb.n_id, nb.adjs, histories)
    assert pruned_adjs[0].edge_index.tolist() == [[1, 2, 4], [0, 0, 2]]
    assert pruned_adjs[1].edge_index.tolist() == [[1, 2], [0, 0]]
