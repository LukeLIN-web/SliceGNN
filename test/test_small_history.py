import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.testing.decorators import withCUDA

from microGNN import History
from microGNN.models import SAGE, ScaleSAGE
from microGNN.prune import prune_computation_graph
from microGNN.utils import get_nano_batch_histories
from microGNN.utils.common_class import Adj

hop = [-1, -1]
num_layers = 2
in_channels = 8
hidden_channels = 4
out_channels = 2
node_num = 8
features = [[i for j in range(in_channels)] for i in range(node_num)]


def test_small_hist():
    mb_n_id = torch.arange(node_num)
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
                                                      node_num=node_num,
                                                      num_nano_batch=2,
                                                      relabel_nodes=True)
    his = []
    for i, cacheid in enumerate(cached_id):
        his.append(History(cacheid, node_num, hidden_channels, 'cpu'))
    histories = torch.nn.ModuleList(his)
    nb = nano_batchs[1]

    x = torch.tensor(features, dtype=torch.float)
    x = x[mb_n_id][nb.n_id]
    histories[0].cached_nodes = torch.tensor(
        [False, False, False, True, False, False, False, False])
    histories[0].emb[0] = torch.tensor([3.3, 3.4, 3.5, 3.6])  # should be pull

    for i, adj in enumerate(nb.adjs):
        batch_size = adj.size[1]
        x_target = x[:adj.size[1]]  # require 前size[0]个节点是 layer nodes
        x = convs[i]((x, x_target),
                     adj.edge_index)  # compute the non cached nodes embedding
        if i != num_layers - 1:  # last layer is not saved
            history = histories[i]
            a_cat_b, counts = torch.cat(
                [nb.n_id[:batch_size],
                 history.global_idx]).unique(return_counts=True)
            interid = a_cat_b[torch.where(counts.gt(1))]
            for j, id in enumerate(interid):
                if history.cached_nodes[id] == True:
                    assert j == 0
                    assert id == 3
                    x[j] = history.emb[torch.where(history.global_idx == id)]
                else:
                    history.emb[id] = x[j].detach()
                    history.cached_nodes[id] = True
            print(history.cached_nodes)
            assert torch.equal(
                history.cached_nodes,
                torch.tensor(
                    [False, False, False, True, False, False, False, False]))
    # loss = F.nll_loss(x, torch.tensor([1]))
    # loss.backward()
    # for param in convs.parameters():
    #     assert param.grad is not None
