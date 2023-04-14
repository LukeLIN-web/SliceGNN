import torch

from microGNN import History
from microGNN.prune import prune_computation_graph
from microGNN.utils import get_nano_batch, get_nano_batch_histories
from microGNN.utils.common_class import Adj, Nanobatch

hop = [-1, -1]
num_layers = 2
in_channels = 8
hidden_channels = 4
out_channels = 2
node_num = 8


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
                                                      node_num=node_num,
                                                      num_nano_batch=2,
                                                      relabel_nodes=True)
    histories = torch.nn.ModuleList([
        History(cacheid, node_num, hidden_channels, 'cpu')
        for cacheid in cached_id
    ])
    histories[0].cached_nodes = torch.tensor(
        [False, False, False, True, False, False, False, False])

    nb = nano_batchs[0]
    pruned_adjs, layernodes = prune_computation_graph(nb.n_id, nb.adjs,
                                                      histories)
    assert pruned_adjs[0].edge_index.tolist() == [[1, 2, 3], [0, 0, 1]]
    assert pruned_adjs[1].edge_index.tolist() == [[1, 2], [0, 0]]
    assert layernodes[0].tolist() == [0, 2, 3, 5]
    assert layernodes[1].tolist() == [0, 2, 3]

    nb = nano_batchs[1]
    pruned_adjs, layernodes = prune_computation_graph(nb.n_id, nb.adjs,
                                                      histories)
    assert pruned_adjs[0].edge_index.tolist() == [[1, 2, 4], [0, 0, 2]]
    assert pruned_adjs[1].edge_index.tolist() == [[1, 2], [0, 0]]
    assert layernodes[0].tolist() == [1, 4, 3, 7]
    assert layernodes[1].tolist() == [1, 3, 4]
