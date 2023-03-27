import torch
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn.conv import SAGEConv

from microGNN import History
from microGNN.models import SAGE, ScaleSAGE
from microGNN.prune import prune_computation_graph
from microGNN.utils import get_nano_batch
from microGNN.utils.common_class import Adj, Nanobatch


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
