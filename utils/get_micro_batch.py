from collections import namedtuple
from torch import Tensor
from typing import List, NamedTuple, Optional, Tuple, Union
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborSampler
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch.nn.functional as F
import torch
torch.set_printoptions(profile="full")


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


class Adj(NamedTuple):
    edge_index: torch.Tensor
    e_id: torch.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(self.edge_index.to(*args, **kwargs),
                   e_id, self.size)


def slice_adj(
    node_idx: Union[int, List[int], Tensor],
    edge_index: Tensor,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = 'source_to_target',
) -> Tuple[Tensor, Tensor,  Tensor]:
    r"""Computes the microbatch edge_index of origin edge_index.

    The :attr:`flow` argument denotes the direction of edges for finding
    If set to :obj:`"source_to_target"`, then the
    method will find all neighbors that point to the initial set of seed nodes
    in :attr:`node_idx.`
    This mimics the natural flow of message passing in Graph Neural Networks.

    The method returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity,  and (3) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central seed
            node(s).
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, 
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        target, source = edge_index
    else:
        source, target = edge_index

    node_mask = target.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = target.new_empty(target.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=target.device).flatten()
    else:
        node_idx = node_idx.to(target.device)

    subsets = [node_idx]

    node_mask.fill_(False)
    node_mask[subsets[-1]] = True
    torch.index_select(node_mask, 0, target, out=edge_mask)  # select edge
    subsets.append(source[edge_mask])
    # remove all target nodes from array .
    # subsets[0] is the target nodes , and we need place it at first.
    mask = torch.isin(subsets[1], subsets[0])
    subsets[1] = subsets[1][~mask]
    subset = torch.cat(subsets[1:]).unique()
    subset = torch.cat((subsets[0], subset), 0)

    node_mask.fill_(False)
    node_mask[subset] = True  # the subgraph nodes after hop

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = target.new_full((num_nodes, ), -1)
        # tensor([ 0,  1,  2,  3, -1, -1,  4,  5, -1, -1])
        node_idx[subset] = torch.arange(subset.size(0), device=target.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index,  edge_mask


def get_micro_batch(
    adjs,
    n_id: Tensor,
    batch_size: int,
    num_micro_batch: int = 2,
) -> List:
    r"""Returns the micro batchs

    Args:
        batch:  mini batch graph
        hop: subgraph hop times
        num_micro_batch: micro_batch number

    :rtype: List[List]
    """
    n_id = torch.arange(len(n_id))  # relabel for mini batch
    if batch_size < num_micro_batch:
        return [batch_size, n_id, adjs]
    assert batch_size % num_micro_batch == 0
    adjs.reverse()
    micro_batch_size = batch_size // num_micro_batch     # TODO: or padding last batch
    micro_batchs = []
    for i in range(num_micro_batch):
        sub_nid = n_id[i * micro_batch_size:(i + 1) * micro_batch_size]
        subadjs = []
        for adj in adjs:
            target_size = len(sub_nid)
            sub_nid, sub_adjs,  edge_mask = slice_adj(
                sub_nid, adj.edge_index, relabel_nodes=True)
            subadjs.append(Adj(sub_adjs, None, (
                len(sub_nid), target_size)))
        subadjs.reverse()  # O(n)
        micro_batchs.append([sub_nid, micro_batch_size, subadjs])
    return micro_batchs


def onehop(data):
    num_features, hidden_size, num_classes = 1, 16, 1
    model = SAGE(num_features, hidden_size, num_classes, num_layers=2)
    train_loader = NeighborSampler(edge_index,
                                   sizes=[-1], batch_size=6,
                                   shuffle=False, num_workers=6)
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
                model(x[n_id][micro_batch.nid], micro_batch.adjs))
        subgraphout = torch.cat(subgraphout, 0)
        assert torch.abs((out - subgraphout).mean()) < 0.01


# 0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,6,6,6,7,7,7,8,8,8,9,9,9
# 1,6,0,2,6,1,3,7,2,4,8,3,5,9,4,9,0,1,7,2,6,8,3,7,9,4,5,8
if __name__ == '__main__':
    edge_index = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9],
                              [1, 6, 0, 2, 6, 1, 3, 7, 2, 4, 8, 3, 5, 9, 4, 9, 0, 1, 7, 2, 6, 8, 3, 7, 9, 4, 5, 8]], dtype=torch.long)
    x = torch.tensor([[1], [2], [3], [4], [5], [6], [7],
                     [8], [9], [10]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    onehop(data)
