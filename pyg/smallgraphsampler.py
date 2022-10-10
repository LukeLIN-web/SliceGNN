from collections import namedtuple
from functools import reduce
from torch import Tensor
from typing import List, NamedTuple, Optional, Tuple, Union
from timeit import default_timer
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborSampler
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes
from tqdm import tqdm
import torch.nn.functional as F
import argparse
from statistics import mean
import numpy as np
import torch
torch.set_printoptions(profile="full")


def our_k_hop_subgraph(
    node_idx: Union[int, List[int], Tensor],
    num_hops: int,
    edge_index: Tensor,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = 'source_to_target',
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Computes the induced subgraph of :obj:`edge_index` around all nodes in
    :attr:`node_idx` reachable within :math:`k` hops.

    The :attr:`flow` argument denotes the direction of edges for finding
    :math:`k`-hop neighbors. If set to :obj:`"source_to_target"`, then the
    method will find all neighbors that point to the initial set of seed nodes
    in :attr:`node_idx.`
    This mimics the natural flow of message passing in Graph Neural Networks.

    The method returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central seed
            node(s).
        num_hops (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
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

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, target, out=edge_mask)  # select edge
        subsets.append(source[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True  # select 0  1 2  # the subgraph nodes after hop

    # edge_mask = node_mask[target] & node_mask[source]
    print(edge_index[:, node_mask[target]])
    print(edge_index[:, node_mask[source]])

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = target.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=target.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask


class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


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


def get_microbatch(
    edge_index,
    n_id,
    batch_size,
    hop: int = 1,
    num_microbatch=2
) -> List[namedtuple('microbatch', ['bach_size', 'nid', 'adjs'])]:
    r"""Returns the micro batchs

    Args:
        batch:  mini batch graph
        hop: subgraph hop times
        num_microbatch: microbatch number

    :rtype: (:class:`List[Data]`)
    """
    microbach_size = batch_size // num_microbatch
    microbatchs = []
    microbatch = namedtuple('microbatch', ['bach_size', 'nid', 'adjs'])
    for i in range(num_microbatch):
        subset = n_id[i * microbach_size:(i + 1) * microbach_size]
        sub_nid, sub_adjs, inv, edge_mask = our_k_hop_subgraph(
            subset, 1, edge_index, relabel_nodes=True)
        subadjs = [EdgeIndex(sub_adjs, None, (
            len(sub_nid), microbach_size))]
        for _ in range(1, hop):
            sub_nid, sub_adjs, inv, edge_mask = k_hop_subgraph(
                sub_nid, 1, edge_index, relabel_nodes=False)
            subadjs.append(EdgeIndex(sub_adjs, None, (
                len(sub_nid), microbach_size)))
        microbatchs.append(microbatch(microbach_size, sub_nid, subadjs))
    return microbatchs


def twohop(data: Data):
    hop = [-1, -1]
    train_loader = NeighborSampler(data.edge_index,
                                   sizes=hop, batch_size=4,
                                   shuffle=False, num_workers=0)
    num_features = 1
    num_classes = 1
    x = data.x
    model = SAGE(num_features, 16, num_classes)
    for epoch in range(1, 2):
        model.train()
        for batch_size, n_id, adjs in train_loader:
            num_microbatch = 2
            microbatchs = get_microbatch(adjs[0].edge_index,  # input biggest graph
                                         n_id,
                                         batch_size, len(hop), num_microbatch)
            leftbatch, rightbatch = microbatchs[0], microbatchs[1]
            assert leftbatch.nid.tolist() == [0, 1, 2, 3, 6, 7]
            assert leftbatch.adjs[0].edge_index.tolist() == [[1, 4, 0, 2, 4, 1, 3, 5, 2],
                                                             [0, 0, 1, 1, 1, 2, 2, 2, 3]]
            # assert rightbatch.nid.tolist() == [2, 3, 4, 5, 8, 9]
            # assert rightbatch.adjs[0].edge_index.tolist() == [[1, 0, 2, 4, 1, 3, 5, 2, 5],
            #                                                   [0, 1, 1, 1, 2, 2, 2, 3, 3]]
            out = model(x[n_id], adjs)
            leftout = model(x[leftbatch.nid], leftbatch.adjs)
            rightout = model(x[rightbatch.nid], rightbatch.adjs)
            subgraphout = torch.cat((leftout, rightout), 0)
            assert torch.abs((out - subgraphout).mean()) < 0.01
            print(out)


def onehop(data: Data):
    train_loader = NeighborSampler(data.edge_index,
                                   sizes=[-1], batch_size=6,
                                   shuffle=False, num_workers=0)
    num_features = 1
    num_classes = 1
    x = data.x
    model = SAGE(num_features, 16, num_classes)
    for epoch in range(1, 2):
        model.train()
        for batch_size, n_id, adjs in train_loader:
            a = x[n_id]
            if isinstance(adjs[0], Tensor):
                # when hop = 1 , adjs is a EdgeIndex, we need convert it to list.
                adjs = [adjs]
            assert adjs[0].edge_index.tolist() == [[1, 6, 0, 2, 6, 1, 3, 7, 2, 4, 8, 3, 5, 9, 4, 9],
                                                   [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5]]
            hop = 1
            num_microbatch = 2
            microbatchs = get_microbatch(adjs[0].edge_index,
                                         n_id,
                                         batch_size, hop, num_microbatch)
            leftbatch, rightbatch = microbatchs[0], microbatchs[1]
            assert leftbatch.nid.tolist() == [0, 1, 2, 3, 6, 7]
            assert leftbatch.adjs[0].edge_index.tolist() == [[1, 4, 0, 2, 4, 1, 3, 5],
                                                             [0, 0, 1, 1, 1, 2, 2, 2]]
            assert rightbatch.nid.tolist() == [2, 3, 4, 5, 8, 9]
            assert rightbatch.adjs[0].edge_index.tolist() == [[0, 2, 4, 1, 3, 5, 2, 5],
                                                              [1, 1, 1, 2, 2, 2, 3, 3]]
            out = model(x[n_id], adjs)
            # there is some bug
            leftout = model(x[leftbatch.nid], leftbatch.adjs)
            rightout = model(x[rightbatch.nid], rightbatch.adjs)
            subgraphout = torch.cat((leftout, rightout), 0)
            assert torch.abs((out - subgraphout).mean()) < 0.01
            print(out)


# 0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,6,6,6,7,7,7,8,8,8,9,9,9
# 1,6,0,2,6,1,3,7,2,4,8,3,5,9,4,9,0,1,7,2,6,8,3,7,9,4,5,8
if __name__ == '__main__':
    edge_index = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9],
                              [1, 6, 0, 2, 6, 1, 3, 7, 2, 4, 8, 3, 5, 9, 4, 9, 0, 1, 7, 2, 6, 8, 3, 7, 9, 4, 5, 8]], dtype=torch.long)
    x = torch.tensor([[1], [2], [3], [4], [5], [6], [7],
                     [8], [9], [10]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    assert data.validate() == True
    onehop(data)
    # twohop(data)
