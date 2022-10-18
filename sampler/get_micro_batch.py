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


class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


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

    # subset, inv = torch.cat(subsets).unique(return_inverse=True)
    # inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True  # the subgraph nodes after hop

    # edge_mask = node_mask[target] & node_mask[source]
    # print(edge_index[:, node_mask[target]])
    # print(edge_index[:, node_mask[source]])

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = target.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=target.device) #tensor([ 0,  1,  2,  3, -1, -1,  4,  5, -1, -1])
        edge_index = node_idx[edge_index]

    return subset, edge_index,  edge_mask


def get_micro_batch(
    adjs: List[EdgeIndex],
    n_id: Tensor,
    batch_size: int,
    num_micro_batch: int = 2,
) -> List[namedtuple('micro_batch', ['bach_size', 'nid', 'adjs'])]:
    r"""Returns the micro batchs

    Args:
        batch:  mini batch graph
        hop: subgraph hop times
        num_micro_batch: micro_batch number

    :rtype: List[namedtuple('micro_batch', ['bach_size', 'nid', 'adjs'])]
    """
    assert batch_size % num_micro_batch == 0
    n_id = torch.arange(len(n_id)) # relabel for mini batch 
    hop = len(adjs)
    micro_batch_size = batch_size // num_micro_batch     # TODO: or padding last batch
    micro_batchs = []
    micro_batch = namedtuple('micro_batch', ['bach_size', 'nid', 'adjs'])
    for i in range(num_micro_batch):
        sub_nid = n_id[i * micro_batch_size:(i + 1) * micro_batch_size]
        subadjs = []
        for j in range(hop):
            target_size = len(sub_nid)
            sub_nid, sub_adjs,  edge_mask = slice_adj(
                sub_nid, adjs[-j-1].edge_index, relabel_nodes=True)  
            subadjs.append(EdgeIndex(sub_adjs, None, (
                len(sub_nid), target_size)))
        subadjs.reverse() # O(n)
        # layer1, layer2 = subadjs[0], subadjs[1]
        # assert layer1.size == (6,4)
        # assert layer2.size == (4,2)
        # assert sub_nid.tolist() == [0,1,2,4,3,5]
        # assert layer1.edge_index.tolist() == [[1, 4, 0, 2, 4, 1, 3, 5, 0, 1, 5],
        #                                                  [0, 0, 1, 1, 1, 2, 2, 2, 4, 4, 4]]
        # assert layer2.edge_index.tolist() == [[1, 4, 0, 2, 4], [0, 0, 1, 1, 1]]
        micro_batchs.append(micro_batch(micro_batch_size, sub_nid, subadjs))
    return micro_batchs


def two_hop(data: Data):
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
            out = model(x[n_id], adjs)
            num_micro_batch = 2
            micro_batchs = get_micro_batch(adjs,
                                           n_id,
                                           batch_size, num_micro_batch)
            leftbatch, rightbatch = micro_batchs[0], micro_batchs[1]
            # assert rightbatch.nid.tolist() == [2, 3, 4, 5, 8, 9]
            # assert rightbatch.adjs[0].edge_index.tolist() == [[1, 0, 2, 4, 1, 3, 5, 2, 5],
            #                                                   [0, 1, 1, 1, 2, 2, 2, 3, 3]]
            leftout = model(x[n_id][leftbatch.nid], leftbatch.adjs)
            rightout = model(x[n_id][rightbatch.nid], rightbatch.adjs)
            subgraphout = torch.cat((leftout, rightout), 0)
            assert torch.abs((out - subgraphout).mean()) < 0.01
            print(out)


def one_hop(data: Data):
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
            # if batch_size == 6:
            #     continue
            if isinstance(adjs[0], Tensor):
                # when hop = 1 , adjs is a EdgeIndex, we need convert it to list.
                adjs = [adjs]
            # assert adjs[0].edge_index.tolist() == [[1, 6, 0, 2, 6, 1, 3, 7, 2, 4, 8, 3, 5, 9, 4, 9],
            #                                        [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5]]
            num_micro_batch = 2
            micro_batchs = get_micro_batch(adjs,
                                           n_id,
                                           batch_size, num_micro_batch)
            leftbatch, rightbatch = micro_batchs[0], micro_batchs[1]
            # assert leftbatch.nid.tolist() == [0, 1, 2, 3, 6, 7]
            # assert leftbatch.adjs[0].edge_index.tolist() == [[1, 4, 0, 2, 4, 1, 3, 5],
            #                                                  [0, 0, 1, 1, 1, 2, 2, 2]]
            # assert rightbatch.nid.tolist() == [3, 4, 5, 2, 8, 9]
            # assert rightbatch.adjs[0].edge_index.tolist() == [[3, 1, 4, 0, 2, 5, 1, 5],
            #                                                   [0, 0, 0, 1, 1, 1, 2, 2]]
            out = model(x[n_id], adjs)
            leftout = model(x[n_id][leftbatch.nid], leftbatch.adjs)
            rightout = model(x[n_id][rightbatch.nid], rightbatch.adjs)
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
    # one_hop(data)
    two_hop(data)
