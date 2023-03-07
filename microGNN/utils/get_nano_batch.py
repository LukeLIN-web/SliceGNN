from torch import Tensor
from typing import List, Optional, Tuple, Union
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch
from microGNN.utils.common_class import Adj, Nanobatch

torch.set_printoptions(profile="full")


def slice_adj(
    node_idx: Union[int, List[int], Tensor],
    edge_index: Tensor,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = "source_to_target",
) -> Tuple[Tensor, Tensor, Tensor]:
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

    assert flow in ["source_to_target", "target_to_source"]
    if flow == "target_to_source":
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
        node_idx = target.new_full((num_nodes,), -1)
        # tensor([ 0,  1,  2,  3, -1, -1,  4,  5, -1, -1])
        node_idx[subset] = torch.arange(subset.size(0), device=target.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, edge_mask


def get_nano_batch(
    adjs: List[Adj],
    n_id: Tensor,
    batch_size: int,
    num_micro_batch: int = 2,
) -> List[Nanobatch]:
    r"""Returns the micro batchs

    Args:
        adjs: each layer adj
        n_id : the node id of the batch
        batch_size: mini batch size
        num_micro_batch: micro_batch number

    :rtype: List[List[Tensor,int,list]]
    """
    n_id = torch.arange(len(n_id))  # relabel for mini batch
    assert batch_size >= num_micro_batch, "batch_size must < num_micro_batch"
    mod = batch_size % num_micro_batch
    if mod != 0:
        batch_size -= mod
    assert batch_size % num_micro_batch == 0
    adjs.reverse()
    micro_batch_size = batch_size // num_micro_batch  # TODO: or padding last batch
    micro_batchs = []
    for i in range(num_micro_batch):
        sub_nid = n_id[i * micro_batch_size : (i + 1) * micro_batch_size]
        subadjs = []
        for adj in adjs:
            target_size = len(sub_nid)
            print(sub_nid)
            sub_nid, sub_adjs, edge_mask = slice_adj(
                sub_nid, adj.edge_index, relabel_nodes=True
            )
            subadjs.append(Adj(sub_adjs, None, (len(sub_nid), target_size)))
        subadjs.reverse()  # O(n)
        micro_batchs.append(Nanobatch(sub_nid, micro_batch_size, subadjs))
    return micro_batchs


def get_nano_batch_withlayer(
    adjs: List[Adj],
    n_id: Tensor,
    batch_size: int,
    num_micro_batch: int = 2,
) -> List[List[Tensor]]:
    r"""Returns each layer node id

    :rtype: List[ each layer node id ]
    """
    n_id = torch.arange(len(n_id))  # relabel for mini batch
    assert batch_size >= num_micro_batch, "batch_size must < num_micro_batch"
    mod = batch_size % num_micro_batch
    if mod != 0:
        batch_size -= mod
    assert batch_size % num_micro_batch == 0
    adjs.reverse()
    micro_batch_size = batch_size // num_micro_batch  # TODO: or padding last batch
    nanobatchs = []
    for i in range(num_micro_batch):
        sub_nid = n_id[i * micro_batch_size : (i + 1) * micro_batch_size]
        subnids = []
        for adj in adjs:
            sub_nid, sub_adjs, edge_mask = slice_adj(
                sub_nid, adj.edge_index, relabel_nodes=True
            )
            subnids.append(sub_nid)  # layer 0 is interal
        nanobatchs.append(subnids)
    return nanobatchs
