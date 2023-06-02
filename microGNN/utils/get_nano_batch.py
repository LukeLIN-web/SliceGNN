from timeit import default_timer as timer
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes

from microGNN.utils.common_class import Adj, Nanobatch

torch.set_printoptions(profile="full")


def slice_adj(
    node_idx: Union[int, List[int], Tensor],
    edge_index: Tensor,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = "source_to_target",
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Computes the nano batch edge_index of origin edge_index.

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

    node_mask.fill_(False)
    node_mask[node_idx] = True
    torch.index_select(node_mask, 0, target, out=edge_mask)  # select edge
    subsets = [node_idx, source[edge_mask]]
    # remove all target nodes from subsets[1].
    # subsets[0] is the target nodes , and we need place it at first.
    mask = torch.isin(subsets[1], subsets[0])  # bottleneck
    subsets[1] = subsets[1][~mask]
    subset = subsets[1].unique()
    subset = torch.cat((subsets[0], subset), 0)

    node_mask.fill_(False)
    node_mask[subset] = True  # the subgraph nodes after hop

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = target.new_full((num_nodes, ), -1)
        # tensor([ 0,  1,  2,  3, -1, -1,  4,  5, -1, -1])
        node_idx[subset] = torch.arange(subset.size(0), device=target.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, edge_mask


# because neighbor sampler mappping the node index in edge index.
# so we set sub_nid, adj.edge_index, relabel_nodes=True to get same output.
def get_nano_batch(
    adjs: List[Adj],
    n_id: Tensor,
    batch_size: int,
    num_nano_batch: int = 2,
    relabel_nodes: bool = True,
) -> List[Nanobatch]:
    r"""Create a list of `num_nano_batch` nanobatches
    from a list of adjacency matrices `adjs`.

    Args:
        adjs (List[Adj]): List of each layer adjacency matrices.
        n_id (torch.Tensor): Node indices.
        batch_size: mini batch size
        num_nano_batch:  nano batch number

    :rtype: List[List[Tensor,int,list]]
    """
    assert (batch_size >= num_nano_batch
            ), "batch_size must be bigger than num_nano_batch"  # noqa
    n_id = torch.arange(len(n_id))  # relabel for mini batch
    mod = batch_size % num_nano_batch
    if mod != 0:
        batch_size -= mod
    assert batch_size % num_nano_batch == 0
    if not isinstance(adjs, list):
        adjs = [adjs]
    adjs.reverse()
    nano_batch_size = batch_size // num_nano_batch
    nano_batchs = []
    for i in range(num_nano_batch):
        sub_nid = n_id[i * nano_batch_size:(i + 1) *
                       nano_batch_size]  # 从target node开始
        subadjs = []
        for adj in adjs:
            target_size = len(sub_nid)
            sub_nid, sub_adjs, edge_mask = slice_adj(
                sub_nid,
                adj.edge_index,
                relabel_nodes=relabel_nodes,
            )
            subadjs.append(Adj(sub_adjs, None, (len(sub_nid), target_size)))
        subadjs.reverse()  # O(n) 大的在前面
        nano_batchs.append(Nanobatch(sub_nid, nano_batch_size, subadjs))
    return nano_batchs


# get nano batch for neighbor loader
def get_loader_nano_batch(batch: Data, num_nano_batch: int,
                          hop: int) -> List[Data]:
    r"""Create a list of `num_nano_batch` nanobatches
    from Data.
    Args:
        num_nano_batch:  nano batch number
    :rtype: List[Data]
    """
    batch_size = batch.batch_size
    assert (batch_size >= num_nano_batch
            ), "batch_size must be bigger than num_nano_batch"  # noqa
    n_id = torch.arange(len(batch.n_id))  # relabel for mini batch
    mod = batch_size % num_nano_batch
    if mod != 0:
        batch_size -= mod
    assert batch_size % num_nano_batch == 0
    nano_batch_size = batch_size // num_nano_batch
    nano_batchs = []
    for i in range(num_nano_batch):
        sub_nid = n_id[i * nano_batch_size:(i + 1) *
                       nano_batch_size]  # 从target node开始
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            sub_nid, hop, batch.edge_index, relabel_nodes=True)
        nano_batchs.append(Data(x=batch.x[subset], edge_index=edge_index))
    return nano_batchs


def get_nano_batch_withlayer(
    adjs: List[Adj],
    n_id: Tensor,
    batch_size: int,
    num_micro_batch: int = 2,
) -> List[List[Tensor]]:
    r"""Returns each layer node id

    Args:
        adjs (List[Adj]): List of adjacency matrices.
        n_id (torch.Tensor): Node indices.
        batch_size (int): mini batch size
        num_nano_batch (int ): Number of micro-batches to create. Defaults 2

    :rtype: List[ each layer node id ]
    """
    assert batch_size >= num_micro_batch, "batch_size must < num_micro_batch"
    n_id = torch.arange(len(n_id))  # relabel for mini batch
    mod = batch_size % num_micro_batch
    if mod != 0:
        batch_size -= mod
    assert batch_size % num_micro_batch == 0
    adjs.reverse()
    micro_batch_size = batch_size // num_micro_batch
    nanobatchs = []
    for i in range(num_micro_batch):
        sub_nid = n_id[i * micro_batch_size:(i + 1) * micro_batch_size]
        subnids = []
        for adj in adjs:
            sub_nid, sub_adjs, edge_mask = slice_adj(sub_nid,
                                                     adj.edge_index,
                                                     relabel_nodes=True)
            subnids.append(sub_nid)  # layer 0 is interal
        nanobatchs.append(subnids)
    return nanobatchs


def get_nano_batch_histories(
    adjs: List[Adj],
    n_id: Tensor,
    batch_size: int,
    num_nano_batch: int = 2,
):
    r"""Create a list of `num_nano_batch` nanobatches
    from a list of adjacency matrices `adjs`.

    Args:
        adjs (List[Adj]): List of each layer adjacency matrices.
        n_id (torch.Tensor): Node indices.
        batch_size: mini batch size
        num_nano_batch:  nano batch number
    """
    assert (batch_size >= num_nano_batch
            ), "batch_size must be bigger than num_nano_batch"  # noqa
    mod = batch_size % num_nano_batch
    if mod != 0:
        batch_size -= mod
    assert batch_size % num_nano_batch == 0, "batch_size must be divisible by num_nano_batch"
    assert isinstance(adjs, list), "adjs must be a list"
    adjs.reverse()
    nano_batch_size = batch_size // num_nano_batch
    nano_batchs = []
    num_layers = len(adjs)
    pin_memory = n_id.device is None or str(n_id.device) == "cpu"
    cached_nodes = torch.full((num_layers - 1, len(n_id)),
                              False,
                              dtype=torch.bool,
                              device=n_id.device,
                              pin_memory=pin_memory)
    cached_id = [[] for i in range(num_layers - 1)]
    n_id = torch.arange(len(n_id))  # relabel for mini batch
    for i in range(num_nano_batch):
        sub_nid = n_id[i * nano_batch_size:(i + 1) * nano_batch_size]
        subadjs = []
        for j, adj in enumerate(adjs):
            target_size = len(sub_nid)
            sub_nid, sub_adjs, _ = slice_adj(
                sub_nid,
                adj.edge_index,
                relabel_nodes=True,
            )  # bottleneck
            if j != num_layers - 1:
                cache_mask = torch.logical_not(cached_nodes[j][sub_nid])
                cached_nodes[j][sub_nid[cache_mask]] = True  # bottleneck
                cached_id[j].append(sub_nid[torch.logical_not(cache_mask)])
            subadjs.append(Adj(sub_adjs, None, (len(sub_nid), target_size)))
        subadjs.reverse()  # O(n) 大的在前面
        nano_batchs.append(Nanobatch(sub_nid, nano_batch_size, subadjs))
    cached_tensor = [torch.cat(ids) for ids in cached_id]
    return nano_batchs, cached_tensor
