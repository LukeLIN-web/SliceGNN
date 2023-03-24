from typing import List, Tuple

import torch
from torch import Tensor

from microGNN.utils import slice_adj
from microGNN.utils.common_class import Adj, Nanobatch


def prune_computation_graph(nb: Nanobatch,
                            histories: torch.nn.ModuleList) -> List[Adj]:
    adjs = nb.adjs
    adjs.reverse()
    pruned_adjs = [adjs[0]]
    layernode = nb.n_id[:adjs[0].size[0]]  # get 1hop nodes
    for i in range(1, len(adjs)):
        adj = adjs[i]
        # 如果hisotry 里面的cached_nodes 为true,则说明这个节点已经计算过了,不需要再计算了
        sub_nid = torch.empty(0, dtype=torch.long)
        for j, id in enumerate(layernode):
            if histories[i - 1].cached_nodes[id] == False:
                sub_nid = torch.cat((sub_nid, torch.tensor([j])))
        layernode, sub_adjs, edge_mask = slice_adj(sub_nid,
                                                   adj.edge_index,
                                                   relabel_nodes=False)
        pruned_adjs.append(Adj(sub_adjs, None, (len(layernode), len(sub_nid))))
    pruned_adjs.reverse()
    return pruned_adjs


def prune(target_node: Tensor, adjs: List[Adj],
          cached_nodes: Tensor) -> Tuple[Tensor, List[Adj]]:
    r"""
    Prune the nano batch graph,
    only keep the nodes that are not in the cached_nodes.

    Args:
        target_node (Tensor): The target node.
        adjs (list): A list of adjacency matrices.
        cached_nodes (Tensor): The cached nodes.

    :rtype: (Tensor, list)
    """
    adjs.reverse()
    new_adjs = []
    layernode = target_node
    for i, adj in enumerate(adjs):
        mask = ~torch.isin(layernode, cached_nodes[i])
        sub_nid = layernode[mask]
        # TODO: 或者可以unique一下, 对比一下unique快还是不unique快
        layernode, sub_adjs, edge_mask = slice_adj(sub_nid,
                                                   adj.edge_index,
                                                   relabel_nodes=False)
        new_adjs.append(Adj(sub_adjs, None, (len(layernode), len(sub_nid))))
    new_adjs.reverse()
    return layernode, adjs


# if __name__ == "__main__":
#     prune()
