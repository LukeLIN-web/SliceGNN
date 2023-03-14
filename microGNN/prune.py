from typing import List, Tuple
from torch import Tensor
import torch
from microGNN.utils import slice_adj
from microGNN.utils.common_class import Adj


def prune(target_node: Tensor, adjs: list, cached_nodes: Tensor) -> Tuple[Tensor, list]:
    r"""
    Prune the nano batch graph, only keep the nodes that are not in the cached_nodes.

    Args:
        target_node (Tensor): The target node.
        adjs (list): A list of adjacency matrices.
        cached_nodes (Tensor): The cached nodes.

    :rtype: (all node, list)
    """
    adjs.reverse()
    new_adjs = []
    layernode = target_node
    for i, adj in enumerate(adjs):
        mask = ~torch.isin(layernode, cached_nodes[i])
        sub_nid = layernode[mask]
        # target node in layer i 或者可以unique一下, 对比一下unique快还是不unique快
        # find the node in layer i that is in cached_nodes
        target_size = len(sub_nid)
        layernode, sub_adjs, edge_mask = slice_adj(
            sub_nid, adj.edge_index, relabel_nodes=False
        )
        new_adjs.append(Adj(sub_adjs, None, (len(layernode), target_size)))
    new_adjs.reverse()
    return layernode, adjs


# if __name__ == "__main__":
#     prune()
