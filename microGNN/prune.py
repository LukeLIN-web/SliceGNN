from typing import List, Tuple
from torch import Tensor
import torch


def remove_common_nodes(tensorA: Tensor, tensorB: Tensor) -> Tensor:
    mask = ~torch.isin(tensorA, tensorB)
    indices = torch.arange(len(tensorA))[mask]

    return indices


def prune(n_id: Tensor, adjs: list, cached_nodes: Tensor) -> Tuple[Tensor, Tensor]:
    adjs.reverse()
    for i, adj in enumerate(adjs):
        adj = adjs[i]
        layernode = adj.edge_index[0]  # target node in layer i
        mask = ~torch.isin(layernode, cached_nodes[i])
        print(mask)  # finnd the node in layer i that is in cached_nodes
        print(adj.edge_index)
        newadj = adj.edge_index[:, mask]
        print(newadj)
        # adjs[i].size =

    return n_id, adjs


# def pruneTree(root: TreeNode) -> TreeNode:
#     queue = [root]
#     while queue:
#         node = queue.pop(0)
#         if should_prune(node):
#             if node.left:
#                 queue.remove(node.left)
#             if node.right:
#                 queue.remove(node.right)
#             if node == root:
#                 return None
#             else:
#                 return node
#         if node.left:
#             queue.append(node.left)
#         if node.right:
#             queue.append(node.right)
#     return root

# def should_prune(node: TreeNode) -> bool:
#     # Example pruning criterion: remove nodes with value less than 5
#     return node.val < 5


# if __name__ == "__main__":
#     prune()
