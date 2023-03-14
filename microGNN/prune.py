from typing import List, Tuple
from torch import Tensor
import torch
from microGNN.utils import slice_adj
from microGNN.utils.common_class import Adj


def remove_common_nodes(tensorA: Tensor, tensorB: Tensor) -> Tensor:
    mask = ~torch.isin(tensorA, tensorB)
    indices = torch.arange(len(tensorA))[mask]

    return indices


def prune(n_id: Tensor, adjs: list, cached_nodes: Tensor) -> Tuple[Tensor, Tensor]:
    adjs.reverse()
    new_adjs = []
    for i, adj in enumerate(adjs):
        adj = adjs[i]
        layernode = adj.edge_index[
            0
        ]  # target node in layer i 或者可以unique一下, 对比一下unique快还是不unique快
        mask = ~torch.isin(layernode, cached_nodes[i])  # remove reused node
        print(mask)  # finnd the node in layer i that is in cached_nodes
        print(adj.edge_index)
        print(adj.edge_index[:, mask])
        sub_nid = adj.edge_index[:, mask][0]
        print(sub_nid)
        target_size = len(sub_nid)
        sub_nid, sub_adjs, edge_mask = slice_adj(
            sub_nid, adj.edge_index, relabel_nodes=True
        )
        print(sub_nid)
        print(sub_adjs)
        new_adjs.append(Adj(sub_adjs, None, (len(sub_nid), target_size)))

    return n_id, adjs


# if __name__ == "__main__":
#     prune()
