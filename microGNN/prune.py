import copy
from typing import List

import torch
from torch import Tensor

from microGNN.utils import slice_adj
from microGNN.utils.common_class import Adj


def prune_computation_graph(n_id: Tensor, nb_adjs: List,
                            histories: torch.nn.ModuleList) -> List[Adj]:
    adjs = copy.deepcopy(nb_adjs)
    adjs.reverse()
    pruned_adjs = [adjs[0]]
    layernode = n_id[:adjs[0].size[0]]  # get 1 hop nodes
    layernodes = [layernode]
    for i in range(1, len(adjs)):
        adj = adjs[i]
        layer_idx = torch.arange(len(layernode))  # nano batch id
        cached_nodes_mask = histories[i - 1].cached_nodes[[layernode]]
        sub_nid = layer_idx[~cached_nodes_mask]
        layernode, sub_adjs, edge_mask = slice_adj(sub_nid,
                                                   adj.edge_index,
                                                   relabel_nodes=False)
        pruned_adjs.append(Adj(sub_adjs, None, (len(layernode), len(sub_nid))))
        layernodes.append(n_id[layernode])
    pruned_adjs.reverse()
    layernodes.reverse()
    return pruned_adjs, layernodes
