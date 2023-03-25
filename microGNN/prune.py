import copy
from typing import List, Tuple

import torch
from torch import Tensor

from microGNN.utils import slice_adj
from microGNN.utils.common_class import Adj, Nanobatch


def prune_computation_graph(nb: Nanobatch,
                            histories: torch.nn.ModuleList) -> List[Adj]:
    adjs = copy.deepcopy(nb.adjs)
    adjs.reverse()
    pruned_adjs = [adjs[0]]
    layernode = nb.n_id[:adjs[0].size[0]]  # get 1 hop nodes
    for i in range(1, len(adjs)):
        adj = adjs[i]
        layer_idx = torch.arange(len(layernode))  # nano batch id
        cached_nodes_mask = histories[i - 1].cached_nodes[layernode]
        sub_nid = layer_idx[~cached_nodes_mask]
        layernode, sub_adjs, edge_mask = slice_adj(sub_nid,
                                                   adj.edge_index,
                                                   relabel_nodes=False)
        pruned_adjs.append(Adj(sub_adjs, None, (len(layernode), len(sub_nid))))
    pruned_adjs.reverse()
    return pruned_adjs
