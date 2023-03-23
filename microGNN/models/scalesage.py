from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SAGEConv

from microGNN import History
from microGNN.prune import prune_computation_graph
from microGNN.utils import slice_adj
from microGNN.utils.common_class import Adj, Nanobatch

from .base import ScalableGNN


class ScaleSAGE(ScalableGNN):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        pool_size: Optional[int] = None,
        buffer_size: Optional[int] = None,
        device=None,
        train=False,
    ):
        super().__init__(hidden_channels, num_layers, pool_size, buffer_size,
                         device)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        super().reset_parameters
        for conv in self.convs:
            conv.reset_parameters()

    # history [0] is 1 hop, [1] 2 hop.
    def forward(self, x: Tensor, nb, histories: torch.nn.ModuleList) -> Tensor:
        n_id = nb.n_id
        adjs = nb.adjs
        pruned_adjs = [adjs[0]]
        layernode = nb.n_id[:nb.size]
        for i in range(1, len(adjs)):
            adj = adjs[i]
            # 如果hisotry 里面的cached_nodes 为true,则说明这个节点已经计算过了,不需要再计算了
            sub_nid = layernode[histories[
                i - 1].cached_nodes[layernode].logical_not()]  # 没有cached 的节点
            print("sub_nid", sub_nid)
            layernode, sub_adjs, edge_mask = slice_adj(sub_nid,
                                                       adj.edge_index,
                                                       relabel_nodes=False)
            pruned_adjs.append(
                Adj(sub_adjs, None, (len(layernode), len(sub_nid))))
        print("pruned_adjs", pruned_adjs)
        for i, (edge_index, _, size) in enumerate(pruned_adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:  # last layer is not saved
                x = F.relu(x)
                history: History = histories[-i]
                batch_size = size[1]
                for i, id in enumerate(n_id[:batch_size]):
                    if history.cached_nodes[id] == True:
                        x[i] = history.emb[id]
                        print("hit", id)
                history.push(x, n_id[:batch_size])  # push 所有的, 包括刚刚pull的
                # require 前size[1]个节点是 next layer nodes
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)
