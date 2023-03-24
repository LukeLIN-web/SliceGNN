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

    # history [0] is outer hop, [1] inner hop, [-1] is 1hop
    def forward(self, x: Tensor, nb, histories: torch.nn.ModuleList) -> Tensor:
        pruned_adjs = prune_computation_graph(nb, histories)
        for i, (edge_index, _, size) in enumerate(pruned_adjs):
            h = self.convs[i](
                x, edge_index)  # compute the non cached nodes embedding
            non_empty_indices = (h != 0).nonzero()
            x[non_empty_indices] = h[non_empty_indices]
            if i != self.num_layers - 1:  # last layer is not saved
                x = F.relu(x)
                history = histories[i]
                batch_size = nb.adjs[i].size[
                    1]  # require 前size[0]个节点是 layer nodes
                history.pull(x, nb.n_id[:batch_size])
                history.push(
                    x[:batch_size],
                    nb.n_id[:batch_size])  # Push all, including just pulled.
                # require 前size[1]个节点是 next layer nodes
                x = F.dropout(x, p=0.5, training=self.training)
        x = x[:nb.adjs[-1].size[1]]
        return x.log_softmax(dim=-1)
