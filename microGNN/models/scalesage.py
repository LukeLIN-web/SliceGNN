from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SAGEConv

from microGNN import History

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

    # history [0] 1 hop, [1] 2 hop.
    def forward(self, x: Tensor, adjs: list, n_id: Tensor,
                histories: torch.nn.ModuleList) -> Tensor:
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:  # last layer is not saved
                x = F.relu(x)
                x = self.push_and_pull(histories[-i], x, n_id[:size[1]],
                                       size[1])
                # require 前size[1]个节点是 next layer nodes
                # x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def push_and_pull(
        self,
        history: History,
        x: Tensor,
        n_id: Tensor,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        pull_node = n_id[history.cached_nodes[n_id]].squeeze()
        x = history.pull(x, pull_node)
        if pull_node.numel() == 0:
            push_node = n_id[:batch_size]
        else:
            push_node = torch.masked_select(
                n_id[:batch_size], ~torch.eq(n_id[:batch_size], pull_node))
        history.push(x, push_node)
        return x
