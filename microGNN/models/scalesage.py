from typing import Optional
from torch_geometric.nn import SAGEConv
import torch
from torch import Tensor
import torch.nn.functional as F
from .base import ScalableGNN


class ScaleSAGE(ScalableGNN):
    def __init__(
        self,
        num_nodes: int,
        in_channels,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        drop_input: bool = True,
        batch_norm: bool = False,
        residual: bool = False,
        linear: bool = False,
        pool_size: Optional[int] = None,
        buffer_size: Optional[int] = None,
        device=None,
    ):
        super().__init__(
            num_nodes, hidden_channels, num_layers, pool_size, buffer_size, device
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        super().reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, adjs: list) -> Tensor:
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)
