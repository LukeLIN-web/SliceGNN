from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SAGEConv

from microGNN import History
from microGNN.prune import prune_computation_graph
from microGNN.utils import get_intersection, get_nano_batch_histories

from .base import ScalableGNN


class ScaleSAGE(ScalableGNN):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int,
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None):
        super().__init__(hidden_channels, num_layers, pool_size, buffer_size)

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
    def forward(self, x: Tensor, n_id: Tensor, adjs: List,
                histories: torch.nn.ModuleList) -> Tensor:
        # pruned_adjs = prune_computation_graph(n_id, adjs, histories)
        for i, (edge_index, _, size) in enumerate(adjs):
            batch_size = adjs[i].size[1]
            x_target = x[:batch_size]  # nano batch layer nodes
            x = self.convs[i]((x, x_target),
                              edge_index)  # non cached nodes embedding
            if i != self.num_layers - 1:  # last layer is not saved
                x = F.relu(x)
                history: History = histories[i]
                interid = get_intersection(n_id[:batch_size],
                                           history.global_idx)
                history.pull_push(x, interid)
                # x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    @torch.no_grad()
    def inference(self, x_all, device, subgraph_loader):
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x)

            x_all = torch.cat(xs, dim=0)

        return x_all
