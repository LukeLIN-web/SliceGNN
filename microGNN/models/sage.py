from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SAGEConv

from microGNN import History
from microGNN.prune import prune_computation_graph


# use for quiver sampler
class SAGE(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers=2):
        super(SAGE, self).__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: Tensor, adjs: list) -> Tensor:
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
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


class ScaleSAGE(torch.nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int,
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None):
        super(ScaleSAGE, self).__init__()

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
        pruned_adjs = prune_computation_graph(n_id, adjs, histories)
        for i, adj in enumerate(pruned_adjs):
            batch_size = adjs[i].size[1]  # original batch size
            x_target = x[:batch_size]  # nano batch layer nodes
            x = self.convs[i]((x, x_target),
                              adj.edge_index)  # non cached nodes embedding
            if i != self.num_layers - 1:  # last layer is not saved
                x = F.relu(x)
                history: History = histories[i]
                x = history.pull(x, n_id[:batch_size])
                history.push(x, n_id[:batch_size])
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


# use for neighborloader
class newSAGE(torch.nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 2):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all: Tensor, device: torch.device,
                  subgraph_loader: NeighborLoader) -> Tensor:

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.node_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                x = x[:batch.batch_size]
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)
        return x_all
