from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear as Lin
from torch_geometric.nn import GATConv

from microGNN import History
from microGNN.prune import prune_computation_graph


class GAT(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(heads * hidden_channels, hidden_channels, heads))
        self.convs.append(
            GATConv(heads * hidden_channels, out_channels, heads,
                    concat=False))

        self.skips = torch.nn.ModuleList()
        self.skips.append(Lin(in_channels, hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.skips.append(
                Lin(hidden_channels * heads, hidden_channels * heads))
        self.skips.append(Lin(hidden_channels * heads, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for skip in self.skips:
            skip.reset_parameters()

    def forward(self, x: Tensor, adjs: List):
        for i, adj in enumerate(adjs):
            batch_size = adjs[i].size[1]  # original batch size
            x_target = x[:batch_size]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), adj.edge_index)
            x = x + self.skips[i](x_target)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, device, subgraph_loader):

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                x = x + self.skips[i](x_target)

                if i != self.num_layers - 1:
                    x = F.elu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all


class ScaleGAT(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(heads * hidden_channels, hidden_channels, heads))
        self.convs.append(
            GATConv(heads * hidden_channels, out_channels, heads,
                    concat=False))

        self.skips = torch.nn.ModuleList()
        self.skips.append(Lin(in_channels, hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.skips.append(
                Lin(hidden_channels * heads, hidden_channels * heads))
        self.skips.append(Lin(hidden_channels * heads, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for skip in self.skips:
            skip.reset_parameters()

    def forward(self, x: Tensor, n_id: Tensor, adjs: List,
                histories: torch.nn.ModuleList):
        pruned_adjs = prune_computation_graph(n_id, adjs, histories)  # 12%用时
        for i, adj in enumerate(pruned_adjs):
            batch_size = adjs[i].size[1]  # original batch size
            x_target = x[:batch_size]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), adj.edge_index)
            x = x + self.skips[i](x_target)
            if i != self.num_layers - 1:
                x = F.elu(x)
                history: History = histories[i]
                x = history.pull(x, n_id[:batch_size])
                history.push(x, n_id[:batch_size])
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, device, subgraph_loader):

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                x = x + self.skips[i](x_target)

                if i != self.num_layers - 1:
                    x = F.elu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all
