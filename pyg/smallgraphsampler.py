from functools import reduce
from torch import Tensor
from typing import List, NamedTuple, Optional, Tuple, Union
from timeit import default_timer
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborSampler
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes
from tqdm import tqdm
import torch.nn.functional as F
import argparse
from statistics import mean
import numpy as np
import torch
torch.set_printoptions(profile="full")


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2):
        super().__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                # x = F.dropout(x, p=0.5, training=self.training)
        return x


def ourk_hop_subgraph(
    node_idx: Union[int, List[int], Tensor],
    num_hops: int,
    edge_index: Tensor,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = 'source_to_target',
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    for i in range(num_hops):
        mask = np.isin(subsets[i+1], subsets[0]) #  numpy是在cpu上运行的, 就用torch isin
        subsets[i+1] = subsets[i+1][~mask]
    subset = torch.cat(subsets[1:]).unique()

    subset = torch.cat((subsets[0], subset), 0)

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, edge_mask


def get_microbatch(
    edge_index,
    n_id,
    batch_size,
    hop: int = 1,
    num_microbatch=2
) -> List[Data]:
    r"""Returns the micro batchs

    Args:
        batch:  mini batch graph
        hop: subgraph hop times
        num_microbatch: microbatch number

    :rtype: (:class:`List[Data]`)
    """
    microbach_size = batch_size // num_microbatch
    microbatchs = []
    for i in range(num_microbatch):
        subset = n_id[i * microbach_size:(i + 1) * microbach_size]
        subbatch_size, subn_id, subadjs = ourk_hop_subgraph(
            subset, hop, edge_index, relabel_nodes=False)
        assert subdata.validate() == True
        microbatchs.append((subbatch_size, subn_id, subadjs))
    return microbatchs


def onehop(data: Data):
    train_loader = NeighborSampler(data.edge_index,
                                   sizes=[-1], batch_size=4,
                                   shuffle=False, num_workers=0)
    num_features = 1
    num_classes = 1
    x = data.x
    model = SAGE(num_features, 256, num_classes)
    assert data.validate() == True
    for epoch in range(1, 2):
        model.train()
        for batch_size, n_id, adjs in train_loader:
            assert adjs[0].tolist() == [[1, 6, 0, 2, 6, 1, 3, 7, 2, 4, 8, 3, 5, 9, 4, 9],
                                        [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5]]
            hop = 1
            num_microbatch = 2
            microbatchs = get_microbatch(edge_index,
                                         n_id,
                                         batch_size, hop, num_microbatch)
            leftbatch, rightbatch = microbatchs[0], microbatchs[1]
            assert leftbatch.validate() == True
            assert leftbatch.n_id.tolist() == [0, 1, 2, 3, 6, 7]
            assert rightbatch.validate() == True
            assert rightbatch.n_id.tolist() == [3, 4, 5, 2, 8, 9]
            assert rightbatch.edge_index.tolist() == [[3, 2, 4, 8, 3, 5, 9, 4, 9],
                                                      [2, 3, 3, 3, 4, 4, 4, 5, 5]]
            assert leftbatch.edge_index.tolist() == [[1, 6, 0, 2, 6, 1, 3, 7, 2],
                                                     [0, 0, 1, 1, 1, 2, 2, 2, 3]]
            out = model(x[n_id], adjs)
            leftout = model(x[leftbatch[1]], leftbatch[2])
            rightout = model(x[rightbatch[1]], rightbatch[2])
            subgraphout = torch.cat((leftout, rightout), 0)
            assert torch.abs((out - subgraphout).mean()) < 0.01
            print(out)


# 0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,6,6,6,7,7,7,8,8,8,9,9,9
# 1,6,0,2,6,1,3,7,2,4,8,3,5,9,4,9,0,1,7,2,6,8,3,7,9,4,5,8
if __name__ == '__main__':
    edge_index = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9],
                               [1, 6, 0, 2, 6, 1, 3, 7, 2, 4, 8, 3, 5, 9, 4, 9, 0, 1, 7, 2, 6, 8, 3, 7, 9, 4, 5, 8]], dtype=torch.long)
    x = torch.tensor([[1], [2], [3], [4], [5], [6], [7],
                     [8], [9], [10]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    onehop(data)
    # twohop(data, args)
