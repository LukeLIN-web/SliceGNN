from functools import reduce
from torch import Tensor
from typing import List, NamedTuple, Optional, Tuple
from timeit import default_timer
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm
import torch.nn.functional as F
import argparse
from statistics import mean

import torch
torch.set_printoptions(profile="full")


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)]
                x = conv(x, batch.edge_index)
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all


def getSublayer(batch: Data, part: int) -> Data:
    hop =1 
    subdata = batch.clone()
    print(batch.n_id)
    subdata.batch_size = batch.batch_size//2
    if part == 0:
        subset = batch.n_id[:subdata.batch_size]
    else:
        subset = batch.n_id[subdata.batch_size:]
    smallset, edge_index, mapping, edge_mask = k_hop_subgraph(subset, hop, batch.edge_index, relabel_nodes=False)
    subdata.edge_index = edge_index 
    subdata.n_id = smallset
    return subdata


def run(data, args):
    kwargs = {'batch_size': 6, 'num_workers': 0}
    train_loader = NeighborLoader(
        data, num_neighbors=[-1], shuffle=False, **kwargs)
    num_features = 1
    num_classes = 2
    model = SAGE(num_features, 256, num_classes)

    for epoch in range(1, 2):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            leftbatch, rightbatch = getSublayer(batch,batch_idx)
            # optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            out = out[:batch.batch_size]
            # our sub graph
            # leftout = model(leftbatch.x, leftbatch.edge_index)[
            #     :leftbatch.batch_size]
            # rightout = model(rightbatch.x, rightbatch.edge_index)[
            #     :rightbatch.batch_size]
            # print(leftout)
            # print(rightout)
            #  merge() 手动合起来.
            # result = torch.allclose(suygraphout,out)
            # 求L2 distance. print
            # torch.cdist(out,suygraphout,p=2)


# 0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,6,6,6,7,7,7,8,8,8,9,9,9
# 1,6,0,2,6,1,3,7,2,4,8,3,5,9,4,9,0,1,7,2,6,8,3,7,9,4,5,8
if __name__ == '__main__':
    edge_index = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9],
                               [1, 6, 0, 2, 6, 1, 3, 7, 2, 4, 8, 3, 5, 9, 4, 9, 0, 1, 7, 2, 6, 8, 3, 7, 9, 4, 5, 8]], dtype=torch.long)
    x = torch.tensor([[1], [2], [3], [4], [5], [6], [7],
                     [8], [9], [10]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    data.n_id = torch.arange(data.num_nodes)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=6)
    args = parser.parse_args()

    run(data, args)
