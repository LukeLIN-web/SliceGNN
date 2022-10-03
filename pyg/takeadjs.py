from torch import Tensor
from typing import List, NamedTuple, Optional, Tuple
from timeit import default_timer
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborSampler
from torch_geometric.datasets import Reddit
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
import os
from statistics import mean

import torch
torch.set_printoptions(profile="full")
from functools import reduce


class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


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
            x_target = x[:size[1]]  # Target num_nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    @torch.no_grad()
    def inference(self, x_all, device, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all



def getSublayer(batch_size: int, n_id: "list[torch.long]", adjs: "list[ EdgeIndex]", args) -> "tuple[int,list[torch.long], list[ EdgeIndex]]":
    sub_batch_size = args.num_nodes//2
    layer0, layer1 = adjs[0], adjs[1]
    # print(layer1.edge_index[0])
    # layer1 process
    i = 0
    target_node = set()
    indices = []
    while len(target_node) < sub_batch_size:
        target_node.add((layer1.edge_index[1])[i].item())
        indices.append(i)
        i += 1  # subgraph have args.num_nodes
    indices = torch.tensor(indices)
    sublayer1_size = (i, sub_batch_size)
    sublayer1_target = torch.index_select(layer1.edge_index[1], 0, indices)
    sublayer1_source = torch.index_select(layer1.edge_index[0], 0, indices)
    sublayer1_e_id = torch.index_select(layer1.e_id, 0, indices)
    sublayer1_edge_index = torch.stack((sublayer1_source, sublayer1_target), 0)
    subadj1 = EdgeIndex(sublayer1_edge_index, sublayer1_e_id, sublayer1_size)
    # layer0 process
    indices = []
    target_node.clear()
    i = 0
    while len(target_node) < sublayer1_size[0]:
        target_node.add((layer0.edge_index[1])[i].item())
        indices.append(i)
        i += 1
    indices = torch.tensor(indices)
    sublayer0_target = torch.index_select(layer0.edge_index[1], 0, indices)
    sublayer0_source = torch.index_select(layer0.edge_index[0], 0, indices)
    sublayer0_e_id = torch.index_select(layer0.e_id, 0, indices)
    sublayer0_size = (i, sublayer1_size[0])
    sublayer0_edge_index = torch.stack((sublayer0_source, sublayer0_target), 0)
    subadj0 = EdgeIndex(sublayer0_edge_index, sublayer0_e_id, tuple(sublayer0_size))
    sub_adjs = [subadj0, subadj1]
    return args.num_nodes//2, sublayer0_source, sub_adjs


def run(dataset, args):
    data = dataset[0]
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)

    train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                                   sizes=[25, 10], batch_size=args.num_nodes,
                                   shuffle=True, num_workers=0)
    # [-1,-1]# 采集所有nerighbor, 画一个10个点的图. 一个采样6个
    # 手动取出3个,
    # subgraph_loader = NeighborSampler(data.edge_index, node_idx=None,
    #                                       sizes=[-1], batch_size=2048,
    #                                       shuffle=False, num_workers=6)
    # 第一层每个node 25个neibor, 第二层每个node 访问10个.
    torch.manual_seed(12345)
    # rank = torch.device('cuda:0')
    # model = SAGE(dataset.num_features, 256, dataset.num_classes).to(rank)
    model = SAGE(dataset.num_features, 256, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # x, y = data.x.to(rank), data.y.to(rank)
    x, y = data.x, data.y
    for epoch in range(1, 4):
        model.train()
        for batch_size, n_id, adjs in train_loader:
            sub_batch_size, sub_n_id, sub_adjs = getSublayer(batch_size, n_id, adjs, args)
            # adjs = [adj.to(rank) for adj in adjs]
            # optimizer.zero_grad()
            # out = model(x[n_id], adjs)
            # print(out.shape)
            # print(out.dtype)
            # exit()
            # loss = F.nll_loss(out, y[n_id[:batch_size]])
            # loss.backward()
            # optimizer.step()

            # our sub graph
            # sub_adjs = [adj.to(rank) for adj in sub_adjs]
            # import pdb
            # pdb.set_trace()
            suygraphout = model(x[sub_n_id], sub_adjs)
            # forward 两次 model
            # merge() 手动合起来. 
            # result = torch.allclose(suygraphout,out)
            # 求L2 distance. print .

        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        # if epoch % 1 == 0:  # We evaluate on a single GPU for now
        # # if False:
        #     model.eval()
        #     with torch.no_grad():
        #         out = model.module.inference(x, rank, subgraph_loader)
        #     res = out.argmax(dim=-1) == data.y

        #     acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
        #     acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
        #     acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
        #     print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')


if __name__ == '__main__':
    datapath = "/root/share/pytorch_geometric/examples/data/Reddit"
    dataset = Reddit(datapath)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=64)
    args = parser.parse_args()

    run(dataset, args)
