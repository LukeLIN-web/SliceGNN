from timeit import default_timer
from torch_geometric.loader import EdgeIndex, Adj
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


def getSublayer(batch_size, n_id, adjs : "int, list[torch.long], list[ EdgeIndex]") -> "tuple[int,list[torch.long], list[ EdgeIndex]]":
    # 首先取lay1的32个中的, 然后再取 layer2中的 source.
    # print( 'Target node num: {},  sampled node num: {}'.format(batch_size,n_id.shape)) # 基本上都是1024,最后几个是469,
    # 取出对应的adjs
    sub_batch_size  = args.num_nodes//2
    print(n_id.shape)  # 64*25*10 , 有14821
    layer1 = adjs[0]
    print(layer1.edge_index.shape)
    layer1 = adjs[1]
    print(layer1.edge_index[0])
    print(layer1.edge_index[1])
    print(layer1.edge_index.dtype)
    # layer1 process
    sublayer1_target = []  # subgraph have args.num_nodes
    sublayer1_source = []
    sublayer1_e_id = []
    sublayer1_size = [0, sub_batch_size]
    i = 0
    while len(sublayer1_target) < sub_batch_size:
        if sublayer1_target[-1] != (layer1.edge_index[1])[i]:
            sublayer1_target.append((layer1.edge_index[1])[i])
            sublayer1_source.append((layer1.edge_index[0])[i])
            sublayer1_e_id.append(layer1.e_id[i])
            sublayer1_size[0] += 1
        i += 1
    layer1_edge_index = torch.tensor(
        [sublayer1_source, sublayer1_target], dtype=layer1.edge_index.dtype)
    subadj1 = EdgeIndex(layer1_edge_index, sublayer1_e_id, sublayer1_size)
    # layer2 process
    sublayer2_target = []  # subgraph have args.num_nodes
    sublayer2_source = []
    sublayer2_e_id = []
    sublayer2_size = [0, sub_batch_size]
    i = 0
    while len(sublayer2_target) < sub_batch_size:
        if sublayer2_target[-1] != (layer1.edge_index[1])[i]:
            sublayer2_target.append((layer1.edge_index[1])[i])
            sublayer2_source.append((layer1.edge_index[0])[i])
            sublayer2_e_id.append(layer1.e_id[i])
            sublayer2_size[0] += 1
        i += 1
    layer1_edge_index = torch.tensor(
        [sublayer2_source, sublayer2_target], dtype=layer1.edge_index.dtype)
    subadj1 = EdgeIndex(layer1_edge_index, sublayer2_e_id, sublayer2_size)
    sub_adjs = [subadj1,subadj2]

    return args.num_nodes//2, sub_n_id, sub_adjs


def run(dataset, args):
    data = dataset[0]
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)

    train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                                   sizes=[25, 10], batch_size=args.num_nodes,
                                   shuffle=True, num_workers=0)
    # subgraph_loader = NeighborSampler(data.edge_index, node_idx=None,
    #                                       sizes=[-1], batch_size=2048,
    #                                       shuffle=False, num_workers=6)
    # 第一层每个node 25个neibor, 第二层每个node 访问10个.
    torch.manual_seed(12345)
    rank = torch.device('cuda:0')
    model = SAGE(dataset.num_features, 256, dataset.num_classes).to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x, y = data.x.to(rank), data.y.to(rank)

    for epoch in range(1, 4):
        model.train()
        for batch_size, n_id, adjs in train_loader:
            sub_batch_size, sub_n_id, sub_adjs = getSublayer(batch_size,n_id,adjs,args)
            adjs = [adj.to(rank) for adj in adjs]
            optimizer.zero_grad()
            out = model(x[n_id], adjs)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()

            # our sub graph
            sub_adjs = [adj.to(rank) for adj in sub_adjs]
            optimizer.zero_grad()
            out = model(x[sub_n_id], sub_adjs)

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
