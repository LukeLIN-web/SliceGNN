"""
sample : quiver
1gpu
"""
import logging
from timeit import default_timer

import hydra
import torch
from ogb.nodeproppred import Evaluator
from omegaconf import OmegaConf
from torch_geometric.loader import NeighborSampler

import quiver
from microGNN.models import SAGE, criterion
from microGNN.utils import cal_metrics, check_memory, get_dataset

log = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class SimpleDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        assert self.x.size(0) == self.y.size(0)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MLP(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)


def train(model, device, train_loader, optimizer):
    model.train()

    total_loss = 0
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(out, y.squeeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model, device, loader, evaluator):
    model.eval()

    y_pred, y_true = [], []
    for x, y in tqdm(loader):
        x = x.to(device)
        out = model(x)

        y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
        y_true.append(y)

    return evaluator.eval({
        "y_true": torch.cat(y_true, dim=0),
        "y_pred": torch.cat(y_pred, dim=0),
    })['acc']


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(conf):
    dataset_name = conf.dataset.name
    params = conf.model.params[dataset_name]
    print(OmegaConf.to_yaml(conf))
    dataset = get_dataset(dataset_name, conf.root)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    x = dataset[0].x

    model = MLP(x.size(-1), 256, 172, 2, 0).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(1, 2):
        train(model, device, train_loader, optimizer)


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main2(conf):
    torch.cuda.reset_peak_memory_stats()
    dataset_name = conf.dataset.name
    params = conf.model.params[dataset_name]
    print(OmegaConf.to_yaml(conf))
    dataset = get_dataset(dataset_name, conf.root)
    data = dataset[0]
    rank = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(rank)
    torch.manual_seed(12345)
    gpu_num, layers = conf.num_train_worker, params.num_layers
    model = SAGE(data.num_features, conf.hidden_channels, dataset.num_classes,
                 layers).to(rank)
    csr_topo = quiver.CSRTopo(data.edge_index)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo,
                                                 sizes=params.hop,
                                                 device=1,
                                                 mode="GPU")
    # x = quiver.Feature(rank=1,
    #                    device_list=[1],
    #                    device_cache_size="4G",
    #                    cache_policy="device_replicate",
    #                    csr_topo=csr_topo)
    # feature = torch.zeros(data.x.shape)
    # feature[:] = data.x
    # x.from_cpu_tensor(feature)
    y = data.y
    x = data.x

    if dataset_name == "ogbn-products" or dataset_name == "papers100M":
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = (
            split_idx["train"],
            split_idx["valid"],
            split_idx["test"],
        )
    train_loader = torch.utils.data.DataLoader(train_idx,
                                               batch_size=params.batch_size *
                                               gpu_num,
                                               num_workers=14,
                                               shuffle=False,
                                               drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochtimes = []
    for epoch in range(1, conf.num_epoch + 1):
        model.train()
        epoch_start = default_timer()
        for seeds in train_loader:
            optimizer.zero_grad()
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            target_node = n_id[:batch_size]
            adjs = [adj.to(rank) for adj in adjs]
            out = model(x[n_id].to(rank), adjs)
            loss = F.nll_loss(out, y[target_node].squeeze(1).to(rank))
            loss.backward()
            optimizer.step()
        epochtime = default_timer() - epoch_start
        if epoch > 1:
            epochtimes.append(epochtime)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {epochtime}")
    maxgpu = torch.cuda.max_memory_allocated() / 10**9
    print("train finished")


if __name__ == "__main__":
    main2()
