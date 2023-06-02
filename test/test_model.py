import copy
from timeit import default_timer

import quiver
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader, NeighborSampler
from torch_geometric.nn.models.basic_gnn import GraphSAGE
from torch_geometric.testing.decorators import onlyCUDA, withCUDA

from microGNN import History
from microGNN.models import SAGE, ScaleSAGE, loaderSAGE
from microGNN.utils import (get_dataset, get_loader_nano_batch, get_nano_batch,
                            get_nano_batch_histories)

device = torch.device("cuda:0")


@onlyCUDA
def test_sageacc():
    dataset = get_dataset("reddit", "/data/")
    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo,
                                                 sizes=[10, 5],
                                                 device=0,
                                                 mode="GPU")
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_loader = torch.utils.data.DataLoader(train_idx,
                                               batch_size=1024,
                                               shuffle=False,
                                               drop_last=True)
    subgraph_loader = NeighborSampler(
        data.edge_index,
        node_idx=None,
        sizes=[-1],
        batch_size=2048,
        shuffle=False,
        num_workers=2,
    )

    num_layers = 2
    hidden_channels = 256
    model = SAGE(in_channels=data.num_features,
                 hidden_channels=hidden_channels,
                 out_channels=dataset.num_classes,
                 num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x, y = data.x.to(device), data.y.to(device)
    model.train()
    for seeds in train_loader:
        n_id, batch_size, adjs = quiver_sampler.sample(seeds)
        target_node = n_id[:batch_size]
        nano_batchs = get_nano_batch(adjs, n_id, batch_size, 2)
        for i, nb in enumerate(nano_batchs):
            adjs = [adj.to(device) for adj in nb.adjs]
            out = model(x[n_id][nb.n_id], adjs)
            loss = F.nll_loss(
                out, y[target_node][i * (nb.size):(i + 1) * (nb.size)])
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        out = model.inference(x, device, subgraph_loader)
    res = out.argmax(dim=-1) == y
    acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
    acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
    acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
    print(f"Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}")
    assert acc1 > 0.94, "Sanity check , Low training accuracy."


@onlyCUDA
def test_loader():
    device = torch.device("cuda:0")
    dataset = get_dataset("reddit", "/data/")
    data = dataset[0].to(device, 'x', 'y')
    train_loader = NeighborLoader(data,
                                  input_nodes=data.train_mask,
                                  num_neighbors=[25, 10],
                                  batch_size=1024,
                                  shuffle=True)
    subgraph_loader = NeighborLoader(copy.copy(data),
                                     input_nodes=None,
                                     num_neighbors=[-1],
                                     batch_size=1024,
                                     shuffle=False)
    subgraph_loader.data.num_nodes = data.num_nodes
    subgraph_loader.data.n_id = torch.arange(data.num_nodes)
    model = GraphSAGE(
        dataset.num_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
        num_layers=2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        nano_batchs = get_loader_nano_batch(batch, num_nano_batch=2, hop=2)
        for nano_batch in nano_batchs:
            nano_batch.to(device)
            out = model(nano_batch.x, nano_batch.edge_index)
            out = out[:nano_batch.batch_size]
            y = nano_batch.y[:nano_batch.batch_size]

            loss = F.cross_entropy(out, y)
            loss.backward()
        optimizer.step()
    print("Training done")

    model.eval()
    y_hat = model.inference(subgraph_loader, device).argmax(dim=-1)
    y = data.y.to(y_hat.device)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum()))
    assert accs[0] > 0.94, "Sanity check , Low training accuracy."


@onlyCUDA
def test_histories_speed():
    dataset = get_dataset("reddit", "/data/")
    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo,
                                                 sizes=[10, 5, 5],
                                                 device=0,
                                                 mode="GPU")
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_loader = torch.utils.data.DataLoader(train_idx,
                                               batch_size=1024,
                                               shuffle=False,
                                               drop_last=True)
    num_layers = 3
    hidden_channels = 256
    model = ScaleSAGE(in_channels=data.num_features,
                      hidden_channels=hidden_channels,
                      out_channels=dataset.num_classes,
                      num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x, y = data.x.to(device), data.y.to(device)
    model.train()
    for seeds in train_loader:
        n_id, batch_size, adjs = quiver_sampler.sample(seeds)
        target_node = n_id[:batch_size]
        nano_batchs, cached_id = get_nano_batch_histories(
            adjs, n_id, batch_size)
        histories = torch.nn.ModuleList([
            History(cacheid, len(n_id), hidden_channels, device)
            for cacheid in cached_id
        ])
        for i, nb in enumerate(nano_batchs):
            adjs = [adj.to(device) for adj in nb.adjs]
            nbid = nb.n_id.to(device)
            out = model(x[n_id][nb.n_id], nbid, adjs, histories)
            loss = F.nll_loss(
                out, y[target_node][i * (nb.size):(i + 1) * (nb.size)])
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()


@withCUDA
def test_histories_sanity(device):
    dataset = Planetoid("/data/Planetoid", name="Cora")
    data = dataset[0]
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_loader = NeighborSampler(data.edge_index,
                                   sizes=[10, 5],
                                   node_idx=train_idx,
                                   batch_size=1024,
                                   shuffle=False,
                                   num_workers=2)
    subgraph_loader = NeighborSampler(
        data.edge_index,
        node_idx=None,
        sizes=[-1],
        batch_size=2048,
        shuffle=False,
        num_workers=6,
    )
    num_layers = 2
    hidden_channels = 16
    model = ScaleSAGE(in_channels=data.num_features,
                      hidden_channels=hidden_channels,
                      out_channels=dataset.num_classes,
                      num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x, y = data.x.to(device), data.y.to(device)
    model.train()
    for batch_size, n_id, adjs in train_loader:
        target_node = n_id[:batch_size]
        nano_batchs, cached_id = get_nano_batch_histories(adjs,
                                                          n_id,
                                                          batch_size=2,
                                                          num_nano_batch=2)
        histories = torch.nn.ModuleList([
            History(cacheid, len(n_id), hidden_channels, device)
            for cacheid in cached_id
        ])
        for i, nb in enumerate(nano_batchs):
            adjs = [adj.to(device) for adj in nb.adjs]
            nbid = nb.n_id.to(device)
            out = model(x[n_id][nb.n_id], nbid, adjs, histories)
            loss = F.cross_entropy(
                out, y[target_node][i * (nb.size):(i + 1) * (nb.size)])
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        out = model.inference(x, device, subgraph_loader)
    res = out.argmax(dim=-1) == y
    acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
    acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
    acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
    print(f"Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}")


if __name__ == "__main__":
    test_loader()
