from timeit import default_timer

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler
from torch_geometric.testing.decorators import onlyCUDA, withCUDA

import quiver
from microGNN import History
from microGNN.models import ScaleSAGE
from microGNN.utils import get_dataset, get_nano_batch


@onlyCUDA
def test_acc():
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
        num_workers=6,
    )
    device = torch.device("cuda:0")
    torch.manual_seed(12345)
    num_layers = 2
    hidden_channels = 256
    model = ScaleSAGE(in_channels=data.num_features,
                      hidden_channels=hidden_channels,
                      out_channels=dataset.num_classes,
                      num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x, y = data.x.to(device), data.y.to(device)
    print("Start training...")
    for epoch in range(5):
        model.train()
        epoch_start = default_timer()
        for seeds in train_loader:
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            target_node = n_id[:batch_size]
            nano_batchs = get_nano_batch(adjs, n_id, batch_size, 2)
            histories = torch.nn.ModuleList([
                History(len(n_id), hidden_channels, device)
                for _ in range(num_layers - 1)
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
        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {default_timer() - epoch_start}"
        )

    model.eval()
    with torch.no_grad():
        out = model.inference(x, device, subgraph_loader)
    res = out.argmax(dim=-1) == y
    acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
    assert acc1 > 0.90, "Sanity check , Low training accuracy."
    acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
    acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
    print(f"Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}")


@withCUDA
def test_real_dataset(device):
    dataset = Planetoid("/data/Planetoid", name="Cora")
    data = dataset[0]
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_loader = NeighborSampler(data.edge_index,
                                   sizes=[10, 5],
                                   node_idx=train_idx,
                                   batch_size=1024,
                                   shuffle=False,
                                   num_workers=12)
    subgraph_loader = NeighborSampler(
        data.edge_index,
        node_idx=None,
        sizes=[-1],
        batch_size=2048,
        shuffle=False,
        num_workers=6,
    )
    torch.manual_seed(12345)
    num_layers = 2
    hidden_channels = 16
    model = ScaleSAGE(in_channels=data.num_features,
                      hidden_channels=hidden_channels,
                      out_channels=dataset.num_classes,
                      num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x, y = data.x.to(device), data.y.to(device)
    print("Start training...")
    for epoch in range(1):
        model.train()
        for batch_size, n_id, adjs in train_loader:
            target_node = n_id[:batch_size]
            nano_batchs = get_nano_batch(adjs, n_id, batch_size, 2)
            histories = torch.nn.ModuleList([
                History(len(n_id), hidden_channels, device)
                for _ in range(num_layers - 1)
            ])
            for i, nb in enumerate(nano_batchs):
                adjs = [adj.to(device) for adj in nb.adjs]
                nbid = nb.n_id.to(device)
                print(x[n_id][nb.n_id].device)
                out = model(x[n_id][nb.n_id], nbid, adjs, histories)
                loss = F.cross_entropy(
                    out, y[target_node][i * (nb.size):(i + 1) * (nb.size)])
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

    model.eval()
    with torch.no_grad():
        out = model.inference(x, device, subgraph_loader)
    res = out.argmax(dim=-1) == y
    acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
    # assert acc1 > 0.90, "Sanity check , Low training accuracy."
    acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
    acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
    print(f"Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}")


if __name__ == "__main__":
    # test_real_dataset("cpu")
    # test_real_dataset("cuda:0")
    test_acc()
