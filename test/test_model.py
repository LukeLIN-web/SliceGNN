import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler

from microGNN import History
from microGNN.models import ScaleSAGE
from microGNN.utils import get_dataset, get_nano_batch


def test_real_dataset():
    # dataset = Planetoid("/data/Planetoid", name="Cora")
    dataset = get_dataset("reddit", "/data/")
    data = dataset[0]
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_loader = NeighborSampler(data.edge_index,
                                   sizes=[10, 5],
                                   node_idx=train_idx,
                                   batch_size=1024,
                                   shuffle=False,
                                   num_workers=12)
    rank = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # rank = torch.device("cpu")
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
                      num_layers=num_layers).to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x, y = data.x.to(rank), data.y.to(rank)
    print("Start training...")
    for epoch in range(6):
        model.train()
        for batch_size, n_id, adjs in train_loader:
            target_node = n_id[:batch_size]
            nano_batchs = get_nano_batch(adjs, n_id, batch_size, 2)
            histories = torch.nn.ModuleList([
                History(len(n_id), hidden_channels, rank)
                for _ in range(num_layers - 1)
            ])
            for i, nb in enumerate(nano_batchs):
                adjs = [adj.to(rank) for adj in nb.adjs]
                nb.n_id.to(rank)
                out = model(x[n_id][nb.n_id], nb.n_id, adjs, histories)
                loss = F.cross_entropy(
                    out, y[target_node][i * (nb.size):(i + 1) * (nb.size)])
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

    model.eval()
    with torch.no_grad():
        out = model.inference(x, rank, subgraph_loader)
    res = out.argmax(dim=-1) == y
    acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
    # assert acc1 > 0.90, "Sanity check , Low training accuracy."
    acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
    acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
    print(f"Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}")


if __name__ == "__main__":
    test_real_dataset()
