import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler

from microGNN import History
from microGNN.models import ScaleSAGE
from microGNN.utils import get_nano_batch


def test_real_dataset():
    dataset = Planetoid("/data/Planetoid", name="Cora")
    data = dataset[0]
    train_loader = NeighborSampler(data.edge_index,
                                   sizes=[10, 5],
                                   batch_size=1024,
                                   shuffle=False,
                                   num_workers=12)
    num_layers = 2
    hidden_channels = 16
    model = ScaleSAGE(in_channels=data.num_features,
                      hidden_channels=hidden_channels,
                      out_channels=dataset.num_classes,
                      num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    x, y = data.x, data.y

    for epoch in range(1):
        model.train()
        for batch_size, n_id, adjs in train_loader:
            target_node = n_id[:batch_size]
            nano_batchs = get_nano_batch(adjs, n_id, batch_size, 2)
            histories = torch.nn.ModuleList([
                History(len(n_id), hidden_channels, 'cpu')
                for _ in range(num_layers - 1)
            ])
            for i, nano_batch in enumerate(nano_batchs):
                out = model(x[n_id][nano_batch.n_id], nano_batch, histories)
                loss = F.cross_entropy(
                    out, y[target_node][i * (nano_batch.size):(i + 1) *
                                        (nano_batch.size)])
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

    # def test():
    #     model.eval()
    #     logits, accs = model(data.x, data.edge_index), []
    #     for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    #         pred = logits[mask].max(1)[1]
    #         acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    #         accs.append(acc)
    #     return accs

    # train_acc, val_acc, test_acc = test()
    # print(f'Train accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}, Test accuracy: {test_acc:.4f}')


if __name__ == "__main__":
    test_real_dataset()
