import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler

from microGNN.models import ScaleSAGE
from microGNN.prune import prune, prune_computation_graph
from microGNN.utils import get_nano_batch


def test_real_dataset():

    model = ScaleSAGE(1433, 16, 7, 2)
    dataset = Planetoid("/data/Planetoid", name="Cora")
    data = dataset[0]
    loader = NeighborSampler(data.edge_index,
                             sizes=[10, 5],
                             batch_size=1024,
                             shuffle=False,
                             num_workers=12)
    out = model(data.x, data.edge_index)

    model = ScaleSAGE(dataset.num_features, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    def train():
        model.train()

        total_loss = total_nodes = 0
        for batch in loader:
            batch = batch
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()

            nodes = batch.train_mask.sum().item()
            total_loss += loss.item() * nodes
            total_nodes += nodes

        return total_loss / total_nodes

    @torch.no_grad()
    def test():  # Inference should be performed on the full graph.
        model.eval()

        out = model.inference(data.x)
        y_pred = out.argmax(dim=-1)

        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            correct = y_pred[mask].eq(data.y[mask]).sum().item()
            accs.append(correct / mask.sum().item())
        return accs

    for epoch in range(1):
        loss = train()
        if epoch % 5 == 0:
            train_acc, val_acc, test_acc = test()
            print(
                f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                f'Val: {val_acc:.4f}, test: {test_acc:.4f}')
        else:
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
