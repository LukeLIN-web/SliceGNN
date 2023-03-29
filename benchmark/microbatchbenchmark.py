"""
sample : quiver
1gpu
"""
import logging
from timeit import default_timer

import hydra
import torch
from omegaconf import OmegaConf
from torch_geometric.loader import NeighborSampler

import quiver
from microGNN.models import SAGE, criterion
from microGNN.utils import get_dataset, get_nano_batch

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def train(conf):
    dataset_name = conf.dataset.name
    params = conf.model.params
    print(OmegaConf.to_yaml(conf))
    dataset = get_dataset(dataset_name, conf.root)
    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo,
                                                 sizes=params.hop,
                                                 device=0,
                                                 mode="GPU")
    rank = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_num, per_gpu = params.num_train_worker, params.nano_pergpu
    if dataset_name == "ogbn-products" or dataset_name == "papers100M":
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = (
            split_idx["train"],
            split_idx["valid"],
            split_idx["test"],
        )
    else:
        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_loader = torch.utils.data.DataLoader(train_idx,
                                               batch_size=params.batch_size *
                                               gpu_num,
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
    torch.manual_seed(12345)
    model = SAGE(data.num_features, 256, dataset.num_classes).to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x, y = data.x.to(rank), data.y.to(rank)

    for epoch in range(1, params.num_epoch + 1):
        model.train()
        epoch_start = default_timer()
        for seeds in train_loader:
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            target_node = n_id[:batch_size]
            nano_batchs = get_nano_batch(adjs, n_id, batch_size,
                                         gpu_num * per_gpu)
            for i, nano_batch in enumerate(nano_batchs):
                nano_batch_adjs = [adj.to(rank) for adj in nano_batch.adjs]
                out = model(x[n_id][nano_batch.n_id], nano_batch_adjs)
                loss = criterion(
                    out,
                    y[target_node][i * (nano_batch.size):(i + 1) *
                                   (nano_batch.size)],
                    dataset_name,
                )
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {default_timer() - epoch_start}"
        )
        if epoch % 5 == 0:  # We evaluate on a single GPU for now
            model.eval()
            with torch.no_grad():
                out = model.inference(x, rank, subgraph_loader)
            res = out.argmax(dim=-1) == y
            acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
            assert acc1 > 0.90, "Sanity check , Low training accuracy."
            acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
            acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
            print(f"Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}")
    print(
        f"Maximum GPU memory usage: {torch.cuda.max_memory_allocated()/10**9} G bytes"
    )


if __name__ == "__main__":
    train()
