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
from microGNN.utils import cal_metrics, check_memory, get_dataset

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def train(conf):
    dataset_name = conf.dataset.name
    params = conf.model.params[dataset_name]
    print(OmegaConf.to_yaml(conf))
    dataset = get_dataset(dataset_name, conf.root)
    data = dataset[0]
    rank = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_num, per_gpu, layers = conf.num_train_worker, conf.nano_pergpu, params.num_layers
    model = SAGE(data.num_features, conf.hidden_channels, dataset.num_classes,
                 layers).to(rank)
    if dataset_name == "ogbn-products" or dataset_name == "papers100M":
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = (
            split_idx["train"],
            split_idx["valid"],
            split_idx["test"],
        )
    else:
        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_loader = NeighborSampler(data.edge_index,
                                   sizes=params.hop,
                                   batch_size=conf.batch_size,
                                   num_workers=14,
                                   shuffle=False,
                                   drop_last=True)
    torch.manual_seed(12345)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    check_memory()
    y = data.y
    x = data.x
    epochtimes = []
    acc3 = -1
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, conf.num_epoch + 1):
        model.train()
        epoch_start = default_timer()
        for n_id, batch_size, adjs in train_loader:
            optimizer.zero_grad()
            print(n_id.shape)
            print(batch_size)
            target_node = n_id[:batch_size]
            adjs = [adj.to(rank) for adj in adjs]
            out = model(x[n_id], adjs)
            loss = criterion(
                out,
                y[target_node],
                dataset_name,
            )
            loss.backward()
            optimizer.step()
        epochtime = default_timer() - epoch_start
        if epoch > 1:
            epochtimes.append(epochtime)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {epochtime}")
        check_memory()
    maxgpu = torch.cuda.max_memory_allocated() / 10**9
    print("train finished")
    # if dataset_name == "ogbn-products" or dataset_name == "papers100M":
    #     pass
    # else:
    #     model.eval()
    #     with torch.no_grad():
    #         out = model.inference(x, rank, subgraph_loader)
    #     res = out.argmax(dim=-1) == y  #  big graph may oom
    #     acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
    #     assert acc1 > 0.90, "Sanity check , Low training accuracy."
    #     acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
    #     acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
    #     print(f"Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}")

    # metric = cal_metrics(epochtimes)
    # log.log(
    #     logging.INFO,
    #     f',origin,{dataset_name},{gpu_num * per_gpu},{layers},{metric["mean"]:.2f}, {maxgpu:.2f}, {acc3:.4f}',
    # )


if __name__ == "__main__":
    train()
