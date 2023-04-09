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
from torch_geometric.datasets import AttributedGraphDataset
from torch_geometric.loader import NeighborSampler

import quiver
from microGNN.models import SAGE, criterion
from microGNN.utils import cal_metrics, check_memory, get_dataset

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def train(conf):
    torch.cuda.reset_peak_memory_stats()
    dataset_name = conf.dataset.name
    params = conf.model.params[dataset_name]
    print(OmegaConf.to_yaml(conf))
    dataset = AttributedGraphDataset(conf.root, "mag")
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
            out = model(x[n_id], adjs)
            loss = criterion(
                out,
                y[target_node].to(rank),
                dataset_name,
            )
            loss.backward()
            optimizer.step()
        epochtime = default_timer() - epoch_start
        if epoch > 1:
            epochtimes.append(epochtime)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {epochtime}")
    maxgpu = torch.cuda.max_memory_allocated() / 10**9
    print("train finished")


if __name__ == "__main__":
    train()
