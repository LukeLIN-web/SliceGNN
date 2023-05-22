import logging
from timeit import default_timer

import hydra
import torch
from ogb.nodeproppred import Evaluator
from omegaconf import OmegaConf
from torch_geometric.loader import NeighborSampler
from utils import get_model

import quiver
from microGNN.models import criterion
from microGNN.utils import cal_metrics, get_dataset

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def train(conf):
    torch.cuda.reset_peak_memory_stats()
    dataset_name = conf.dataset.name
    params = conf.model.params[dataset_name]
    print(OmegaConf.to_yaml(conf))
    dataset = get_dataset(dataset_name, conf.root)
    data = dataset[0]

    rank = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(rank)
    torch.manual_seed(12345)
    gpu_num, per_gpu, layers = conf.num_train_worker, conf.nano_pergpu, len(
        params.hop)
    model_params = {
        'inputs_channels': data.num_features,
        'hidden_channels': params.hidden_channels,
        'output_channels': dataset.num_classes,
        'num_layers': layers,
    }
    if conf.model.name == "gat":
        model_params['num_heads'] = params.heads

    model = get_model(conf.model.name, model_params, scale=False).to(rank)
    csr_topo = quiver.CSRTopo(data.edge_index)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo,
                                                 sizes=params.hop,
                                                 device=1,
                                                 mode="GPU")
    x = quiver.Feature(rank=1,
                       device_list=[1],
                       device_cache_size="4G",
                       cache_policy="device_replicate",
                       csr_topo=csr_topo)
    feature = torch.zeros(data.x.shape)
    feature[:] = data.x
    x.from_cpu_tensor(feature)
    y = data.y.to(rank)

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
                                               num_workers=14,
                                               shuffle=False,
                                               drop_last=True)
    if dataset_name != "papers100M":
        subgraph_loader = NeighborSampler(
            data.edge_index,
            node_idx=None,
            sizes=[-1],
            batch_size=1024,
            shuffle=False,
            num_workers=14,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    epochtimes = []

    for epoch in range(1, conf.num_epoch + 1):
        model.train()
        samplertimes = []
        traintimes = []
        for seeds in train_loader:
            epoch_start = default_timer()
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            sampletime = default_timer()
            optimizer.zero_grad()
            target_node = n_id[:batch_size]
            adjs = [adj.to(rank) for adj in adjs]
            out = model(x[n_id].to(rank), adjs)
            loss = criterion(
                out,
                y[target_node],
                dataset_name,
            )
            loss.backward()
            optimizer.step()
            traintime = default_timer()
            samplertimes.append(sampletime - epoch_start)
            traintimes.append(traintime - sampletime)  # train时间是sampler两倍.
        epoch_end = default_timer()
        if epoch > 1:
            epochtimes.append(epoch_end - epoch_start)
        # print(f"Epoch: {epoch:03d}, Time: {epoch_end - epoch_start:.4f}")
        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, sampler Time: {sum(samplertimes):.4f}, Train Time: {sum(traintimes):.4f}"
        )
    maxgpu = torch.cuda.max_memory_allocated() / 10**9
    print(f"max gpu memory {maxgpu:.2f} GB")
    metric = cal_metrics(epochtimes)
    log.log(
        logging.INFO,
        f',origin,{dataset_name},{gpu_num * per_gpu},{layers},{metric["mean"]:.2f},{params.batch_size} ,{maxgpu:.2f}',
    )


if __name__ == "__main__":
    train()
