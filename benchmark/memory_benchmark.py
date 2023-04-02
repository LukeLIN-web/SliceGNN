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
from microGNN import History
from microGNN.models import ScaleSAGE, criterion
from microGNN.utils import (cal_metrics, check_memory, get_dataset,
                            get_nano_batch)

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def train(conf):
    dataset_name = conf.dataset.name
    params = conf.model.params[dataset_name]
    print(OmegaConf.to_yaml(conf))
    dataset = get_dataset(dataset_name, conf.root)
    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)
    # quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo,
    #                                              sizes=params.hop,
    #                                              device=0,
    #                                              mode="GPU")
    x = quiver.Feature(rank=0,
                       device_list=[0],
                       device_cache_size="4G",
                       cache_policy="device_replicate",
                       csr_topo=csr_topo)
    feature = torch.zeros(data.x.shape)  # 感觉这种paper100要出问题.
    feature[:] = data.x
    x.from_cpu_tensor(feature)
    check_memory()
    # rank = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if dataset_name == "ogbn-products" or dataset_name == "papers100M":
    #     split_idx = dataset.get_idx_split()
    #     train_idx, valid_idx, test_idx = (
    #         split_idx["train"],
    #         split_idx["valid"],
    #         split_idx["test"],
    #     )
    # else:
    #     train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    # gpu_num, per_gpu, layers = conf.num_train_worker, conf.nano_pergpu, params.num_layers
    # train_loader = torch.utils.data.DataLoader(train_idx,
    #                                            batch_size=conf.batch_size *
    #                                            gpu_num,
    #                                            num_workers=14,
    #                                            shuffle=False,
    #                                            drop_last=True)
    # torch.manual_seed(12345)
    # model = ScaleSAGE(data.num_features, conf.hidden_channels,
    #                   dataset.num_classes, layers).to(rank)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # y = data.y.to(rank)
    # torch.cuda.reset_max_memory_allocated()
    # for epoch in range(1, conf.num_epoch + 1):
    #     model.train()
    #     for seeds in train_loader:
    #         optimizer.zero_grad()
    #         n_id, batch_size, adjs = quiver_sampler.sample(seeds)
    #         target_node = n_id[:batch_size]
    #         nano_batchs = get_nano_batch(adjs, n_id, batch_size,
    #                                      gpu_num * per_gpu)
    #         histories = torch.nn.ModuleList([
    #             History(len(n_id), conf.hidden_channels, rank)
    #             for _ in range(layers - 1)
    #         ])
    #         for i, nb in enumerate(nano_batchs):
    #             adjs = [adj.to(rank) for adj in nb.adjs]
    #             nbid = nb.n_id.to(rank)
    #             out = model(x[n_id][nb.n_id], nbid, adjs, histories)
    #             loss = criterion(
    #                 out, y[target_node][i * (nb.size):(i + 1) * (nb.size)],
    #                 dataset_name)
    #             loss.backward()
    #         optimizer.step()
    #     check_memory()
    # maxgpu = torch.cuda.max_memory_allocated() / 10**9
    # log.log(
    #     logging.INFO,
    #     f',scalesage,{dataset_name},{gpu_num * per_gpu},{layers}, {maxgpu:.2f}',
    # )


if __name__ == "__main__":
    train()
