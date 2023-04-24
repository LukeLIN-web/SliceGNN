# Reaches around 0.7945 ± 0.0059 test accuracy.

from timeit import default_timer

import hydra
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from omegaconf import OmegaConf
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler

import quiver
from microGNN import History
from microGNN.models import GAT
from microGNN.prune import prune_computation_graph
from microGNN.utils import cal_metrics, get_dataset, get_nano_batch_histories

# train_loader = NeighborSampler(data.edge_index,
#                                node_idx=train_idx,
#                                sizes=[10, 10, 10],
#                                batch_size=512,
#                                shuffle=True,
#                                num_workers=12)
# subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
#                                   batch_size=1024, shuffle=False,
#                                   num_workers=12)

# y = data.y.squeeze().to(rank)


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc


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

    if dataset_name == "ogbn-products" or dataset_name == "papers100M":
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = (
            split_idx["train"],
            split_idx["valid"],
            split_idx["test"],
        )
    else:
        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    gpu_num, per_gpu, layers = conf.num_train_worker, conf.nano_pergpu, len(
        params.hop)
    train_loader = torch.utils.data.DataLoader(train_idx,
                                               batch_size=params.batch_size *
                                               gpu_num,
                                               num_workers=14,
                                               shuffle=False,
                                               drop_last=True)
    model = GAT(dataset.num_features,
                conf.hidden_channels,
                dataset.num_classes,
                num_layers=3,
                heads=params.heads)
    model = model.to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    y = data.y.squeeze()
    epochtimes = []
    acc3 = -1
    model.train()
    for epoch in range(1, conf.num_epoch + 1):

        for seeds in train_loader:
            optimizer.zero_grad()
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            epoch_start = default_timer()
            target_node = n_id[:batch_size]
            nano_batchs, cached_id = get_nano_batch_histories(
                adjs, n_id, batch_size)
            histories = torch.nn.ModuleList([
                History(cacheid, len(n_id),
                        conf.hidden_channels * params.heads, rank)
                for cacheid in cached_id
            ])
            for i, nb in enumerate(nano_batchs):
                adjs = [adj.to(rank) for adj in nb.adjs]
                nbid = nb.n_id.to(rank)
                out = model(x[n_id][nb.n_id], adjs, nbid, histories)
                loss = F.nll_loss(
                    out,
                    y[target_node][i * (nb.size):(i + 1) * (nb.size)].to(rank))
                loss.backward()
            optimizer.step()
        epochtime = default_timer() - epoch_start
        if epoch > 1:
            epochtimes.append(epochtime)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {epochtime}")


if __name__ == "__main__":
    train()
