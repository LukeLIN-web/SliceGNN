import logging
from timeit import default_timer

import hydra
import torch
from ogb.nodeproppred import Evaluator
from omegaconf import OmegaConf
from torch_geometric.loader import NeighborSampler
from utils import get_model

import quiver
from microGNN import History
from microGNN.models import criterion
from microGNN.utils import cal_metrics, get_dataset, get_nano_batch_histories

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
    # if dataset_name != "papers100M":
    #     subgraph_loader = NeighborSampler(
    #         data.edge_index,
    #         node_idx=None,
    #         sizes=[-1],
    #         batch_size=2048,
    #         shuffle=False,
    #         num_workers=14,
    #     )
    model_params = {
        'inputs_channels': data.num_features,
        'hidden_channels': params.hidden_channels,
        'output_channels': dataset.num_classes,
        'num_layers': layers,
    }
    if conf.model.name == "gat":
        emb_dim = params.hidden_channels * params.heads
        model_params['num_heads'] = params.heads
    else:
        emb_dim = params.hidden_channels
    model = get_model(conf.model.name, model_params, scale=True).to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    y = data.y
    epochtimes = []
    acc3 = -1

    for epoch in range(1, conf.num_epoch + 1):
        model.train()
        epoch_start = default_timer()
        for seeds in train_loader:
            optimizer.zero_grad()
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            target_node = n_id[:batch_size]
            nano_batchs, cached_id = get_nano_batch_histories(
                adjs, n_id, batch_size)
            histories = torch.nn.ModuleList([
                History(cacheid, len(n_id), emb_dim, rank)
                for cacheid in cached_id
            ])
            for i, nb in enumerate(nano_batchs):
                adjs = [adj.to(rank) for adj in nb.adjs]
                nbid = nb.n_id.to(rank)
                out = model(x[n_id][nb.n_id], nbid, adjs, histories)
                loss = criterion(
                    out,
                    y[target_node][i * (nb.size):(i + 1) * (nb.size)].to(rank),
                    dataset_name)
                loss.backward()
            optimizer.step()
        epochtime = default_timer() - epoch_start
        if epoch > 1:
            epochtimes.append(epochtime)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {epochtime}")
    maxgpu = torch.cuda.max_memory_allocated() / 10**9
    metric = cal_metrics(epochtimes)
    log.log(
        logging.INFO,
        f',scale+{conf.model.name},{dataset_name},{gpu_num * per_gpu},{layers},{metric["mean"]:.2f},{params.batch_size}, {maxgpu:.2f}',
    )

    # model.eval()
    # if dataset_name == "ogbn-products":
    #     evaluator = Evaluator(name=dataset_name)
    #     out = model.inference(x, rank, subgraph_loader)

    #     y_true = y.cpu()
    #     y_pred = out.argmax(dim=-1, keepdim=True)

    #     acc1 = evaluator.eval({
    #         'y_true': y_true[train_idx],
    #         'y_pred': y_pred[train_idx],
    #     })['acc']
    #     acc2 = evaluator.eval({
    #         'y_true': y_true[valid_idx],
    #         'y_pred': y_pred[valid_idx],
    #     })['acc']
    #     acc3 = evaluator.eval({
    #         'y_true': y_true[test_idx],
    #         'y_pred': y_pred[test_idx],
    #     })['acc']
    #     print(f"Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}")
    # elif dataset_name == "papers100M":
    #     pass
    # else:
    #     with torch.no_grad():
    #         out = model.inference(x, rank, subgraph_loader).cpu()
    #     res = out.argmax(dim=-1) == y.cpu()
    #     acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
    #     acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
    #     acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
    #     print(f"Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}")

    # metric = cal_metrics(epochtimes)
    # log.log(
    #     logging.INFO,
    #     f',scale+{conf.model.name},{dataset_name},{gpu_num * per_gpu},{layers},{metric["mean"]:.2f},{params.batch_size}, {maxgpu:.2f}, {acc3:.4f}',
    # )


if __name__ == "__main__":
    train()
