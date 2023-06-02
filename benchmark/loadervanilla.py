import logging
from timeit import default_timer

import hydra
import quiver
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
from omegaconf import OmegaConf
from torch_geometric.loader import NeighborLoader, NeighborSampler
from torch_geometric.nn.models.basic_gnn import GraphSAGE
from utils import get_model

from microGNN.models import criterion
from microGNN.utils import cal_metrics, get_dataset, get_loader_nano_batch

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def train(conf):
    torch.cuda.reset_peak_memory_stats()
    dataset_name = conf.dataset.name
    params = conf.model.params[dataset_name]
    print(OmegaConf.to_yaml(conf))
    dataset = get_dataset(dataset_name, conf.root)
    data = dataset[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(12345)
    train_loader = NeighborLoader(data,
                                  input_nodes=data.train_mask,
                                  num_neighbors=[25, 10],
                                  batch_size=1024,
                                  shuffle=True)
    # subgraph_loader = NeighborLoader(copy.copy(data),
    #                                  input_nodes=None,
    #                                  num_neighbors=[-1],
    #                                  batch_size=1024,
    #                                  shuffle=False)

    # if dataset_name == "ogbn-products" or dataset_name == "papers100M":
    #     split_idx = dataset.get_idx_split()
    #     train_idx, valid_idx, test_idx = (
    #         split_idx["train"],
    #         split_idx["valid"],
    #         split_idx["test"],
    #     )
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

    model = GraphSAGE(
        dataset.num_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
        num_layers=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    y = data.y.to(device)
    epochtimes = []
    acc3 = -1
    for epoch in range(1, conf.num_epoch + 1):
        model.train()
        epoch_start = default_timer()
        for batch in train_loader:
            optimizer.zero_grad()
            batch.to(device)
            out = model(batch.x, batch.edge_index)
            out = out[:batch.batch_size]
            y = batch.y[:batch.batch_size]
            loss = F.cross_entropy(out, y)
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
        f',nano,{dataset_name},{gpu_num * per_gpu},{layers},{metric["mean"]:.2f},{params.batch_size} ,{maxgpu:.2f}',
    )

    # if dataset_name == "ogbn-products" or dataset_name == "papers100M":
    #     evaluator = Evaluator(name=dataset_name)
    #     model.eval()
    #     out = model.inference(x, device, subgraph_loader)

    #     y_true = y.cpu()
    #     y_pred = out.argmax(dim=-1, keepdim=True)

    #     acc1 = evaluator.eval({
    #         'y_true': y_true[split_idx['train']],
    #         'y_pred': y_pred[split_idx['train']],
    #     })['acc']
    #     acc2 = evaluator.eval({
    #         'y_true': y_true[split_idx['valid']],
    #         'y_pred': y_pred[split_idx['valid']],
    #     })['acc']
    #     acc3 = evaluator.eval({
    #         'y_true': y_true[split_idx['test']],
    #         'y_pred': y_pred[split_idx['test']],
    #     })['acc']
    # else:
    #     model.eval()
    #     with torch.no_grad():
    #         out = model.inference(x, device, subgraph_loader)
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
