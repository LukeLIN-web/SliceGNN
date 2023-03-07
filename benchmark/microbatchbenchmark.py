"""
sample : quiver
1gpu
getmicrobatch : yes
"""
import logging
import torch
import hydra
from omegaconf import OmegaConf
import quiver
from microGNN.utils import get_nano_batch, cal_metrics, get_dataset
from microGNN.utils.model import SAGE
from timeit import default_timer
import torch.nn.functional as F

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def train(conf):
    dataset_name = conf.dataset.name
    params = conf.model.params
    print(OmegaConf.to_yaml(conf))
    dataset = get_dataset(dataset_name, conf.root)
    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)
    quiver_sampler = quiver.pyg.GraphSageSampler(
        csr_topo, sizes=params.hop, device=0, mode="GPU"
    )
    rank = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_num = params.num_train_worker
    if dataset_name == "ogbn-products" or dataset_name == "papers100M":
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = (
            split_idx["train"],
            split_idx["valid"],
            split_idx["test"],
        )
    else:
        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_loader = torch.utils.data.DataLoader(
        train_idx, batch_size=params.batch_size * gpu_num, shuffle=False, drop_last=True
    )
    layer_num, per_gpu = params.architecture.num_layers, params.micro_pergpu
    nanobatch_num = gpu_num * per_gpu
    torch.manual_seed(12345)
    model = SAGE(data.num_features, 256, dataset.num_classes).to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x, y = data.x, data.y.to(rank)

    for epoch in range(1, 5):
        model.train()
        epoch_start = default_timer()
        for seeds in train_loader:
            optimizer.zero_grad()
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            target_node = n_id[: len(seeds)]
            nano_batchs = get_nano_batch(adjs, n_id, batch_size, gpu_num * per_gpu)
            for n_id, size, adjs in nano_batchs:
                out = model(x[n_id], adjs)  # forward
                loss = F.nll_loss(out, y[target_node][:size])  # reddit loss
                # loss = F.cross_entropy(out, y[target_node][:batch_size]) #yelp loss
                # loss = F.cross_entropy(
                #     out, y[target_node][:batch_size].squeeze()
                # )  # products loss
                loss.backward()
            optimizer.step()

        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {default_timer() - epoch_start}"
        )
    print(f"Maximum GPU memory usage: {torch.cuda.max_memory_allocated()} bytes")


if __name__ == "__main__":
    train()
