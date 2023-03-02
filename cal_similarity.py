"""
sample : quiver
dataset: reddit
getmicrobatch : yes
"""

import itertools
import logging
import torch
import hydra
from omegaconf import OmegaConf
import quiver
from microGNN.utils import get_nano_batch_withlayer, cal_metrics, get_dataset
import microGNN.utils.calu_similarity as sim

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def onebyone(conf):
    dataset_name = conf.dataset.name
    params = conf.model.params
    print(OmegaConf.to_yaml(conf))
    dataset = get_dataset(dataset_name, conf.root)
    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)
    quiver_sampler = quiver.pyg.GraphSageSampler(
        csr_topo, sizes=params.hop, device=0, mode="GPU"
    )
    gpu_num = params.num_train_worker
    if dataset_name == "ogbn-products":
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
    maxrate = [[] for i in range(layer_num)]
    minrate = [[] for i in range(layer_num)]
    random = True
    for seeds in train_loader:
        n_id, batch_size, adjs = quiver_sampler.sample(seeds)  # there is gloabl n_id
        nano_batchs = get_nano_batch_withlayer(adjs, n_id, batch_size, nanobatch_num)
        layernode_num, max_sum_common_nodes, min_sum_common_nodes = (
            [0] * layer_num,
            [0] * layer_num,
            [10] * layer_num,
        )
        # calculate all nodes in each layer, sum all nano batchs up
        for nano_batch in nano_batchs:
            for layer in range(layer_num):
                layernode_num[layer] += len(nano_batch[layer])
        if random is True:
            for layer in range(layer_num):
                for i in range(nanobatch_num - 1):
                    max_sum_common_nodes[layer] += sim.common_nodes_num(
                        nano_batchs[i][layer], nano_batchs[i + 1][layer]
                    )
                maxrate[layer].append(
                    max_sum_common_nodes[layer] / layernode_num[layer]
                )
        else:
            for perm in itertools.permutations(nano_batchs):
                sum_nodes = [0] * layer_num
                for layer in range(layer_num):
                    for i in range(nanobatch_num - 1):
                        sum_nodes[layer] += sim.common_nodes_num(
                            perm[i][layer], perm[i + 1][layer]
                        )
                for layer in range(layer_num):
                    if sum_nodes[layer] > max_sum_common_nodes[layer]:
                        max_sum_common_nodes[layer] = sum_nodes[layer]
                    if sum_nodes[layer] < min_sum_common_nodes[layer]:
                        min_sum_common_nodes[layer] = sum_nodes[layer]
            for layer in range(layer_num):
                maxrate[layer].append(
                    max_sum_common_nodes[layer] / layernode_num[layer]
                )
                minrate[layer].append(
                    min_sum_common_nodes[layer] / layernode_num[layer]
                )
    for layer in range(layer_num):
        max_metrics = cal_metrics(maxrate[layer])
        if random is True:
            log.log(
                logging.INFO,
                f',{random},{gpu_num},{gpu_num * per_gpu},{layer},{max_metrics["mean"]:.2f}',
            )
        else:
            min_metrics = cal_metrics(minrate[layer])
            log.log(
                logging.INFO,
                f',{random},{gpu_num},{gpu_num * per_gpu},{layer},{max_metrics["mean"]:.2f}, {min_metrics["mean"]:.2f}',
            )


if __name__ == "__main__":
    onebyone()
