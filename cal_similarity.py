'''
sample : quiver
dataset: reddit
getmicrobatch : yes
'''

import itertools
import hydra
from omegaconf import OmegaConf
import torch
from torch_geometric.datasets import Reddit
import quiver
from timeit import default_timer
from microGNN.utils.get_micro_batch import get_micro_batch_withlayer
from microGNN.utils.common_config import gpu
import microGNN.utils.calu_similarity as sim
import logging
# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(config_path='conf', config_name='config', version_base='1.1')
def main(conf):
    model_name, dataset_name = conf.model.name, conf.dataset.name
    params = conf.model.params[dataset_name]
    print(OmegaConf.to_yaml(conf))
    dataset = Reddit('/data/Reddit')
    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)
    quiver_sampler = quiver.pyg.GraphSageSampler(
        csr_topo, sizes=[25, 10], device=0, mode='GPU')
    num_train_worker = params.num_train_worker
    data = dataset[0]
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)

    train_loader = torch.utils.data.DataLoader(
        train_idx, batch_size=params.batch_size*num_train_worker, shuffle=False, drop_last=True)

    torch.manual_seed(12345)

    seeds = next(iter(train_loader))
    n_id, batch_size, adjs = quiver_sampler.sample(seeds)
    per_gpu = params.micro_pergpu
    micro_batchs = get_micro_batch_withlayer(adjs,
                                             n_id,
                                             batch_size, num_train_worker*per_gpu)
    layernode_num = [0] * params.architecture.num_layers
    for micro_batch in micro_batchs:
        for layer in range(params.architecture.num_layers):
            layernode_num[layer] += len(micro_batch[layer])
    random = False
    layer_num = params.architecture.num_layers
    max_sum_common_nodes = [0] * layer_num
    reuse_rate = [0] * layer_num
    if random == True:
        for gpu_idx in range(num_train_worker):
            start = gpu_idx * per_gpu
            end = start + per_gpu
            gpu = micro_batchs[start:end]
            for layer in range(layer_num):
                for i in range(len(gpu) - 1):
                    max_sum_common_nodes[layer] += sim.common_nodes(
                        gpu[i][layer], gpu[i + 1][layer])
        for layer in range(layer_num):
            reuse_rate[layer] = max_sum_common_nodes[layer] / \
                layernode_num[layer]
    else:
        for perm in itertools.permutations(micro_batchs):
            sum_common_nodes = [0] * layer_num
            for gpu_idx in range(num_train_worker):
                start = gpu_idx * per_gpu
                end = start + per_gpu
                gpu = perm[start:end]
                for layer in range(layer_num):
                    for i in range(len(gpu) - 1):
                        sum_common_nodes[layer] += sim.common_nodes(
                            gpu[i][layer], gpu[i + 1][layer])
            for layer in range(layer_num):
                if sum_common_nodes[layer] > max_sum_common_nodes[layer]:
                    max_sum_common_nodes[layer] = sum_common_nodes[layer]
        for layer in range(layer_num):
            reuse_rate[layer] = max_sum_common_nodes[layer] / \
                layernode_num[layer]
    for layer in range(layer_num):
        log.log(logging.INFO, ',{},{},{},{},{}'.format(
            random, num_train_worker, num_train_worker*per_gpu, layer, reuse_rate[layer]))


if __name__ == '__main__':
    main()
