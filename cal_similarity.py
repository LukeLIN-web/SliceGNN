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
from microGNN.utils.get_micro_batch import get_micro_batch
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
    micro_batchs = get_micro_batch(adjs,
                                   n_id,
                                   batch_size, num_train_worker*per_gpu)
    print(micro_batchs[0][0].shape)
    random = True
    layer_max_sum_similiarity = [0, 0]
    if random == True:
        per_gpu = len(micro_batchs)//num_train_worker
        layer_max_sum_similiarity = [0] * params.architecture.num_layers

        for gpu_idx in range(num_train_worker):
            start = gpu_idx * per_gpu
            end = start + per_gpu
            gpu = micro_batchs[start:end]

            for layer in range(2):
                for i in range(len(gpu) - 1):
                    layer_max_sum_similiarity[layer] += sim.Ochiai(
                        gpu[i][layer], gpu[i + 1][layer])
    else:
        layer_max_sum_similiarity = [0, 0]
        for perm in itertools.permutations(micro_batchs):
            gpu0 = perm[:per_gpu//2]
            gpu1 = perm[per_gpu//2:per_gpu]
            sum_similiarity = 0
            for layer in range(2):
                for i in range(len(gpu0)-1):
                    max_sum_similiarity += sim.Ochiai(
                        gpu0[i][layer], gpu0[i+1][layer])
                for i in range(len(gpu1)-1):
                    max_sum_similiarity += sim.Ochiai(
                        gpu1[i][layer], gpu1[i+1][layer])
                if sum_similiarity > layer_max_sum_similiarity[layer]:
                    layer_max_sum_similiarity[layer] = sum_similiarity
    for layer in range(2):
        log.log(logging.INFO, 'layer {} max_sum_similiarity: {}'.format(
            layer, layer_max_sum_similiarity[layer]))


if __name__ == '__main__':
    main()
