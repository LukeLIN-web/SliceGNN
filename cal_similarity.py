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

    num_train_workers = params.num_train_worker
    num_sample_worker = params.num_sample_worker
    print(OmegaConf.to_yaml(conf))

    dataset = Reddit('/data/Reddit')
    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)

    quiver_sampler = quiver.pyg.GraphSageSampler(
        csr_topo, sizes=[25, 10], device=0, mode='GPU')  

    per_gpu = params.micro_pergpu

    num_train_worker = params.num_train_worker
    data = dataset[0]
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)

    train_loader = torch.utils.data.DataLoader(
        train_idx, batch_size=params.batch_size*num_train_worker, shuffle=False, drop_last=True)

    torch.manual_seed(12345)

    seeds = next(iter(train_loader))
    n_id, batch_size, adjs = quiver_sampler.sample(seeds)
    micro_batchs = get_micro_batch(adjs,
                                   n_id,
                                   batch_size, num_train_worker*per_gpu)

    layer0_max_sum_similiarity = 0
    random = True
    if random == True:
        gpu0 = micro_batchs[:per_gpu//2]
        gpu1 = micro_batchs[per_gpu//2:per_gpu]
        for i in range(len(gpu0)-1):
            max_sum_similiarity += sim.Ochiai(gpu0[i].n_id, gpu0[i+1].n_id)
        for i in range(len(gpu1)-1):
            max_sum_similiarity += sim.Ochiai(gpu1[i].n_id, gpu1[i+1].n_id)
    else:
        for perm in itertools.permutations(micro_batchs):
            gpu0 = perm[:per_gpu//2]
            gpu1 = perm[per_gpu//2:per_gpu]
            sum_similiarity = 0
            for i in range(len(gpu0)-1):
                sum_similiarity += sim.Ochiai(gpu0[i].n_id, gpu0[i+1].n_id)
            for i in range(len(gpu1)-1):
                sum_similiarity += sim.Ochiai(gpu1[i].n_id, gpu1[i+1].n_id)
            if sum_similiarity > max_sum_similiarity:
                max_sum_similiarity = sum_similiarity
    log.log(logging.INFO, 'max_sum_similiarity: {}'.format(max_sum_similiarity))


if __name__ == '__main__':
    main()
