import logging
import os
from timeit import default_timer

import hydra
import quiver
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch_geometric.loader import NeighborSampler

from microGNN.models import SAGE
from microGNN.utils import cal_metrics, get_dataset, get_nano_batch
from microGNN.utils.common_config import gpu

log = logging.getLogger(__name__)
from enum import Enum


def get_global_model(run_config):
    params = run_config['params']
    dataset = run_config['dataset']
    num_features, num_classes = dataset.num_features, dataset.num_classes
    model = SAGE(num_features, params.hidden_channels, num_classes)
    model.share_memory(
    )  # gradients are allocated lazily, so they are not shared here
    return model


def get_run_config(conf, params):
    run_config = {}
    run_config['conf'] = conf
    run_config['params'] = params
    run_config['train_workers'] = [
        gpu(i) for i in range(conf.num_train_worker)
    ]
    run_config['sample_workers'] = [
        gpu(conf.num_train_worker + i) for i in range(conf.num_sample_worker)
    ]
    run_config['queue'] = mp.Queue(maxsize=conf.queue_size)
    run_config['ratio'] = 3
    return run_config


def run_sample(worker_id, run_config, quiver_sampler):
    conf = run_config['conf']
    params = run_config['params']
    ratio = run_config['ratio']

    data = run_config['dataset'][0]
    nano_queues = run_config['queue']
    num_nano = conf.num_sample_worker + ratio * conf.num_train_worker
    mq_sem = run_config['mq_sem']
    sampler_stop_event = run_config['sampler_stop_event'][worker_id]
    ctx = run_config['sample_workers'][worker_id]
    torch.cuda.set_device(ctx)

    print('[Sample Worker {:d}/{:d}] Started with PID {:d} Device {:s}'.format(
        worker_id, conf.num_sample_worker, os.getpid(), ctx))

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)

    train_loader = torch.utils.data.DataLoader(
        train_idx,
        batch_size=params.batch_size * num_nano,
        shuffle=False,
        drop_last=True,
    )

    epochtimes = []
    for epoch in range(1, conf.num_epoch + 1):
        tic = default_timer()
        sampler_stop_event.clear()
        for seeds in train_loader:
            mq_sem.release()
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            nano_batchs = get_nano_batch(adjs, n_id, batch_size, num_nano)
            nano_queues.put((n_id, nano_batchs))
        sampler_stop_event.set()
        toc = default_timer()
        epochtimes.append(toc - tic)
        # Get the current number of elements in the queue
        num_elements = nano_queues.qsize()
        print("Number of elements in the queue:", num_elements)
    metric = cal_metrics(epochtimes)
    print(
        f'sample finished. {conf.model.name},{metric["mean"]:.2f},{params.batch_size}'
    )


class TrainerType(Enum):
    Trainer = 1
    Switcher = 2


def run_train(worker_id, run_config, x, trainer_type):
    conf = run_config['conf']
    params = run_config['params']
    num_worker = conf.num_train_worker
    dataset = run_config['dataset']
    queue = run_config['queue']

    ctx = None
    if (trainer_type == TrainerType.Trainer):
        ctx = run_config['train_workers'][worker_id]
    elif (trainer_type == TrainerType.Switcher):
        ctx = run_config['sample_workers'][worker_id]
        sampler_stop_event = run_config['sampler_stop_event'][worker_id]
    global_cpu_model = run_config['global_cpu_model']

    mq_sem = run_config['mq_sem']
    train_device = torch.device(ctx)
    torch.cuda.set_device(train_device)
    print(
        '[{:10s} Worker  {:d}/{:d}] Started with PID {:d} Train Device '.
        format(trainer_type.name,
               worker_id, conf.num_train_worker + conf.num_sample_worker,
               os.getpid()), train_device)

    num_features, num_classes = dataset.num_features, dataset.num_classes
    model = SAGE(num_features, params.hidden_channels, num_classes)
    for (param, cpu_param) in zip(model.parameters(),
                                  global_cpu_model.parameters()):
        param.data = cpu_param.data.clone()
    model = model.to(train_device)
    model_copy = SAGE(num_features, params.hidden_channels, num_classes)
    for (param, cpu_param) in zip(model_copy.parameters(),
                                  global_cpu_model.parameters()):
        param.data = cpu_param.data.clone()
    model_copy = model_copy.to(train_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    data = dataset[0]
    batch_size = params.batch_size * num_worker

    model.train()
    model_copy.train()

    # if worker_id == 0:
    #     subgraph_loader = NeighborSampler(
    #         data.edge_index,
    #         node_idx=None,
    #         sizes=[-1],
    #         batch_size=2048,
    #         shuffle=False,
    #         num_workers=6,
    #     )

    y = data.y.to(train_device)
    epochtimes = []
    for epoch in range(1, conf.num_epoch + 1):
        model.train()
        while mq_sem.acquire(timeout=0.01):
            (n_id, nano_batchs) = queue.get()  #
            epoch_start = default_timer()
            optimizer.zero_grad()
            target_node = n_id[:batch_size][worker_id *
                                            params.batch_size:(worker_id + 1) *
                                            params.batch_size]
            for i, nano_batch in enumerate(nano_batchs):
                nano_batch_adjs = [
                    adj.to(train_device) for adj in nano_batch.adjs
                ]
                out = model(x[n_id][nano_batch.n_id], nano_batch_adjs)
                loss = F.nll_loss(
                    out,
                    y[target_node][i * (nano_batch.size):(i + 1) *
                                   (nano_batch.size)],
                )
                loss.backward()

            optimizer.step()
            epoch_end = default_timer()

    metric = cal_metrics(epochtimes)
    print(
        f'train finished + {conf.model.name},{metric["mean"]:.2f},{params.batch_size}'
    )

    # if worker_id == 0:
    #     model.eval()
    #     with torch.no_grad():
    #         if num_worker > 1:
    #             out = model.module.inference(x, worker_id, subgraph_loader)
    #         else:
    #             out = model.inference(x, worker_id, subgraph_loader)
    #     res = out.argmax(dim=-1) == y
    #     acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
    #     assert acc1 > 0.90, "Sanity check , Low training accuracy."
    #     acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
    #     acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
    #     print(f"Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}")


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(conf):
    torch.manual_seed(12345)
    dataset_name = conf.dataset.name
    params = conf.model.params[dataset_name]
    run_config = get_run_config(conf, params)

    num_train_workers = conf.num_train_worker
    num_sample_worker = conf.num_sample_worker
    print(OmegaConf.to_yaml(conf))

    dataset_name = conf.dataset.name
    dataset = get_dataset(dataset_name, conf.root)
    data = dataset[0]
    run_config['dataset'] = dataset
    csr_topo = quiver.CSRTopo(data.edge_index)
    run_config['global_cpu_model'] = get_global_model(run_config)
    run_config['mq_sem'] = mp.Semaphore(0)
    quiver_sampler = quiver.pyg.GraphSageSampler(
        csr_topo, sizes=[25, 10], device=0,
        mode="GPU")  # 这里是0, 但是spawn之后会变成fake,然后再lazy init 赋值

    quiver_feature = quiver.Feature(
        rank=0,
        device_list=list(range(num_train_workers)),
        device_cache_size="2G",
        cache_policy="device_replicate",
        csr_topo=csr_topo,
    )
    quiver_feature.from_cpu_tensor(data.x)

    workers = []

    # sampler_stop_event is used to notify each train_switcher
    run_config['sampler_stop_event'] = []
    for woker_id in range(num_sample_worker):
        run_config['sampler_stop_event'].append(mp.Event())

    for worker_id in range(num_sample_worker):
        p = mp.Process(
            target=run_sample,
            args=(worker_id, run_config, quiver_sampler),
        )
        p.start()
        workers.append(p)
        # sampler switcher
        p = mp.Process(target=run_train,
                       args=(worker_id, run_config, quiver_feature,
                             TrainerType.Switcher))
        p.start()
        workers.append(p)

    for worker_id in range(num_train_workers):
        p = mp.Process(
            target=run_train,
            args=(worker_id, run_config, quiver_feature, TrainerType.Trainer),
        )
        p.start()
        workers.append(p)

    for p in workers:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
