import logging
import os
from timeit import default_timer

import hydra
import quiver
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.loader import NeighborSampler

from microGNN.models import SAGE
from microGNN.utils import cal_metrics, get_dataset, get_nano_batch
from microGNN.utils.common_config import gpu

log = logging.getLogger(__name__)


def run_sample(worker_id, params, conf, data, quiver_sampler, nano_queues):
    sample_workers = [
        gpu(conf.num_train_worker + i) for i in range(conf.num_sample_worker)
    ]

    num_sample_worker = conf.num_sample_worker
    per_gpu = conf.nano_pergpu
    ctx = sample_workers[worker_id]
    torch.cuda.set_device(ctx)
    num_train_worker = conf.num_train_worker
    print(
        f"[Sample Worker {worker_id}/{num_sample_worker}] Started with PID {os.getpid()}({torch.cuda.get_device_name(ctx)})"
    )

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)

    train_loader = torch.utils.data.DataLoader(
        train_idx,
        batch_size=params.batch_size * num_train_worker,
        shuffle=False,
        drop_last=True,
    )
    epochtimes = []
    for epoch in range(1, conf.num_epoch + 1):
        epoch_start = default_timer()
        for seeds in train_loader:
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            nano_batchs = get_nano_batch(adjs, n_id, batch_size,
                                         num_train_worker * per_gpu)
            for i in range(num_train_worker):
                nano_queues[i].put(
                    (n_id, nano_batchs[i * per_gpu:(i + 1) * per_gpu]))
        epoch_end = default_timer()
        epochtime = epoch_end - epoch_start
        if epoch > 1:
            epochtimes.append(epochtime)
        print(f"Epoch: {epoch:03d}, Sample Epoch Time: {epochtime}")
    metric = cal_metrics(epochtimes)
    print(
        f'sample finished + {conf.model.name},{metric["mean"]:.2f},{params.batch_size}'
    )


def run_train(worker_id, params, conf, x, dataset, queue):
    train_workers = [gpu(i) for i in range(conf.num_train_worker)]
    ctx = train_workers[worker_id]
    num_worker = conf.num_train_worker

    if num_worker > 1:
        dist_init_method = "tcp://{master_ip}:{master_port}".format(
            master_ip="127.0.0.1", master_port="12345")
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=dist_init_method,
            world_size=num_worker,
            rank=worker_id,
        )
    train_device = torch.device(ctx)
    torch.cuda.set_device(train_device)

    print(
        f"[Train Worker { worker_id}/{num_worker}] Started with PID {os.getpid()}({torch.cuda.get_device_name(ctx)})"
    )

    data = dataset[0]
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    batch_size = params.batch_size * num_worker

    if worker_id == 0:
        subgraph_loader = NeighborSampler(
            data.edge_index,
            node_idx=None,
            sizes=[-1],
            batch_size=2048,
            shuffle=False,
            num_workers=6,
        )
    num_features, num_classes = dataset.num_features, dataset.num_classes
    model = SAGE(num_features, params.hidden_channels,
                 num_classes).to(train_device)
    if num_worker > 1:
        model = DistributedDataParallel(model,
                                        device_ids=[train_device],
                                        output_device=train_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    y = data.y.to(train_device)
    length = train_idx.size(dim=0) // (1024 * num_worker)
    epochtimes = []
    for epoch in range(1, conf.num_epoch + 1):
        model.train()
        itetime = []
        for _ in range(length):
            (n_id, nano_batchs) = queue.get()
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
            itetime.append(epoch_end - epoch_start)
        if epoch > 1:
            epochtimes.append(sum(itetime))

        if worker_id == 0:
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Epoch Time: {sum(itetime)}"
            )

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

    num_train_workers = conf.num_train_worker
    num_sample_worker = conf.num_sample_worker
    print(OmegaConf.to_yaml(conf))

    dataset_name = conf.dataset.name
    dataset = get_dataset(dataset_name, conf.root)
    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)

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
    mp.set_start_method("spawn")
    nanobatchs_qs = [mp.Queue(30) for i in range(num_train_workers)]
    for worker_id in range(num_sample_worker):
        p = mp.Process(
            target=run_sample,
            args=(worker_id, params, conf, data, quiver_sampler,
                  nanobatchs_qs),
        )
        p.start()
        workers.append(p)

    for worker_id in range(num_train_workers):
        p = mp.Process(
            target=run_train,
            args=(
                worker_id,
                params,
                conf,
                quiver_feature,
                dataset,
                nanobatchs_qs[worker_id],
            ),
        )
        p.start()
        workers.append(p)

    for p in workers:
        p.join()


if __name__ == "__main__":
    main()
