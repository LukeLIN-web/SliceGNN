import argparse
import ast

import torch
import torch.nn.functional as F
from tqdm import tqdm


import os
from utils import get_dataset, get_micro_batch
from torch_geometric.nn import SAGEConv
from torch_geometric.profile import rename_profile_file, timeit
import quiver
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
from torch.profiler import profile


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2):
        super(SAGE, self).__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)


def run(rank, world_size, data, x, quiver_sampler: quiver.pyg.GraphSageSampler,num_classes, args: argparse.ArgumentParser):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)

    train_loader = torch.utils.data.DataLoader(
        train_idx, batch_size=1024, shuffle=False, drop_last=True)

    torch.manual_seed(12345)
    model = SAGE(data.num_features, 256, num_classes)
    model = model.to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    y = data.y.to(rank)
    model.train()
    # Container to hold: event -> list of events in milliseconds
    MODEL_SAMPLE = "sample"
    MODEL_TRAIN = "train"
    metrics = {MODEL_SAMPLE: [], MODEL_TRAIN: []}

    for epoch in range(1, args.warmup + args.num_epochs + 1):
        if epoch <= args.warmup:
            for seeds in train_loader:
                optimizer.zero_grad()
                microseeds = seeds[rank * seeds.size(0) // world_size: (
                    rank + 1) * seeds.size(0) // world_size]
                n_id, batch_size, adjs = quiver_sampler.sample(microseeds)
                target_node = n_id[:len(seeds)]
                micro_batch_adjs = [adj.to(rank)
                                    for adj in adjs]  # load topo
                out = model(x[n_id], micro_batch_adjs)  # forward
                loss = F.nll_loss(out, y[target_node][:batch_size])
                loss.backward()
                optimizer.step()
        else:
            with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                    with_flops=True
                ) as prof:
                for seeds in train_loader:

                        optimizer.zero_grad()
                        microseeds = seeds[rank * seeds.size(0) // world_size: (
                            rank + 1) * seeds.size(0) // world_size]
                        n_id, batch_size, adjs = quiver_sampler.sample(microseeds)
                        target_node = n_id[:len(seeds)]
                        micro_batch_adjs = [adj.to(rank)
                                            for adj in adjs]  # load topo
                        out = model(x[n_id], micro_batch_adjs)  # forward
                        loss = F.nll_loss(out, y[target_node][:batch_size])
                        loss.backward()
                        optimizer.step()
            print(prof.key_averages().table(
                        sort_by="cpu_time_total", row_limit=10))

        dist.barrier()


def main(args: argparse.ArgumentParser) -> None:
    world_size = args.gpu_num
    dataset_name = "Reddit"
    data, num_classes = get_dataset(dataset_name, args.root,
                                    args.use_sparse_tensor, args.bf16)
    csr_topo = quiver.CSRTopo(data.edge_index)

    quiver_sampler = quiver.pyg.GraphSageSampler(
        csr_topo, sizes=[25, 10], device=0, mode='GPU')
    # cache feature 到rank0
    quiver_feature = quiver.Feature(rank=0, device_list=list(range(
        world_size)), device_cache_size="2G", cache_policy="device_replicate", csr_topo=csr_topo)
    quiver_feature.from_cpu_tensor(data.x)

    print('Let\'s use', world_size, 'GPUs!')
    print('BENCHMARK STARTS')

    mp.spawn(
        run,
        args=(world_size, data, quiver_feature,
              quiver_sampler,num_classes,  args),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('GNN training benchmark')
    argparser.add_argument('--datasets', nargs='+',
                           default=['Reddit'], type=str)
    argparser.add_argument(
        '--use-sparse-tensor', action='store_true',
        help='use torch_sparse.SparseTensor as graph storage format')
    argparser.add_argument('--root', default='../../data', type=str,
                           help='relative path to look for the datasets')
    argparser.add_argument('--batch-sizes', nargs='+',
                           default=[512, 1024, 2048, 4096, 8192], type=int)
    argparser.add_argument('--num-layers', nargs='+', default=[2, 3], type=int)
    argparser.add_argument('--num-neighbors', default=[10],
                           type=ast.literal_eval,
                           help='number of neighbors to sample per layer')
    argparser.add_argument('--num-workers', default=2, type=int)
    argparser.add_argument('--warmup', default=1, type=int)
    argparser.add_argument('--profile', action='store_true')
    argparser.add_argument('--vtune-profile', action='store_true')
    argparser.add_argument('--bf16', action='store_true')
    argparser.add_argument('--no-progress-bar', action='store_true',
                           default=False, help='turn off using progress bar')
    argparser.add_argument('--num-epochs', default=1, type=int)
    argparser.add_argument('--gpu_num', type=int, default=2)
    argparser.add_argument('--micro_pergpu', type=int, default=1)
    args = argparser.parse_args()

    main(args)
