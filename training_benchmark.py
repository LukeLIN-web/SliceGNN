import argparse
import ast

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import emit_itt, get_dataset, get_model
from utils.microbatch_reddit_quiver_gloo import SAGE
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import PNAConv
from torch_geometric.profile import rename_profile_file, timeit, torch_profile


def train_homo(model, loader, optimizer, device, progress_bar=True,
               desc="") -> None:
    if progress_bar:
        loader = tqdm(loader, desc=desc)
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        if hasattr(batch, 'adj_t'):
            edge_index = batch.adj_t
        else:
            edge_index = batch.edge_index
        out = model(batch.x, edge_index)
        batch_size = batch.batch_size
        out = out[:batch_size]
        target = batch.y[:batch_size]
        loss = F.cross_entropy(out, target)
        loss.backward()
        optimizer.step()


def main(args: argparse.ArgumentParser) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # If we use a custom number of steps, then we need to use RandomSampler,
    # which already does shuffle.
    shuffle = False if args.num_steps != -1 else True

    print('BENCHMARK STARTS')
    for dataset_name in args.datasets:
        print(f'Dataset: {dataset_name}')
        data, num_classes = get_dataset(dataset_name, args.root,
                                        args.use_sparse_tensor, args.bf16)
        if torch.cuda.is_available():
            amp = torch.cuda.amp.autocast(enabled=False)
        else:
            amp = torch.cpu.amp.autocast(enabled=args.bf16)
        inputs_channels = data.num_features

        for batch_size in args.batch_sizes:
            for layers in args.num_layers:
                num_neighbors = args.num_neighbors
                if type(num_neighbors) is list:
                    if len(num_neighbors) == 1:
                        num_neighbors = num_neighbors * layers
                elif type(num_neighbors) is int:
                    num_neighbors = [num_neighbors] * layers

                assert len(
                    num_neighbors) == layers, \
                    f'''num_neighbors={num_neighbors} length
                        != num of layers={layers}'''

                subgraph_loader = NeighborLoader(
                    data,
                    num_neighbors=num_neighbors,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=args.num_workers,
                )

                for hidden_channels in args.num_hidden_channels:
                    print('----------------------------------------------')
                    print(f'Batch size={batch_size}, '
                          f'Layers amount={layers}, '
                          f'Num_neighbors={num_neighbors}, '
                          f'Hidden features size={hidden_channels}, '
                          f'Sparse tensor={args.use_sparse_tensor}')

                    model = SAGE(inputs_channels,
                                 hidden_channels, num_classes)
                    model = model.to(device)
                    model.train()
                    optimizer = torch.optim.Adam(model.parameters(),
                                                 lr=0.001)

                    progress_bar = False if args.no_progress_bar else True
                    train = train_homo
                    with amp:
                        for _ in range(args.warmup):
                            train(model, subgraph_loader, optimizer,
                                  device, progress_bar=progress_bar,
                                  desc="Warmup")
                        with timeit(avg_time_divisor=args.num_epochs):
                            # becomes a no-op if vtune_profile == False
                            with emit_itt(args.vtune_profile):
                                for epoch in range(args.num_epochs):
                                    train(model, subgraph_loader,
                                          optimizer, device,
                                          progress_bar=progress_bar,
                                          desc=f"Epoch={epoch}")

                        if args.profile:
                            with torch_profile():
                                train(model, subgraph_loader, optimizer,
                                      device, progress_bar=progress_bar,
                                      desc="Profile training")
                                rename_profile_file(dataset_name,
                                                    str(batch_size),
                                                    str(layers),
                                                    str(hidden_channels),
                                                    str(num_neighbors))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('GNN training benchmark')
    argparser.add_argument('--datasets', nargs='+',
                           default=['ogbn-products',
                                    'Reddit'], type=str)
    argparser.add_argument(
        '--use-sparse-tensor', action='store_true',
        help='use torch_sparse.SparseTensor as graph storage format')
    argparser.add_argument(
        '--models', nargs='+',
        default=['edge_cnn', 'gat', 'gcn', 'pna', 'rgat', 'rgcn'], type=str)
    argparser.add_argument('--root', default='../../data', type=str,
                           help='relative path to look for the datasets')
    argparser.add_argument('--batch-sizes', nargs='+',
                           default=[512, 1024, 2048, 4096, 8192], type=int)
    argparser.add_argument('--num-layers', nargs='+', default=[2, 3], type=int)
    argparser.add_argument('--num-hidden-channels', nargs='+',
                           default=[64, 128, 256], type=int)
    argparser.add_argument(
        '--num-heads', default=2, type=int,
        help='number of hidden attention heads, applies only for gat and rgat')
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
    argparser.add_argument(
        '--num-steps', default=-1, type=int,
        help='number of steps, -1 means iterating through all the data')
    argparser.add_argument('--gpu_num', type=int, default=2)
    argparser.add_argument('--micro_pergpu', type=int, default=1)
    args = argparser.parse_args()

    main(args)
