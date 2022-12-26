import os.path as osp

import torch
from ogb.nodeproppred import PygNodePropPredDataset

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG, Reddit
from torch_geometric.utils import index_to_mask


try:
    from torch.autograd.profiler import emit_itt
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def emit_itt(*args, **kwargs):
        yield

def get_dataset(name, root, use_sparse_tensor=False, bf16=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), root, name)
    # transform = T.ToSparseTensor(
    #     remove_edge_index=False) if use_sparse_tensor else None
    # if name == 'ogbn-products':
    #     dataset = PygNodePropPredDataset('ogbn-products', root=path,
    #                                      transform=transform)
    if name == 'Reddit':
        dataset = Reddit(root=path)

    data = dataset[0]

    # if name == 'ogbn-products':
    #     split_idx = dataset.get_idx_split()
    #     data.train_mask = index_to_mask(split_idx['train'],
    #                                     size=data.num_nodes)
    #     data.val_mask = index_to_mask(split_idx['valid'], size=data.num_nodes)
    #     data.test_mask = index_to_mask(split_idx['test'], size=data.num_nodes)
    #     data.y = data.y.squeeze()

    # if bf16:
    #     data.x = data.x.to(torch.bfloat16)

    return data, dataset.num_classes
