from torch import Tensor
from typing import List, NamedTuple, Optional, Tuple, Union
from torch_geometric.loader import NeighborSampler
from torch_geometric.data import Data
import torch
from sampler.get_micro_batch import *
import pytest


def test_two_hop():
    edge_index = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9],
                         [1, 6, 0, 2, 6, 1, 3, 7, 2, 4, 8, 3, 5, 9, 4, 9, 0, 1, 7, 2, 6, 8, 3, 7, 9, 4, 5, 8]], dtype=torch.long)
    x = Tensor([[1], [2], [3], [4], [5], [6], [7],
                [8], [9], [10]])
    data = Data(x=x, edge_index=edge_index)
    hop = [-1, -1]
    train_loader = NeighborSampler(data.edge_index,
                                   sizes=hop, batch_size=4,
                                   shuffle=False, num_workers=0)
    num_features, num_classes = 1, 1
    x = data.x
    model = SAGE(num_features, 16, num_classes)
    for epoch in range(1):
        for batch_size, n_id, adjs in train_loader:
            out = model(x[n_id], adjs)
            num_micro_batch = 2
            micro_batchs = get_micro_batch(adjs,
                                           n_id,
                                           batch_size, num_micro_batch)
            leftbatch, rightbatch = micro_batchs[0], micro_batchs[1]
            leftout = model(x[n_id][leftbatch.nid], leftbatch.adjs)
            rightout = model(x[n_id][rightbatch.nid], rightbatch.adjs)
            subgraphout = torch.cat((leftout, rightout), 0)
            assert torch.abs((out - subgraphout).mean()) < 0.01


def test_one_hop():
    edge_index = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9],
                         [1, 6, 0, 2, 6, 1, 3, 7, 2, 4, 8, 3, 5, 9, 4, 9, 0, 1, 7, 2, 6, 8, 3, 7, 9, 4, 5, 8]], dtype=torch.long)
    x = Tensor([[1], [2], [3], [4], [5], [6], [7],
                [8], [9], [10]])
    data = Data(x=x, edge_index=edge_index)
    train_loader = NeighborSampler(data.edge_index,
                                   sizes=[-1], batch_size=6,
                                   shuffle=False, num_workers=0)
    num_features, num_classes = 1, 1
    x = data.x
    model = SAGE(num_features, 16, num_classes)
    for epoch in range(1):
        model.train()
        for batch_size, n_id, adjs in train_loader:
            if isinstance(adjs[0], Tensor):
                # when hop = 1 , adjs is a EdgeIndex, we need convert it to list.
                adjs = [adjs]
            num_micro_batch = 2
            micro_batchs = get_micro_batch(adjs,
                                           n_id,
                                           batch_size, num_micro_batch)
            leftbatch, rightbatch = micro_batchs[0], micro_batchs[1]
            out = model(x[n_id], adjs)
            leftout = model(x[n_id][leftbatch.nid], leftbatch.adjs)
            rightout = model(x[n_id][rightbatch.nid], rightbatch.adjs)
            subgraphout = torch.cat((leftout, rightout), 0)
            assert torch.abs((out - subgraphout).mean()) < 0.01
