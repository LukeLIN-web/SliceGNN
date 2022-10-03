
from torch import Tensor
from typing import List, NamedTuple, Optional, Tuple
from timeit import default_timer
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborSampler
from torch_geometric.datasets import Reddit
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
import os
from statistics import mean

import torch
torch.set_printoptions(profile="full")
from functools import reduce

# datapath = "/root/share/pytorch_geometric/examples/data/Reddit"
# dataset = Reddit(datapath)

#torch.Size([10, 1])
# torch.float32
# data = dataset[0]
# test = data.x
# print(test.shape)
# print(test.dtype)
a = torch.tensor([[ 4.],
        [ 6.],
        [ 7.],
        [ 3.],
        [ 4.],
        [ 8.],
        [ 2.],
        [ 8.],
        [ 1.],
        [ 2.],
        [ 9.],
        [ 6.],
        [ 7.],
        [10.],
        [ 1.],
        [ 5.],
        [10.],
        [ 1.]])
indices = torch.tensor([0, 2])
# b = a.index_select(-2,indices)
b = a[:,:1]
print(b)