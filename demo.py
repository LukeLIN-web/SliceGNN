
# import matplotlib.pyplot as plt
# import networkx as nx
# from functools import reduce
# from torch import Tensor
# from typing import List, NamedTuple, Optional, Tuple
# from timeit import default_timer
# from torch_geometric.nn import SAGEConv
# from torch_geometric.loader import NeighborSampler
# from torch_geometric.datasets import Reddit
# from tqdm import tqdm
# from torch.nn.parallel import DistributedDataParallel
# import torch.nn.functional as F
# import torch.multiprocessing as mp
# import torch.distributed as dist
# import argparse
# import os
# from statistics import mean

# import torch
# torch.set_printoptions(profile="full")

# G = nx.Graph()
# G.add_nodes_from([i for i in range(10)])
# e = zip(range(0, 3), range(1, 4))
# G.add_edges_from(e)
# nx.draw(G, pos=nx.circular_layout(G), node_color='r', edge_color='b')
# plt.savefig("Graph.png", format="PNG")
import torch

from torch_geometric.data.storage import BaseStorage
from torch_geometric.profile import get_cpu_memory_from_gc


class A(BaseStorage):
    def __init__(self, a, **kwargs):
        self.a = a
        for key, value in kwargs.items():
            setattr(self, key, value)


class B:
    def __init__(self, b):
        self.__dict__['_storage'] = A(b, _parent=self)


def main():
    a = torch.rand(10000, 10000)
    print(get_cpu_memory_from_gc())
    B(a)


if __name__ == '__main__':
    print(get_cpu_memory_from_gc())
    main()
    print(get_cpu_memory_from_gc())