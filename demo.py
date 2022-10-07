
import matplotlib.pyplot as plt
import networkx as nx
from functools import reduce
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

G = nx.Graph()
G.add_nodes_from([i for i in range(10)])
e = zip(range(0, 3), range(1, 4))
G.add_edges_from(e)
nx.draw(G, pos=nx.circular_layout(G), node_color='r', edge_color='b')
plt.savefig("Graph.png", format="PNG")