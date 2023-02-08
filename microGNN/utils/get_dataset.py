import os.path as osp
from torch_geometric.datasets import  Reddit

def get_dataset(name, root, use_sparse_tensor=False, bf16=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), root, name)
    
    if name == 'Reddit':
        dataset = Reddit(root=path)

    data = dataset[0]


    return data, dataset.num_classes