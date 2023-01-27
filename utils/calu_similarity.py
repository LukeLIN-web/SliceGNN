from .common_class import Adj, Microbatch
import torch
from torch import Tensor
from timeit import default_timer 

def calu_similarity(nid1: Tensor, nid2: Tensor) -> float:
    """
    calu the similarity of two microbatch
    """
    len1, len2 = len(nid1), len(nid2)
    if len1 == 0 or len2 == 0:
        return 0
    else:
        start = default_timer()
        # a_cat_b, counts = torch.cat([nid1, nid2]).unique(return_counts=True)
        # inter =  a_cat_b[torch.where(counts.gt(1))]
        intersection = nid1[(nid1.view(1, -1) == nid2.view(-1, 1)).any(dim=0)]
        print("calu_similarity time: ", default_timer() - start)
        return intersection
