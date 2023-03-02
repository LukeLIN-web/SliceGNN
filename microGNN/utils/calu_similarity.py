from math import sqrt
import torch
from torch import Tensor
from timeit import default_timer


def common_nodes_num(nid1, nid2: Tensor) -> int:
    """
    calu the common nodes number of two microbatch
    """
    assert len(nid2) > 0
    if isinstance(nid1, set):
        for nid in nid2:
            nid1.add(nid)
        return len(nid1.intersection(nid2))
    else:
        a_cat_b, counts = torch.cat([nid1, nid2]).unique(return_counts=True)
        intersection = a_cat_b[torch.where(counts.gt(1))]
        return len(intersection)


def Ochiai(nid1: Tensor, nid2: Tensor) -> float:
    """
    calu the similarity of two microbatch
    """
    len1, len2 = len(nid1), len(nid2)
    if len1 == 0 or len2 == 0:
        return 0
    else:
        # start = default_timer()
        a_cat_b, counts = torch.cat([nid1, nid2]).unique(return_counts=True)
        intersection = a_cat_b[torch.where(counts.gt(1))]
        # print("calu_similarity time: ", default_timer() - start)
        return len(intersection) / sqrt(len1 * len2)


def Jaccard(nid1: Tensor, nid2: Tensor) -> float:
    """
    calu the similarity of two microbatch
    """
    len1, len2 = len(nid1), len(nid2)
    if len1 == 0 or len2 == 0:
        return 0
    else:
        # start = default_timer()
        a_cat_b, counts = torch.cat([nid1, nid2]).unique(return_counts=True)
        intersection = a_cat_b[torch.where(counts.gt(1))]
        # print("calu_similarity time: ", default_timer() - start)
        return len(intersection) / (len1 + len2 - len(intersection))
