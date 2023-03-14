from typing import List
from torch_geometric.loader import NeighborSampler
import torch
from microGNN.utils.get_nano_batch import *
from microGNN.prune import prune
from microGNN.utils.common_class import Adj, Nanobatch

edge_index = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9],
                           [1, 6, 0, 2, 6, 1, 3, 7, 2, 4, 8, 3, 5, 9, 4, 9, 0, 1, 7, 2, 6, 8, 3, 7, 9, 4, 5, 8]], dtype=torch.long)  # fmt: skip

x = torch.tensor(
    [[1, 2], [2, 3], [3, 3], [4, 3], [5, 3], [6, 3], [7, 3], [8, 3], [9, 3], [10, 3]]
)


def test_getprune():
    """
    1. 生成一个batch
    2. 生成两个nano batch
    3. prune确认正确性
    """
    hop = [1, 3]
    train_loader = NeighborSampler(
        edge_index,
        sizes=hop,
        batch_size=1,
        shuffle=False,
        num_workers=6,
        drop_last=True,
    )
    # for batch_size, n_id, adjs in train_loader:
    batch_size, n_id, adjs = next(iter(train_loader))
    print(batch_size, n_id, adjs)
    print(adjs[0].edge_index.dtype)
    print(n_id.dtype)


def test_prune():
    """
    验证prune正确性
    """
    edge1 = torch.tensor([[3, 4], [2, 2]])
    adjs1 = Adj(edge1, None, (3, 1))
    edge2 = torch.tensor([[3, 4, 6, 7, 5, 8], [2, 2, 3, 3, 4, 4]])
    adjs2 = Adj(edge2, None, (7, 3))
    adjs = [adjs2, adjs1]
    n_id = torch.tensor([2, 3, 4, 5, 6, 7, 8])  # 第二hop会排序吗? 不一定. 不过hop1都在hop2前面
    batch_size = 1
    target_node = torch.tensor([2])
    # print(n_id, adjs)
    sel_nid = torch.tensor([[40, 3, 20], [23, 4, 30]])  # target node, 1hop, 2hop ...
    # print(sel_nid)
    sub_n_id, sub_adjs = prune(target_node, adjs, sel_nid)
    print(sub_adjs)
    assert torch.all(torch.eq(sub_n_id, torch.tensor([2, 3, 4, 6, 7])))
    assert torch.all(
        torch.eq(
            sub_adjs[1].edge_index,
            torch.tensor([[3, 4, 6, 7, 5, 8], [2, 2, 3, 3, 4, 4]]),
        )
    )
    assert torch.all(torch.eq(sub_adjs[0].edge_index, torch.tensor([[3, 4], [2, 2]])))


if __name__ == "__main__":
    test_prune()
    # test_getprune()
