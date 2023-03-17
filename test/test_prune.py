import torch
from torch_geometric.loader import NeighborSampler

from microGNN.prune import prune
from microGNN.utils import get_nano_batch
from microGNN.utils.common_class import Adj

edge_index = torch.tensor([[0, 0, 1, 1, 2, 6], [1, 6, 0, 2, 1, 0]],
                          dtype=torch.long)  # fmt: skip

x = torch.tensor([[1, 2], [2, 3], [3, 3], [4, 3], [5, 3], [6, 3], [7, 3],
                  [8, 3], [9, 3], [10, 3]])


def test_getprune():
    hop = [-1, -1]
    train_loader = NeighborSampler(
        edge_index,
        sizes=hop,
        batch_size=2,
        shuffle=False,
        drop_last=True,
    )
    # for batch_size, n_id, adjs in train_loader:
    batch_size, n_id, adjs = next(iter(train_loader))
    print(batch_size, n_id, adjs)
    # target_node = n_id[:batch_size]
    nano_batchs = get_nano_batch(adjs,
                                 n_id,
                                 batch_size,
                                 num_nano_batch=2,
                                 relabel_nodes=False)
    for nano_batch in nano_batchs:
        print(nano_batch)


# 验证prune正确性
def test_prune():
    edge1 = torch.tensor([[3, 4], [2, 2]])
    adjs1 = Adj(edge1, None, (3, 1))
    edge2 = torch.tensor([[3, 4, 6, 7, 5, 8], [2, 2, 3, 3, 4, 4]])
    adjs2 = Adj(edge2, None, (7, 3))
    adjs = [adjs2, adjs1]  # hop2会排序吗? 不一定. 不过hop1节点都在hop2前面
    target_node = torch.tensor([2])
    cached_nodes = torch.tensor([[3], [4]])  # target node, 1hop, 2hop ...
    sub_n_id, sub_adjs = prune(target_node, adjs, cached_nodes)
    assert torch.all(torch.eq(sub_n_id, torch.tensor([2, 3, 4, 6, 7])))
    assert torch.all(
        torch.eq(
            sub_adjs[1].edge_index,
            torch.tensor([[3, 4, 6, 7, 5, 8], [2, 2, 3, 3, 4, 4]]),
        ))
    assert torch.all(
        torch.eq(sub_adjs[0].edge_index, torch.tensor([[3, 4], [2, 2]])))


if __name__ == "__main__":
    # test_prune()
    test_getprune()
