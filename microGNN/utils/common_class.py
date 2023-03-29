from typing import List, NamedTuple, Tuple

from torch import Tensor


class Adj(NamedTuple):
    edge_index: Tensor
    e_id: Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(self.edge_index.to(*args, **kwargs), e_id, self.size)


class Nanobatch(NamedTuple):
    n_id: Tensor
    size: int
    adjs: List[Adj]

    def to(self, *args, **kwargs):
        n_id = self.n_id.to(*args, **kwargs) if self.n_id is not None else None
        for adj in self.adjs:
            adj.to(*args, **kwargs)
        return Nanobatch(n_id, self.size, self.adjs.to(*args, **kwargs))
