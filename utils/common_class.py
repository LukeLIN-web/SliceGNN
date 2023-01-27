from typing import List, NamedTuple, Optional, Tuple, Union
import torch
class Adj(NamedTuple):
    edge_index: torch.Tensor
    e_id: torch.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(self.edge_index.to(*args, **kwargs),
                   e_id, self.size)


class Microbatch(NamedTuple):
    n_id: torch.Tensor
    size: int
    adjs : List[Adj]

    def to(self, *args, **kwargs):
        n_id = self.n_id.to(*args, **kwargs) if self.n_id is not None else None
        return Microbatch(self.n_id.to(*args, **kwargs),
                   self.adjs.to(*args, **kwargs), self.size)