from typing import Optional

import torch
from torch import Tensor


class History(torch.nn.Module):
    r"""A historical embedding storage module."""

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        pin_memory = device is None or str(device) == "cpu"
        self.emb = torch.empty(num_embeddings,
                               embedding_dim,
                               device=device,
                               pin_memory=pin_memory)
        self.cached_nodes = torch.full((num_embeddings, ),
                                       False,
                                       dtype=torch.bool,
                                       device=device,
                                       pin_memory=pin_memory)

        self._device = torch.device("cpu")

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.fill_(0)
        self.cached_nodes.fill_(False)  # 0 is a valid node id.

    def _apply(self, fn):
        # Set the `_device` of the module without transfering `self.emb`.
        self._device = fn(torch.zeros(1)).device
        return self

    @torch.no_grad()
    def pull(self, n_id: Optional[Tensor] = None) -> Tensor:
        out = self.emb
        if n_id is not None:
            assert n_id.device == self.emb.device
            out = out.index_select(0, n_id)
        return out.to(device=self._device)

    @torch.no_grad()
    def push(
        self,
        x: Tensor,
        n_id: Tensor,
    ):

        if n_id is None and x.size(0) != self.num_embeddings:
            raise ValueError

        assert n_id.device == self.emb.device
        self.emb[n_id] = x.to(self.emb.device)
        self.cached_nodes[n_id] = True

    def forward(self, *args, **kwargs):
        """"""
        raise NotImplementedError  # hisotry不是model,只是用了数据结构.

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({self.num_embeddings}, "
                f"{self.embedding_dim}, emb_device={self.emb.device}, "
                f"device={self._device})"
                f"{self.emb}")
