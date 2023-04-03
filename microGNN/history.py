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
        self.cached_nodes.fill_(False)

    def _apply(self, fn):
        # Set the `_device` of the module without transfering `self.emb`.
        self._device = fn(torch.zeros(1)).device
        return self

    def pull(self, x: Tensor, n_id: Tensor) -> Tensor:
        cached_nodes = self.cached_nodes[
            n_id]  # get cached_nodes for the given node ids
        emb = self.emb[n_id]  # get embeddings for the cached nodes
        cached_nodes = cached_nodes.unsqueeze(1).expand(
            x.size(0), x.size(1))  # expand to the same shape as x
        out = torch.where(
            cached_nodes, emb,
            x)  # replace the values of cached nodes in x with their embeddings
        return out.to(device=x.device)

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
