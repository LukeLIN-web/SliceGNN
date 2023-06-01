from typing import List

import torch
from torch import Tensor


class History(torch.nn.Module):
    r"""A historical embedding storage module."""

    def __init__(self,
                 cached_id: Tensor,
                 num_embeddings: int,
                 embedding_dim: int,
                 device=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        pin_memory = device is None or str(device) == "cpu"
        self.emb_idx = torch.full(
            (num_embeddings, ),
            -1,
            dtype=torch.long,
            device=device,
            pin_memory=pin_memory)  # corrsponding index in self.emb

        cached_id = cached_id.to(device)
        values = torch.arange(len(cached_id), device=device)
        self.emb_idx.scatter_(0, cached_id, values)

        self.emb = torch.empty(len(cached_id),
                               embedding_dim,
                               device=device,
                               pin_memory=pin_memory)

        self.cached_nodes = torch.full(
            (num_embeddings, ),
            False,
            dtype=torch.bool,
            device=device,
            pin_memory=pin_memory)  # pushed embedding or not

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.fill_(0)
        self.cached_nodes.fill_(False)

    def pull_push(self, x: Tensor, inter_id: Tensor):
        self.pull(x, inter_id)
        self.push(x, inter_id)

    def pull(self, x: Tensor, target_id: Tensor) -> Tensor:
        is_cached = self.cached_nodes[target_id]
        cached_id = target_id[is_cached]  # bottleneck
        emb_indices = self.emb_idx[cached_id]
        embeddings = self.emb[emb_indices]
        out = x.clone()
        out[is_cached] = embeddings
        return out

    @torch.no_grad()
    def push(self, x: Tensor, target_id: Tensor) -> Tensor:
        should_cache = (self.emb_idx[target_id] != -1)
        tocacheid = target_id[should_cache]
        emb_indices = self.emb_idx[tocacheid]
        self.emb[emb_indices] = x[should_cache]
        self.cached_nodes[tocacheid] = True

    def forward(self, *args, **kwargs):
        """"""
        raise NotImplementedError  # history is not model, only use for storage

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({self.num_embeddings}, "
                f"{self.embedding_dim}, emb_device={self.emb.device}, "
                f"device={self._device})"
                f"{self.emb}")
