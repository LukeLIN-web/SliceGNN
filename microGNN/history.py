from typing import List

import torch
from torch import Tensor


class History(torch.nn.Module):
    r"""A historical embedding storage module."""

    def __init__(self,
                 cached_id: List,
                 num_embeddings: int,
                 embedding_dim: int,
                 device=None):
        super().__init__()

        num_cache = len(cached_id)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        pin_memory = device is None or str(device) == "cpu"
        self.global_idx = torch.unique(
            torch.tensor(cached_id,
                         device=device,
                         dtype=torch.int64,
                         pin_memory=pin_memory))

        self.emb = torch.empty(num_cache,
                               embedding_dim,
                               device=device,
                               pin_memory=pin_memory)

        self.cached_nodes = torch.full((num_embeddings, ),
                                       False,
                                       dtype=torch.bool,
                                       device=device,
                                       pin_memory=pin_memory)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.fill_(0)
        self.cached_nodes.fill_(False)

    def pull_push(self, x: Tensor, inter_id: Tensor):
        self.pull(x, inter_id)
        self.push(x, inter_id)

    def pull(self, x: Tensor, inter_id: Tensor) -> Tensor:
        cached_idxs = self.global_idx == inter_id
        cached_idxs = torch.where(cached_idxs)  # select true index
        cached_embs = self.emb[cached_idxs]
        x.copy_(cached_embs)

    @torch.no_grad()
    def push(self, x: Tensor, inter_id: Tensor):
        cached_nodes = self.cached_nodes[inter_id]
        uncached_idxs = torch.where(~cached_nodes)
        uncached_ids = inter_id[uncached_idxs]
        uncached_embs = x.detach()[uncached_idxs]
        print("uncached_ids", uncached_ids)
        print("uncached_embs", uncached_embs)
        print(self.emb)
        self.emb[uncached_ids] = uncached_embs
        self.cached_nodes[uncached_ids] = True

    def forward(self, *args, **kwargs):
        """"""
        raise NotImplementedError  # hisotry不是model,只是用了数据结构.

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({self.num_embeddings}, "
                f"{self.embedding_dim}, emb_device={self.emb.device}, "
                f"device={self._device})"
                f"{self.emb}")
