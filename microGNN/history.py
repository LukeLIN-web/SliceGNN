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
        self.global_idx = torch.tensor(cached_id,
                                       device=device,
                                       dtype=torch.int64,
                                       pin_memory=pin_memory)

        self.emb = torch.empty(num_cache,
                               embedding_dim,
                               device=device,
                               pin_memory=pin_memory)

        self.cached_nodes = torch.full(
            (num_embeddings, ),
            False,
            dtype=torch.bool,
            device=device,
            pin_memory=pin_memory)  # push embedding or not

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.fill_(0)
        self.cached_nodes.fill_(False)

    def pull_push(self, x: Tensor, inter_id: Tensor):
        self.pull(x, inter_id)
        self.push(x, inter_id)

    def pull(self, x: Tensor, inter_id: Tensor, layer_id: Tensor) -> Tensor:
        # out = x.clone()
        # for id in inter_id:
        #     if self.cached_nodes[id]:
        #         # print("pulling")
        #         embidx = torch.where(self.global_idx == id)[0]
        #         emb = self.emb[embidx]
        #         xidx = torch.where(layer_id == id)[0]
        #         out[xidx] = emb
        #     else:
        #         # print("need cache , but not pushed")
        #         pass
        # return out
        out = x.clone()
        cached_id = inter_id[self.cached_nodes[inter_id]]
        cached_embidx = torch.where(
            self.global_idx.unsqueeze(-1) == cached_id)[0]
        if len(cached_embidx) > 0:
            cached_xidx = torch.where(layer_id.unsqueeze(-1) == cached_id)[0]
            out[cached_xidx, :] = self.emb[cached_embidx[:len(cached_xidx)], :]
        return out

    @torch.no_grad()
    def push(self, x: Tensor, inter_id: Tensor, layer_id: Tensor) -> Tensor:
        # for id in inter_id:
        #     if self.cached_nodes[id]:
        #         # print("have pushed")
        #         pass
        #     else:
        #         # print("pushing")
        #         embidx = torch.where(self.global_idx == id)[0]
        #         xidx = torch.where(layer_id == id)[0]
        #         self.emb[embidx] = x[xidx]
        #         self.cached_nodes[id] = True
        uncached_id = inter_id[~self.cached_nodes[inter_id]]
        uncached_embidx = torch.where(
            self.global_idx.unsqueeze(-1) == uncached_id)[0]
        if len(uncached_embidx) > 0:
            uncached_xidx = torch.where(
                layer_id.unsqueeze(-1) == uncached_id)[0]
            self.emb[uncached_embidx] = x[uncached_xidx]
            self.cached_nodes[uncached_id] = True

    def forward(self, *args, **kwargs):
        """"""
        raise NotImplementedError  # hisotry不是model,只是用了数据结构.

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({self.num_embeddings}, "
                f"{self.embedding_dim}, emb_device={self.emb.device}, "
                f"device={self._device})"
                f"{self.emb}")
