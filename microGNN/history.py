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
        self.need_cache_nodes = torch.full(
            (num_embeddings, ),
            -1,
            dtype=torch.long,
            device=device,
            pin_memory=pin_memory)  # push embedding or not

        tensor_1d = torch.tensor(cached_id, device=device)
        values = torch.arange(len(tensor_1d), device=device)
        self.need_cache_nodes.scatter_(0, tensor_1d, values)

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

    def pull(self, x: Tensor, layer_id: Tensor) -> Tensor:
        # Assuming that `layer_id` is a tensor with shape (n,) and `x` is a tensor with shape (n, d).

        # create a boolean tensor to identify which nodes are cached
        is_cached = self.cached_nodes[layer_id]

        # select the indices of nodes that are cached
        cache_indices = self.need_cache_nodes[layer_id[is_cached]]

        # clone the input tensor
        out = x.clone()
        # copy the cached embeddings to the output tensor
        print(is_cached.shape, out.shape)
        out[is_cached] = self.emb[cache_indices]

        return out

    @torch.no_grad()
    def push(self, x: Tensor, layer_id: Tensor) -> Tensor:
        # Assuming that `layer_id` is a tensor with shape (n,) and `x` is a tensor with shape (n, d).

        # create a boolean tensor to identify which nodes need to be cached
        need_cache = (self.cached_nodes[layer_id]
                      == False) & (self.need_cache_nodes[layer_id] != -1)

        # select the indices of nodes that need to be cached
        cache_indices = self.need_cache_nodes[layer_id][need_cache]

        # copy the corresponding input tensor values to the embeddings
        self.emb[cache_indices] = x[need_cache]
        # update the cached_nodes boolean tensor
        self.cached_nodes[layer_id[need_cache]] = True
        return x

    def forward(self, *args, **kwargs):
        """"""
        raise NotImplementedError  # hisotry不是model,只是用了数据结构.

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({self.num_embeddings}, "
                f"{self.embedding_dim}, emb_device={self.emb.device}, "
                f"device={self._device})"
                f"{self.emb}")
