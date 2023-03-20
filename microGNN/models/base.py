from typing import Optional

import torch


class ScalableGNN(torch.nn.Module):
    r"""An abstract class for implementing scalable GNNs via historical
    embeddings.
    This class will take care of initializing :obj:`num_layers - 1` historical
    embeddings, and provides a convenient interface to push recent node
    embeddings to the history, and to pull previous embeddings from the
    history.
    In case historical embeddings are stored on the CPU, they will reside
    inside pinned memory, which allows for asynchronous memory transfers of
    historical embeddings.
    For this, this class maintains a :class:`AsyncIOPool` object that
    implements the underlying mechanisms of asynchronous memory transfers as
    described in our paper.

    Args:
        num_nodes (int): The number of nodes in the graph.
        hidden_channels (int): The number of hidden channels of the model.
            As a current restriction, all intermediate node embeddings need to
            utilize the same number of features.
        num_layers (int): The number of layers of the model.
        pool_size (int, optional): The number of pinned CPU buffers for pulling
            histories and transfering them to GPU.
            Needs to be set in order to make use of asynchronous memory
            transfers. (default: :obj:`None`)
        buffer_size (int, optional): The size of pinned CPU buffers, i.e. the
            maximum number of out-of-mini-batch nodes pulled at once.
            Needs to be set in order to make use of asynchronous memory
            transfers. (default: :obj:`None`)
    """

    def __init__(
        self,
        hidden_channels: int,
        num_layers: int,
        pool_size: Optional[int] = None,
        buffer_size: Optional[int] = None,
        device=None,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.pool_size = num_layers - 1 if pool_size is None else pool_size
        self.buffer_size = buffer_size

    @property
    def emb_device(self):
        return self.histories[0].emb.device

    @property
    def device(self):
        return self.histories[0]._device

    def reset_parameters(self):
        for history in self.histories:
            history.reset_parameters()
