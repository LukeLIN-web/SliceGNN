# GNN Project
We implment `gnn_project/pyg/smallgraphsampler.py` 


```python
def slice_adj(
    node_idx: Union[int, List[int], Tensor],
    edge_index: Tensor,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = 'source_to_target',
) -> Tuple[Tensor, Tensor,  Tensor]: 

def get_micro_batch(
    adjs: List[EdgeIndex],
    n_id: Tensor,
    batch_size: int,
    num_micro_batch: int = 2,
) -> List[namedtuple('micro_batch', ['bach_size', 'nid', 'adjs'])]:

```
