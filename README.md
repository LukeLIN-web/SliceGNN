# GNN Project
We implment `slice adj and get  micro batch` graph partition 


```python
def slice_adj(
    node_idx: Union[int, List[int], Tensor],
    edge_index: Tensor,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = 'source_to_target',
) -> Tuple[Tensor, Tensor,  Tensor]:

def get_micro_batch(
    adjs : List[Adj],
    n_id: Tensor,
    batch_size: int,
    num_micro_batch: int,
) -> List[Microbatch]:

```
