# Slice GNN

This project implements several key features for efficient Graph Neural Network (GNN) training:

- **Graph Partitioning**: We implement `slice adj` and `get nano batch` for graph partitioning to optimize mini-batch processing.
- **Pipeline with Multi-Processing**: A pipeline is implemented using `torch.multiprocessing.Queue` for parallel data handling and processing.
- **Computation Graph Pruning**: A function to prune the computation graph is implemented to reduce unnecessary computations and improve efficiency.
- **Comprehensive Testing**: We have written extensive tests to ensure the correctness and reliability of the code.

## Environment

The project requires the following dependencies:

```
torch 1.11.0
cuda_11.3
torch_geometric 2.2.0
torch-scatter          2.1.0
torch-sparse           0.6.16
```
## Reference

https://arxiv.org/pdf/2406.00552

https://arxiv.org/pdf/2303.13775 
