import numpy as np
import torch


# Use this function to monitor GPU memory usage
def check_memory():
    print(
        f"GPU memory usage: {torch.cuda.memory_allocated() / 1024**2 : 2f} MB")
    print(
        f"GPU max memory usage: {torch.cuda.max_memory_allocated() / 1024**2 : 2f} MB"
    )
    print(
        f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2 : 2f} MB"
    )
    print(
        f"GPU max memory reserved: {torch.cuda.max_memory_reserved() / 1024**2 : 2f} MB"
    )


def cal_metrics(arr: list) -> dict:
    """Calculate metrics for a list of numbers.
    mean, median, std, min, max, 25%, 75%, 90%, 95%, 99%
    """
    nparr = np.array(arr)

    return {
        "mean": nparr.mean(),
        "median": np.median(nparr),
        "std": nparr.std(),
        "min": nparr.min(),
        "max": nparr.max(),
    }
