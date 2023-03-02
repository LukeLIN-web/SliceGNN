import numpy as np


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
