import torch

def fedavg(deltas, sizes):
    """Weighted average: each client counts in proportion to its data size (n_k / sum n_j)."""
    total = sum(sizes)
    weights = [n / total for n in sizes]
    stacked = torch.stack(deltas)                       # shape: (clients, parameters)
    w = torch.tensor(weights, device=stacked.device).view(-1, 1)
    return (stacked * w).sum(dim=0)

def coordinate_median(deltas):
    """Median of each parameter across clients (ignores data sizes)."""
    stacked = torch.stack(deltas)
    return stacked.median(dim=0).values

def trimmed_mean(deltas, trim_k=1):
    """For each parameter: sort across clients, drop trim_k smallest and trim_k largest, average the rest."""
    stacked = torch.stack(deltas)
    n = stacked.shape[0]
    if 2 * trim_k >= n:
        raise ValueError(f"trim_k={trim_k} is not feasible with {n} clients (need 2*k < n)")
    sorted_vals, _ = torch.sort(stacked, dim=0)
    return sorted_vals[trim_k:n - trim_k].mean(dim=0)

def aggregate(deltas, sizes, method="fedavg", trim_k=1):
    if method == "fedavg":
        return fedavg(deltas, sizes)
    if method == "median":
        return coordinate_median(deltas)
    if method == "trimmed_mean":
        return trimmed_mean(deltas, trim_k)
    raise ValueError(f"Unknown aggregation method: {method}")
