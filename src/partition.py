import numpy as np

def iid_partition(dataset, num_clients, seed=42):
    """Shuffle all indices, then split them into equal-sized random chunks."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(dataset))
    return [chunk for chunk in np.array_split(indices, num_clients)]

def noniid_partition(dataset, num_clients, shards_per_client=2, seed=42):
    """Sort by label, cut into shards, give each client a few random shards.
    With 2 shards per client, each client mostly holds 1-2 digit classes."""
    rng = np.random.RandomState(seed)
    labels = np.array(dataset.targets)
    sorted_idx = np.argsort(labels, kind="stable")
    num_shards = num_clients * shards_per_client
    shards = np.array_split(sorted_idx, num_shards)
    shard_order = rng.permutation(num_shards)
    parts = []
    for c in range(num_clients):
        chosen = shard_order[c * shards_per_client:(c + 1) * shards_per_client]
        parts.append(np.concatenate([shards[s] for s in chosen]))
    return parts
