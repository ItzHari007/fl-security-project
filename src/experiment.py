import time
import torch
from torch.utils.data import DataLoader
from src.utils import set_seed
from src.model import SimpleNet
from src.data import load_mnist
from src.partition import iid_partition, noniid_partition
from src.client import Client
from src.server import Server, run_federated
from src.security import generate_keys, SecureServerChannel

def run_experiment(seed=42, num_clients=5, rounds=10, local_epochs=1, lr=0.01,
                   verbose=False, attack="none", num_malicious=0, scale_factor=10.0,
                   partition="iid", aggregation="fedavg", trim_k=1, security=False):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, test_data = load_mnist()
    test_loader = DataLoader(test_data, batch_size=1000)

    if partition == "iid":
        parts = iid_partition(train_data, num_clients, seed=seed)
    elif partition == "noniid":
        parts = noniid_partition(train_data, num_clients, seed=seed)
    else:
        raise ValueError(f"Unknown partition: {partition}")
    clients = [Client(i, train_data, parts[i]) for i in range(num_clients)]
    server = Server(SimpleNet(), test_loader, device)

    malicious_ids = list(range(num_malicious))
    keys = channel = None
    if security:
        keys = generate_keys(range(num_clients))
        channel = SecureServerChannel(keys)

    start = time.time()
    history = run_federated(server, clients, rounds, local_epochs, lr, verbose=verbose,
                            attack=attack, malicious_ids=malicious_ids,
                            scale_factor=scale_factor,
                            aggregation=aggregation, trim_k=trim_k,
                            keys=keys, channel=channel)
    runtime = time.time() - start

    if security:
        channel.save_log(f"logs/secure_run_{partition}_{aggregation}_{attack}_m{num_malicious}_seed{seed}.csv")

    for row in history:
        row.update({"seed": seed, "num_clients": num_clients,
                    "local_epochs": local_epochs, "lr": lr,
                    "partition": partition, "attack": attack,
                    "num_malicious": num_malicious, "scale_factor": scale_factor,
                    "aggregation": aggregation, "trim_k": trim_k,
                    "security": security, "runtime_sec": runtime})
    return history
