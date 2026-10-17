import time
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from src.train_utils import evaluate
from src.aggregation import aggregate
from src.attacks import apply_attack
from src.security import make_message

def bytes_to_delta(body, device):
    """Rebuild the update tensor from the verified message bytes."""
    arr = np.frombuffer(body, dtype="<f4").copy()
    return torch.from_numpy(arr).to(device)

class Server:
    def __init__(self, model, test_loader, device):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device

    def apply_update(self, aggregated_delta):
        params = list(self.model.parameters())
        with torch.no_grad():
            new_vec = parameters_to_vector(params) + aggregated_delta
            vector_to_parameters(new_vec, params)

    def evaluate(self):
        return evaluate(self.model, self.test_loader, self.device)

def run_federated(server, clients, rounds, local_epochs=1, lr=0.01, verbose=True,
                  attack="none", malicious_ids=(), scale_factor=10.0,
                  aggregation="fedavg", trim_k=1, keys=None, channel=None):
    history = []
    for r in range(1, rounds + 1):
        deltas, sizes = [], []
        tag_time = verify_time = 0.0
        msg_bytes = rejected = 0

        for c in clients:
            delta = c.train(server.model, server.device, local_epochs, lr)
            if c.client_id in malicious_ids:
                delta = apply_attack(delta, attack, scale_factor)   # poison first

            if channel is not None:                                  # then authenticate
                t0 = time.perf_counter()
                msg = make_message(c.client_id, r, delta, keys[c.client_id])
                tag_time += time.perf_counter() - t0
                msg_bytes += 16 + len(msg["delta_bytes"]) + len(msg["tag"])

                t0 = time.perf_counter()
                ok, reason = channel.receive(msg, r)
                verify_time += time.perf_counter() - t0
                if not ok:
                    rejected += 1
                    continue
                delta = bytes_to_delta(msg["delta_bytes"], server.device)

            deltas.append(delta)
            sizes.append(c.num_samples)

        if deltas:
            server.apply_update(aggregate(deltas, sizes, aggregation, trim_k))

        loss, acc = server.evaluate()
        history.append({"round": r, "test_loss": loss, "test_accuracy": acc,
                        "tag_time_sec": tag_time, "verify_time_sec": verify_time,
                        "msg_bytes": msg_bytes, "rejected": rejected})
        if verbose:
            print(f"Round {r}: test loss = {loss:.4f}, test accuracy = {acc:.2f}%")
    return history
