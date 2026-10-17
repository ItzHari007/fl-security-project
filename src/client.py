import copy
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import parameters_to_vector
from src.train_utils import train_one_epoch

class Client:
    def __init__(self, client_id, dataset, indices, batch_size=32):
        self.client_id = client_id
        self.num_samples = len(indices)
        self.loader = DataLoader(Subset(dataset, indices),
                                 batch_size=batch_size, shuffle=True)

    def train(self, global_model, device, local_epochs=1, lr=0.01):
        """Train a copy of the global model locally; return the update (delta)."""
        model = copy.deepcopy(global_model).to(device)
        before = parameters_to_vector(model.parameters()).detach().clone()

        for _ in range(local_epochs):
            train_one_epoch(model, self.loader, device, lr)

        after = parameters_to_vector(model.parameters()).detach()
        return after - before    # one long vector of weight changes
