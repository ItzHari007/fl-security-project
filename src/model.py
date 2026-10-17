import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),            # 1x28x28 image -> 784 numbers
            nn.Linear(784, 128),     # 784 inputs -> 128 hidden units
            nn.ReLU(),
            nn.Linear(128, 10),      # 128 -> 10 outputs (digits 0-9)
        )

    def forward(self, x):
        return self.net(x)
