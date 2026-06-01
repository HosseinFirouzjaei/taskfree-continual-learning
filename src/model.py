import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 32)    # 4 inputs -> 32 numbers
        self.layer2 = nn.Linear(32, 32)   # 32 -> 32
        self.layer3 = nn.Linear(32, 2)    # 32 -> 2 outputs (inside / outside)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x
