import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()                # required setup, always include
        self.layer1 = nn.Linear(4, 32)    # 4 inputs -> 32 numbers
        self.layer2 = nn.Linear(32, 32)   # 32 -> 32
        self.layer3 = nn.Linear(32, 2)    # 32 -> 2 outputs (inside / outside)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


if __name__ == "__main__":
    from data import make_quadrant_data

    X, y = make_quadrant_data(2000, quadrant=1)

    # turn the numpy arrays into torch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    model = Net()
    outputs = model(X)                    # run all points through the model
    print("output shape:", outputs.shape)  # should be (2000, 2)

    predictions = outputs.argmax(dim=1)     # pick the bigger of the 2 numbers
    accuracy = (predictions == y).float().mean()
    print("accuracy before training:", accuracy.item())