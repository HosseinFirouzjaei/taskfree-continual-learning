import numpy as np
import torch
import torch.nn as nn

from data import make_quadrant_data
from model import Net


def to_tensors(X, y):
    """Turn numpy arrays into torch tensors."""
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def evaluate(model, X, y):
    """Return how often the model is correct on data X with labels y."""
    model.eval()                       # test mode
    with torch.no_grad():              # do not track gradients (faster)
        outputs = model(X)
        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == y).float().mean()
    return accuracy.item()


def train_online(model, X, y, batch_size=10, lr=0.03, steps_per_batch=5):
    """Train one small batch at a time. Practice each batch a few times."""
    model.train()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for start in range(0, len(X), batch_size):
        batch_X = X[start:start + batch_size]
        batch_y = y[start:start + batch_size]

        for _ in range(steps_per_batch):       # practice this batch a few times
            outputs = model(batch_X)
            loss = loss_function(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":

    torch.manual_seed(0)


    # training data for both quadrants
    X1, y1 = make_quadrant_data(2000, quadrant=1, seed=1)
    X2, y2 = make_quadrant_data(2000, quadrant=2, seed=2)

    # separate test data
    X1_test, y1_test = make_quadrant_data(500, quadrant=1, seed=11)
    X2_test, y2_test = make_quadrant_data(500, quadrant=2, seed=22)

    # shuffle each quadrant so every batch has a mix of inside and outside
    rng = np.random.default_rng(0)
    o1 = rng.permutation(len(X1)); X1, y1 = X1[o1], y1[o1]
    o2 = rng.permutation(len(X2)); X2, y2 = X2[o2], y2[o2]

    # convert all to tensors
    X1, y1 = to_tensors(X1, y1)
    X2, y2 = to_tensors(X2, y2)
    X1_test, y1_test = to_tensors(X1_test, y1_test)
    X2_test, y2_test = to_tensors(X2_test, y2_test)

    model = Net()

    train_online(model, X1, y1)        # learn quadrant 1
    print("After task 1:")
    print("  quadrant 1 accuracy:", evaluate(model, X1_test, y1_test))
    print("  quadrant 2 accuracy:", evaluate(model, X2_test, y2_test))

    train_online(model, X2, y2)        # now learn quadrant 2
    print("After task 2:")
    print("  quadrant 1 accuracy:", evaluate(model, X1_test, y1_test))
    print("  quadrant 2 accuracy:", evaluate(model, X2_test, y2_test))