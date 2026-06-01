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

def update_buffer(buffer_X, buffer_y, new_X, new_y, model, max_size):
    """Keep the hardest (highest-loss) points from the buffer plus the new batch."""
    all_X = torch.cat([buffer_X, new_X], dim=0)
    all_y = torch.cat([buffer_y, new_y], dim=0)

    loss_function = nn.CrossEntropyLoss(reduction='none')  # one loss per point
    model.eval()
    with torch.no_grad():
        losses = loss_function(model(all_X), all_y)

    order = torch.argsort(losses, descending=True)   # hardest first
    keep = order[:max_size]                           # keep the top few
    return all_X[keep], all_y[keep]

def train_online(model, X, y, buffer_X, buffer_y,
                 batch_size=10, lr=0.03, steps_per_batch=5, buffer_size=100):
    """Train one batch at a time, mixing in the hard buffer. Returns the updated buffer."""
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for start in range(0, len(X), batch_size):
        batch_X = X[start:start + batch_size]
        batch_y = y[start:start + batch_size]

        # train on the new batch PLUS the buffer
        if len(buffer_X) > 0:
            train_X = torch.cat([batch_X, buffer_X], dim=0)
            train_y = torch.cat([batch_y, buffer_y], dim=0)
        else:
            train_X, train_y = batch_X, batch_y

        model.train()
        for _ in range(steps_per_batch):
            outputs = model(train_X)
            loss = loss_function(outputs, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # update the buffer with the hardest points
        buffer_X, buffer_y = update_buffer(buffer_X, buffer_y, batch_X, batch_y, model, buffer_size)

    return buffer_X, buffer_y


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

    # one buffer for the whole stream, starts empty
    buffer_X = torch.empty(0, 4)
    buffer_y = torch.empty(0, dtype=torch.long)

    buffer_X, buffer_y = train_online(model, X1, y1, buffer_X, buffer_y)   # task 1
    print("After task 1:")
    print("  quadrant 1 accuracy:", evaluate(model, X1_test, y1_test))
    print("  quadrant 2 accuracy:", evaluate(model, X2_test, y2_test))

    buffer_X, buffer_y = train_online(model, X2, y2, buffer_X, buffer_y)   # task 2
    print("After task 2:")
    print("  quadrant 1 accuracy:", evaluate(model, X1_test, y1_test))
    print("  quadrant 2 accuracy:", evaluate(model, X2_test, y2_test))