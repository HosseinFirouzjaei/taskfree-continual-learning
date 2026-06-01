"""
Task-free online continual learning - synthetic experiment.
Reproduces the idea behind Figure 2 of Aljundi et al. (2019).

run() processes ONE long stream of data: all of task 1 (quadrant 1),
then all of task 2 (quadrant 2). It supports three modes:

  plain online            : use_buffer=False, use_mas=False
  online + hard buffer    : use_buffer=True,  use_mas=False
  online + buffer + MAS   : use_buffer=True,  use_mas=True   <- the full method
"""

import numpy as np
import torch
import torch.nn as nn

from data import make_quadrant_data
from model import Net


def to_tensors(X, y):
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def evaluate(model, X, y):
    """Fraction of correct predictions on data X with labels y."""
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y).float().mean().item()


def update_buffer(buffer_X, buffer_y, new_X, new_y, model, max_size):
    """Keep the hardest (highest-loss) points from the buffer plus the new batch."""
    all_X = torch.cat([buffer_X, new_X])
    all_y = torch.cat([buffer_y, new_y])
    per_point_loss = nn.CrossEntropyLoss(reduction='none')   # one loss per point
    model.eval()
    with torch.no_grad():
        losses = per_point_loss(model(all_X), all_y)
    keep = torch.argsort(losses, descending=True)[:max_size]  # hardest first
    return all_X[keep], all_y[keep]


def estimate_importance(model, X):
    """MAS: how sensitive the model's output is to each parameter.
    A large value means the parameter matters a lot -> avoid changing it later."""
    model.eval()
    model.zero_grad()
    output_size = (model(X) ** 2).sum()      # squared length of the output vector
    output_size.backward()                   # gradient w.r.t. every parameter
    importance = {}
    for name, p in model.named_parameters():
        importance[name] = (p.grad.abs() / len(X)).detach().clone()
    model.zero_grad()
    return importance


def run(use_buffer, use_mas,
        lam=3.0, plateau_mean=0.05, plateau_std=0.02,
        batch_size=10, lr=0.03, steps_per_batch=5,
        buffer_size=20, window_size=5):

    # --- build the stream: all of task 1, then all of task 2 ---
    X1, y1 = to_tensors(*make_quadrant_data(2000, quadrant=1, seed=1))
    X2, y2 = to_tensors(*make_quadrant_data(2000, quadrant=2, seed=2))
    X1_test, y1_test = to_tensors(*make_quadrant_data(500, quadrant=1, seed=11))
    X2_test, y2_test = to_tensors(*make_quadrant_data(500, quadrant=2, seed=22))

    rng = np.random.default_rng(0)                       # shuffle inside each task
    o1 = rng.permutation(len(X1)); X1, y1 = X1[o1], y1[o1]
    o2 = rng.permutation(len(X2)); X2, y2 = X2[o2], y2[o2]

    stream_X = torch.cat([X1, X2])
    stream_y = torch.cat([y1, y2])
    switch_batch = len(X1) // batch_size                 # where task 2 begins

    # --- model and training tools ---
    torch.manual_seed(0)
    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    # the hard buffer (starts empty)
    buffer_X = torch.empty(0, 4)
    buffer_y = torch.empty(0, dtype=torch.long)

    # MAS memory: importance of each parameter, and the values to stay close to
    importance = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    anchor = {n: p.detach().clone() for n, p in model.named_parameters()}
    n_consolidations = 0

    # plateau / peak detector
    loss_window = []
    on_plateau = False
    plateau_mean_old = 0.0
    plateau_std_old = 0.0

    q1_curve = []                                        # task-1 accuracy over time

    for start in range(0, len(stream_X), batch_size):
        batch_X = stream_X[start:start + batch_size]
        batch_y = stream_y[start:start + batch_size]

        # mix the new batch with the buffer
        if use_buffer and len(buffer_X) > 0:
            train_X = torch.cat([batch_X, buffer_X])
            train_y = torch.cat([batch_y, buffer_y])
        else:
            train_X, train_y = batch_X, batch_y

        # a few gradient steps on this batch
        model.train()
        for _ in range(steps_per_batch):
            task_loss = loss_function(model(train_X), train_y)
            if use_mas:
                penalty = 0.0
                for name, p in model.named_parameters():
                    penalty = penalty + (importance[name] * (p - anchor[name]) ** 2).sum()
                loss = task_loss + (lam / 2.0) * penalty
            else:
                loss = task_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # MAS: consolidate only when the loss is on a low, steady plateau
        if use_mas:
            model.eval()
            with torch.no_grad():
                new_batch_loss = loss_function(model(batch_X), batch_y).item()
            loss_window.append(new_batch_loss)
            if len(loss_window) > window_size:
                loss_window.pop(0)

            mean = float(np.mean(loss_window))
            std = float(np.std(loss_window))

            # plateau -> save which parameters matter now
            if (not on_plateau) and len(loss_window) == window_size \
                    and mean < plateau_mean and std < plateau_std:
                points = buffer_X if len(buffer_X) > 0 else batch_X
                new_importance = estimate_importance(model, points)
                n_consolidations += 1
                for name in importance:                  # cumulative moving average
                    importance[name] = importance[name] \
                        + (new_importance[name] - importance[name]) / n_consolidations
                anchor = {n: p.detach().clone() for n, p in model.named_parameters()}
                plateau_mean_old, plateau_std_old = mean, std
                loss_window = []
                on_plateau = True

            # peak (loss jumped up) -> allow a new plateau to be found later
            if len(loss_window) > 0 and float(np.mean(loss_window)) > plateau_mean_old + plateau_std_old:
                on_plateau = False

        # update the buffer with the hardest points
        if use_buffer:
            buffer_X, buffer_y = update_buffer(buffer_X, buffer_y, batch_X, batch_y, model, buffer_size)

        # record task-1 accuracy every few batches (for plotting)
        if (start // batch_size) % 8 == 0:
            q1_curve.append((start // batch_size, evaluate(model, X1_test, y1_test)))

    q1 = evaluate(model, X1_test, y1_test)
    q2 = evaluate(model, X2_test, y2_test)
    return {"q1": q1, "q2": q2, "total": (q1 + q2) / 2,
            "consolidations": n_consolidations,
            "switch": switch_batch, "q1_curve": q1_curve}


if __name__ == "__main__":
    r = run(use_buffer=True, use_mas=True)        # the full method
    print("Full method (buffer + MAS):")
    print(f"  quadrant 1 accuracy: {r['q1'] * 100:.1f}%")
    print(f"  quadrant 2 accuracy: {r['q2'] * 100:.1f}%")
    print(f"  importance consolidations: {r['consolidations']}")
