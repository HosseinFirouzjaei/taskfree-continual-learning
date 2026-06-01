# Task-Free Continual Learning — Reproduction

A from-scratch reproduction of the **synthetic experiment** in
*"Task-Free Continual Learning"* by Aljundi, Kelchtermans, and Tuytelaars (2019).

A small network learns two tasks one after another (points inside/outside a
4D sphere, first in one quadrant, then in another). Plain online learning
**forgets** the first task while learning the second. The paper's method —
a small **hard buffer** plus **Memory Aware Synapses (MAS)** with a
loss-plateau detector — keeps the old knowledge.

## Result (reproduces Figure 2)

| Method | Quadrant-1 acc | Total acc |
|---|---|---|
| Online (no buffer) | ~60% | ~80% |
| Online + hard buffer | ~51% | ~76% |
| **Online + buffer + MAS (ours)** | **~100%** | **~100%** |

Paper Figure 2: 50(67), 52(74), 100(100). Both baselines forget task 1;
only the full method keeps it. See `figure2_repro.png`.

## Files (in `src/`)
- `data.py`  — makes the 4D points (inside/outside the sphere, per quadrant)
- `model.py` — a small neural network
- `train.py` — the full method: streaming training, hard buffer, MAS, plateau detector
- `compare.py` — runs all three methods and saves `figure2_repro.png`

## Setup (conda)
```
conda create -n taskfree python=3.11 pytorch numpy matplotlib -c conda-forge
conda activate taskfree
```

## Run
```
python src/train.py      # run the full method once
python src/compare.py    # compare all three methods + save the figure
```

## How the method works (short version)
1. **Stream**: data arrives in small batches; the model sees a little at a time.
2. **Hard buffer**: keep the few highest-loss points seen and re-train on them.
3. **MAS**: when the loss settles on a low, steady plateau, measure how
   important each weight is, then penalise future changes to important weights.
4. **Plateau/peak detector**: decides *when* to do step 3, without ever being
   told where one task ends and the next begins (this is the "task-free" part).
