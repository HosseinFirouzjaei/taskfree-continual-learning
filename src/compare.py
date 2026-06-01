"""Run the three methods and reproduce Figure 2. Saves figure2_repro.png."""

import matplotlib
matplotlib.use("Agg")                 # save to a file instead of opening a window
import matplotlib.pyplot as plt

from train import run

methods = [
    ("Online No Hard Buffer", dict(use_buffer=False, use_mas=False), "tab:green"),
    ("Online (hard buffer)",  dict(use_buffer=True,  use_mas=False), "tab:orange"),
    ("Continual (ours)",      dict(use_buffer=True,  use_mas=True),  "tab:blue"),
]

print("=== Reproducing Figure 2 ===\n")
results = {}
for name, cfg, color in methods:
    r = run(**cfg)
    results[name] = (r, color)
    print(f"  {name:24s}: quadrant 1 = {r['q1'] * 100:5.1f}%   total = {r['total'] * 100:5.1f}%"
          f"   ->  {r['q1'] * 100:.0f}({r['total'] * 100:.0f})")

print("\nPaper Figure 2:           50(67), 52(74), 100(100)")

plt.figure(figsize=(8, 5))
for name, (r, color) in results.items():
    xs = [b for b, _ in r["q1_curve"]]
    ys = [a * 100 for _, a in r["q1_curve"]]
    plt.plot(xs, ys, label=name, color=color, linewidth=2)
plt.axvline(results["Continual (ours)"][0]["switch"], color="grey",
            linestyle="--", label="task switch (q1 -> q2)")
plt.xlabel("training step (batch number)")
plt.ylabel("Quadrant 1 test accuracy (%)")
plt.title("Synthetic experiment: task-1 accuracy while learning task 2")
plt.legend(loc="lower left")
plt.ylim(40, 105)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figure2_repro.png", dpi=120)
print("\nSaved figure2_repro.png")
