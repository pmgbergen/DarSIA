"""Plot cell transport density |q_K| for the wall test (reads results/wall_test_flux_<TAG>.npz).

One row per eps, one column per wall strength K.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "results"
TAG = "lagged"
VARIANT = "F"
SOLVERS = {0.05: "lagged+AA5", 0.0005: "lagged+AA5+cont"}

data = np.load(OUT / f"wall_test_flux_{TAG}.npz")
Ks = sorted({float(k.split("_")[0][1:]) for k in data})
fig, axs = plt.subplots(
    len(SOLVERS), len(Ks), figsize=(3.0 * len(Ks), 3.0 * len(SOLVERS)), constrained_layout=True,
    squeeze=False,
)
for row, (eps, solver) in enumerate(SOLVERS.items()):
    for col, K in enumerate(Ks):
        q = data[f"K{K}_{VARIANT}_{solver}_eps{eps}"]
        ax = axs[row, col]
        ax.imshow(q.T, origin="lower", extent=(0, 8, 0, 8), cmap="magma", vmax=np.quantile(q, 0.995))
        ax.set_title(f"K = {K:g}, eps = {eps:g}  ({VARIANT})")
        ax.set_xticks([0, 4, 8])
        ax.set_yticks([0, 4, 8])
fig.savefig(OUT / f"wall_test_flux_{VARIANT}.png", dpi=150)
