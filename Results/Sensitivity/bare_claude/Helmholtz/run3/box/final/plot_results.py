import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = np.load("solution_hard_seed0.npz"); X, Y, U, Ue = d["X"], d["Y"], d["U_pinn"], d["U_exact"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for a, Z, t, cm in zip(ax, [U, Ue, np.abs(U-Ue)], ["PINN u", "exact u = sin(πx)sin(4πy)", "|error|"], ["RdBu_r", "RdBu_r", "viridis"]):
    im = a.pcolormesh(X, Y, Z, cmap=cm, shading="auto"); a.set_title(t); a.set_xlabel("x"); a.set_ylabel("y"); a.set_aspect("equal"); fig.colorbar(im, ax=a)
fig.tight_layout(); fig.savefig("solution_error.png", dpi=130)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for s in [0, 1, 2]:
    h = json.load(open(f"results_hard_seed{s}.json"))["history"]
    step = [r["it"] + (20000 if r["phase"] == "lbfgs" else 0) for r in h]
    ax[0].semilogy(step, [r["loss"] for r in h], label=f"seed {s}"); ax[1].semilogy(step, [r["rel_l2"] for r in h], label=f"seed {s}")
for a, t in zip(ax, ["training loss (normalised PDE residual²)", "relative L2 error vs exact"]):
    a.axvline(20000, color="gray", ls="--"); a.set_title(t); a.set_xlabel("iteration (Adam | L-BFGS)"); a.legend(); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig("training_history.png", dpi=130)
