import sys, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
pre = sys.argv[1] if len(sys.argv) > 1 else "/box/final/run"
d = np.load(pre + "_pred.npz"); h = json.load(open(pre + "_hist.json"))
x, t, up, ur = d["x"], d["t"], d["u_pred"], d["u_ref"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, Z, ttl, cm in [(ax[0], ur, "reference", "RdBu_r"), (ax[1], up, "PINN", "RdBu_r"), (ax[2], np.abs(up - ur), "|error|", "viridis")]:
    im = a.pcolormesh(t, x, Z.T, cmap=cm, shading="auto"); a.set_xlabel("t"); a.set_ylabel("x"); a.set_title(ttl); fig.colorbar(im, ax=a)
fig.tight_layout(); fig.savefig(pre + "_fields.png", dpi=120)
fig, ax = plt.subplots(1, 4, figsize=(16, 3.5))
for a, tt in zip(ax, [0.25, 0.5, 0.75, 1.0]):
    i = np.argmin(abs(t - tt)); a.plot(x, ur[i], "k-", label="reference"); a.plot(x, up[i], "r--", label="PINN"); a.set_title(f"t={t[i]:.2f}"); a.set_xlabel("x")
ax[0].legend(); fig.tight_layout(); fig.savefig(pre + "_slices.png", dpi=120)
fig, ax = plt.subplots(figsize=(6, 4))
it = [e["it"] for e in h["hist"]]
ax.semilogy(it, [e["res"] for e in h["hist"]], label="mean residual loss"); ax.semilogy(it, [e["rel_l2"] for e in h["hist"]], label="rel. L2 error (eval only)")
ax.set_xlabel("step"); ax.legend(); fig.tight_layout(); fig.savefig(pre + "_history.png", dpi=120)
rel_t = np.linalg.norm(up - ur, axis=1) / np.linalg.norm(ur, axis=1)
print("rel L2 at t=0.25,0.5,0.75,1:", [f"{rel_t[np.argmin(abs(t-s))]:.2e}" for s in [0.25, .5, .75, 1]])
