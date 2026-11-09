"""Builds summary_all.png from results.json + snapshots_N512.npz (produced by run.py)."""
import numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from burgers2d import L, exact_1d, exact_nodes
res = json.load(open("results.json")); conv = res["convergence"]
Ns = sorted(int(k) for k in conv); times = ["1.0", "1.5", "2.0"]
d = np.load("snapshots_N512.npz"); ts = [0.0, 0.5, 1.0, 1.5, 1.75, 2.0]
snaps = {t: d[f"t{t}"].astype(float) for t in ts}; N = 512; xc = (np.arange(N)+0.5)*L/N
fig = plt.figure(figsize=(20, 11), layout="constrained")
gs = fig.add_gridspec(3, 6)
for k, t in enumerate(ts):
    ax = fig.add_subplot(gs[0, k])
    im = ax.imshow(snaps[t].T, origin="lower", extent=[0, L, 0, L], cmap="viridis", vmin=-1/3, vmax=1)
    ax.set_title(f"u(x,y), t={t}"); ax.set_xlabel("x"); ax.set_ylabel("y")
fig.colorbar(im, cax=ax.inset_axes([1.03, 0, 0.05, 1]), label="u")
for k, t in enumerate(ts):
    ax = fig.add_subplot(gs[1, k])
    sf = np.linspace(0, L, 4000); ax.plot(sf, exact_1d(sf, t), "k-", lw=1, label="exact")
    ax.plot(xc[::2], np.diag(snaps[t])[::2], "o", ms=2, color="C0", label="WENO5, N=512")
    ax.set_title(f"slice y=x, t={t}"); ax.set_xlabel("x"); ax.set_ylim(-0.45, 1.1); ax.grid(alpha=.3)
    if k == 0: ax.legend(fontsize=8, loc="lower left")
ax = fig.add_subplot(gs[2, 0:2])
for i, t in enumerate(times):
    ax.loglog(Ns, [conv[str(n)][t]["L1"] for n in Ns], "o-", color=f"C{i}", label=f"L1 error, t={t}")
n = np.array(Ns, float)
ax.loglog(n, conv["64"]["1.0"]["L1"]*(64/n)**5, "k--", lw=1, label="slope -5")
ax.loglog(n, conv["64"]["2.0"]["L1"]*(64/n)**1, "k:", lw=1, label="slope -1")
ax.set_xlabel("N (grid points per direction)"); ax.set_ylabel("L1 error vs exact"); ax.grid(alpha=.3, which="both")
ax.legend(fontsize=8); ax.set_title("grid convergence (t=1 smooth; t>=1.5 shocked)")
ax = fig.add_subplot(gs[2, 2:4])
e = snaps[2.0] - exact_nodes(N, 2.0)
im2 = ax.imshow(np.log10(np.abs(e).T+1e-16), origin="lower", extent=[0, L, 0, L], cmap="magma", vmin=-12, vmax=0)
fig.colorbar(im2, cax=ax.inset_axes([1.03, 0, 0.04, 1]), label="log10 |u - u_exact|"); ax.set_title("pointwise error, t=2, N=512")
ax = fig.add_subplot(gs[2, 4:6])
for t in ts: ax.plot(xc, np.diag(snaps[t]), label=f"t={t}")
ax.axvline(np.pi+2/3, color="gray", ls=":", lw=1, label="exact shock, t=2")
ax.set_xlim(2.5, 5.5); ax.set_xlabel("x (on y=x)"); ax.set_title("zoom: steepening and shock (y=x)")
ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.suptitle(r"2D inviscid Burgers $u_t+(u^2/2)_x+(u^2/2)_y=0$, $u_0=\frac{1}{3}+\frac{2}{3}\sin\frac{x+y}{2}$, periodic $[0,4\pi]^2$ — FD-WENO5 + SSP-RK3", fontsize=14)
fig.savefig("summary_all.png", dpi=110)
