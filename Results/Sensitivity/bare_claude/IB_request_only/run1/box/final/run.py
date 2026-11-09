import numpy as np, json, time, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, burgers2d
from burgers2d import solve, exact_grid as exact_cellavg, exact_point, L

res = {"convergence": {}}
burgers2d.SCHEME["name"] = sys.argv[1] if len(sys.argv) > 1 else "fd"
res["scheme"] = burgers2d.SCHEME["name"]
for N in [64, 128, 256, 512]:
    t0 = time.time()
    _, sn = solve(N, T=2.0, save_times=[1.0, 2.0])
    dx = L / N
    row = {}
    for tt in [1.0, 2.0]:
        e = sn[tt] - exact_cellavg(N, tt)
        row[str(tt)] = {"L1": float(np.mean(np.abs(e))), "L2": float(np.sqrt(np.mean(e**2))),
                        "Linf": float(np.max(np.abs(e)))}
        if tt > 1.5:   # error away from the shock: exclude |xi - pi| < 0.25 (fixed physical band)
            xc_ = (np.arange(N) + 0.5) * dx
            X, Y = np.meshgrid(xc_, xc_, indexing="ij")
            xi = np.mod(0.5*(X+Y) - tt/3 + np.pi, 2*np.pi) - np.pi
            m = np.abs(np.abs(xi) - np.pi) > 0.25
            row[str(tt) + "_away_from_shock"] = {"L1": float(np.mean(np.abs(e[m]))), "Linf": float(np.max(np.abs(e[m])))}
            # numerical shock position along diagonal: where u crosses mean C=1/3 with negative slope
            d = np.diag(sn[tt]); xs = 2*xc_
            k_ = [i for i in range(N-1) if d[i] > 1/3 >= d[i+1]]
            pos = [xs[i] + (d[i]-1/3)/(d[i]-d[i+1])*(xs[i+1]-xs[i]) for i in k_]
            row["shock_x_plus_y_numeric_" + str(tt)] = [float(p_) for p_ in pos]
            row["shock_x_plus_y_exact_" + str(tt)] = [float(2*np.pi + 2*tt/3 + 4*np.pi*j) for j in range(2)]
        row["mass_drift_" + str(tt)] = float(abs(sn[tt].mean() - 1/3))
        row["max_" + str(tt)] = float(sn[tt].max()); row["min_" + str(tt)] = float(sn[tt].min())
    res["convergence"][N] = row
    print(N, json.dumps(row), f"{time.time()-t0:.1f}s", flush=True)

Ns = sorted(res["convergence"])
for tt in ["1.0", "2.0"]:
    for nrm in ["L1", "L2", "Linf"]:
        es = [res["convergence"][n][tt][nrm] for n in Ns]
        res.setdefault("orders", {})[f"{tt}_{nrm}"] = [float(np.log2(es[i]/es[i+1])) for i in range(len(es)-1)]
es = [res["convergence"][n]["2.0_away_from_shock"]["L1"] for n in Ns]
res["orders"]["2.0_L1_away_from_shock"] = [float(np.log2(es[i]/es[i+1])) for i in range(len(es)-1)]
print(json.dumps(res["orders"], indent=1))

# snapshots at finest grid
N = 512
times = [0.5, 1.0, 1.5, 1.75, 2.0]
uT, sn = solve(N, T=2.0, save_times=times)
sn[0.0] = exact_cellavg(N, 0.0)
np.savez_compressed("/box/final/snapshots_N512.npz", **{f"t{k:.2f}": v for k, v in sn.items()})
json.dump(res, open("/box/final/results.json", "w"), indent=1)

xc = (np.arange(N) + 0.5) * L / N
alltimes = [0.0] + times
fig = plt.figure(figsize=(22, 9.5), layout="constrained")
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.1])
for i, tt in enumerate(alltimes):
    ax = fig.add_subplot(gs[0, i])
    im = ax.imshow(sn[tt].T, origin="lower", extent=[0, L, 0, L], cmap="RdBu_r", vmin=-0.35, vmax=1.0)
    ax.set_title(f"u(x,y,t={tt:g})"); ax.set_xlabel("x")
    if i == 0: ax.set_ylabel("y")
fig.colorbar(im, ax=fig.axes[:6], shrink=0.9, location="right", pad=0.01, label="u")
# diagonal cut y = x  (x+y = 2x): numerics vs exact
ax = fig.add_subplot(gs[1, :4])
xf = np.linspace(0, L, 4000)
cols = plt.cm.viridis(np.linspace(0, 0.9, len(alltimes)))
for c, tt in zip(cols, alltimes):
    ax.plot(xc, np.diag(sn[tt]), ".", ms=2.5, color=c)
    ax.plot(xf, exact_point(xf.copy(), xf.copy(), tt), "-", lw=0.8, color=c, label=f"t={tt:g}")
ax.set_xlabel("x  (along diagonal y = x)"); ax.set_ylabel("u")
ax.set_title("Diagonal cut: dots = FD-WENO5 N=512 (every grid point on y=x), lines = exact entropy solution (shock forms at t=1.5)")
ax.legend(ncol=6, fontsize=8, loc="lower left")
ax = fig.add_subplot(gs[1, 4:])
for tt, mk in [("1.0", "o"), ("2.0", "s")]:
    ax.loglog(Ns, [res["convergence"][n][tt]["L1"] for n in Ns], mk + "-", label=f"L1 error, t={tt}")
ax.loglog(Ns, [res["convergence"][n]["2.0_away_from_shock"]["L1"] for n in Ns], "^-", label="L1, t=2, |dist to shock|>0.25")
ax.loglog(Ns, 3e2*np.array(Ns, float)**-5, "k:", label="slope -5")
ax.loglog(Ns, 3*np.array(Ns, float)**-1, "k--", label="slope -1")
ax.set_xlabel("N"); ax.set_ylabel("L1 error vs exact"); ax.legend(fontsize=8); ax.set_title("Grid convergence")
fig.suptitle(r"2D inviscid Burgers, $u_0=1/3+(2/3)\sin\frac{x+y}{2}$, periodic $[0,4\pi]^2$", fontsize=14)
plt.savefig("/box/final/summary_all.png", dpi=110, bbox_inches="tight")
print("done")
