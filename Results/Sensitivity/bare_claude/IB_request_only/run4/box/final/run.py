import numpy as np, json, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from burgers2d import solve, exact_s, exact_xy, L

times = [0.0, 0.5, 1.0, 1.5, 1.75, 2.0]
res = {"convergence": {}}

# --- convergence study ---
for N in [32, 64, 128, 256, 512]:
    t0 = time.time()
    snaps = solve(N, 2.0, snap_times=times)
    row = {}
    for t in [1.0, 2.0]:
        xc = (np.arange(N) + 0.5) * L / N
        e = snaps[t] - exact_xy(xc[:, None], xc[None, :], t)
        row[f"t={t}"] = dict(L1=float(np.mean(np.abs(e))), L2=float(np.sqrt(np.mean(e**2))),
                             Linf=float(np.max(np.abs(e))))
    m0 = snaps[0.0].mean(); m2 = snaps[2.0].mean()
    row["mass_drift"] = float(abs(m2 - m0))
    row["max_u_t2"] = float(snaps[2.0].max()); row["min_u_t2"] = float(snaps[2.0].min())
    # symmetry check: solution should depend only on x+y -> invariant under (i,j)->(i+1,j-1)
    row["xy_symmetry_dev"] = float(np.max(np.abs(snaps[2.0] - np.roll(np.roll(snaps[2.0], 1, 0), -1, 1))))
    row["wall_s"] = time.time() - t0
    res["convergence"][N] = row
    print(N, json.dumps(row))
    if N == 512:
        final = snaps

Ns = sorted(res["convergence"])
for t in ["t=1.0", "t=2.0"]:
    for nrm in ["L1", "L2", "Linf"]:
        e = [res["convergence"][N][t][nrm] for N in Ns]
        rates = [float(np.log2(e[i] / e[i+1])) for i in range(len(e)-1)]
        res.setdefault("rates", {}).setdefault(t, {})[nrm] = rates
print(json.dumps(res["rates"], indent=1))
json.dump(res, open("results.json", "w"), indent=1)

# --- save final-grid solution ---
N = 512; h = L / N; xc = (np.arange(N) + 0.5) * h
np.savez_compressed("solution_N512.npz", x=xc, y=xc, times=np.array(times),
                    u=np.array([final[t] for t in times]))

# --- summary plot ---
fig = plt.figure(figsize=(18, 13))
gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.9])
X, Y = np.meshgrid(xc, xc, indexing="ij")
for k, t in enumerate(times):
    ax = fig.add_subplot(gs[k // 3, (k % 3) * 2:(k % 3) * 2 + 2])
    im = ax.pcolormesh(xc, xc, final[t].T, cmap="viridis", vmin=-1/3, vmax=1, shading="auto")
    ax.set_aspect("equal"); ax.set_title(f"u(x,y,t={t})  [WENO5, {N}²]")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046)
# diagonal profile vs exact
ax = fig.add_subplot(gs[2, 0:3])
s_fine = np.linspace(0, 2*np.pi, 2000)
cols = plt.cm.plasma(np.linspace(0, 0.85, len(times)))
for c, t in zip(cols, times):
    # cells along diagonal j=i : s = x
    ax.plot(xc, final[t].diagonal(), '.', ms=2, color=c)           # on y=x, s=x
    ax.plot(s_fine, exact_s(s_fine, t), '-', lw=1, color=c, label=f"t={t}")
ax.set_xlim(0, 2*np.pi)
ax.set_xlabel("s = (x+y)/2   (along diagonal y=x, s=x)"); ax.set_ylabel("u")
ax.set_title("Diagonal profile: dots = numerical, lines = exact (characteristics + shock)")
ax.legend(fontsize=8, ncol=2)
ax = fig.add_subplot(gs[2, 3:6])
for t, mk in [("t=1.0", "o-"), ("t=2.0", "s-")]:
    for nrm in ["L1", "Linf"]:
        ax.loglog(Ns, [res["convergence"][n][t][nrm] for n in Ns], mk, label=f"{nrm}, {t}")
sc = json.load(open("shock_check.json")); Ns2 = sorted(int(k) for k in sc)
ax.loglog(Ns2, [sc[str(n)]["L1_away"] for n in Ns2], "^-", label="L1, t=2.0, |s-s_shock|>0.5")
ax.loglog(Ns, 3e2*np.array(Ns, float)**-5, 'k--', lw=0.8, label="O(N⁻⁵)")
ax.loglog(Ns, 2*np.array(Ns, float)**-1, 'k:', lw=0.8, label="O(N⁻¹)")
ax.set_xlabel("N (cells per direction)"); ax.set_ylabel("error vs exact point values")
ax.set_title("Grid convergence"); ax.legend(fontsize=8)
fig.suptitle("2D inviscid Burgers  u_t + u u_x + u u_y = 0,  u0 = 1/3 + 2/3 sin((x+y)/2), periodic on [0,4π]²"
             "\nshock forms at t* = 1.5", fontsize=13)
fig.tight_layout()
fig.savefig("summary_all.png", dpi=120)
