"""Make summary_all.png and convergence table from rti_*.npz outputs."""
import glob, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

files = sorted(glob.glob("rti_*x*.npz"), key=lambda f: int(f.split("_")[1].split("x")[0]))
D = {int(f.split("_")[1].split("x")[0]): dict(np.load(f)) for f in files}
res = sorted(D)
fine = res[-1]
d = D[fine]
ts = d["snap_t"]


def coarsen(a, f):
    nx, ny = a.shape
    return a.reshape(nx//f, f, ny//f, f).mean(axis=(1, 3))


def tips(rho, y, thr=1.5):
    heavy = rho > thr
    ys = np.broadcast_to(y, rho.shape)
    spike = ys[heavy].max()           # heavy fluid penetrating toward +y (gravity direction)
    bubble = ys[~heavy].min()         # light fluid penetrating toward -y
    return spike, bubble

lines = []
lines.append("L1 density difference between successive grids (finer block-averaged onto coarser)")
lines.append("t     " + "  ".join(f"|{a}-{b}|" for a, b in zip(res[:-1], res[1:])))
conv = {}
for k, t in enumerate(ts):
    errs = []
    for a, b in zip(res[:-1], res[1:]):
        ra = D[a]["snap_rho"][k]; rb = coarsen(D[b]["snap_rho"][k], b // a)
        errs.append(np.abs(ra - rb).mean())
    conv[t] = errs
    lines.append(f"{t:.1f}  " + "  ".join(f"{e:.3e}" for e in errs))
lines.append("")
lines.append("Spike / bubble tip positions (rho=1.5 contour extremes)")
for r in res:
    s = [tips(D[r]["snap_rho"][k], D[r]["y"][None, :]) for k in range(len(ts))]
    lines.append(f"{r}x{4*r}: " + "  ".join(f"t={t:.1f}:({a:.3f},{b:.3f})" for t, (a, b) in zip(ts, s)))
lines.append("")
lines.append("Conservation (finest relative drift over [0,3]):")
for r in res:
    m = D[r]["h_mass"]; e = D[r]["h_etot"]
    lines.append(f"{r}x{4*r}: mass {abs(m[-1]-m[0])/m[0]:.2e}  E+PE {abs(e[-1]-e[0])/e[0]:.2e}  "
                 f"x-mirror asym max {D[r]['h_sym'].max():.2e}  min p {D[r]['h_pmin'].min():.4f}  steps {int(D[r]['nsteps'])}")
txt = "\n".join(lines)
print(txt)
open("diagnostics.txt", "w").write(txt + "\n")

# ---------------- figure ----------------
fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1], hspace=0.28)
top = gs[0].subgridspec(1, len(ts) + len(res) + 1, width_ratios=[1]*(len(ts)+len(res)) + [0.08], wspace=0.08)
ext = [0, 0.25, 0, 1]
for k, t in enumerate(ts):
    ax = fig.add_subplot(top[k])
    im = ax.imshow(d["snap_rho"][k].T, origin="lower", extent=ext, cmap="RdYlBu_r", vmin=0.9, vmax=2.1, aspect="equal")
    ax.set_title(f"t = {t:.1f}", fontsize=10)
    ax.set_xticks([0.05, 0.2]); ax.tick_params(labelsize=8);
    if k: ax.set_yticks([])
    else: ax.set_ylabel("y")
for j, r in enumerate(res):
    ax = fig.add_subplot(top[len(ts) + j])
    ax.imshow(D[r]["snap_rho"][-1].T, origin="lower", extent=ext, cmap="RdYlBu_r", vmin=0.9, vmax=2.1, aspect="equal")
    ax.set_title(f"t=3, {r}×{4*r}", fontsize=9); ax.set_yticks([]); ax.set_xticks([0.05, 0.2]); ax.tick_params(labelsize=8)
cax = fig.add_subplot(top[-1]); fig.colorbar(im, cax=cax, label="density ρ")
fig.text(0.5, 0.955, f"Rayleigh–Taylor instability, compressible Euler (γ=1.4, g=+1 ŷ), WENO5+HLLC+SSP-RK3. "
         f"Left: ρ evolution on {fine}×{4*fine}; right: t=3 at each resolution", ha="center", fontsize=12)

bot = gs[1].subgridspec(1, 4, wspace=0.32)
cols = plt.cm.viridis(np.linspace(0, 0.85, len(res)))
ax = fig.add_subplot(bot[0])
for c, r in zip(cols, res):
    s = np.array([tips(D[r]["snap_rho"][k], D[r]["y"][None, :]) for k in range(len(ts))])
    ax.plot(ts, s[:, 0], "o-", color=c, label=f"{r}×{4*r} spike")
    ax.plot(ts, s[:, 1], "s--", color=c, label=f"{r}×{4*r} bubble")
ax.set_xlabel("t"); ax.set_ylabel("y of ρ=1.5 extreme"); ax.set_title("Spike (heavy→+y) / bubble tips")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

ax = fig.add_subplot(bot[1])
for c, r in zip(cols, res):
    rho = D[r]["snap_rho"]; y = D[r]["y"]
    for k, ls in zip([2, 4, 6], ["-", "--", ":"]):
        ax.plot(rho[k].mean(axis=0), y, ls, color=c, lw=1.2, label=f"{r}, t={ts[k]:.0f}" if True else None)
ax.set_xlabel("x-averaged ρ"); ax.set_ylabel("y"); ax.set_title("Horizontally averaged density")
ax.legend(fontsize=6, ncol=1); ax.grid(alpha=0.3)

ax = fig.add_subplot(bot[2])
E = np.array([conv[t] for t in ts])
for j, (a, b) in enumerate(zip(res[:-1], res[1:])):
    ax.semilogy(ts[1:], E[1:, j], "o-", label=f"‖ρ{a}−ρ{b}‖₁")
ax.set_xlabel("t"); ax.set_ylabel("L1 difference"); ax.set_title("Grid convergence (successive grids)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

ax = fig.add_subplot(bot[3])
for c, r in zip(cols, res):
    e = D[r]["h_etot"]; m = D[r]["h_mass"]
    ax.semilogy(D[r]["h_t"][1:], np.abs(e[1:]-e[0])/e[0] + 1e-17, "-", color=c, label=f"{r}: total energy (E−ρgy)")
    ax.semilogy(D[r]["h_t"][1:], np.abs(m[1:]-m[0])/m[0] + 1e-17, ":", color=c, label=f"{r}: mass")
ax.set_xlabel("t"); ax.set_ylabel("relative drift"); ax.set_title("Conservation")
ax.legend(fontsize=6); ax.grid(alpha=0.3, which="both")
fig.savefig("summary_all.png", dpi=130, bbox_inches="tight")
print("saved summary_all.png")
