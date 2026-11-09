"""Builds summary_all.png from the solver outputs in runs/."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os, sys

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C = {"n64": "#2a78d6", "n128": "#eb6834", "n256": "#1baf7a"}
LBL = {"n64": "64×256", "n128": "128×512", "n256": "256×1024"}
cmap = LinearSegmentedColormap.from_list("blue_seq", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
SIG_LIN = float(sys.argv[1]) if len(sys.argv) > 1 else 2.4507

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2})


def snaps(name, NX, NY):
    raw = np.fromfile(os.path.join(D, f"{name}_snap.bin"))
    rec = 1 + 5 * NX * NY
    raw = raw[: (raw.size // rec) * rec].reshape(-1, rec)
    return raw[:, 0], raw[:, 1:].reshape(-1, 5, NY, NX)


def diag(name):
    return np.loadtxt(os.path.join(D, f"{name}_diag.txt"))


fin = "n256" if os.path.exists(os.path.join(D, "n256_snap.bin")) else "n128"
NX = {"n64": 64, "n128": 128, "n256": 256}[fin]
ts, S = snaps(fin, NX, 4 * NX)
show = [0, 2, 4, 6, 8, 10, 12]

fig = plt.figure(figsize=(18, 10.5), facecolor="#fcfcfb")
gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1], hspace=0.28)
top = gs[0].subgridspec(1, len(show) + 1, width_ratios=[1] * len(show) + [0.08], wspace=0.15)
for n, k in enumerate(show):
    ax = fig.add_subplot(top[n])
    im = ax.imshow(S[k, 0], origin="lower", extent=[0, 0.25, 0, 1], cmap=cmap, vmin=0.8, vmax=2.2,
                   interpolation="nearest")
    ax.contour(np.linspace(0, .25, NX), np.linspace(0, 1, 4 * NX), S[k, 4], levels=[0.5], colors=INK,
               linewidths=0.5)
    ax.set_title(f"t = {ts[k]:.1f}", color=INK)
    ax.grid(False); ax.set_xticks([0, 0.25])
    if n: ax.set_yticklabels([])
    else: ax.set_ylabel("y  (gravity → +y)")
    ax.set_xlabel("x")
cax = fig.add_subplot(top[-1]); cb = fig.colorbar(im, cax=cax); cb.set_label("density ρ")
fig.text(0.5, 0.93, f"Rayleigh–Taylor instability, compressible Euler (γ=1.4, g=+1 ŷ, A=1/3) — density, "
         f"{LBL[fin]} grid; black line: tracer = 0.5 (material interface)", ha="center", fontsize=12, color=INK)

bot = gs[1].subgridspec(1, 5, wspace=0.36)
runs = [r for r in ["n64", "n128", "n256"] if os.path.exists(os.path.join(D, f"{r}_diag.txt"))]

# (a) bubble / spike fronts
ax = fig.add_subplot(bot[0])
for r in runs:
    d = diag(r)
    ax.plot(d[:, 0], d[:, 11], color=C[r], label=LBL[r])
    ax.plot(d[:, 0], d[:, 10], color=C[r], ls="--")
ax.set_xlabel("t"); ax.set_ylabel("y of mixing-zone edge")
ax.set_title("(a) Spike (solid) & bubble (dashed) fronts\n(5%/95% tracer)", color=INK)
ax.legend(frameon=False, loc="upper left")

# (b) mixing width
ax = fig.add_subplot(bot[1])
for r in runs:
    d = diag(r); ax.plot(d[:, 0], d[:, 9], color=C[r], label=LBL[r])
ax.set_xlabel("t"); ax.set_ylabel("W = 6∫⟨f(1−f)⟩ dy")
ax.set_title("(b) Integral mixing width", color=INK); ax.legend(frameon=False)

# (c) linear growth check
ax = fig.add_subplot(bot[2])
for name, lab, col in [("lin128", "amp×1e-3, 128×512", "#6250d6"), (f"{fin}", f"task amp, {LBL[fin]}", C[fin])]:
    p = os.path.join(D, f"{name}_diag.txt")
    if not os.path.exists(p): continue
    d = np.loadtxt(p); a = np.abs(d[:, 8]); ax.semilogy(d[:, 0], a / a[0], color=col, label=lab, lw=2 if name != "lin128" else 3, alpha=1 if name != "lin128" else 0.6)
t = np.linspace(0.3, 2.0, 50)
dl = np.loadtxt(os.path.join(D, "lin128_diag.txt")); al = np.abs(dl[:, 8]) / abs(dl[0, 8])
i0 = np.searchsorted(dl[:, 0], 0.8)
ax.semilogy(t, al[i0] * np.exp(SIG_LIN * (t - dl[i0, 0])), color=INK, ls="--", lw=1.3,
            label=f"linear theory σ={SIG_LIN:.3f}")
ax.set_xlim(0, 3); ax.set_xlabel("t"); ax.set_ylabel("|v̂(k=8π)| / initial")
ax.set_title("(c) Mode growth vs linear stability", color=INK); ax.legend(frameon=False, fontsize=8)

# (d) conservation
ax = fig.add_subplot(bot[3])
for r in runs:
    d = diag(r); Et = d[:, 4] + d[:, 5]
    ax.semilogy(d[1:, 0], np.abs(Et[1:] - Et[0]) / abs(Et[0]) + 1e-17, color=C[r], label=f"energy, {LBL[r]}")
for r in runs:
    d = diag(r); ax.semilogy(d[1:, 0], np.abs(d[1:, 1] - d[0, 1]) / d[0, 1] + 1e-17, color=C[r], ls=":", lw=1.2)
ax.set_xlabel("t"); ax.set_ylabel("relative drift")
ax.set_title("(d) Conservation: total energy incl. potential\n(solid), mass (dotted)", color=INK)
ax.legend(frameon=False, fontsize=7.5, loc="center right", bbox_to_anchor=(1, 0.45))

# (e) hydrostatic rest test
ax = fig.add_subplot(bot[4])
d = diag("n128_rest"); ax.plot(d[:, 0], d[:, 7], color="#2a78d6", label="unperturbed, max|v|")
d = diag("n128"); ax.semilogy(d[:, 0], d[:, 7], color="#eb6834", label="perturbed, max|v|")
ax.set_xlabel("t"); ax.set_ylabel("max |v|  (128×512)")
ax.set_title("(e) Hydrostatic test: spurious vs\nphysical velocity", color=INK); ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1, 0.3))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "summary_all.png")
fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
print("saved", out)
