"""Diagnostics + summary figure for the RT runs. Usage: python analyze.py RUNDIR OUTDIR"""
import sys, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

run, out = sys.argv[1], sys.argv[2]
GAM, G, LX, LY = 1.4, 1.0, 0.25, 1.0
K = 8 * np.pi
SIGMA_LIN = None
try:
    sys.path.insert(0, out)
    from linear_stability import compressible
    SIGMA_LIN = compressible(1600)
except Exception as e:
    print("linear stability failed", e)


def rd(f):
    with open(f, "rb") as fh:
        nx, ny = np.fromfile(fh, np.int32, 2)
        t = np.fromfile(fh, np.float64, 1)[0]
        d = np.fromfile(fh, np.float64).reshape(4, ny, nx)
    return t, d


def series(pre):
    return [rd(f) for f in sorted(glob.glob(f"{run}/{pre}_*.bin"))]


def prim(d):
    r = d[0]; u = d[1] / r; v = d[2] / r
    p = (GAM - 1) * (d[3] - 0.5 * r * (u * u + v * v))
    return r, u, v, p


def integrals(d):
    ny, nx = d.shape[1:]
    dA = (LX / nx) * (LY / ny)
    y = (np.arange(ny) + 0.5) / ny * LY
    m = d[0].sum() * dA
    mx = d[1].sum() * dA
    etot = (d[3] - d[0] * G * y[:, None]).sum() * dA  # E + rho*phi, phi = -g y
    return m, mx, etot


def tips(d, thr=1.5):
    r = d[0]; ny = r.shape[0]
    y = (np.arange(ny) + 0.5) / ny
    heavy = r > thr
    spike = y[np.any(heavy, axis=1)].max()  # heavy fluid rising (+y, along gravity)
    bubble = y[np.any(~heavy, axis=1)].min()  # light fluid sinking
    return spike, bubble


def coarsen(a, f):
    ny, nx = a.shape
    return a.reshape(ny // f, f, nx // f, f).mean(axis=(1, 3))


res = {}
S = {n: series(f"r{n}") for n in (64, 128, 256)}
S = {n: s for n, s in S.items() if len(s) > 0}
nfin = max(S)
report = {}

# ---- conservation & symmetry
for n, s in S.items():
    I0 = integrals(s[0][1])
    drift = {"mass": 0, "xmom": 0, "energy": 0}
    sym = 0
    for t, d in s:
        I = integrals(d)
        drift["mass"] = max(drift["mass"], abs(I[0] - I0[0]) / I0[0])
        drift["xmom"] = max(drift["xmom"], abs(I[1]) / I0[0])
        drift["energy"] = max(drift["energy"], abs(I[2] - I0[2]) / abs(I0[2]))
        r = d[0]
        sym = max(sym, np.abs(r - r[:, ::-1]).max())
    report[f"N{n}"] = dict(t_final=s[-1][0], max_rel_mass_drift=drift["mass"],
                           max_abs_xmom_over_mass=drift["xmom"],
                           max_rel_total_energy_drift=drift["energy"],
                           max_mirror_asym_rho=sym)

# ---- grid convergence: L1 density error vs finest grid
conv = {}
times = [s[0] for s in S[nfin]]
for n in S:
    if n == nfin: continue
    f = nfin // n
    e = []
    for (t, d), (tf, df) in zip(S[n], S[nfin]):
        e.append(np.abs(d[0] - coarsen(df[0], f)).mean())
    conv[n] = np.array(e)
report["L1_rho_vs_finest"] = {str(n): {f"t={times[i]:.1f}": float(conv[n][i]) for i in (10, 20, 30) if i < len(conv[n])} for n in conv}

# ---- tips
tipdat = {n: np.array([[t, *tips(d)] for t, d in s]) for n, s in S.items()}
report["tips_t3"] = {str(n): dict(spike=float(v[-1, 1]), bubble=float(v[-1, 2])) for n, v in tipdat.items()}

# ---- linear growth rate (small-amplitude runs)
lin = {}
for n in (64, 128):
    s = series(f"lin{n}")
    if not s: continue
    A = []
    for t, d in s:
        r, u, v, p = prim(d)
        nx = d.shape[2]
        x = (np.arange(nx) + 0.5) / nx * LX
        vk = 2 * (v * np.cos(K * x)[None, :]).mean(axis=1)  # cos-mode amplitude per row
        A.append([t, np.abs(vk).max()])
    A = np.array(A)
    msk = (A[:, 0] >= 0.8) & (A[:, 0] <= 1.6)
    sl = np.polyfit(A[msk, 0], np.log(A[msk, 1]), 1)[0]
    lin[n] = (A, sl)
report["linear_growth"] = {str(n): float(v[1]) for n, v in lin.items()}
report["linear_theory_compressible"] = SIGMA_LIN
print(json.dumps(report, indent=1))
json.dump(report, open(f"{out}/diagnostics.json", "w"), indent=1)

# ---------------- figure
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SER = {64: "#eb6834", 128: "#1baf7a", 256: "#2a78d6"}
cmap = LinearSegmentedColormap.from_list("seqblue", ["#f4f8fd", "#9cc3ee", "#2a78d6", "#123f78", "#06152b"])
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(16, 10), facecolor="#fcfcfb")
gs = fig.add_gridspec(2, 10, height_ratios=[1.35, 1], hspace=0.32, wspace=0.55)
snap_idx = [0, 5, 10, 15, 20, 25, 30]
sf = S[nfin]
for k, i in enumerate(snap_idx):
    ax = fig.add_subplot(gs[0, k])
    t, d = sf[i]
    im = ax.imshow(d[0], origin="lower", extent=[0, LX, 0, LY], cmap=cmap, vmin=0.8, vmax=2.6,
                   interpolation="nearest")
    ax.set_title(f"t = {t:.1f}", fontsize=10)
    ax.set_xticks([0, 0.25]); ax.set_xticklabels(["0", ".25"])
    if k: ax.set_yticklabels([])
    else: ax.set_ylabel("y")
    ax.set_xlabel("x")
# final density at each resolution
ax = fig.add_subplot(gs[0, 7:10])
cols = []
for n in sorted(S):
    cols.append(S[n][-1][1][0])
ny = cols[-1].shape[0]
# show side by side, upsampled to finest grid
panel = np.concatenate([np.kron(c, np.ones((nfin // c.shape[1],) * 2)) for c in cols], axis=1)
ax.imshow(panel, origin="lower", extent=[0, LX * len(cols), 0, LY], cmap=cmap, vmin=0.8, vmax=2.6,
          interpolation="nearest")
ax.set_xticks([LX * (j + 0.5) for j in range(len(cols))])
ax.set_xticklabels([f"{n}×{4*n}" for n in sorted(S)])
ax.set_yticklabels([])
ax.set_title("t = 3.0, grid comparison", fontsize=10)
cb = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04)
cb.set_label("density ρ")
fig.text(0.5, 0.965, f"Rayleigh–Taylor instability, compressible Euler (γ=1.4, g=+1 ŷ) — density, "
         f"{nfin}×{4*nfin} grid, WENO5-Z + HLLC + SSP-RK3", ha="center", fontsize=13, color=INK)

def style(ax):
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)

# tips
ax = fig.add_subplot(gs[1, 0:3]); style(ax)
for n in sorted(S):
    v = tipdat[n]
    ax.plot(v[:, 0], v[:, 1] - 0.5, color=SER[n], lw=2, label=f"{n}×{4*n}")
    ax.plot(v[:, 0], 0.5 - v[:, 2], color=SER[n], lw=2, ls="--")
ax.set_xlabel("t"); ax.set_ylabel("tip distance from y=0.5")
ax.set_title("Interface tips (ρ=1.5 contour)\nsolid: heavy spike (+y), dashed: light bubble (−y)", fontsize=9)
ax.legend(frameon=False)

# linear growth
ax = fig.add_subplot(gs[1, 3:6]); style(ax)
for n, (A, sl) in lin.items():
    ax.semilogy(A[:, 0], A[:, 1], color=SER[n], lw=2, label=f"{n}×{4*n}: fit σ = {sl:.3f}")
if SIGMA_LIN and lin:
    A = lin[max(lin)][0]
    t0 = 1.2; a0 = np.interp(t0, A[:, 0], A[:, 1])
    tt = np.linspace(0.3, 2.0, 50)
    ax.semilogy(tt, a0 * np.exp(SIGMA_LIN * (tt - t0)), color=INK2, lw=1.2, ls=":",
                label=f"linear theory σ = {SIGMA_LIN:.3f}")
ax.axvspan(0.8, 1.6, color="#eeedea", zorder=0)
ax.set_xlabel("t"); ax.set_ylabel("max_y |v̂ₖ|  (k = 8π)")
ax.set_title("Linear regime: perturbation amp. 1e-4\nk=8π mode growth vs compressible linear theory", fontsize=9)
ax.legend(frameon=False, fontsize=8)

# convergence + conservation
ax = fig.add_subplot(gs[1, 7:10]); style(ax)
for n in sorted(conv):
    ax.semilogy(times, np.maximum(conv[n], 1e-16), color=SER[n], lw=2, label=f"{n}×{4*n} vs {nfin}×{4*nfin}")
ax.set_xlabel("t"); ax.set_ylabel("L1 density difference")
ax.set_title("Grid convergence: L1 ‖ρ_N − ρ_finest‖\n(finest grid averaged to coarse cells)", fontsize=9)
ax.legend(frameon=False, fontsize=8)
r = report[f"N{nfin}"]
ax.text(0.40, 0.04, f"{nfin}×{4*nfin}: max rel. mass drift {r['max_rel_mass_drift']:.1e}\n"
        f"max rel. (E+ρφ) drift {r['max_rel_total_energy_drift']:.1e}\n"
        f"mirror asymmetry max|Δρ| {r['max_mirror_asym_rho']:.1e}",
        transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
fig.savefig(f"{out}/summary_all.png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
print("saved")
