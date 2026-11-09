import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, glob, json, os
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import cumulative_trapezoid as ct
W = os.path.dirname(os.path.abspath(__file__))
cmap = LinearSegmentedColormap.from_list("blue", ["#f7f9fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
COL = {16: "#eb6834", 32: "#1baf7a", 64: "#2a78d6"}
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False})
runs = {}
for N in (16, 32, 64):
    f = f"{W}/rti_N{N}_p3.npz"
    if os.path.exists(f): runs[N] = np.load(f)
Nf = max(runs); D = runs[Nf]
sig_lin = float(np.load(f"{W}/sigma_lin.npy"))
def extents(rho):          # spike tip (heavy fluid furthest into +y) / bubble tip (light fluid furthest into -y)
    ny = rho.shape[1]; y = (np.arange(ny) + 0.5) / ny
    heavy = (rho > 1.5).any(axis=0); light = (rho < 1.5).any(axis=0)
    return y[heavy].max(), y[light].min()
metrics = {}
for N, d in runs.items():
    t, M, E, P, A, Ek, Ps = d["hist"].T
    S = d["snaps"]; st = d["snap_t"]
    ex = np.array([extents(S[0, :, :, i]) for i in range(len(st))])
    metrics[N] = dict(mass_rel_drift=float(abs(M - M[0]).max() / M[0]),
                      energy_budget_rel_err=float(abs(E - E[0] - ct(Ps, t, initial=0)).max() / E[0]),
                      symmetry_err_rho_t3=float(abs(S[0] - S[0, ::-1]).max()),
                      spike_tip=dict(zip(map(float, st), map(float, ex[:, 0]))),
                      bubble_tip=dict(zip(map(float, st), map(float, ex[:, 1]))),
                      rho_min_t3=float(S[0, ..., -1].min()), rho_max_t3=float(S[0, ..., -1].max()))
lin = {}
for f in sorted(glob.glob(f"{W}/lin_N*_p3.npz")):
    d = np.load(f); t, *_ , A, Ek, Ps = d["hist"].T
    m = (t > 0.8) & (t < 1.5); s = np.polyfit(t[m], np.log(abs(A[m])), 1)[0]
    lin[int(d["NEL"])] = dict(t=t, A=A, sigma=float(s))
json.dump(dict(metrics=metrics, sigma_linear_theory=sig_lin,
               sigma_measured_small_amp={k: v["sigma"] for k, v in lin.items()}),
          open(f"{W}/metrics.json", "w"), indent=1)

fig = plt.figure(figsize=(16, 11), facecolor="white")
gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 1], hspace=0.28)
top = gs[0].subgridspec(1, 8, width_ratios=[1]*7 + [0.08], wspace=0.08)
st = D["snap_t"]; show = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
for j, ts in enumerate(show):
    i = int(np.argmin(abs(st - ts))); ax = fig.add_subplot(top[j])
    im = ax.imshow(D["snaps"][0, :, :, i].T, origin="lower", extent=[0, .25, 0, 1], cmap=cmap, vmin=0.8, vmax=2.4,
                   aspect="equal", interpolation="nearest")
    ax.set_title(f"t = {st[i]:.1f}", color=INK); ax.set_xticks([0, .25]); ax.set_xticklabels(["0", "0.25"])
    if j: ax.set_yticklabels([])
    else: ax.set_ylabel("y")
    ax.set_xlabel("x")
cb = fig.colorbar(im, cax=fig.add_subplot(top[7])); cb.set_label("density ρ")
fig.text(0.5, 0.965, f"Compressible Rayleigh–Taylor instability (γ=1.4, g=+1 ŷ, Atwood 1/3) — density, DGSEM p=3, "
         f"{Nf}×{4*Nf} elements ({4*Nf}×{16*Nf} nodes)", ha="center", fontsize=13, color=INK)
bot = gs[1].subgridspec(1, 4, wspace=0.35, width_ratios=[1, 1, 1, 1.25])
# (a) t=3 resolution comparison
sub = bot[0].subgridspec(1, len(runs), wspace=0.05)
for j, (N, d) in enumerate(sorted(runs.items())):
    ax = fig.add_subplot(sub[j]); s = d["snaps"]
    ax.imshow(s[0, :, :, -1].T, origin="lower", extent=[0, .25, 0, 1], cmap=cmap, vmin=0.8, vmax=2.4, aspect="equal")
    ax.set_title(f"{N}×{4*N}\nt=3", fontsize=8, color=INK); ax.set_xticks([]); ax.set_yticks([] if j else [0, .5, 1])
# (b) interface extents
ax = fig.add_subplot(bot[1])
for N, d in sorted(runs.items()):
    s = d["snap_t"]; sp = [metrics[N]["spike_tip"][float(x)] for x in s]; bu = [metrics[N]["bubble_tip"][float(x)] for x in s]
    ax.plot(s, sp, "-o", color=COL[N], lw=2, ms=4, label=f"{N}×{4*N}"); ax.plot(s, bu, "--s", color=COL[N], lw=2, ms=4)
ax.set_xlabel("t"); ax.set_ylabel("y of ρ=1.5 front"); ax.set_title("Mixing-layer extent\n(solid: heavy spike tip, dashed: light bubble tip)", color=INK)
ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.25)
# (c) linear growth
ax = fig.add_subplot(bot[2])
for N, v in sorted(lin.items()):
    ax.semilogy(v["t"], abs(v["A"]), color=COL.get(N, "k"), lw=2, label=f"ε=1e-6, {N}×{4*N}: σ={v['sigma']:.3f}")
if lin:
    v = lin[max(lin)]; t0 = 1.2; a0 = np.interp(t0, v["t"], abs(v["A"]))
    tt = np.linspace(0.3, 1.5, 10); ax.semilogy(tt, a0*np.exp(sig_lin*(tt - t0)), ":", color=INK, lw=1.5, label=f"normal-mode theory σ={sig_lin:.3f}")
ax.set_xlabel("t"); ax.set_ylabel("|v̂(k=8π)|"); ax.set_title("Linear regime check (tiny perturbation)", color=INK)
ax.legend(frameon=False, fontsize=7); ax.grid(alpha=.25)
# (d) energetics
ax = fig.add_subplot(bot[3])
for N, d in sorted(runs.items()):
    t, M, E, P, A, Ek, Ps = d["hist"].T
    ax.semilogy(t, Ek, color=COL[N], lw=2, label=f"{N}×{4*N}")
ax.set_xlabel("t"); ax.set_ylabel("kinetic energy ∫½ρ|u|²"); ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8)
m = metrics[Nf]
ax.set_title("Kinetic energy", color=INK)
ax.text(0.98, 0.04, f"finest grid:\nmass drift {m['mass_rel_drift']:.1e}\nenergy-budget err {m['energy_budget_rel_err']:.1e}\n"
        f"x-symmetry err {m['symmetry_err_rho_t3']:.1e}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2)
fig.savefig(f"{W}/summary_all.png", dpi=150, bbox_inches="tight")
print(json.dumps({k: {kk: vv for kk, vv in v.items() if 'tip' not in kk} for k, v in metrics.items()}, indent=1))
print({k: v["sigma"] for k, v in lin.items()}, sig_lin)
