import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, sys, os, json
from linear import growth
res = [n for n in (16, 32, 64) if os.path.exists(f"rti_N{n}.npz")]
D = {n: dict(np.load(f"rti_N{n}.npz")) for n in res}
sig_lin = growth(n=2000)
w = D[res[0]]["w"]; N = len(w)

def colavg(d, f):        # quadrature-weighted horizontal average of field f -> (nt, ny_nodes)
    nx = d["X"].shape[0] // N
    wx = np.tile(w, nx) / w.sum() / nx
    return np.einsum("tij,i->tj", f, wx)

def mode_amp(d):          # |cos(8 pi x)| Fourier coefficient of v, max over y
    X = d["X"]; nx = X.shape[0] // N
    wx = np.tile(w, nx) * (0.25 / nx / 2)
    c = np.cos(8*np.pi*X[:, 0])
    a = np.einsum("tij,i->tj", d["v"], wx * c) * 2 / 0.25
    return np.abs(a).max(axis=1)

def extents(d):          # bubble/spike tip from horizontal min/max density crossing 1.5
    y = d["Y"][0]; r = d["rho"]
    rmin, rmax = r.min(axis=1), r.max(axis=1)
    # bubble (light fluid penetrating down): lowest y where min density < 1.5
    yb = np.array([y[np.argmax(rm < 1.5)] for rm in rmin])
    # spike (heavy fluid penetrating up): highest y where max density > 1.5
    ys = np.array([y[len(y)-1-np.argmax(rm[::-1] > 1.5)] for rm in rmax])
    return yb, ys

summary = {"sigma_linear_incompressible": sig_lin}
for n in res:
    d = D[n]; t = d["t"]; A = mode_amp(d)
    m = (t >= 0.3) & (t <= 1.0)
    s = np.polyfit(t[m], np.log(A[m]), 1)[0]
    yb, ys = extents(d)
    summary[f"N{n}"] = dict(sigma_fit=s, mass_drift=float(np.abs(d["mass"]/d["mass"][0]-1).max()),
        energy_drift=float(np.abs(d["etot"]/d["etot"][0]-1).max()),
        bubble_y_t3=float(yb[-1]), spike_y_t3=float(ys[-1]),
        rho_min=float(d["rho"].min()), rho_max=float(d["rho"].max()))
# resolution difference: horizontally averaged density at final time vs finest
fine = res[-1]; yf = D[fine]["Y"][0]; pf = colavg(D[fine], D[fine]["rho"])
for n in res[:-1]:
    d = D[n]; p = colavg(d, d["rho"])
    for k, tt in enumerate(d["t"]):
        pass
    for tt in (1.0, 2.0, 3.0):
        k = np.argmin(abs(d["t"]-tt))
        e = np.sqrt(np.mean((np.interp(yf, d["Y"][0], p[k]) - pf[k])**2))
        summary[f"N{n}"][f"L2diff_avg_rho_vs_N{fine}_t{tt:g}"] = float(e)
print(json.dumps(summary, indent=1))
json.dump(summary, open("summary.json", "w"), indent=1)

# ---------------- figure ----------------
cols = {16: "#9aa5b1", 32: "#2a7ab9", 64: "#c4432b"}
d = D[fine]; t = d["t"]
snaps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
fig = plt.figure(figsize=(17, 11.5))
gs = fig.add_gridspec(2, 1, height_ratios=[1.55, 1], hspace=0.22)
gt = gs[0].subgridspec(1, len(snaps)+1, width_ratios=[1]*len(snaps)+[0.08], wspace=0.08)
for i, ts in enumerate(snaps):
    ax = fig.add_subplot(gt[i]); k = np.argmin(abs(t-ts))
    # tile 2 periods for readability? keep one period, true aspect
    pc = ax.pcolormesh(d["X"], d["Y"], d["rho"][k], cmap="cividis", vmin=0.95, vmax=2.05, shading="gouraud")
    ax.set_aspect("equal"); ax.set_title(f"t = {t[k]:.1f}", fontsize=11)
    ax.set_xticks([0, 0.25]); 
    if i: ax.set_yticks([])
    else: ax.set_ylabel("y")
    ax.set_xlabel("x")
cb = fig.colorbar(pc, cax=fig.add_subplot(gt[-1])); cb.set_label(r"density $\rho$")
fig.text(0.5, 0.965, f"Rayleigh–Taylor instability, compressible Euler (γ=1.4, g=+1 in y) — "
         f"DGSEM P3, {fine}×{4*fine} elements ({4*fine}×{16*fine} nodes)", ha="center", fontsize=13)
gb = gs[1].subgridspec(1, 4, wspace=0.32)
# (a) linear growth
ax = fig.add_subplot(gb[0])
for n in res:
    ax.semilogy(D[n]["t"], mode_amp(D[n]), color=cols[n], lw=2, label=f"{n}×{4*n}")
A = mode_amp(d); k0 = np.argmin(abs(t-0.3))
tt = np.linspace(0, 1.6, 50)
ax.semilogy(tt, A[k0]*np.exp(sig_lin*(tt-t[k0])), "k--", lw=1.2, label=f"linear theory σ={sig_lin:.3f}")
ax.set_xlim(0, 3); ax.set_xlabel("t"); ax.set_ylabel(r"max$_y$ |$\hat v_{k=8\pi}$|")
ax.set_title("(a) mode amplitude & linear growth"); ax.legend(fontsize=8, frameon=False)
# (b) bubble/spike
ax = fig.add_subplot(gb[1])
for n in res:
    yb, ys = extents(D[n])
    ax.plot(D[n]["t"], ys, color=cols[n], lw=2, label=f"{n}×{4*n}")
    ax.plot(D[n]["t"], yb, color=cols[n], lw=2, ls=":")
ax.axhline(0.5, color="0.7", lw=0.8)
ax.text(0.1, 0.92, "spike tip (heavy ↑, solid)", transform=ax.transAxes, fontsize=8)
ax.text(0.1, 0.05, "bubble tip (light ↓, dotted)", transform=ax.transAxes, fontsize=8)
ax.set_xlabel("t"); ax.set_ylabel("y  (ρ = 1.5 front)"); ax.set_title("(b) mixing-zone fronts")
ax.legend(fontsize=8, frameon=False, loc="center left")
# (c) averaged density profile
ax = fig.add_subplot(gb[2])
for n in res:
    p = colavg(D[n], D[n]["rho"])
    ax.plot(p[-1], D[n]["Y"][0], color=cols[n], lw=2, label=f"{n}×{4*n}, t=3")
p0 = colavg(d, d["rho"])[0]
ax.plot(p0, d["Y"][0], color="k", lw=1, ls="--", label="t=0")
ax.set_xlabel(r"$\langle\rho\rangle_x$"); ax.set_ylabel("y"); ax.set_title("(c) horizontally averaged density")
ax.legend(fontsize=8, frameon=False)
# (d) conservation + energy partition
ax = fig.add_subplot(gb[3])
for n in res:
    ax.semilogy(D[n]["t"][1:], np.abs(D[n]["etot"][1:]/D[n]["etot"][0]-1), color=cols[n], lw=2, label=f"{n}×{4*n}: total energy")
    ax.semilogy(D[n]["t"][1:], np.abs(D[n]["mass"][1:]/D[n]["mass"][0]-1)+1e-17, color=cols[n], lw=1, ls=":")
ax.text(0.03, 0.04, "dotted: mass (round-off)", transform=ax.transAxes, fontsize=8)
ax.set_xlabel("t"); ax.set_ylabel("relative drift"); ax.set_title(r"(d) conservation of $\int\rho$ and $\int(E-\rho g y)$")
ax.legend(fontsize=8, frameon=False, loc="center right")
fig.savefig("summary_all.png", dpi=130, bbox_inches="tight")
