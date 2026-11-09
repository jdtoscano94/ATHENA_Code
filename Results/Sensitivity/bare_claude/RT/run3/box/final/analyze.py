"""Post-processing: linear-theory growth rate, convergence, conservation, summary figure."""
import numpy as np, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eig
from linear_compressible import growth as comp_growth

GAM, G, LX, LY = 1.4, 1.0, 0.25, 1.0
D = "/box/final/data"
RES = [n for n in (32, 64, 128, 256) if os.path.exists(f"{D}/rt_{n}.npz")]
COL = {32: "#2a78d6", 64: "#eb6834", 128: "#1baf7a", 256: "#eda100"}
LS = {32: ":", 64: "-.", 128: "--", 256: "-"}
INK, INK2 = "#0b0b0b", "#52514e"


def linear_growth(k=8 * np.pi, N=2000):
    """Incompressible linear RT eigenproblem for the tanh profile with walls:
    (rho w')' - k^2 rho w = (g k^2 / n^2) rho_y w,  w(0)=w(1)=0.  Returns max n."""
    y = np.linspace(0, 1, N + 1); h = y[1] - y[0]
    rho = lambda s: 2.0 - 0.5 * (1 + np.tanh(50 * (s - 0.5)))
    yi = y[1:-1]; rp = rho(yi + h / 2); rm = rho(yi - h / 2)
    A = (np.diag(-(rp + rm)) + np.diag(rp[:-1], 1) + np.diag(rm[1:], -1)) / h ** 2 - np.diag(k ** 2 * rho(yi))
    ry = -25 * (1 - np.tanh(50 * (yi - 0.5)) ** 2)
    B = np.diag(G * k ** 2 * ry)
    lam = eig(A, B, right=False).real  # lambda = 1/n^2
    lam = lam[np.isfinite(lam) & (lam > 0)]
    return 1 / np.sqrt(lam.min())


def fit_rate(d, t0=0.5, t1=1.0):
    t = d["h_t"]; m = (t >= t0) & (t <= t1)
    return 0.5 * np.polyfit(t[m], np.log(d["h_KE"][m]), 1)[0]  # KE ~ exp(2 n t)


def main():
    runs = {n: np.load(f"{D}/rt_{n}.npz") for n in RES}
    stat = {n: np.load(f"{D}/static_{n}.npz") for n in (32, 64, 128) if os.path.exists(f"{D}/static_{n}.npz")}
    n_lin = linear_growth()
    n_sharp = np.sqrt((2 - 1) / (2 + 1) * 8 * np.pi * G)
    lines = []
    P = lambda s: (print(s), lines.append(s))
    n_comp = comp_growth(4000)
    lin = {n: np.load(f"{D}/lin_{n}.npz") for n in (64, 128) if os.path.exists(f"{D}/lin_{n}.npz")}
    sh = {n: np.load(f"{D}/statichydro_{n}.npz") for n in (64, 128) if os.path.exists(f"{D}/statichydro_{n}.npz")}
    P(f"Linear theory (compressible, exact hydrostatic tanh background, walls): n = {n_comp:.5f}")
    P(f"Linear theory (incompressible, tanh profile, walls): n = {n_lin:.4f};  sharp interface sqrt(Akg) = {n_sharp:.4f}")
    nf = max(RES); ref = runs[nf]
    for n in RES:
        d = runs[n]
        dm = abs(d["h_mass"][-1] - d["h_mass"][0]) / d["h_mass"][0]
        dE = np.abs(d["h_Etot"] - d["h_Etot"][0]).max() / d["h_Etot"][0]
        P(f"{n}x{4*n}: fitted growth n(t in [0.5,1.0]) = {fit_rate(d):.4f}; |dM/M|={dm:.2e}; max|dE/E|={dE:.2e}; "
          f"max|momx|={np.abs(d['h_momx']).max():.1e}; mirror-sym err={d['h_sym'].max():.1e}; "
          f"t=3: spike y={d['h_spike'][-1]:.4f}, bubble y={d['h_bubble'][-1]:.4f}, "
          f"min p={d['h_pmin'].min():.3f}, rho in [{d['h_rmin'].min():.4f},{d['h_rmax'].max():.4f}]")
    # convergence of horizontally averaged density profiles vs finest grid (coarsen finest)
    for tt in ("1.00", "2.00", "3.00"):
        rf = ref[f"snap_{tt}"][0]
        for n in RES[:-1]:
            r = runs[n][f"snap_{tt}"][0]; c = nf // n
            rfc = rf.reshape(n, c, 4 * n, c).mean(axis=(1, 3))
            P(f"t={tt}: L1 density diff {n} vs coarsened {nf}: {np.abs(r - rfc).mean():.4e}; "
              f"mean-profile L1 diff: {np.abs(r.mean(0) - rfc.mean(0)).mean():.4e}")
    for n, d in lin.items():
        t = d["h_t"]; m = (t >= 2) & (t <= 3)
        nf_ = np.polyfit(t[m], np.log(d["h_vhat"][m]), 1)[0]
        P(f"small-amplitude (amp x1e-4) {n}x{4*n}: fitted n of k=8pi v-mode, t in [2,3] = {nf_:.5f}; rel. err vs compressible theory = {abs(nf_-n_comp)/n_comp:.2e}")
    for n, s in sh.items():
        P(f"static, exactly hydrostatic p (verification) {n}x{4*n}: max|v| = {s['h_vmax'].max():.2e}")
    for n, s in stat.items():
        P(f"static (unperturbed) {n}x{4*n}: max|v| over t in [0,3] = {s['h_vmax'].max():.2e}, final max|v| = {s['h_vmax'][-1]:.2e}")
    open("/box/final/results.txt", "w").write("\n".join(lines) + "\n")

    # ---------------- figure ----------------
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(17, 11), facecolor="#fcfcfb")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 1], hspace=0.25)
    gt = gs[0].subgridspec(1, 8, width_ratios=[1] * 7 + [0.08], wspace=0.12)
    times = ["0.00", "0.50", "1.00", "1.50", "2.00", "2.50", "3.00"]
    X, Y = ref["X"], ref["Y"]
    for i, tt in enumerate(times):
        ax = fig.add_subplot(gt[i])
        im = ax.imshow(ref[f"snap_{tt}"][0].T, origin="lower", extent=[0, LX, 0, LY], cmap="cividis",
                       vmin=0.75, vmax=2.75, interpolation="nearest")
        ax.set_title(f"t = {float(tt):.1f}", color=INK); ax.set_xticks([0, 0.125, 0.25]); ax.set_xticklabels(["0", "", "0.25"])
        if i: ax.set_yticklabels([])
        else: ax.set_ylabel("y")
        ax.set_xlabel("x")
    cax = fig.add_subplot(gt[7]); cb = fig.colorbar(im, cax=cax); cb.set_label("density ρ")
    fig.text(0.5, 0.965, f"Rayleigh–Taylor instability, compressible Euler (γ=1.4, g=+1 ŷ), density at {nf}×{4*nf}  "
             f"(MUSCL-HLLC-RK3)", ha="center", fontsize=12, color=INK)

    gb = gs[1].subgridspec(1, 5, wspace=0.36)
    # (a) bubble / spike positions
    ax = fig.add_subplot(gb[0])
    for n in RES:
        d = runs[n]
        ax.plot(d["h_t"], d["h_spike"] - 0.5, color=COL[n], ls=LS[n], lw=2, label=f"{n}×{4*n}")
        ax.plot(d["h_t"], 0.5 - d["h_bubble"], color=COL[n], ls=LS[n], lw=1.2)
    ax.text(2.05, 0.33, "spike", color=INK2); ax.text(2.3, 0.19, "bubble", color=INK2)
    ax.set_xlabel("t"); ax.set_ylabel("distance from y=0.5 (⟨f⟩ₓ=5%/95%)")
    ax.set_title("(a) spike / bubble penetration", color=INK); ax.legend(frameon=False, title="grid")
    # (b) kinetic energy + linear theory
    ax = fig.add_subplot(gb[1])
    for n in RES:
        ax.semilogy(runs[n]["h_t"], runs[n]["h_KE"], color=COL[n], ls=LS[n], lw=2, label=f"{n}×{4*n}")
    ax.set_xlabel("t"); ax.set_ylabel("kinetic energy ∫½ρ|u|² dA")
    ax.set_title("(b) kinetic energy", color=INK); ax.legend(frameon=False)
    # (e) linear verification
    axe = fig.add_subplot(gb[4])
    for n, d in lin.items():
        axe.semilogy(d["h_t"], d["h_vhat"], color=COL[n], ls=LS[n], lw=2, label=f"{n}×{4*n}, amp×1e-4")
    if lin:
        d = lin[max(lin)]; a3 = np.interp(3.0, d["h_t"], d["h_vhat"]); tl = np.linspace(0.8, 3, 20)
        axe.semilogy(tl, 0.5 * a3 * np.exp(n_comp * (tl - 3)), color=INK, lw=1, alpha=0.6,
                     label=f"compressible theory n={n_comp:.3f}")
        axe.semilogy(tl, 0.25 * a3 * np.exp(n_lin * (tl - 3)), color=INK2, lw=1, ls=":",
                     label=f"incompressible n={n_lin:.3f}")
    axe.set_xlabel("t"); axe.set_ylabel("max_y |v̂(k=8π)|"); axe.legend(frameon=False, fontsize=7.5)
    axe.set_title("(e) linear growth (amp×1e-4)", color=INK)
    # (c) mean profile at t=3
    ax = fig.add_subplot(gb[2])
    for n in RES:
        r = runs[n]["snap_3.00"][0]; y = runs[n]["Y"][0]
        ax.plot(r.mean(0), y, color=COL[n], ls=LS[n], lw=2, label=f"{n}×{4*n}")
    r0 = ref["snap_0.00"][0].mean(0); ax.plot(r0, ref["Y"][0], color=INK2, lw=1, ls=":", label="t = 0")
    ax.set_xlabel("horizontally averaged ρ"); ax.set_ylabel("y"); ax.set_title("(c) ⟨ρ⟩ₓ(y) at t = 3", color=INK)
    ax.legend(frameon=False, loc="center right", fontsize=7.5)
    # (d) conservation
    ax = fig.add_subplot(gb[3])
    for n in RES:
        d = runs[n]
        ax.semilogy(d["h_t"], np.abs(d["h_Etot"] - d["h_Etot"][0]) / d["h_Etot"][0] + 1e-17, color=COL[n], ls=LS[n],
                    lw=2, label=f"{n}×{4*n}")
    ax.set_xlabel("t"); ax.set_ylabel("|ΔE_tot|/E_tot,  E_tot = ∫(E − ρgy)dA")
    ax.set_title("(d) energy conservation", color=INK); ax.legend(frameon=False)
    fig.savefig("/box/final/summary_all.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
