"""2D inviscid Burgers u_t + (u^2/2)_x + (u^2/2)_y = 0 on [0,4pi]^2, periodic,
u0 = 1/3 + 2/3 sin((x+y)/2), t in [0,2].
Finite-difference WENO5 (Jiang-Shu, Shu-Osher form) + Lax-Friedrichs flux splitting + SSP-RK3.
Validated against the exact entropy solution (characteristics + shock)."""
import numpy as np, json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = 4 * np.pi
T = 2.0

# ---------------- exact entropy solution ----------------
def exact(x, y, t):
    """Reduce to 1D: s=(x+y)/2 -> u_t + u u_s = 0, u0 = 1/3 + 2/3 sin s.
    Frame z = s - t/3, v = u - 1/3: v_t + v v_z = 0, v0 = (2/3) sin z (odd, 2pi-periodic).
    Shock (t>1.5) sits at z = pi (mod 2pi) by symmetry. For z in (0,pi): the
    characteristic foot z0 is the unique root of F(z0)=z0+(2t/3)sin z0 - z in [0,z]."""
    s = 0.5 * (x + y)
    z = np.mod(s - t / 3.0 + np.pi, 2 * np.pi) - np.pi      # z in [-pi, pi)
    sg = np.sign(z); za = np.abs(z)
    a, b = np.zeros_like(za), za.copy()
    for _ in range(60):                                      # bisection
        m = 0.5 * (a + b)
        F = m + (2 * t / 3) * np.sin(m) - za
        a = np.where(F < 0, m, a); b = np.where(F < 0, b, m)
    z0 = 0.5 * (a + b)
    return 1.0 / 3.0 + sg * (2.0 / 3.0) * np.sin(z0)

# ---------------- WENO5 FV solver ----------------
def weno5_left(v, ax):
    """Left-biased reconstruction at interface i+1/2 from cell averages (periodic)."""
    vm2, vm1, v0 = np.roll(v, 2, ax), np.roll(v, 1, ax), v
    vp1, vp2 = np.roll(v, -1, ax), np.roll(v, -2, ax)
    eps = 1e-6
    b0 = 13/12*(vm2-2*vm1+v0)**2 + 0.25*(vm2-4*vm1+3*v0)**2
    b1 = 13/12*(vm1-2*v0+vp1)**2 + 0.25*(vm1-vp1)**2
    b2 = 13/12*(v0-2*vp1+vp2)**2 + 0.25*(3*v0-4*vp1+vp2)**2
    a0, a1, a2 = 0.1/(eps+b0)**2, 0.6/(eps+b1)**2, 0.3/(eps+b2)**2
    p0 = (2*vm2 - 7*vm1 + 11*v0) / 6
    p1 = (-vm1 + 5*v0 + 2*vp1) / 6
    p2 = (2*v0 + 5*vp1 - vp2) / 6
    return (a0*p0 + a1*p1 + a2*p2) / (a0 + a1 + a2)

def rhs(u, h):
    """Finite-difference (Shu-Osher) WENO5 with global Lax-Friedrichs splitting:
    f^± = (u^2/2 ± a u)/2, a = max|u|; numerical flux at i+1/2 = WENO-left(f^+) + WENO-right(f^-).
    5th order in multi-D (unlike dimension-by-dimension FV with midpoint face fluxes)."""
    a = np.abs(u).max()
    f = 0.5 * u**2
    fp, fm = 0.5 * (f + a * u), 0.5 * (f - a * u)
    r = np.zeros_like(u)
    for ax in (0, 1):
        Fp = weno5_left(fp, ax)                                       # at i+1/2
        Fm = np.roll(np.flip(weno5_left(np.flip(fm, ax), ax), ax), -1, ax)  # at i+1/2
        F = Fp + Fm
        r -= (F - np.roll(F, 1, ax)) / h
    return r

def solve(N, out_times, cfl=0.4):
    h = L / N
    xc = (np.arange(N) + 0.5) * h
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    u = exact(X, Y, 0.0)                          # point values at grid nodes
    t, snaps, k = 0.0, {}, 0
    out_times = sorted(out_times)
    if out_times[0] == 0.0: snaps[0.0] = u.copy(); k = 1
    while k < len(out_times):
        dt = cfl * h / (2 * np.abs(u).max())      # |f'|/dx + |g'|/dy
        dt = min(dt, out_times[k] - t)
        u1 = u + dt * rhs(u, h)
        u2 = 0.75*u + 0.25*(u1 + dt*rhs(u1, h))
        u = u/3 + 2/3*(u2 + dt*rhs(u2, h))
        t += dt
        if abs(t - out_times[k]) < 1e-12:
            snaps[out_times[k]] = u.copy(); k += 1
    return snaps

if __name__ == "__main__":
    times = [0.0, 0.5, 1.0, 1.5, 1.75, 2.0]
    Ns = [64, 128, 256, 512, 1024]
    results, sols = {}, {}
    for N in Ns:
        snaps = solve(N, times)
        sols[N] = snaps
        res = {}
        for t in times:
            xc = (np.arange(N) + 0.5) * (L / N)
            X, Y = np.meshgrid(xc, xc, indexing="ij")
            ue = exact(X, Y, t)
            e = snaps[t] - ue
            res[t] = dict(L1=float(np.mean(np.abs(e))), Linf=float(np.abs(e).max()),
                          mass_err=float(abs(snaps[t].mean() - 1/3)),
                          umin=float(snaps[t].min()), umax=float(snaps[t].max()))
        results[N] = res
        print(N, {t: f"L1={r['L1']:.2e} Linf={r['Linf']:.2e}" for t, r in res.items()}, flush=True)
    # observed orders
    lines = ["| t | " + " | ".join(f"L1 N={N}" for N in Ns) + " | order (last pair) |",
             "|---|" + "---|" * (len(Ns) + 1)]
    for t in times[1:]:
        e = [results[N][t]["L1"] for N in Ns]
        p = np.log2(e[-2] / e[-1])
        lines.append(f"| {t} | " + " | ".join(f"{v:.3e}" for v in e) + f" | {p:.2f} |")
    tab = "\n".join(lines); print(tab)
    open("/box/final/convergence.md", "w").write(tab + "\n")
    json.dump({str(N): {str(t): r for t, r in res.items()} for N, res in results.items()},
              open("/box/final/errors.json", "w"), indent=1)
    Nf = Ns[-1]
    np.savez_compressed("/box/final/solution_N1024.npz",
                        **{f"t{t}": sols[Nf][t] for t in times})

    # ---------------- summary figure ----------------
    h = L / Nf; xc = (np.arange(Nf) + 0.5) * h
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.9], hspace=0.38, wspace=0.3)
    for i, t in enumerate(times):
        ax = fig.add_subplot(gs[i // 3, (i % 3) * 2:(i % 3) * 2 + 2])
        im = ax.imshow(sols[Nf][t].T, origin="lower", extent=[0, L, 0, L],
                       cmap="viridis", vmin=-1/3, vmax=1)
        ax.set_title(f"u(x,y,t={t})  (N={Nf})" + ("  shock forms" if t == 1.5 else ""))
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.046)
    # diagonal cut: x=y line, s = x
    ax = fig.add_subplot(gs[2, 0:3])
    xs = np.linspace(0, L, 2000)
    for i, t in enumerate(times):
        c = plt.cm.plasma(i / len(times))
        ax.plot(xc, np.diag(sols[Nf][t]), ".", ms=2, color=c)
        ax.plot(xs, exact(xs, xs, t), "-", lw=0.8, color=c, label=f"t={t}")
    ax.set_title("Diagonal cut y=x: numerical (dots) vs exact (lines)")
    ax.set_xlabel("x (=y)"); ax.set_ylabel("u"); ax.legend(fontsize=8, ncol=3)
    ax = fig.add_subplot(gs[2, 3:6])
    for t in times[1:]:
        ax.loglog(Ns, [results[N][t]["L1"] for N in Ns], "o-", label=f"t={t}")
    ax.loglog(Ns, 1e-1 * (np.array(Ns) / 64.) ** -5, "k--", lw=0.8, label="N^-5")
    ax.loglog(Ns, 3e-3 * (np.array(Ns) / 64.) ** -1, "k:", lw=0.8, label="N^-1")
    ax.set_xlabel("N (cells per direction)"); ax.set_ylabel("L1 error vs exact")
    ax.set_title("Grid convergence"); ax.legend(fontsize=8, ncol=2)
    fig.suptitle("2D inviscid Burgers, u0 = 1/3 + 2/3 sin((x+y)/2), periodic on [0,4π]², FD-WENO5-LF-RK3")
    fig.savefig("/box/final/summary_all.png", dpi=120, bbox_inches="tight")
