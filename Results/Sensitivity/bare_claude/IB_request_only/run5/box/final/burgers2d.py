"""2D inviscid Burgers u_t + (u^2/2)_x + (u^2/2)_y = 0 on [0,4pi]^2, periodic.
FD-WENO5 (Shu-Osher finite-difference on point values, component-wise, global Lax-Friedrichs flux splitting) + SSP-RK3.
Verified against the exact entropy solution (characteristics + stationary shock in co-moving frame)."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, json, sys
L = 4*np.pi; T = 2.0
def u0(x, y): return 1/3 + 2/3*np.sin((x+y)/2)

def exact(x, y, t):
    """u = v(eta,t), eta=(x+y)/2; v = 1/3 + w, w=(2/3)sin(xi0), eta' = eta - t/3 = xi0 + (2t/3) sin xi0.
    Entropy branch: xi0 in [0, xc] for eta' in [0,pi), odd symmetry; shock at eta'=pi (t>1.5)."""
    a = 2*t/3
    e = np.mod((x+y)/2 - t/3 + np.pi, 2*np.pi) - np.pi      # in [-pi, pi)
    s = np.sign(e); e = np.abs(e)
    hi = np.full_like(e, np.pi) if a <= 1 else np.full_like(e, np.arccos(-1/a))
    lo = np.zeros_like(e)
    for _ in range(60):
        m = 0.5*(lo+hi); f = m + a*np.sin(m) - e
        lo = np.where(f < 0, m, lo); hi = np.where(f >= 0, m, hi)
    return 1/3 + s*(2/3)*np.sin(0.5*(lo+hi))

def weno5(vm2, vm1, v0, vp1, vp2):  # reconstruct at i+1/2 from the left
    eps = 1e-6
    b0 = 13/12*(vm2-2*vm1+v0)**2 + 0.25*(vm2-4*vm1+3*v0)**2
    b1 = 13/12*(vm1-2*v0+vp1)**2 + 0.25*(vm1-vp1)**2
    b2 = 13/12*(v0-2*vp1+vp2)**2 + 0.25*(3*v0-4*vp1+vp2)**2
    a0, a1, a2 = 0.1/(eps+b0)**2, 0.6/(eps+b1)**2, 0.3/(eps+b2)**2
    q0 = (2*vm2-7*vm1+11*v0)/6; q1 = (-vm1+5*v0+2*vp1)/6; q2 = (2*v0+5*vp1-vp2)/6
    return (a0*q0+a1*q1+a2*q2)/(a0+a1+a2)

def dflux(u, ax, h):
    alpha = np.abs(u).max()
    f = 0.5*u*u; fp = 0.5*(f+alpha*u); fm = 0.5*(f-alpha*u)
    r = lambda k: np.roll(fp, -k, axis=ax); q = lambda k: np.roll(fm, -k, axis=ax)
    Fh = weno5(r(-2), r(-1), r(0), r(1), r(2)) + weno5(q(3), q(2), q(1), q(0), q(-1))  # flux at i+1/2
    return (Fh - np.roll(Fh, 1, axis=ax))/h

def rhs(u, h): return -dflux(u, 0, h) - dflux(u, 1, h)

def cell_avg_exact(N, t, ng=4):
    """cell averages of exact solution via ng x ng Gauss quadrature (used for IC and errors)."""
    h = L/N; g, w = np.polynomial.legendre.leggauss(ng)
    c = (np.arange(N)+0.5)*h; acc = np.zeros((N, N))
    for gi, wi in zip(g, w):
        for gj, wj in zip(g, w):
            X, Y = np.meshgrid(c+gi*h/2, c+gj*h/2, indexing="ij")
            acc += wi*wj*(exact(X, Y, t) if t > 0 else u0(X, Y))
    return acc/4

def grid(N):
    c = np.arange(N)*L/N
    return np.meshgrid(c, c, indexing="ij")

def ex_pts(N, t):
    X, Y = grid(N); return exact(X, Y, t) if t > 0 else u0(X, Y)

def solve(N, times, cfl=0.4):
    h = L/N; X, Y = grid(N); u = u0(X, Y); t = 0.0; out = {}
    for tt in sorted(times):
        while t < tt - 1e-14:
            dt = min(cfl*h/(2*np.abs(u).max()), tt - t)   # 2D CFL: dt*(|f'|+|g'|)/h
            u1 = u + dt*rhs(u, h)
            u2 = 0.75*u + 0.25*(u1 + dt*rhs(u1, h))
            u = u/3 + 2/3*(u2 + dt*rhs(u2, h)); t += dt
        out[tt] = u.copy()
    return out

if __name__ == "__main__":
    snaps = [0.0, 0.5, 1.0, 1.5, 1.75, 2.0]
    res = {}
    for N in [32, 64, 128, 256, 512]:
        sol = solve(N, [t for t in snaps if t > 0])
        res[N] = {}
        for t in [1.0, 2.0]:
            e = sol[t] - ex_pts(N, t)
            res[N][t] = dict(L1=np.abs(e).mean(), L2=np.sqrt((e**2).mean()), Linf=np.abs(e).max())
        print(N, res[N], "min/max u(T)", sol[2.0].min(), sol[2.0].max(), "mass err", sol[2.0].mean()-1/3, flush=True)
        if N == 512: fine = sol
    with open("errors.json", "w") as f: json.dump({str(k): {str(t): v for t, v in d.items()} for k, d in res.items()}, f, indent=1)
    Ns = sorted(res)
    for t in [1.0, 2.0]:
        print(f"t={t} observed orders (L1):", [np.log2(res[Ns[i]][t]['L1']/res[Ns[i+1]][t]['L1']).round(2) for i in range(len(Ns)-1)])
    # ---- summary figure
    N = 512; h = L/N; c = np.arange(N)*h
    fine[0.0] = ex_pts(N, 0.0)
    fig, axs = plt.subplots(3, 3, figsize=(16, 15))
    for k, t in enumerate(snaps):
        ax = axs.flat[k]
        im = ax.imshow(fine[t].T, origin="lower", extent=[0, L, 0, L], cmap="viridis", interpolation="nearest", vmin=-1/3, vmax=1)
        ax.set_title(f"u(x,y,t={t})  [WENO5, {N}²]"); ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axs.flat[6]   # diagonal slice vs exact
    xs = np.linspace(0, L, 2000)
    for t, col in zip([0.0, 1.0, 1.5, 2.0], ["C0", "C1", "C2", "C3"]):
        ax.plot(c, np.diag(fine[t]), ".", ms=2, color=col)
        ax.plot(xs, exact(xs, xs, t) if t > 0 else u0(xs, xs), "-", lw=0.8, color=col, label=f"t={t}")
    ax.set_title("Diagonal slice y=x: numerical (dots) vs exact (lines)"); ax.set_xlabel("x (=y)"); ax.legend()
    ax = axs.flat[7]
    for t, m in [(1.0, "o-"), (2.0, "s-")]:
        ax.loglog(Ns, [res[n][t]["L1"] for n in Ns], m, label=f"L1 error t={t}")
    ax.loglog(Ns, 3e2*np.array(Ns, float)**-5., "k--", lw=0.8, label="N^-5")
    ax.loglog(Ns, 2*np.array(Ns, float)**-1., "k:", lw=0.8, label="N^-1")
    ax.set_xlabel("N"); ax.set_ylabel("L1 error"); ax.set_title("Grid convergence vs exact solution"); ax.legend()
    ax = axs.flat[8]
    e = fine[2.0] - ex_pts(N, 2.0)
    im = ax.imshow(e.T, origin="lower", extent=[0, L, 0, L], cmap="RdBu_r", vmin=-0.05, vmax=0.05)
    ax.set_title("Error at t=2 (num − exact), 512²"); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("2D inviscid Burgers, u0=1/3+2/3 sin((x+y)/2), periodic [0,4π]²; shock forms at t*=1.5", fontsize=14)
    fig.tight_layout(); fig.savefig("summary_all.png", dpi=110)
