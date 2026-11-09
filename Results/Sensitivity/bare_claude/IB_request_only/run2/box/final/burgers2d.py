"""2D inviscid Burgers: u_t + (u^2/2)_x + (u^2/2)_y = 0 on [0,4pi]^2 periodic.
FV WENO5 (component-wise, dimension-by-dimension) + Godunov flux + SSP-RK3.
Exact reference: 1D reduction in s=(x+y)/2, Hopf-Lax entropy solution."""
import numpy as np, sys, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

L = 4*np.pi
def u0_cellavg(N):
    # exact cell averages of 1/3 + 2/3 sin((x+y)/2)
    dx = L/N; xe = np.arange(N+1)*dx
    # avg of sin((x+y)/2) over cell = [product formula]
    # ∫∫ sin(a+b) with a=x/2,b=y/2: use complex exp
    ex = (np.exp(1j*xe[1:]/2)-np.exp(1j*xe[:-1]/2))/(1j/2)/dx
    return 1/3 + 2/3*np.imag(np.outer(ex, ex))   # index [i(x), j(y)]

def weno5_lr(v, axis):
    """Return left/right states at interface i+1/2 (between cell i and i+1)."""
    r = lambda k: np.roll(v, -k, axis=axis)
    vm2, vm1, v0, vp1, vp2, vp3 = r(-2), r(-1), v, r(1), r(2), r(3)
    eps = 1e-6
    def rec(a, b, c, d, e):  # reconstruct at right face of c using a..e
        b0 = 13/12*(a-2*b+c)**2 + 0.25*(a-4*b+3*c)**2
        b1 = 13/12*(b-2*c+d)**2 + 0.25*(b-d)**2
        b2 = 13/12*(c-2*d+e)**2 + 0.25*(3*c-4*d+e)**2
        a0, a1, a2 = 0.1/(eps+b0)**2, 0.6/(eps+b1)**2, 0.3/(eps+b2)**2
        s = a0+a1+a2
        return (a0*(2*a-7*b+11*c) + a1*(-b+5*c+2*d) + a2*(2*c+5*d-e))/(6*s)
    uL = rec(vm2, vm1, v0, vp1, vp2)
    uR = rec(vp3, vp2, vp1, v0, vm1)
    return uL, uR

def godunov(uL, uR):
    f = lambda u: 0.5*u*u
    return np.maximum(f(np.maximum(uL, 0)), f(np.minimum(uR, 0)))

def rhs(u, dx):
    """Finite-difference (Shu-Osher) WENO5 with global Lax-Friedrichs flux splitting:
    high order in multi-D (unlike dimension-by-dimension FV)."""
    out = np.zeros_like(u); a = np.abs(u).max(); f = 0.5*u*u
    fp, fm = 0.5*(f + a*u), 0.5*(f - a*u)
    for ax in (0, 1):
        Fp, _ = weno5_lr(fp, ax)      # left-biased reconstruction of f+
        _, Fm = weno5_lr(fm, ax)      # right-biased reconstruction of f-
        F = Fp + Fm
        out -= (F - np.roll(F, 1, axis=ax))/dx
    return out

def solve(N, times, cfl=0.4):
    dx = L/N; xc = (np.arange(N)+0.5)*dx
    u = 1/3 + 2/3*np.sin((xc[:, None]+xc[None, :])/2); t = 0.0; snaps = {}
    times = sorted(times)
    for T in times:
        while t < T - 1e-14:
            dt = min(cfl*dx/(2*np.abs(u).max()), T - t)
            u1 = u + dt*rhs(u, dx)
            u2 = 0.75*u + 0.25*(u1 + dt*rhs(u1, dx))
            u = u/3 + 2/3*(u2 + dt*rhs(u2, dx)); t += dt
        snaps[T] = u.copy()
    return snaps

# ---------- exact entropy solution (1D in s=(x+y)/2) ----------
def exact_1d(s, t):
    """v_t + v v_s = 0, v0 = 1/3 + 2/3 sin s.  Galilean shift w=v-1/3, z=s-t/3,
    then Hopf-Lax: z0* = argmin (z-z0)^2/(2t) - (2/3)cos z0; w = (z-z0*)/t."""
    s = np.asarray(s, float)
    if t == 0: return 1/3 + 2/3*np.sin(s)
    z = s - t/3
    z = (z + np.pi) % (2*np.pi) - np.pi                      # in [-pi,pi)
    zg = np.linspace(-3*np.pi, 3*np.pi, 6001)
    J = (z[:, None]-zg[None, :])**2/(2*t) - 2/3*np.cos(zg[None, :])
    z0 = zg[np.argmin(J, axis=1)]
    for _ in range(50):  # Newton refine on dJ/dz0 = -(z-z0)/t + 2/3 sin z0
        g = -(z-z0)/t + 2/3*np.sin(z0); h = 1/t + 2/3*np.cos(z0)
        z0 = z0 - g/np.where(np.abs(h) > 1e-12, h, 1e-12)
    return 1/3 + (z - z0)/t

def exact_point(N, t):
    xc = (np.arange(N)+0.5)*L/N; k = np.arange(2*N-1)
    v = exact_1d((2*xc[0] + k*L/N)/2, t)
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing='ij'); return v[I+J]

def exact_cellavg(N, t, q=8):
    """cell averages of exact solution (Gauss quadrature in the diagonal variable)."""
    dx = L/N
    gx, gw = np.polynomial.legendre.leggauss(q)
    # u depends on xi=x+y only; average over a square cell = average over xi with
    # triangular weight on [xc+yc-dx, xc+yc+dx]. Use tensor Gauss for simplicity.
    xc = (np.arange(N)+0.5)*dx
    # values depend on i+j only -> compute for k=i+j
    k = np.arange(2*N-1); xi_c = (k+1)*dx  # xc_i+yc_j = (i+j+1)dx
    a = gx*dx/2
    XI = xi_c[:, None, None] + a[None, :, None] + a[None, None, :]
    U = exact_1d(XI.ravel()/2, t).reshape(XI.shape)
    avg = np.einsum('kab,a,b->k', U, gw, gw)/4
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    return avg[I+J]

if __name__ == "__main__":
    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    res = {}
    for N in [64, 128, 256, 512]:
        sn = solve(N, times)
        dx = L/N; res[N] = {}
        for T in times:
            ue = exact_point(N, T); e = sn[T]-ue
            res[N][T] = dict(L1=float(np.mean(np.abs(e))), L2=float(np.sqrt(np.mean(e**2))),
                             Linf=float(np.abs(e).max()))
        print(N, {T: f"{res[N][T]['L1']:.3e}" for T in times}, flush=True)
        if N == 512: snaps = sn
    json.dump({str(k): {str(t): v for t, v in d.items()} for k, d in res.items()},
              open("/box/final/errors.json", "w"), indent=1)
    np.savez_compressed("/box/final/solution_N512.npz", **{f"t{T}": snaps[T] for T in times})
    # ---- plot ----
    N = 512; dx = L/N; xc = (np.arange(N)+0.5)*dx
    fig, ax = plt.subplots(3, len(times), figsize=(4*len(times), 11.5), constrained_layout=True)
    for c, T in enumerate(times):
        im = ax[0, c].imshow(snaps[T].T, origin='lower', extent=[0, L, 0, L], cmap='RdBu_r',
                             vmin=-1/3, vmax=1)
        ax[0, c].set_title(f"u(x,y,t={T})  N={N}"); ax[0, c].set_xlabel("x"); ax[0, c].set_ylabel("y")
        ue = exact_point(N, T)
        im2 = ax[1, c].imshow(np.clip(np.abs(snaps[T]-ue),1e-12,None).T, origin='lower', extent=[0, L, 0, L],
                              cmap='magma', norm=matplotlib.colors.LogNorm(1e-12, 1))
        ax[1, c].set_title(f"|u - u_exact|, t={T}")
        # diagonal-ish cut: y = x (cells i=j), xi = 2x
        d = np.diag(snaps[T]); xf = np.linspace(0, L, 2000)
        ax[2, c].plot(xf, exact_1d(xf, T), 'k-', lw=1.2, label='exact (entropy)')
        ax[2, c].plot(xc, d, 'r.', ms=2.5, label='FD-WENO5 N=512')
        ax[2, c].set_title(f"cut along y=x, t={T}"); ax[2, c].set_xlabel("x (=y)")
        ax[2, c].grid(alpha=.3)
    ax[2, 0].legend()
    fig.colorbar(im, ax=ax[0, :], shrink=0.8, label="u")
    fig.colorbar(im2, ax=ax[1, :], shrink=0.8, label="abs error")
    fig.suptitle("2D inviscid Burgers  u_t+u u_x+u u_y=0,  u0=1/3+2/3 sin((x+y)/2), periodic [0,4π]²;"
                 "  shock forms at t*=1.5", fontsize=13)
    fig.savefig("/box/final/summary_all.png", dpi=110)
