"""2D inviscid Burgers  u_t + (u^2/2)_x + (u^2/2)_y = 0 on [0,4pi]^2, periodic, t in [0,2].
IC: u0 = 1/3 + 2/3 sin((x+y)/2).
Numerics (default scheme='fd'): Shu-Osher finite-difference WENO5-JS on point values with global
Lax-Friedrichs flux splitting, SSP-RK3. (scheme='fv': dimension-by-dimension finite-volume WENO5 +
Godunov flux; kept for comparison - it is only 2nd order in 2D for nonlinear fluxes.)
Reference: exact entropy solution (characteristics + shock); point values ('fd') or cell averages ('fv').
"""
import numpy as np, sys, json

L = 4 * np.pi
A, C = 2.0 / 3.0, 1.0 / 3.0          # amplitude, mean


def exact_point(x, y, t):
    """Exact entropy solution. With eta=(x+y)/2: u_t + u u_eta = 0, u0 = C + A sin(eta).
    Moving frame xi = eta - C t: v_t + v v_xi = 0, v0 = A sin(xi); shock (t>1.5) stays at xi = pi."""
    eta = 0.5 * (x + y)
    xi = np.mod(eta - C * t + np.pi, 2 * np.pi) - np.pi      # in [-pi, pi)
    s = np.sign(xi); s[s == 0] = 1.0
    z = np.abs(xi)                                           # solve z0 + A t sin z0 = z on [0, pi] (unique root)
    lo, hi = np.zeros_like(z), np.full_like(z, np.pi)
    for _ in range(60):                                      # bisection (robust; F(lo)<=0<=F(hi))
        mid = 0.5 * (lo + hi)
        F = mid + A * t * np.sin(mid) - z
        lo = np.where(F < 0, mid, lo); hi = np.where(F >= 0, mid, hi)
    z0 = 0.5 * (lo + hi)
    for _ in range(3):                                       # Newton polish
        d = 1 + A * t * np.cos(z0)
        ok = np.abs(d) > 1e-3
        z0 = np.where(ok, z0 - (z0 + A * t * np.sin(z0) - z) / np.where(ok, d, 1), z0)
    v = A * np.sin(s * z0)
    v = np.where(np.abs(np.abs(xi) - np.pi) < 1e-14, 0.0, v)
    return C + v


def exact_cellavg(N, t, ng=5):
    """Cell averages of the exact solution. u depends only on x+y, so a 1D average along the
    diagonal is not exact for squares; use ng x ng Gauss-Legendre quadrature per cell."""
    dx = L / N
    g, w = np.polynomial.legendre.leggauss(ng)
    xc = (np.arange(N) + 0.5) * dx
    out = np.zeros((N, N))
    for i in range(ng):
        for j in range(ng):
            X = xc[:, None] + 0.5 * dx * g[i]
            Y = xc[None, :] + 0.5 * dx * g[j]
            out += 0.25 * w[i] * w[j] * exact_point(np.broadcast_to(X, (N, N)).copy(),
                                                    np.broadcast_to(Y, (N, N)).copy(), t)
    return out


def weno5_minus(vm2, vm1, v0, vp1, vp2, eps=1e-6):
    """Left-biased WENO5 value at interface i+1/2."""
    b0 = 13/12*(vm2-2*vm1+v0)**2 + 0.25*(vm2-4*vm1+3*v0)**2
    b1 = 13/12*(vm1-2*v0+vp1)**2 + 0.25*(vm1-vp1)**2
    b2 = 13/12*(v0-2*vp1+vp2)**2 + 0.25*(3*v0-4*vp1+vp2)**2
    a0, a1, a2 = 0.1/(eps+b0)**2, 0.6/(eps+b1)**2, 0.3/(eps+b2)**2
    q0 = (2*vm2 - 7*vm1 + 11*v0) / 6
    q1 = (-vm1 + 5*v0 + 2*vp1) / 6
    q2 = (2*v0 + 5*vp1 - vp2) / 6
    return (a0*q0 + a1*q1 + a2*q2) / (a0 + a1 + a2)


def godunov(uL, uR):
    """Exact Riemann flux for f=u^2/2."""
    fL, fR = 0.5*uL**2, 0.5*uR**2
    shock = np.maximum(fL, fR)                               # uL > uR
    rare = np.where((uL <= 0) & (uR >= 0), 0.0, np.minimum(fL, fR))
    return np.where(uL > uR, shock, rare)


def dflux(u, axis, dx):
    r = lambda k: np.roll(u, -k, axis=axis)
    uL = weno5_minus(r(-2), r(-1), r(0), r(1), r(2))         # left state at i+1/2
    uR = weno5_minus(r(3), r(2), r(1), r(0), r(-1))          # right state at i+1/2 (mirrored)
    F = godunov(uL, uR)
    return (F - np.roll(F, 1, axis=axis)) / dx


def dflux_fd(u, axis, dx, alpha):
    r = lambda w, k: np.roll(w, -k, axis=axis)
    fp = 0.5 * (0.5*u**2 + alpha*u)                          # f+ (upwind to the right)
    fm = 0.5 * (0.5*u**2 - alpha*u)                          # f- (upwind to the left)
    Fp = weno5_minus(r(fp,-2), r(fp,-1), fp, r(fp,1), r(fp,2))
    Fm = weno5_minus(r(fm,3), r(fm,2), r(fm,1), fm, r(fm,-1))
    F = Fp + Fm                                              # numerical flux at i+1/2
    return (F - np.roll(F, 1, axis=axis)) / dx


SCHEME = {"name": "fd"}


def rhs(u, dx):
    if SCHEME["name"] == "fd":
        a = 1.1 * np.max(np.abs(u))
        return -(dflux_fd(u, 0, dx, a) + dflux_fd(u, 1, dx, a))
    return -(dflux(u, 0, dx) + dflux(u, 1, dx))


def exact_grid(N, t):
    """Reference matching the scheme's unknowns."""
    if SCHEME["name"] == "fd":
        xc = (np.arange(N) + 0.5) * L / N
        X, Y = np.meshgrid(xc, xc, indexing="ij")
        return exact_point(X, Y, t)
    return exact_cellavg(N, t)


def solve(N, T=2.0, cfl=0.4, save_times=()):
    dx = L / N
    u = exact_grid(N, 0.0)
    t, snaps = 0.0, {}
    save_times = sorted(save_times)
    k = 0
    while t < T - 1e-14:
        dt = cfl * dx / (2 * np.max(np.abs(u)))              # 2D: |f'|+|g'| <= 2 max|u|
        if k < len(save_times):
            dt = min(dt, save_times[k] - t) if save_times[k] > t + 1e-14 else dt
        dt = min(dt, T - t)
        u1 = u + dt * rhs(u, dx)
        u2 = 0.75*u + 0.25*(u1 + dt*rhs(u1, dx))
        u = u/3 + 2/3*(u2 + dt*rhs(u2, dx))
        t += dt
        while k < len(save_times) and abs(t - save_times[k]) < 1e-12:
            snaps[save_times[k]] = u.copy(); k += 1
    return u, snaps
