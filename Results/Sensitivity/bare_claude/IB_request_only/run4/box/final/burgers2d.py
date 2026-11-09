"""2D inviscid Burgers: u_t + (u^2/2)_x + (u^2/2)_y = 0 on [0,4pi]^2, periodic.
IC u0 = 1/3 + 2/3 sin((x+y)/2).
Finite-difference (Shu-Osher flux form, point values) WENO5 (Jiang-Shu) reconstruction + global Lax-Friedrichs flux splitting,
SSP-RK3 time stepping. Exact solution via characteristics + shock (equal-area by symmetry).
"""
import numpy as np, sys, json

L = 4 * np.pi
A = 2.0 / 3.0   # amplitude
C = 1.0 / 3.0   # mean


# ---------------- exact solution ----------------
def exact_s(s, t):
    """Exact entropy solution as function of s=(x+y)/2 for u_t + u u_s = 0."""
    # Galilean frame: v = u - C, sp = s - C t, v_t + v v_sp = 0, v0 = A sin(sp)
    sp = np.mod(s - C * t + np.pi, 2 * np.pi) - np.pi      # in [-pi, pi)
    sgn = np.sign(sp); q = np.abs(sp)                       # q in [0, pi]
    a = A * t
    xc = np.arccos(-1.0 / a) if a > 1 else np.pi           # end of monotone branch
    lo = np.zeros_like(q); hi = np.full_like(q, xc)
    for _ in range(80):                                     # bisection: xi + a sin xi = q
        mid = 0.5 * (lo + hi)
        f = mid + a * np.sin(mid) - q
        lo = np.where(f < 0, mid, lo); hi = np.where(f >= 0, mid, hi)
    xi = 0.5 * (lo + hi)
    v = sgn * A * np.sin(xi)
    return C + v


def exact_xy(x, y, t):
    return exact_s(0.5 * (x + y), t)


def exact_cell_avg(N, t, ng=5):
    """Cell averages of exact solution using ng-point Gauss in each direction."""
    h = L / N
    g, w = np.polynomial.legendre.leggauss(ng)
    xc = (np.arange(N) + 0.5) * h
    acc = np.zeros((N, N))
    for gi, wi in zip(g, w):
        for gj, wj in zip(g, w):
            X = xc[:, None] + 0.5 * h * gi
            Y = xc[None, :] + 0.5 * h * gj
            acc += wi * wj * exact_xy(X, Y, t)
    return acc / 4.0


# ---------------- WENO5 solver ----------------
def weno5_minus(vm2, vm1, v0, vp1, vp2):
    """Left-biased reconstruction at i+1/2 from stencil i-2..i+2."""
    eps = 1e-6
    b0 = 13/12*(vm2-2*vm1+v0)**2 + 0.25*(vm2-4*vm1+3*v0)**2
    b1 = 13/12*(vm1-2*v0+vp1)**2 + 0.25*(vm1-vp1)**2
    b2 = 13/12*(v0-2*vp1+vp2)**2 + 0.25*(3*v0-4*vp1+vp2)**2
    a0 = 0.1/(eps+b0)**2; a1 = 0.6/(eps+b1)**2; a2 = 0.3/(eps+b2)**2
    p0 = (2*vm2 - 7*vm1 + 11*v0)/6
    p1 = (-vm1 + 5*v0 + 2*vp1)/6
    p2 = (2*v0 + 5*vp1 - vp2)/6
    return (a0*p0 + a1*p1 + a2*p2)/(a0+a1+a2)


def flux_div(u, h, axis, alpha):
    f = 0.5 * u * u
    fp = 0.5 * (f + alpha * u); fm = 0.5 * (f - alpha * u)
    r = lambda a, k: np.roll(a, -k, axis=axis)   # r(a,k)[i] = a[i+k]
    # interface i+1/2
    Fp = weno5_minus(r(fp, -2), r(fp, -1), fp, r(fp, 1), r(fp, 2))
    Fm = weno5_minus(r(fm, 3), r(fm, 2), r(fm, 1), fm, r(fm, -1))
    F = Fp + Fm
    return (F - np.roll(F, 1, axis=axis)) / h


def rhs(u, h):
    alpha = np.max(np.abs(u))
    return -(flux_div(u, h, 0, alpha) + flux_div(u, h, 1, alpha))


def solve(N, T=2.0, cfl=0.4, snap_times=()):
    h = L / N
    xc = (np.arange(N) + 0.5) * h
    u = exact_xy(xc[:, None], xc[None, :], 0.0)
    t = 0.0; snaps = {}
    targets = sorted(set(list(snap_times) + [T]))
    k = 0
    while k < len(targets):
        if abs(t - targets[k]) < 1e-12:
            snaps[targets[k]] = u.copy(); k += 1; continue
        dt = cfl * h / (2 * np.max(np.abs(u)))
        dt = min(dt, targets[k] - t)
        u1 = u + dt * rhs(u, h)
        u2 = 0.75 * u + 0.25 * (u1 + dt * rhs(u1, h))
        u = u / 3 + 2 / 3 * (u2 + dt * rhs(u2, h))
        t += dt
    return snaps
