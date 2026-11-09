"""2D inviscid Burgers: u_t + (u^2/2)_x + (u^2/2)_y = 0 on [0,4pi]^2, periodic, t in [0,2].
IC u0 = 1/3 + 2/3 sin((x+y)/2).  Finite-difference (Shu-Osher) WENO5-JS on point values, global Lax-Friedrichs flux splitting, SSP-RK3.
(Dimension-by-dimension FD-WENO is genuinely 5th order in 2D, unlike dimension-by-dimension FV-WENO.)
Exact (entropy) solution from characteristics + shock (see exact_1d)."""
import numpy as np, sys, json, time

L = 4 * np.pi


def u0(x, y):
    return 1/3 + 2/3 * np.sin((x + y) / 2)


# ---------------- exact solution ----------------
def exact_1d(s, t):
    """Entropy solution of u_t + u u_s = 0, u(s,0)=1/3+2/3 sin s (s = (x+y)/2).
    In the frame z = s - t/3, w = u - 1/3 solves w_t + w w_z = 0, w0 = 2/3 sin z (odd), so a
    stationary shock sits at z = pi (mod 2pi) for t > 1.5. For z in [0,pi) the characteristic
    foot zeta solves zeta + (2t/3) sin zeta = z on the monotone branch [0, zeta_m]."""
    z = np.mod(s - t / 3 + np.pi, 2 * np.pi) - np.pi      # in [-pi, pi)
    sg = np.where(z < 0, -1.0, 1.0)
    za = np.abs(z)
    a = 2 * t / 3
    zm = np.pi if a <= 1 else np.arccos(-1 / a)
    lo = np.zeros_like(za); hi = np.minimum(za, zm)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        g = mid + a * np.sin(mid) - za
        lo = np.where(g < 0, mid, lo); hi = np.where(g < 0, hi, mid)
    zeta = 0.5 * (lo + hi)
    w = sg * (2 / 3) * np.sin(zeta)
    return 1/3 + w


def exact_nodes(N, t):
    xc = (np.arange(N) + 0.5) * (L / N)
    return exact_1d((xc[:, None] + xc[None, :]) / 2, t)


def exact_cellavg(N, t, nq=6):
    """Cell averages of exact solution on N x N grid via nq x nq Gauss-Legendre (it depends on x+y)."""
    dx = L / N
    gp, gw = np.polynomial.legendre.leggauss(nq)
    xc = (np.arange(N) + 0.5) * dx
    out = np.zeros((N, N))
    for i, (pi_, wi) in enumerate(zip(gp, gw)):
        for j, (pj, wj) in enumerate(zip(gp, gw)):
            X = xc[:, None] + 0.5 * dx * pi_
            Y = xc[None, :] + 0.5 * dx * pj
            out += 0.25 * wi * wj * exact_1d((X + Y) / 2, t)
    return out


# ---------------- WENO5 FV solver ----------------
def weno5_left(vm2, vm1, v0, vp1, vp2):
    """Reconstruct value at i+1/2 from the left using cells i-2..i+2."""
    eps = 1e-6
    b0 = 13/12*(vm2-2*vm1+v0)**2 + 0.25*(vm2-4*vm1+3*v0)**2
    b1 = 13/12*(vm1-2*v0+vp1)**2 + 0.25*(vm1-vp1)**2
    b2 = 13/12*(v0-2*vp1+vp2)**2 + 0.25*(3*v0-4*vp1+vp2)**2
    a0 = 0.1/(eps+b0)**2; a1 = 0.6/(eps+b1)**2; a2 = 0.3/(eps+b2)**2
    q0 = (2*vm2 - 7*vm1 + 11*v0)/6
    q1 = (-vm1 + 5*v0 + 2*vp1)/6
    q2 = (2*v0 + 5*vp1 - vp2)/6
    return (a0*q0 + a1*q1 + a2*q2)/(a0+a1+a2)


def dflux(u, axis, dx, alpha):
    """-(F_{i+1/2}-F_{i-1/2})/dx along axis, LF splitting f± = (u^2/2 ± alpha u)/2."""
    f = 0.5 * u * u
    fp = 0.5 * (f + alpha * u); fm = 0.5 * (f - alpha * u)
    r = lambda a, k: np.roll(a, -k, axis=axis)  # r(a,k)[i] = a[i+k]
    # F^+_{i+1/2}: left-biased, cells i-2..i+2 ; F^-_{i+1/2}: right-biased, mirror of cells i+3..i-1
    Fp = weno5_left(r(fp, -2), r(fp, -1), fp, r(fp, 1), r(fp, 2))
    Fm = weno5_left(r(fm, 3), r(fm, 2), r(fm, 1), fm, r(fm, -1))
    F = Fp + Fm
    return -(F - r(F, -1)) / dx


def rhs(u, dx, alpha):
    return dflux(u, 0, dx, alpha) + dflux(u, 1, dx, alpha)


def solve(N, T, cfl=0.4, snaps=()):
    dx = L / N
    xc = (np.arange(N) + 0.5) * dx
    u = u0(xc[:, None], xc[None, :])  # point values at nodes
    t = 0.0; out = {}
    snaps = sorted(snaps); k = 0
    while t < T - 1e-14:
        alpha = np.abs(u).max()
        dt = cfl * dx / (2 * alpha)
        tn = min(T, snaps[k]) if k < len(snaps) else T
        dt = min(dt, tn - t)
        u1 = u + dt * rhs(u, dx, alpha)
        u2 = 0.75 * u + 0.25 * (u1 + dt * rhs(u1, dx, alpha))
        u = u / 3 + 2 / 3 * (u2 + dt * rhs(u2, dx, alpha))
        t += dt
        if k < len(snaps) and abs(t - snaps[k]) < 1e-12:
            out[snaps[k]] = u.copy(); k += 1
    return u, out
