"""2D inviscid Burgers  u_t + (u^2/2)_x + (u^2/2)_y = 0  on [0,4pi]^2, periodic.
Conservative finite difference (Shu-Osher) on nodes x_i=(i+1/2)h, WENO5-JS reconstruction + local Lax-Friedrichs (Rusanov) flux splitting,
SSP-RK3 time stepping. Exact entropy solution via characteristics + symmetry for validation."""
import numpy as np, sys, json

L = 4*np.pi
def u0(x, y, t=0): return 1/3 + 2/3*np.sin((x+y)/2)

# ---------- exact entropy solution ----------
def exact(x, y, t, iters=60):
    # u depends on eta=(x+y)/2: u_t + u u_eta = 0, period 2pi in eta.
    # moving frame zeta = eta - t/3, v = u - 1/3: v_t + v v_zeta = 0, v0 = 2/3 sin(zeta)
    # shock (t>1.5) stationary at zeta = pi by odd symmetry.
    z = np.mod((x+y)/2 - t/3 + np.pi, 2*np.pi) - np.pi      # in [-pi,pi)
    s = np.sign(z); a = np.abs(z)
    lo = np.zeros_like(a); hi = np.full_like(a, np.pi)
    for _ in range(iters):                                    # bisection for xi0: xi0 + t*(2/3)sin(xi0) = a
        m = 0.5*(lo+hi); f = m + t*(2/3)*np.sin(m) - a
        lo = np.where(f < 0, m, lo); hi = np.where(f < 0, hi, m)
    xi0 = 0.5*(lo+hi)
    return 1/3 + s*(2/3)*np.sin(xi0)

# ---------- WENO5 ----------
def weno5_left(vm2, vm1, v0, vp1, vp2, eps=1e-6):
    # reconstruct value at i+1/2 from the left
    b0 = 13/12*(vm2-2*vm1+v0)**2 + 0.25*(vm2-4*vm1+3*v0)**2
    b1 = 13/12*(vm1-2*v0+vp1)**2 + 0.25*(vm1-vp1)**2
    b2 = 13/12*(v0-2*vp1+vp2)**2 + 0.25*(3*v0-4*vp1+vp2)**2
    a0 = 0.1/(eps+b0)**2; a1 = 0.6/(eps+b1)**2; a2 = 0.3/(eps+b2)**2
    p0 = (2*vm2 - 7*vm1 + 11*v0)/6
    p1 = (-vm1 + 5*v0 + 2*vp1)/6
    p2 = (2*v0 + 5*vp1 - vp2)/6
    return (a0*p0 + a1*p1 + a2*p2)/(a0+a1+a2)

def dflux(u, axis, h):
    f = 0.5*u*u
    r = lambda a, k: np.roll(a, -k, axis=axis)   # r(a,k)[i] = a[i+k]
    # local alpha at interface i+1/2 (max over stencil)
    alpha = np.maximum.reduce([np.abs(r(u, k)) for k in range(-2, 4)])
    Fp = weno5_left(*[0.5*(r(f, k) + alpha*r(u, k)) for k in (-2, -1, 0, 1, 2)])
    Fm = weno5_left(*[0.5*(r(f, k) - alpha*r(u, k)) for k in (3, 2, 1, 0, -1)])
    F = Fp + Fm                                   # numerical flux at i+1/2
    return (F - np.roll(F, 1, axis=axis))/h

def rhs(u, h): return -(dflux(u, 0, h) + dflux(u, 1, h))

def solve(N, T, cfl=0.4, snaps=()):
    h = L/N
    xc = (np.arange(N)+0.5)*h
    u = u0(xc[:, None], xc[None, :])
    t = 0.0; out = {}; snaps = sorted(snaps)
    while t < T - 1e-14:
        dt = cfl*h/(2*np.abs(u).max())             # 2D: |f_x'|+|f_y'| = 2|u|
        nxt = min([s for s in snaps if s > t + 1e-14] + [T])
        dt = min(dt, nxt - t)
        u1 = u + dt*rhs(u, h)
        u2 = 0.75*u + 0.25*(u1 + dt*rhs(u1, h))
        u = u/3 + 2/3*(u2 + dt*rhs(u2, h))
        t += dt
        for s in snaps:
            if abs(t - s) < 1e-12: out[s] = u.copy()
    out[T] = u.copy()
    return out
