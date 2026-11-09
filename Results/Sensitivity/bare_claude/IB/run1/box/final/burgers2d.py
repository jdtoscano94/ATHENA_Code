"""2D inviscid Burgers u_t + (u^2/2)_x + (u^2/2)_y = 0 on [0,4pi]^2, periodic.
Finite volume, WENO5-JS reconstruction, global Lax-Friedrichs flux splitting, SSP-RK3."""
import numpy as np, sys, json
L = 4*np.pi
def u0(x, y): return 1/3 + 2/3*np.sin((x+y)/2)

def weno5_flux(f_p, f_m, ax):
    # interface flux at i+1/2 along axis ax (periodic); f_p upwinded from left, f_m from right
    r = lambda a, k: np.roll(a, -k, axis=ax)
    eps = 1e-6
    def recon(a, b, c, d, e):  # reconstruct at right face of c using a..e = v_{i-2..i+2}
        q0 = (2*a - 7*b + 11*c)/6; q1 = (-b + 5*c + 2*d)/6; q2 = (2*c + 5*d - e)/6
        b0 = 13/12*(a-2*b+c)**2 + 1/4*(a-4*b+3*c)**2
        b1 = 13/12*(b-2*c+d)**2 + 1/4*(b-d)**2
        b2 = 13/12*(c-2*d+e)**2 + 1/4*(3*c-4*d+e)**2
        a0 = 0.1/(eps+b0)**2; a1 = 0.6/(eps+b1)**2; a2 = 0.3/(eps+b2)**2
        return (a0*q0 + a1*q1 + a2*q2)/(a0+a1+a2)
    Fp = recon(r(f_p,-2), r(f_p,-1), f_p, r(f_p,1), r(f_p,2))
    Fm = recon(r(f_m,3), r(f_m,2), r(f_m,1), f_m, r(f_m,-1))
    return Fp + Fm

def rhs(u, dx):
    f = 0.5*u*u; a = np.max(np.abs(u))
    fp = 0.5*(f + a*u); fm = 0.5*(f - a*u)
    out = np.zeros_like(u)
    for ax in (0, 1):
        F = weno5_flux(fp, fm, ax)
        out -= (F - np.roll(F, 1, axis=ax))/dx
    return out

def solve(N, T, cfl=0.4, save_times=()):
    dx = L/N; xc = (np.arange(N)+0.5)*dx
    X, Y = np.meshgrid(xc, xc, indexing='ij')
    u = u0(X, Y); t = 0.0; snaps = {}
    save = sorted(save_times); k = 0
    mass0 = u.sum()*dx*dx
    while t < T - 1e-14:
        while k < len(save) and abs(t - save[k]) < 1e-12: snaps[save[k]] = u.copy(); k += 1
        dt = cfl*dx/(2*np.max(np.abs(u)))
        nxt = min([T] + [s for s in save if s > t + 1e-12])
        dt = min(dt, nxt - t)
        u1 = u + dt*rhs(u, dx)
        u2 = 0.75*u + 0.25*(u1 + dt*rhs(u1, dx))
        u = u/3 + 2/3*(u2 + dt*rhs(u2, dx)); t += dt
    while k < len(save): snaps[save[k]] = u.copy(); k += 1
    return X, Y, u, snaps, abs(u.sum()*dx*dx - mass0)

def exact(X, Y, t):
    """Exact entropy solution: theta=(x+y)/2, v=1/3+w, w solves w_t+w w_phi=0 in frame phi=theta-t/3,
    w0=(2/3)sin; characteristic phi = z + (2/3) t sin z with z chosen in the same half (0,pi)/(-pi,0)
    as phi (shock stationary at phi=pi in moving frame for t>1.5)."""
    phi = np.mod((X+Y)/2 - t/3 + np.pi, 2*np.pi) - np.pi   # in [-pi,pi)
    s = np.sign(phi); p = np.abs(phi)
    lo = np.zeros_like(p); hi = np.full_like(p, np.pi)
    for _ in range(60):  # bisection on g(z)=z+(2/3)t sin z - p on [0,pi]; unique root for p<pi
        m = 0.5*(lo+hi); g = m + 2/3*t*np.sin(m) - p
        lo = np.where(g < 0, m, lo); hi = np.where(g < 0, hi, m)
    z = s*0.5*(lo+hi)
    return 1/3 + 2/3*np.sin(z)
