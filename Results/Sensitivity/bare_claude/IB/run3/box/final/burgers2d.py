"""2D inviscid Burgers u_t + (u^2/2)_x + (u^2/2)_y = 0 on [0,4pi]^2 periodic, t in [0,2].
Conservative finite-difference (Shu-Osher) WENO5 on point values at cell centres (Jiang-Shu) + global Lax-Friedrichs flux splitting + SSP-RK3.
Exact reference: Lax-Oleinik formula for the reduced 1D problem in s=x+y."""
import numpy as np, sys, json
L = 4*np.pi
u0f = lambda s: 1/3 + 2/3*np.sin(s/2)
U0f = lambda s: s/3 - 4/3*np.cos(s/2)          # antiderivative of u0

def exact(s, t):
    """Entropy solution of u_t + 2u u_s = 0 (flux f=u^2, f*(p)=p^2/4):
    u(s,t) = (s - xi*)/(2t), xi* = argmin_xi U0(xi) + (s-xi)^2/(4t)."""
    s = np.asarray(s, float); sh = s.shape; s = s.ravel()
    if t == 0: return u0f(s).reshape(sh)
    out = np.empty_like(s)
    for k0 in range(0, s.size, 2000):
        sk = s[k0:k0+2000][:, None]
        # characteristic feet lie in [s-2t, s+2t/3*... ] ; |2u|<=2 -> |s-xi|<=2t
        xi = sk + np.linspace(-2*t-0.1, 2*t+0.1, 4001)[None, :]
        G = U0f(xi) + (sk-xi)**2/(4*t)
        j = np.argmin(G, axis=1)
        x = xi[np.arange(len(j)), j]
        for _ in range(30):  # Newton refine on G'=u0(xi)-(s-xi)/(2t)
            g = u0f(x) - (sk[:, 0]-x)/(2*t)
            gp = np.cos(x/2)/3 + 1/(2*t)
            x = x - g/np.where(gp > 1e-12, gp, 1e-12)
        out[k0:k0+2000] = (sk[:, 0]-x)/(2*t)
    return out.reshape(sh)

def weno5_minus(vm2, vm1, v0, vp1, vp2):
    eps = 1e-6
    b0 = 13/12*(vm2-2*vm1+v0)**2 + 0.25*(vm2-4*vm1+3*v0)**2
    b1 = 13/12*(vm1-2*v0+vp1)**2 + 0.25*(vm1-vp1)**2
    b2 = 13/12*(v0-2*vp1+vp2)**2 + 0.25*(3*v0-4*vp1+vp2)**2
    a0 = 0.1/(eps+b0)**2; a1 = 0.6/(eps+b1)**2; a2 = 0.3/(eps+b2)**2
    q0 = (2*vm2-7*vm1+11*v0)/6; q1 = (-vm1+5*v0+2*vp1)/6; q2 = (2*v0+5*vp1-vp2)/6
    return (a0*q0+a1*q1+a2*q2)/(a0+a1+a2)

def dflux(u, ax, dx):
    a = np.max(np.abs(u))
    f = 0.5*u*u
    fp = 0.5*(f + a*u); fm = 0.5*(f - a*u)
    r = lambda v, k: np.roll(v, -k, axis=ax)
    # interface i+1/2
    Fp = weno5_minus(r(fp,-2), r(fp,-1), fp, r(fp,1), r(fp,2))
    Fm = weno5_minus(r(fm,3), r(fm,2), r(fm,1), fm, r(fm,-1))
    F = Fp + Fm
    return (F - np.roll(F, 1, axis=ax))/dx

def rhs(u, dx):
    return -(dflux(u, 0, dx) + dflux(u, 1, dx))

def cell_avg_exact(N, t, ng=4):
    """Cell averages of the exact solution (Gauss-Legendre in each direction, ng^2 pts)."""
    dx = L/N; xg, wg = np.polynomial.legendre.leggauss(ng)
    xc = (np.arange(N)+0.5)*dx
    # u depends on s=x+y only -> evaluate on 1D set of s values then average
    acc = np.zeros((N, N))
    for a, wa in zip(xg, wg):
        for b, wb in zip(xg, wg):
            X = xc[:, None] + a*dx/2; Y = xc[None, :] + b*dx/2
            sv = X + Y
            # exploit dependence on s only: unique values of (i+j)
            k = np.arange(2*N-1); s1 = (k+1)*dx + (a+b)*dx/2
            e = exact(s1, t)
            I, J = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
            acc += wa*wb/4*e[I+J]
    return acc

def cell_avg_ic(N):
    dx = L/N; xc = (np.arange(N)+0.5)*dx
    # exact average of sin((x+y)/2) over a cell
    fac = (np.sin(dx/4)/(dx/4))**2
    return 1/3 + 2/3*fac*np.sin((xc[:, None]+xc[None, :])/2)

def solve(N, times, cfl=0.4):
    dx = L/N; xc=(np.arange(N)+0.5)*dx; u = u0f(xc[:,None]+xc[None,:]); t = 0.0; snaps = {}
    times = sorted(times); k = 0
    if times[0] == 0: snaps[0.0] = u.copy(); k = 1
    while k < len(times):
        a = np.max(np.abs(u)); dt = cfl*dx/(2*a)
        if t + dt >= times[k] - 1e-14: dt = times[k] - t; hit = True
        else: hit = False
        u1 = u + dt*rhs(u, dx)
        u2 = 0.75*u + 0.25*(u1 + dt*rhs(u1, dx))
        u = u/3 + 2/3*(u2 + dt*rhs(u2, dx))
        t += dt
        if hit: snaps[times[k]] = u.copy(); k += 1
    return snaps

def point_exact(N, t):
    dx = L/N; k = np.arange(2*N-1); e = exact((k+1)*dx, t)
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    return e[I+J]
