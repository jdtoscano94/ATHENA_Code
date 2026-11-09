"""2D compressible Euler Rayleigh-Taylor instability solver (JAX, float64).

Finite volume: WENO5-JS reconstruction of primitive variables (rho,u,v,p),
HLLC Riemann solver, SSP-RK3 time integration, gravity source (0,0,rho g, rho v g).
BC: periodic in x, reflective (free-slip) walls in y with hydrostatic pressure
extrapolation into ghost cells.

usage: python rti.py NX NY [outfile]
"""
import sys, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

GAM = 1.4
G = 1.0
LX, LY, TEND = 0.25, 1.0, 3.0
NG = 3


def initial(nx, ny):
    dx, dy = LX / nx, LY / ny
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    rho = 2.0 + (1.0 - 2.0) * 0.5 * (1 + np.tanh(50 * (Y - 0.5)))
    p = np.where(Y < 0.5, 2 * Y + 1, Y + 1.5)
    c = np.sqrt(GAM * p / rho)
    u = np.zeros_like(X)
    v = -0.025 * c * np.cos(8 * np.pi * X) * np.sin(np.pi * Y) ** 6
    E = p / (GAM - 1) + 0.5 * rho * (u ** 2 + v ** 2)
    return np.stack([rho, rho * u, rho * v, E]), x, y


def prim(U):
    rho = U[0]
    u = U[1] / rho
    v = U[2] / rho
    p = (GAM - 1) * (U[3] - 0.5 * rho * (u * u + v * v))
    return jnp.stack([rho, u, v, p])


def weno5(vm2, vm1, v0, vp1, vp2):
    """Left-biased WENO5-JS value at interface i+1/2 from stencil i-2..i+2."""
    eps = 1e-6
    b0 = 13/12*(vm2-2*vm1+v0)**2 + 0.25*(vm2-4*vm1+3*v0)**2
    b1 = 13/12*(vm1-2*v0+vp1)**2 + 0.25*(vm1-vp1)**2
    b2 = 13/12*(v0-2*vp1+vp2)**2 + 0.25*(3*v0-4*vp1+vp2)**2
    a0 = 0.1/(eps+b0)**2
    a1 = 0.6/(eps+b1)**2
    a2 = 0.3/(eps+b2)**2
    q0 = (2*vm2 - 7*vm1 + 11*v0)/6
    q1 = (-vm1 + 5*v0 + 2*vp1)/6
    q2 = (2*v0 + 5*vp1 - vp2)/6
    return (a0*q0 + a1*q1 + a2*q2)/(a0+a1+a2)


def hllc(WL, WR, normal):
    """HLLC flux; normal=0 -> x, 1 -> y. W = (rho,u,v,p)."""
    rL, rR = WL[0], WR[0]
    pL, pR = WL[3], WR[3]
    if normal == 0:
        unL, unR, utL, utR = WL[1], WR[1], WL[2], WR[2]
    else:
        unL, unR, utL, utR = WL[2], WR[2], WL[1], WR[1]
    EL = pL/(GAM-1) + 0.5*rL*(unL**2+utL**2)
    ER = pR/(GAM-1) + 0.5*rR*(unR**2+utR**2)
    cL = jnp.sqrt(GAM*pL/rL); cR = jnp.sqrt(GAM*pR/rR)
    SL = jnp.minimum(unL-cL, unR-cR)
    SR = jnp.maximum(unL+cL, unR+cR)
    SM = (pR - pL + rL*unL*(SL-unL) - rR*unR*(SR-unR)) / (rL*(SL-unL) - rR*(SR-unR))

    def flux(r, un, ut, p, E):
        return jnp.stack([r*un, r*un*un+p, r*un*ut, (E+p)*un])

    def ustar(r, un, ut, p, E, S):
        f = r*(S-un)/(S-SM)
        return jnp.stack([f, f*SM, f*ut, f*(E/r + (SM-un)*(SM + p/(r*(S-un))))])

    FL = flux(rL, unL, utL, pL, EL); FR = flux(rR, unR, utR, pR, ER)
    UL = jnp.stack([rL, rL*unL, rL*utL, EL]); UR = jnp.stack([rR, rR*unR, rR*utR, ER])
    FsL = FL + SL*(ustar(rL, unL, utL, pL, EL, SL) - UL)
    FsR = FR + SR*(ustar(rR, unR, utR, pR, ER, SR) - UR)
    F = jnp.where(SL >= 0, FL, jnp.where(SM >= 0, FsL, jnp.where(SR > 0, FsR, FR)))
    if normal == 1:  # swap momentum components back to (x,y)
        F = jnp.stack([F[0], F[2], F[1], F[3]])
    return F


def make_solver(nx, ny):
    dx, dy = LX/nx, LY/ny

    def ghost(W):
        # periodic in x
        W = jnp.concatenate([W[:, -NG:, :], W, W[:, :NG, :]], axis=1)
        # reflective in y: mirror rho,u; flip v; hydrostatic pressure
        # bottom: ghost j=-k mirrors j=k-1, distance (2k-1)dy below
        bot = W[:, :, :NG][:, :, ::-1]      # ghost order -NG..-1 mirrors NG-1..0
        dist_b = (2*jnp.arange(NG, 0, -1) - 1) * dy  # ghost -k at y=-(k-.5)dy, mirror at (k-.5)dy
        bot = jnp.stack([bot[0], bot[1], -bot[2], bot[3] - bot[0]*G*dist_b[None, :]])
        top = W[:, :, -NG:][:, :, ::-1]     # ghost ny..ny+NG-1 mirrors ny-1..ny-NG
        dist_t = (2*jnp.arange(1, NG+1) - 1) * dy
        top = jnp.stack([top[0], top[1], -top[2], top[3] + top[0]*G*dist_t[None, :]])
        return jnp.concatenate([bot, W, top], axis=2)

    def rhs(U):
        W = ghost(prim(U))  # shape (4, nx+6, ny+6)
        # x-direction interfaces i-1/2 for i=0..nx  (nx+1 faces), interior in y
        Wx = W[:, :, NG:-NG]
        s = [Wx[:, k:k+nx+1] for k in range(6)]  # s[k] cell index (k-3)+face offset
        WL = weno5(s[0], s[1], s[2], s[3], s[4])
        WR = weno5(s[5], s[4], s[3], s[2], s[1])
        Fx = hllc(WL, WR, 0)
        Wy = W[:, NG:-NG, :]
        s = [Wy[:, :, k:k+ny+1] for k in range(6)]
        WL = weno5(s[0], s[1], s[2], s[3], s[4])
        WR = weno5(s[5], s[4], s[3], s[2], s[1])
        Fy = hllc(WL, WR, 1)
        # enforce exact zero mass/energy flux at walls
        Fy = Fy.at[jnp.array([0, 3]), :, 0].set(0.0).at[jnp.array([0, 3]), :, -1].set(0.0)
        dU = -(Fx[:, 1:] - Fx[:, :-1])/dx - (Fy[:, :, 1:] - Fy[:, :, :-1])/dy
        src = jnp.stack([jnp.zeros_like(U[0]), jnp.zeros_like(U[0]), U[0]*G, U[2]*G])
        return dU + src

    @jax.jit
    def step(U, dt):
        U1 = U + dt*rhs(U)
        U2 = 0.75*U + 0.25*(U1 + dt*rhs(U1))
        return U/3 + 2/3*(U2 + dt*rhs(U2))

    @jax.jit
    def maxspeed(U):
        W = prim(U)
        c = jnp.sqrt(GAM*W[3]/W[0])
        return jnp.max(jnp.maximum((jnp.abs(W[1])+c)/dx, (jnp.abs(W[2])+c)/dy))

    return step, maxspeed


def run(nx, ny, out, cfl=0.4, snaps=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)):
    U0, x, y = initial(nx, ny)
    dx, dy = LX/nx, LY/ny
    step, maxspeed = make_solver(nx, ny)
    U = jnp.asarray(U0)
    t, n = 0.0, 0
    Y = y[None, :]
    def diag(U):
        Un = np.asarray(U)
        W = np.asarray(prim(U))
        return dict(t=t, mass=Un[0].sum()*dx*dy,
                    etot=(Un[3] - Un[0]*G*Y).sum()*dx*dy,  # E + potential (-rho g y)
                    xmom=Un[1].sum()*dx*dy,
                    ke=(0.5*Un[0]*(W[1]**2+W[2]**2)).sum()*dx*dy,
                    pmin=W[3].min(), rmin=W[0].min(),
                    sym=np.abs(Un[0] - Un[0][::-1, :]).max())
    hist = [diag(U)]
    snapshots = {0.0: np.asarray(U0[0])}
    si = 1
    t0 = time.time()
    while t < TEND - 1e-12:
        dt = cfl/float(maxspeed(U))
        tnext = snaps[si]
        if t + dt > tnext:
            dt = tnext - t
        U = step(U, dt)
        t += dt; n += 1
        if abs(t - tnext) < 1e-12:
            t = tnext
            snapshots[tnext] = np.asarray(U[0])
            hist.append(diag(U))
            h = hist[-1]
            print(f"[{nx}x{ny}] t={t:.2f} n={n} mass={h['mass']:.12f} etot={h['etot']:.12f} "
                  f"pmin={h['pmin']:.4f} sym={h['sym']:.2e} wall={time.time()-t0:.0f}s", flush=True)
            si += 1
            if si >= len(snaps):
                break
        elif n % 50 == 0:
            hist.append(diag(U))
    Uf = np.asarray(U)
    keys = hist[0].keys()
    np.savez_compressed(out, x=x, y=y, U=Uf, snap_t=np.array(sorted(snapshots)),
                        snap_rho=np.stack([snapshots[k] for k in sorted(snapshots)]),
                        **{"h_"+k: np.array([h[k] for h in hist]) for k in keys}, nsteps=n)


if __name__ == "__main__":
    nx, ny = int(sys.argv[1]), int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else f"rti_{nx}x{ny}.npz"
    run(nx, ny, out)
