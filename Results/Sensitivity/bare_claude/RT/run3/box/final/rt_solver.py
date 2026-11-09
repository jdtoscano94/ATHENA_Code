"""2D compressible Euler Rayleigh-Taylor instability solver (JAX, CPU, float64).

Finite volume: MUSCL (MC limiter, primitive variables) + HLLC flux + SSP-RK3.
Gravity g=+1 in +y as a cell-centred source: d(rho v)/dt += rho g, dE/dt += rho v g.
Periodic in x, reflective (free-slip) walls in y with hydrostatically extrapolated ghost pressure.
"""
import os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

GAM = 1.4
G = 1.0
LX, LY = 0.25, 1.0
NG = 2


def prim2cons(r, u, v, p):
    return jnp.stack([r, r * u, r * v, p / (GAM - 1) + 0.5 * r * (u * u + v * v)])


def cons2prim(U):
    r = U[0]; u = U[1] / r; v = U[2] / r
    p = (GAM - 1) * (U[3] - 0.5 * r * (u * u + v * v))
    return r, u, v, p


def initial(nx, ny, perturb=True, amp=1.0, exact_hydro=False):
    dx, dy = LX / nx, LY / ny
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    r = 2.0 + (1.0 - 2.0) * 0.5 * (1 + np.tanh(50 * (Y - 0.5)))
    p = np.where(Y < 0.5, 2 * Y + 1, Y + 1.5)
    if exact_hydro:  # verification only: p from integrating dp/dy = rho(y) g exactly, p(0.5)=2
        from scipy.integrate import quad
        rf = lambda s: 2.0 - 0.5 * (1 + np.tanh(50 * (s - 0.5)))
        py = np.array([2.0 + G * quad(rf, 0.5, yy, limit=200)[0] for yy in y])
        p = np.broadcast_to(py, Y.shape).copy()
    c = np.sqrt(GAM * p / r)
    v = -0.025 * amp * c * np.cos(8 * np.pi * X) * np.sin(np.pi * Y) ** 6 if perturb else 0 * X
    u = 0 * X
    return X, Y, prim2cons(jnp.array(r), jnp.array(u), jnp.array(v), jnp.array(p))


def mc_slope(qm, q0, qp):
    a = q0 - qm; b = qp - q0
    s = 0.5 * (jnp.sign(a) + jnp.sign(b))
    return s * jnp.minimum(jnp.minimum(2 * jnp.abs(a), 2 * jnp.abs(b)), 0.5 * jnp.abs(a + b))


def hllc(rL, unL, utL, pL, rR, unR, utR, pR):
    """HLLC flux, normal velocity un, tangential ut. Returns (F_r, F_mn, F_mt, F_E)."""
    cL = jnp.sqrt(GAM * pL / rL); cR = jnp.sqrt(GAM * pR / rR)
    EL = pL / (GAM - 1) + 0.5 * rL * (unL ** 2 + utL ** 2)
    ER = pR / (GAM - 1) + 0.5 * rR * (unR ** 2 + utR ** 2)
    SL = jnp.minimum(unL - cL, unR - cR)
    SR = jnp.maximum(unL + cL, unR + cR)
    Ss = (pR - pL + rL * unL * (SL - unL) - rR * unR * (SR - unR)) / (rL * (SL - unL) - rR * (SR - unR))
    FL = (rL * unL, rL * unL ** 2 + pL, rL * unL * utL, (EL + pL) * unL)
    FR = (rR * unR, rR * unR ** 2 + pR, rR * unR * utR, (ER + pR) * unR)
    UL = (rL, rL * unL, rL * utL, EL)
    UR = (rR, rR * unR, rR * utR, ER)

    def star(r, un, ut, p, E, S):
        f = r * (S - un) / (S - Ss)
        return (f, f * Ss, f * ut, f * (E / r + (Ss - un) * (Ss + p / (r * (S - un)))))
    UsL = star(rL, unL, utL, pL, EL, SL)
    UsR = star(rR, unR, utR, pR, ER, SR)
    out = []
    for k in range(4):
        fsl = FL[k] + SL * (UsL[k] - UL[k])
        fsr = FR[k] + SR * (UsR[k] - UR[k])
        f = jnp.where(SL >= 0, FL[k], jnp.where(Ss >= 0, fsl, jnp.where(SR > 0, fsr, FR[k])))
        out.append(f)
    return out


def make_rhs(nx, ny):
    dx, dy = LX / nx, LY / ny

    def ghost(W):
        r, u, v, p = W
        # periodic x
        def px(q):
            return jnp.concatenate([q[-NG:], q, q[:NG]], axis=0)
        r, u, v, p = px(r), px(u), px(v), px(p)
        # reflective y, hydrostatic pressure extrapolation (dp/dy = rho g)
        rb = r[:, 1::-1]; ub = u[:, 1::-1]; vb = -v[:, 1::-1]
        dist_b = jnp.array([3.0, 1.0]) * dy  # ghost -2 mirrors 1 (dist 3dy), ghost -1 mirrors 0
        pb = p[:, 1::-1] - rb * G * dist_b[None, :]
        rt = r[:, :-3:-1]; ut = u[:, :-3:-1]; vt = -v[:, :-3:-1]
        dist_t = jnp.array([1.0, 3.0]) * dy
        pt = p[:, :-3:-1] + rt * G * dist_t[None, :]
        cat = lambda a, q, b: jnp.concatenate([a, q, b], axis=1)
        return cat(rb, r, rt), cat(ub, u, ut), cat(vb, v, vt), cat(pb, p, pt)

    def faces(Q, axis):
        # Q padded with NG ghosts along axis; returns left/right states at the n+1 interior faces
        n = Q.shape[axis] - 2 * NG
        sl = lambda a, b: jax.lax.slice_in_dim(Q, a, Q.shape[axis] - b, axis=axis)
        qm, q0, qp = sl(0, 2), sl(1, 1), sl(2, 0)  # cells 1..N+2 (n+2 cells)
        d = mc_slope(qm, q0, qp)
        qL = jax.lax.slice_in_dim(q0 + 0.5 * d, 0, n + 1, axis=axis)
        qR = jax.lax.slice_in_dim(q0 - 0.5 * d, 1, n + 2, axis=axis)
        return qL, qR

    def rhs(U):
        W = cons2prim(U)
        r, u, v, p = ghost(W)
        # x-direction: use only interior y cells
        rx, ux, vx, pxx = [q[:, NG:-NG] for q in (r, u, v, p)]
        L = [faces(q, 0) for q in (rx, ux, vx, pxx)]
        Fr, Fmu, Fmv, FE = hllc(L[0][0], L[1][0], L[2][0], L[3][0], L[0][1], L[1][1], L[2][1], L[3][1])
        Fx = jnp.stack([Fr, Fmu, Fmv, FE])
        ry, uy, vy, py = [q[NG:-NG, :] for q in (r, u, v, p)]
        L = [faces(q, 1) for q in (ry, uy, vy, py)]
        Gr, Gmv, Gmu, GE = hllc(L[0][0], L[2][0], L[1][0], L[3][0], L[0][1], L[2][1], L[1][1], L[3][1])
        # exact wall flux (no penetration): only pressure in normal momentum
        pw_b = L[3][1][:, 0]; pw_t = L[3][0][:, -1]
        z = jnp.zeros_like(pw_b)
        Gr = Gr.at[:, 0].set(z).at[:, -1].set(z)
        GE = GE.at[:, 0].set(z).at[:, -1].set(z)
        Gmu = Gmu.at[:, 0].set(z).at[:, -1].set(z)
        Gmv = Gmv.at[:, 0].set(pw_b).at[:, -1].set(pw_t)
        Gy = jnp.stack([Gr, Gmu, Gmv, GE])
        dU = -(Fx[:, 1:, :] - Fx[:, :-1, :]) / dx - (Gy[:, :, 1:] - Gy[:, :, :-1]) / dy
        rho = U[0]
        S = jnp.stack([0 * rho, 0 * rho, rho * G, U[2] * G])
        return dU + S

    @jax.jit
    def step(U, dt):
        U1 = U + dt * rhs(U)
        U2 = 0.75 * U + 0.25 * (U1 + dt * rhs(U1))
        return U / 3 + 2.0 / 3 * (U2 + dt * rhs(U2))

    @jax.jit
    def maxspeed(U):
        r, u, v, p = cons2prim(U)
        c = jnp.sqrt(GAM * p / r)
        return jnp.max((jnp.abs(u) + c) / dx + (jnp.abs(v) + c) / dy)

    return step, maxspeed


def diagnostics(U, Y, dx, dy):
    U = np.asarray(U)
    r = U[0]; v = U[2] / r; u = U[1] / r
    p = (GAM - 1) * (U[3] - 0.5 * r * (u * u + v * v))
    dA = dx * dy
    mass = r.sum() * dA
    momx = U[1].sum() * dA
    Etot = (U[3] - r * G * Y).sum() * dA  # internal+kinetic+potential(-rho g y)
    KE = (0.5 * r * (u * u + v * v)).sum() * dA
    # heavy-fluid fraction f in [0,1]; horizontal average
    f = np.clip((r - 1.0) / 1.0, 0, 1)
    fbar = f.mean(axis=0)
    y = Y[0]
    # mixing zone: 5%-95% heavy-fraction extent of horizontal mean
    def first_cross(level):
        idx = np.where(fbar < level)[0]
        return y[idx[0]] if len(idx) else y[-1]
    # heavy fluid sits below (small y) and falls towards +y
    spike = y[np.where(fbar > 0.05)[0].max()] if np.any(fbar > 0.05) else 0
    bubble = y[np.where(fbar < 0.95)[0].min()] if np.any(fbar < 0.95) else 1
    # symmetry about x = LX/2 (mirror)
    sym = np.abs(r - r[::-1, :]).max()
    vhat = np.abs(np.fft.rfft(v, axis=0)[1]).max() * 2 / v.shape[0]  # max_y |k=8pi mode of v|
    return dict(mass=mass, momx=momx, Etot=Etot, KE=KE, spike=spike, bubble=bubble,
                vmax=np.abs(v).max(), vhat=vhat, pmin=p.min(), rmin=r.min(), rmax=r.max(), sym=sym)


def run(nx, ny, tend=3.0, cfl=0.4, snap_times=(), perturb=True, diag_every=0.01, tag=None, amp=1.0, exact_hydro=False):
    X, Y, U = initial(nx, ny, perturb, amp, exact_hydro)
    dx, dy = LX / nx, LY / ny
    step, maxspeed = make_rhs(nx, ny)
    t = 0.0; nstep = 0
    snaps = {}; hist = []
    snap_times = sorted(snap_times)
    next_diag = 0.0
    si = 0
    t0 = time.time()
    while True:
        if t >= next_diag - 1e-12:
            d = diagnostics(U, Y, dx, dy); d["t"] = t; hist.append(d)
            next_diag += diag_every
        if si < len(snap_times) and t >= snap_times[si] - 1e-12:
            snaps[snap_times[si]] = np.asarray(U); si += 1
        if t >= tend - 1e-12:
            break
        dt = cfl / float(maxspeed(U))
        tgt = min(tend, next_diag, snap_times[si] if si < len(snap_times) else tend)
        dt = min(dt, tgt - t)
        U = step(U, dt); t += dt; nstep += 1
        if nstep % 2000 == 0:
            print(f"[{nx}x{ny}] t={t:.3f} steps={nstep} wall={time.time()-t0:.0f}s", flush=True)
    print(f"[{nx}x{ny}] done steps={nstep} wall={time.time()-t0:.0f}s", flush=True)
    keys = hist[0].keys()
    H = {k: np.array([h[k] for h in hist]) for k in keys}
    out = dict(X=X, Y=Y, U=np.asarray(U), **{"h_" + k: v for k, v in H.items()})
    for ts, Us in snaps.items():
        out[f"snap_{ts:.2f}"] = Us
    if tag:
        np.savez_compressed(tag, **out)
    return out


if __name__ == "__main__":
    nx = int(sys.argv[1]); ny = 4 * nx
    mode = sys.argv[2] if len(sys.argv) > 2 else "rt"
    os.makedirs("/box/final/data", exist_ok=True)
    if mode == "static":
        run(nx, ny, tend=3.0, perturb=False, diag_every=0.1, tag=f"/box/final/data/static_{nx}.npz")
    elif mode == "statichydro":
        run(nx, ny, tend=3.0, perturb=False, diag_every=0.1, exact_hydro=True, tag=f"/box/final/data/statichydro_{nx}.npz")
    elif mode == "lin":
        run(nx, ny, tend=3.0, amp=1e-4, diag_every=0.02, tag=f"/box/final/data/lin_{nx}.npz")
    else:
        run(nx, ny, tend=3.0, snap_times=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            tag=f"/box/final/data/rt_{nx}.npz")
