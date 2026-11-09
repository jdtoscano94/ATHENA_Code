#!/usr/bin/env python3
"""
2D inviscid Burgers' equation on a periodic domain.

PDE (conservation form):
    u_t + (u^2/2)_x + (u^2/2)_y = 0,   (x,y) in [0, 4*pi]^2,  t in [0, 2]
    u(x,y,0) = 1/3 + (2/3) sin((x+y)/2),   periodic in x and y.

Numerical method
----------------
* 5th-order finite-difference WENO reconstruction (WENO-Z variant of the
  Jiang-Shu smoothness indicators), applied dimension by dimension with
  global Lax-Friedrichs (Rusanov) flux splitting.
* Time stepping: classical 4th-order Runge-Kutta (default) or SSP-RK3,
  with a CFL-based time step  dt = CFL * h / (alpha_x + alpha_y).

Exact solution (used for validation)
------------------------------------
With xi := (x+y)/2 both the initial data and the PDE reduce exactly to the
1D problem
    U_t + U U_xi = 0,   U(xi,0) = m + a sin(xi),   m = 1/3, a = 2/3,
with period 2*pi in xi (the domain [0,4*pi]^2 covers two periods along the
diagonal).  Its entropy solution is evaluated with the Lax-Oleinik /
Hopf-Lax minimization formula (fine-grid bracketing + golden-section
refinement to machine precision):
    Z(eta,t) = d/deta min_s [ (eta-s)^2/(2t) + phi0(s) ],  phi0(s) = -a cos s,
    U(xi,t)  = m + Z(xi - m*t, t).
Shocks form at t_b = 1/a = 3/2 at xi = pi + m*t_b; for t >= t_b a shock sits
exactly at xi_s(t) = pi + m*t (mod 2*pi) with left/right states
u_L/R = m +/- a sin(eta0*) where eta0* solves eta0 + a*t*sin(eta0) = pi on
(pi/2, pi).
"""

import numpy as np

# ----------------------------------------------------------------------
# global constants
# ----------------------------------------------------------------------
L_DOMAIN = 4.0 * np.pi          # domain side
M0 = 1.0 / 3.0                 # mean of IC
A0 = 2.0 / 3.0                 # amplitude of IC
PER = 2.0 * np.pi              # period of solution in xi = (x+y)/2
T_BURST = 1.0 / A0             # shock-formation time = 1.5

EPS_W = 1e-12                  # WENO regularisation epsilon
DW = (0.1, 0.6, 0.3)           # ideal WENO5 weights

# ----------------------------------------------------------------------
# WENO5-Z reconstruction (finite differences, point values)
# ----------------------------------------------------------------------

def _wz_weights(b0, b1, b2):
    """Nonlinear WENO-Z weights."""
    tau = np.abs(b0 - b2)
    a0 = DW[0] * (1.0 + (tau / (b0 + EPS_W)) ** 2)
    a1 = DW[1] * (1.0 + (tau / (b1 + EPS_W)) ** 2)
    a2 = DW[2] * (1.0 + (tau / (b2 + EPS_W)) ** 2)
    s = a0 + a1 + a2
    return a0 / s, a1 / s, a2 / s


def weno5_pos(F, ax):
    """Left-biased WENO5-Z interface values  F_{i+1/2}  from point values F.
    Works on arrays of any shape along axis ax (periodic wrap)."""
    fm2 = np.roll(F, 2, ax)
    fm1 = np.roll(F, 1, ax)
    fp1 = np.roll(F, -1, ax)
    fp2 = np.roll(F, -2, ax)
    p0 = (2.0 * fm2 - 7.0 * fm1 + 11.0 * F) / 6.0
    p1 = (-fm1 + 5.0 * F + 2.0 * fp1) / 6.0
    p2 = (2.0 * F + 5.0 * fp1 - fp2) / 6.0
    b0 = (13.0 / 12.0) * (fm2 - 2.0 * fm1 + F) ** 2 + 0.25 * (fm2 - 4.0 * fm1 + 3.0 * F) ** 2
    b1 = (13.0 / 12.0) * (fm1 - 2.0 * F + fp1) ** 2 + 0.25 * (fm1 - fp1) ** 2
    b2 = (13.0 / 12.0) * (F - 2.0 * fp1 + fp2) ** 2 + 0.25 * (3.0 * F - 4.0 * fp1 + fp2) ** 2
    w0, w1, w2 = _wz_weights(b0, b1, b2)
    return w0 * p0 + w1 * p1 + w2 * p2


def weno5_neg(F, ax):
    """Right-biased (mirror) WENO5-Z interface values  F_{i+1/2}."""
    fm1 = np.roll(F, 1, ax)
    fp1 = np.roll(F, -1, ax)
    fp2 = np.roll(F, -2, ax)
    fp3 = np.roll(F, -3, ax)
    q0 = (2.0 * fp3 - 7.0 * fp2 + 11.0 * fp1) / 6.0
    q1 = (-fp2 + 5.0 * fp1 + 2.0 * F) / 6.0
    q2 = (2.0 * fp1 + 5.0 * F - fm1) / 6.0
    b0 = (13.0 / 12.0) * (fp3 - 2.0 * fp2 + fp1) ** 2 + 0.25 * (fp3 - 4.0 * fp2 + 3.0 * fp1) ** 2
    b1 = (13.0 / 12.0) * (fp2 - 2.0 * fp1 + F) ** 2 + 0.25 * (fp2 - F) ** 2
    b2 = (13.0 / 12.0) * (fp1 - 2.0 * F + fm1) ** 2 + 0.25 * (3.0 * fp1 - 4.0 * F + fm1) ** 2
    w0, w1, w2 = _wz_weights(b0, b1, b2)
    return w0 * q0 + w1 * q1 + w2 * q2


# ----------------------------------------------------------------------
# semi-discrete RHS with Lax-Friedrichs flux splitting
# ----------------------------------------------------------------------

def rhs(u, h, flux_x=None, flux_y=None, alphax=None, alphay=None):
    """du/dt = -( d/dx flux_x(u) + d/dy flux_y(u) ).

    Default: 2D inviscid Burgers (flux = u^2/2, LF alpha = max|u| recomputed
    per call).  For testing, linear advection can be injected via
    flux_x = lambda v: cx*v, alphax = |cx| (same for y).
    """
    if flux_x is None:
        amax = max(float(np.abs(u).max()), 1e-14)
        flux_x = flux_y = lambda v: 0.5 * v * v
        alphax = alphay = amax
    fx = flux_x(u)
    fy = flux_y(u)
    Fx = weno5_pos(0.5 * (fx + alphax * u), 0) + weno5_neg(0.5 * (fx - alphax * u), 0)
    Fy = weno5_pos(0.5 * (fy + alphay * u), 1) + weno5_neg(0.5 * (fy - alphay * u), 1)
    return -(Fx - np.roll(Fx, 1, 0) + Fy - np.roll(Fy, 1, 1)) / h


def step(u, dt, h, method='rk4', **kw):
    if method == 'rk4':
        k1 = rhs(u, h, **kw)
        k2 = rhs(u + (0.5 * dt) * k1, h, **kw)
        k3 = rhs(u + (0.5 * dt) * k2, h, **kw)
        k4 = rhs(u + dt * k3, h, **kw)
        return u + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    if method == 'ssprk3':
        k1 = rhs(u, h, **kw)
        u1 = u + dt * k1
        k2 = rhs(u1, h, **kw)
        u2 = 0.75 * u + 0.25 * (u1 + dt * k2)
        k3 = rhs(u2, h, **kw)
        return u / 3.0 + 2.0 * (u2 + dt * k3) / 3.0
    raise ValueError(method)


def integrate(u0, h, T, dtmax, snap_times=(), method='rk4', **kw):
    """March u0 to time T with time steps <= dtmax, hitting each requested
    snapshot time exactly.  Returns (u_final, snaps, nsteps)."""
    u = np.array(u0, dtype=float)
    t = 0.0
    snaps = {}
    nsteps = 0
    for ts in sorted(snap_times):
        if ts <= t + 1e-14:
            snaps[ts] = u.copy()
            continue
        n = max(1, int(np.ceil((ts - t) / dtmax - 1e-12)))
        dt = (ts - t) / n
        for _ in range(n):
            u = step(u, dt, h, method, **kw)
        nsteps += n
        t = ts
        snaps[ts] = u.copy()
    if t < T - 1e-14:
        n = max(1, int(np.ceil((T - t) / dtmax - 1e-12)))
        dt = (T - t) / n
        for _ in range(n):
            u = step(u, dt, h, method, **kw)
        nsteps += n
        t = T
    return u, snaps, nsteps


# ----------------------------------------------------------------------
# initial condition
# ----------------------------------------------------------------------

def initial_condition(N):
    h = L_DOMAIN / N
    x = np.arange(N) * h
    return x, M0 + A0 * np.sin(0.5 * (x[:, None] + x[None, :]))


# ----------------------------------------------------------------------
# exact entropy solution (Lax-Oleinik / Hopf-Lax)
# ----------------------------------------------------------------------

def exact_solution(xi, t, chunk=2048):
    """Exact entropy solution U(xi, t) of  U_t + U U_xi = 0,
    U(xi,0) = m + a sin(xi) (period 2*pi).  Vectorised over xi."""
    m, a, P = M0, A0, PER
    xi = np.asarray(xi, dtype=float)
    shape = xi.shape
    xf = xi.ravel()
    out = np.empty_like(xf)
    if t <= 0.0:
        out[:] = m + a * np.sin(xf)
        return out.reshape(shape)
    eta = np.mod(xf - m * t, P)
    # minimise G(s) = (eta-s)^2/(2t) - a cos(s) over the periodic window
    # [eta-P/2, eta+P/2]; the entropy solution is U = m + (eta - s*)/t.
    Mgrid = 4096
    offs = (np.arange(Mgrid) / (Mgrid - 1.0) - 0.5) * P
    invphi = 0.5 * (np.sqrt(5.0) - 1.0)
    npts = len(eta)
    for i0 in range(0, npts, chunk):
        et = eta[i0:i0 + chunk]
        S = et[:, None] + offs[None, :]
        G = (et[:, None] - S) ** 2 / (2.0 * t) - a * np.cos(S)
        k = np.argmin(G, axis=1)
        r = np.arange(len(et))
        lo = S[r, np.maximum(k - 1, 0)].copy()
        hi = S[r, np.minimum(k + 1, Mgrid - 1)].copy()
        # if the discrete argmin sits at a grid edge, extend to the window edge
        lo = np.where(k == 0, et - 0.5 * P, lo)
        hi = np.where(k == Mgrid - 1, et + 0.5 * P, hi)
        # golden-section refinement (vectorised) to get near the minimum
        blo, bhi = lo.copy(), hi.copy()          # basin bracket (kept for Newton)
        c = hi - invphi * (hi - lo)
        d = lo + invphi * (hi - lo)
        for _ in range(60):
            gc = (et - c) ** 2 / (2.0 * t) - a * np.cos(c)
            gd = (et - d) ** 2 / (2.0 * t) - a * np.cos(d)
            msk = gc < gd
            hi = np.where(msk, d, hi)
            lo = np.where(msk, lo, c)
            c = hi - invphi * (hi - lo)
            d = lo + invphi * (hi - lo)
        s = 0.5 * (lo + hi)
        # Newton polish on G'(s) = (s-et)/t + a sin(s): the golden search is
        # limited to ~sqrt(eps) accuracy because G is quadratically flat at
        # its minimum; Newton on the derivative recovers machine precision.
        for _ in range(15):
            Gp = (s - et) / t + a * np.sin(s)
            Gpp = 1.0 / t + a * np.cos(s)
            good = Gpp > 1e-6
            step = np.where(good, Gp / np.where(good, Gpp, 1.0), 0.0)
            s = s - np.clip(step, blo - s, bhi - s)
        out[i0:i0 + len(et)] = m + (et - s) / t
    return out.reshape(shape)


def exact_2d(N, t):
    """Exact solution of the 2D task on the N x N node grid."""
    h = L_DOMAIN / N
    k = np.arange(2 * N - 1)
    Uk = exact_solution(k * (h / 2.0), t)
    idx = np.add.outer(np.arange(N), np.arange(N))   # i + j
    return Uk[idx]


def shock_state(t):
    """Analytic shock for t >= t_b: returns (xi_s mod 2pi, u_L, u_R), else None."""
    m, a = M0, A0
    if t < T_BURST - 1e-13:
        return None
    if abs(t - T_BURST) <= 1e-13:
        return (np.pi + m * t) % PER, m, m
    at = a * t
    eta_mid = np.arccos(-1.0 / at)          # local max of eta0 + at sin(eta0)
    lo, hi = 0.5 * np.pi, eta_mid
    f = lambda e: e + at * np.sin(e) - np.pi
    if f(hi) < 0:                          # numerically degenerate bracket
        hi = np.pi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    eta0 = 0.5 * (lo + hi)
    zL = a * np.sin(eta0)
    return (np.pi + m * t) % PER, m + zL, m - zL


def char_solution(xi, t, tol=1e-14, itmax=200):
    """Pre-shock exact solution from characteristics (Newton solve), for
    cross-checking exact_solution().  Valid for t < t_b."""
    m, a = M0, A0
    xi = np.asarray(xi, dtype=float)
    x0 = xi - m * t
    err = None
    for _ in range(itmax):
        u0 = m + a * np.sin(x0)
        g = x0 + u0 * t - xi
        err = np.max(np.abs(g))
        if err < tol:
            break
        dg = 1.0 + (a * np.cos(x0)) * t
        x0 = x0 - g / dg
    return m + a * np.sin(x0), err


# ----------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------

def norms(err):
    return dict(L1=float(np.mean(np.abs(err))),
                L2=float(np.sqrt(np.mean(err * err))),
                Linf=float(np.max(np.abs(err))))


def error_report(u_num, N, t):
    """Errors of the numerical 2D field vs exact solution at time t."""
    ue = exact_2d(N, t)
    nrm = norms(u_num - ue)
    if t >= T_BURST:                       # exclude shock neighbourhood from Linf
        ss = shock_state(t)
        h = L_DOMAIN / N
        x = np.arange(N) * h
        xi = 0.5 * (x[:, None] + x[None, :])
        d = np.abs(((xi - ss[0] + PER / 2) % PER) - PER / 2)
        mask = d > 10 * h
        if mask.any():
            nrm['Linf_smooth'] = float(np.max(np.abs((u_num - ue)[mask])))
    return nrm


def numerical_shock_xi(u, N):
    """Shock position (xi, mod 2pi) from the largest jump of the diagonal."""
    h = L_DOMAIN / N
    v = np.diagonal(u).copy()
    j = np.abs(np.diff(v))
    i = int(np.argmax(j))
    xi = (i + 0.5) * h
    return xi % PER, float(j[i])


def mass(u):
    return float(np.mean(u))


# ----------------------------------------------------------------------
# production run + summary figure
# ----------------------------------------------------------------------

def production(outdir='.', N=512, cfl=0.4, method='rk4',
               snap_times=(0.0, 0.5, 1.0, 1.5, 1.75, 2.0)):
    import os, json, time as _time
    os.makedirs(outdir, exist_ok=True)
    h = L_DOMAIN / N
    x, u0 = initial_condition(N)
    umax = float(np.abs(u0).max())
    dtmax = cfl * h / (2.0 * umax)
    t0 = _time.time()
    uT, snaps, nsteps = integrate(u0, h, 2.0, dtmax,
                                  snap_times=snap_times, method=method)
    wall = _time.time() - t0
    print(f'production N={N} {method} cfl={cfl}: {nsteps} steps, '
          f'dt={dtmax:.6f}, {wall:.1f}s wall')

    # errors at t = 1 (smooth) and t = 2 (post-shock)
    err = {'t=1': error_report(snaps[1.0], N, 1.0),
           't=2': error_report(snaps[2.0], N, 2.0)}
    xi_num, jump = numerical_shock_xi(snaps[2.0], N)
    ss2 = shock_state(2.0)
    shock_err = abs(((xi_num - ss2[0] + PER / 2) % PER) - PER / 2)
    extra = {
        'N': N, 'cfl': cfl, 'method': method, 'nsteps': nsteps,
        'dt': dtmax, 'wall_s': wall,
        'mass_drift': abs(mass(uT) - mass(u0)),
        'range': [float(uT.min()), float(uT.max())],
        'shock_xi_numeric': xi_num, 'shock_xi_exact': float(ss2[0]),
        'shock_pos_err': shock_err, 'shock_jump': jump,
        'shock_states_exact': [float(ss2[1]), float(ss2[2])],
        'anti_diag_invariance': float(np.max(np.abs(
            snaps[2.0] - np.roll(snaps[2.0], (1, -1), axis=(0, 1))))),
    }
    print(json.dumps({'errors': err, 'extra': extra}, indent=2))
    with open(os.path.join(outdir, 'errors.json'), 'w') as f:
        json.dump({'errors': err, 'extra': extra}, f, indent=2)

    np.savez_compressed(os.path.join(outdir, 'snapshots.npz'),
                        x=x, times=np.array(snap_times),
                        u=np.array([snaps[t] for t in snap_times]))
    return x, snaps, err, extra


def _shock_segments(xi_s):
    """Line segments (x0,y0)-(x1,y1) of the shock x+y = 2 xi_s mod 4pi in the
    square [0,4pi]^2."""
    L = L_DOMAIN
    cs = [(2.0 * xi_s) % (2.0 * L)]
    cs.append(cs[0] + L)
    segs = []
    for c in cs:
        if c <= L:
            segs.append(((0.0, c), (c, 0.0)))
        else:
            segs.append(((L, c - L), (c - L, L)))
    return segs


def make_summary(x, snaps, out_png, conv_csv=None,
                 meta=('WENO5-Z + RK4', 512, 0.4)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    L = L_DOMAIN
    times = sorted(snaps)
    vmin, vmax = -0.4, 1.05
    fig = plt.figure(figsize=(17.5, 12.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.95])

    axes = []
    for i, t in enumerate(times):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        im = ax.imshow(snaps[t], origin='lower', extent=[0, L, 0, L],
                       cmap='viridis', vmin=vmin, vmax=vmax, aspect='equal',
                       interpolation='bilinear')
        title = f't = {t:.2f}'
        if abs(t - T_BURST) < 1e-9:
            title += '  (shock forms)'
        elif t > T_BURST:
            title += '  (shocked)'
        ax.set_title(title, fontsize=12)
        if t >= T_BURST:
            ss = shock_state(t)
            for (p0, p1) in _shock_segments(ss[0] if ss else np.pi + M0 * t):
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'w--',
                        lw=1.2, alpha=0.9)
        if i % 3 > 0:
            ax.set_yticklabels([])
        if i // 3 == 0:
            ax.set_xticklabels([])
        axes.append(ax)
    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    cb = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.015)
    cb.set_label('u(x, y, t)')
    cb.ax.tick_params(labelsize=9)

    # ---- profiles along the diagonal -------------------------------------
    axp = fig.add_subplot(gs[2, 0])
    cols = plt.cm.plasma(np.linspace(0.15, 0.92, len(times)))
    for c, t in zip(cols, times):
        axp.plot(x, np.diagonal(snaps[t]), color=c, lw=1.5,
                 label=f't = {t:.2f}')
    axp.set_xlabel(r'$\xi = (x+y)/2$')
    axp.set_ylabel(r'$u(\xi, t)$  along diagonal $x = y$')
    axp.set_title('1D reduction along the diagonal (two periods)', fontsize=12)
    axp.legend(fontsize=8, ncol=2, loc='upper left')
    axp.set_xlim(0, L)
    axp.grid(alpha=0.25)

    # ---- numerical vs exact at t = 2 --------------------------------------
    axe = fig.add_subplot(gs[2, 1])
    xi_f = np.linspace(0.0, L, 2401)
    axe.plot(xi_f, exact_solution(xi_f, 2.0), 'k-', lw=1.4,
             label='exact (Lax–Oleinik)')
    v = np.diagonal(snaps[2.0])
    axe.plot(x[::4], v[::4], 'o', ms=2.4, color='crimson', mfc='none',
             label='numerical (diagonal)')
    ss2 = shock_state(2.0)
    axe.axvline(ss2[0], color='0.5', ls=':', lw=1)
    axe.set_xlabel(r'$\xi = (x+y)/2$')
    axe.set_ylabel('u')
    axe.set_title('t = 2.0: numerical vs exact', fontsize=12)
    axe.legend(fontsize=9, loc='upper left')
    axe.set_xlim(0, L)
    axe.grid(alpha=0.25)
    axin = axe.inset_axes([0.56, 0.10, 0.40, 0.42])
    m = (x >= ss2[0] - 0.25) & (x <= ss2[0] + 0.25)
    axin.plot(x[m], v[m], 'o', ms=2.5, color='crimson', mfc='none')
    axin.plot(xi_f, exact_solution(xi_f, 2.0), 'k-', lw=1.2)
    axin.axvline(ss2[0], color='0.5', ls=':', lw=0.8)
    axin.set_xlim(ss2[0] - 0.25, ss2[0] + 0.25)
    axin.set_xticks([])
    axin.set_yticks([])
    axin.set_title('shock zoom', fontsize=8)

    # ---- convergence -------------------------------------------------------
    axc = fig.add_subplot(gs[2, 2])
    try:
        import csv
        with open(conv_csv) as f:
            rows = list(csv.DictReader(f))
        Ns = np.array([float(r['N']) for r in rows])
        axc.loglog(Ns, [float(r['L1_t1']) for r in rows], 'o-', label='L1, t=1')
        axc.loglog(Ns, [float(r['L2_t1']) for r in rows], 's-', label='L2, t=1')
        axc.loglog(Ns, [float(r['L1_t2']) for r in rows], 'o--', label='L1, t=2')
        axc.loglog(Ns, [float(r['L2_t2']) for r in rows], 's--', label='L2, t=2')
        p1 = np.polyfit(np.log(Ns), np.log([float(r['L1_t1']) for r in rows]), 1)[0]
        p2 = np.polyfit(np.log(Ns), np.log([float(r['L1_t2']) for r in rows]), 1)[0]
        axc.loglog(Ns, [float(rows[0]['L1_t1']) * (Ns[0] / n) ** 5 for n in Ns],
                   color='0.6', lw=0.8)
        axc.text(0.05, 0.05, f'fitted order (L1): t=1: {p1:.2f}, t=2: {p2:.2f}\n'
                             f'grey guide: 5th order',
                 transform=axc.transAxes, fontsize=9,
                 bbox=dict(fc='white', alpha=0.8, ec='0.8'))
        have_conv = True
    except Exception as e:
        axc.text(0.5, 0.5, 'convergence.csv not found', ha='center')
        have_conv = False
    axc.set_xlabel('N (grid points per direction)')
    axc.set_ylabel('error vs exact solution')
    axc.set_title('Grid-convergence study', fontsize=12)
    if have_conv:
        axc.legend(fontsize=8, loc='lower left')
    axc.grid(alpha=0.25, which='both')

    fig.suptitle('2D inviscid Burgers equation  '
                 r'$u_t + (u^2/2)_x + (u^2/2)_y = 0$,  periodic,  '
                 r'$u_0 = \frac{1}{3} + \frac{2}{3}\sin\frac{x+y}{2}$   |   '
                 f'{meta[0]},  N = {meta[1]},  CFL = {meta[2]}',
                 fontsize=13)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print('wrote', out_png)


if __name__ == '__main__':
    import os
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='/box/final')
    ap.add_argument('--N', type=int, default=512)
    ap.add_argument('--cfl', type=float, default=0.4)
    ap.add_argument('--method', default='rk4')
    ap.add_argument('--conv-csv', default='/box/final/convergence.csv')
    a = ap.parse_args()
    x, snaps, err, extra = production(a.outdir, a.N, a.cfl, a.method)
    make_summary(x, snaps, os.path.join(a.outdir, 'summary_all.png'),
                 conv_csv=a.conv_csv,
                 meta=(f'WENO5-Z + {a.method.upper()}', a.N, a.cfl))
