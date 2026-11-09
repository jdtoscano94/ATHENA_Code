#!/usr/bin/env python3
"""
Validation suite for the 2D inviscid Burgers solver in burgers2d.py.

Tests:
 1. Exact solution (Lax-Oleinik) vs pre-shock characteristic solution,
    vs analytic shock position/states, mass over a period.
 2. Linear advection through the same RHS machinery (exercises WENO in both
    directions and both flux signs): convergence order on a smooth periodic IC.
 3. Stationary solution test: any function of (x-y) is an exact steady state
    of u_t + u(u_x + u_y) = 0; the discrete operator must preserve it to
    truncation level (~h^5 drift).
 4. x-only reduction test: u0 = m + a sin(x) reduces to 1D Burgers in x;
    compare with exact solution at t = 1 (smooth) and t = 2 (shocked).
 5. Grid-convergence study for the task IC at t = 1 and t = 2 (+ temporal
    refinement and RK4 vs SSP-RK3 cross-checks).
 6. Conservation, anti-diagonal invariance.

Writes:  validation.txt, convergence.csv
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import burgers2d as b

OUT = []
PASS = []


def log(s=''):
    print(s)
    OUT.append(str(s))


def check(name, cond, detail=''):
    tag = 'PASS' if cond else 'FAIL'
    PASS.append(cond)
    log(f'  [{tag}] {name}  {detail}')


# ----------------------------------------------------------------------

def test_exact():
    log('\n=== 1. Exact (Lax-Oleinik) solution self-checks ===')
    rng = np.random.default_rng(42)

    log('1a. vs characteristic (Newton) solution, pre-shock times:')
    for t in [0.3, 1.0, 1.4]:
        xi = rng.uniform(0.0, 2 * np.pi, 1000)
        ue = b.exact_solution(xi, t)
        uc, res = b.char_solution(xi, t)
        d = float(np.max(np.abs(ue - uc)))
        check(f't={t}: max|diff| < 1e-12', d < 1e-12, f'(diff={d:.2e})')

    log('1b. shock position/states vs analytic formulas (post-shock):')
    for t in [1.75, 2.0]:
        xi_s, uL, uR = b.shock_state(t)
        dlt = 1e-7
        ul = b.exact_solution(np.array([xi_s - dlt]), t)[0]
        ur = b.exact_solution(np.array([xi_s + dlt]), t)[0]
        check(f't={t}: states at xi_s +/- 1e-7', 
              abs(ul - uL) < 1e-6 and abs(ur - uR) < 1e-6,
              f'({ul:.7f} vs {uL:.7f}, {ur:.7f} vs {uR:.7f})')
        # entropy: u_L > u_R (compressive)
        check(f't={t}: compressive (u_L > u_R)', uL > uR)

    log('1c. mass of exact solution over a fine sampling of one period:')
    for t in [0.7, 1.75, 2.0]:
        xi = np.linspace(0.0, 2 * np.pi, 2_000_001)
        u = b.exact_solution(xi[::20], t)      # 100k pts, chunked eval
        # trapezoid with 100k pts: sampling error ~ jump*dxi/period ~ 1e-5
        mean = float(np.mean(u))
        check(f't={t}: |mean - m| < 1e-4 (sampling-limited)',
              abs(mean - b.M0) < 1e-4, f'(mean={mean:.8f})')

    log('1d. shock-formation time: t_b = 1/a = 1.5, first shock at pi + m t_b:')
    ss = b.shock_state(1.5)
    check('t_b = 1.5, xi_b = pi + 1/2, zero-strength shock',
          abs(b.T_BURST - 1.5) < 1e-14 and abs(ss[0] - (np.pi + 0.5)) < 1e-9
          and abs(ss[1] - ss[2]) < 1e-9)


# ----------------------------------------------------------------------

def advection_run(N, cx, cy, T=2.0, cfl=0.4):
    h = b.L_DOMAIN / N
    x = np.arange(N) * h
    f0 = lambda X, Y: (0.6 + 0.35 * np.sin(X / 2) * np.sin(Y / 2)
                       + 0.2 * np.sin(X / 2 + Y / 2 + 0.7))
    u0 = f0(x[:, None], x[None, :])
    kw = dict(flux_x=lambda v: cx * v, flux_y=lambda v: cy * v,
              alphax=abs(cx), alphay=abs(cy))
    dtmax = cfl * h / (abs(cx) + abs(cy))
    uT, _, ns = b.integrate(u0, h, T, dtmax, method='rk4', **kw)
    ue = f0(np.mod(x - cx * T, b.L_DOMAIN)[:, None],
            np.mod(x - cy * T, b.L_DOMAIN)[None, :])
    return b.norms(uT - ue), ns


def test_advection():
    log('\n=== 2. Linear advection through the same operator (flux signs) ===')
    for cx, cy in [(1.0, 0.7), (-1.0, -0.7), (1.0, -0.7), (-0.7, 1.0)]:
        errs = []
        for N in [16, 32, 64, 128]:
            nrm, ns = advection_run(N, cx, cy)
            errs.append(nrm)
            log(f'  c=({cx:+.1f},{cy:+.1f}) N={N:3d}: L2={nrm["L2"]:.3e} '
                f'Linf={nrm["Linf"]:.3e} ({ns} steps)')
        o2 = np.log2(errs[-3]['L2'] / errs[-1]['L2']) / 2.0   # over 2 doublings
        oi = np.log2(errs[-3]['Linf'] / errs[-1]['Linf']) / 2.0
        check(f'c=({cx:+.1f},{cy:+.1f}): L2 order (N=32->128) >= 3.5',
              o2 >= 3.5, f'(L2 order={o2:.2f}, Linf order={oi:.2f})')


# ----------------------------------------------------------------------

def test_stationary():
    log('\n=== 3. Stationary solution f(x-y) (exact steady state) ===')
    # Any function of (x-y) satisfies u_x + u_y = 0 exactly, hence is a steady
    # state.  The discrete RHS is then the *sum of the two directional
    # truncation errors* (each ~h^4 at flux critical points, mirror-symmetric
    # in x/y); it must be small and decay rapidly with N.
    drifts = {}
    for N in [64, 128, 256]:
        h = b.L_DOMAIN / N
        x = np.arange(N) * h
        u0 = b.M0 + b.A0 * np.sin(0.5 * (x[:, None] - x[None, :]))
        dtmax = 0.4 * h / (2.0 * float(np.abs(u0).max()))
        uT, _, ns = b.integrate(u0, h, 2.0, dtmax, method='rk4')
        drifts[N] = float(np.max(np.abs(uT - u0)))
        log(f'  N={N}: max|u(2)-u(0)| = {drifts[N]:.3e}  ({ns} steps)')
    r1 = np.log2(drifts[64] / drifts[128])
    r2 = np.log2(drifts[128] / drifts[256])
    check('stationary drift decays with apparent order >= 2.5',
          r1 >= 2.5 and r2 >= 2.5, f'(orders {r1:.1f}, {r2:.1f})')
    check('stationary drift at N=256 < 5e-7', drifts[256] < 5e-7,
          f'(drift={drifts[256]:.2e})')


# ----------------------------------------------------------------------

def test_xonly():
    log('\n=== 4. x-only reduction: u0 = m + a sin(x) -> 1D Burgers in x ===')
    # For x-only data the scheme is exactly a 1D WENO solve with dx = h, i.e.
    # the SAME effective resolution as the diagonal-IC case at N/2
    # (whose effective diagonal spacing is h/2).  Errors must coincide.
    for N in [64, 128]:
        h = b.L_DOMAIN / N
        x = np.arange(N) * h
        u0 = np.tile((b.M0 + b.A0 * np.sin(x))[:, None], (1, N))
        dtmax = 0.4 * h / (2.0 * 1.0)
        uT, snaps, ns = b.integrate(u0, h, 2.0, dtmax,
                                    snap_times=(1.0, 2.0), method='rk4')
        # reference: diagonal IC at N/2 (same effective 1D resolution)
        Nd = N // 2
        hd = b.L_DOMAIN / Nd
        xd, u0d = b.initial_condition(Nd)
        uTd, snapsd, _ = b.integrate(u0d, hd, 2.0,
                                     0.4 * hd / (2.0 * 1.0),
                                     snap_times=(1.0, 2.0), method='rk4')
        for t in [1.0, 2.0]:
            ue = np.tile(b.exact_solution(x, t)[:, None], (1, N))
            nrm = b.norms(snaps[t] - ue)
            nrm_ref = b.error_report(snapsd[t], Nd, t)
            rel = abs(nrm['L1'] - nrm_ref['L1']) / nrm_ref['L1']
            log(f'  N={N} t={t}: L1={nrm["L1"]:.3e} (diag-IC at N/2: '
                f'{nrm_ref["L1"]:.3e}, rel diff {rel:.1%})')
            check(f'N={N} t={t}: x-only error == diag-IC error at N/2 (<10%)',
                  rel < 0.10)


# ----------------------------------------------------------------------

def convergence_study():
    log('\n=== 5. Grid convergence for the task IC (WENO5-Z + RK4, CFL=0.4) ===')
    rows = []
    for N in [32, 64, 128, 256, 512]:
        h = b.L_DOMAIN / N
        x, u0 = b.initial_condition(N)
        dtmax = 0.4 * h / (2.0 * float(np.abs(u0).max()))
        t0 = time.time()
        uT, snaps, ns = b.integrate(u0, h, 2.0, dtmax,
                                    snap_times=(1.0, 2.0), method='rk4')
        wall = time.time() - t0
        e1 = b.error_report(snaps[1.0], N, 1.0)
        e2 = b.error_report(snaps[2.0], N, 2.0)
        xi_n, jump = b.numerical_shock_xi(snaps[2.0], N)
        ss = b.shock_state(2.0)
        serr = abs(((xi_n - ss[0] + b.PER / 2) % b.PER) - b.PER / 2)
        rows.append(dict(N=N, **{f'{k}_t1': v for k, v in e1.items()},
                         **{f'{k}_t2': v for k, v in e2.items()},
                         shock_pos_err=serr, wall=wall, nsteps=ns))
        log(f'  N={N:4d} ({ns:4d} steps, {wall:5.1f}s): '
            f't=1: L1={e1["L1"]:.3e} L2={e1["L2"]:.3e} Linf={e1["Linf"]:.3e} | '
            f't=2: L1={e2["L1"]:.3e} L2={e2["L2"]:.3e} Linf={e2["Linf"]:.3e} '
            f'(smooth L∞={e2.get("Linf_smooth", float("nan")):.1e}) '
            f'shock pos err={serr:.1e}')
    # orders
    log('  observed orders (L1):')
    for i in range(1, len(rows)):
        p = np.log(rows[i - 1]['L1_t1'] / rows[i]['L1_t1']) / np.log(rows[i]['N'] / rows[i - 1]['N'])
        q = np.log(rows[i - 1]['L1_t2'] / rows[i]['L1_t2']) / np.log(rows[i]['N'] / rows[i - 1]['N'])
        log(f'    N={rows[i-1]["N"]:4d}->{rows[i]["N"]:4d}:  t=1: {p:.2f}   t=2: {q:.2f}')
    check('t=1: L1 order over N=128->512 >= 4.0',
          np.log(rows[-3]['L1_t1'] / rows[-1]['L1_t1']) / np.log(4) >= 4.0)
    check('t=2: L1 monotonically decreasing', 
          all(rows[i]['L1_t2'] < rows[i - 1]['L1_t2'] for i in range(1, len(rows))))

    # temporal refinement + method cross-check at N=256
    log('  temporal checks at N=256:')
    N = 256
    h = b.L_DOMAIN / N
    x, u0 = b.initial_condition(N)
    base = {}
    for cfl, meth in [(0.4, 'rk4'), (0.1, 'rk4'), (0.1, 'ssprk3')]:
        dtmax = cfl * h / (2.0 * float(np.abs(u0).max()))
        uT, snaps, ns = b.integrate(u0, h, 2.0, dtmax,
                                    snap_times=(1.0, 2.0), method=meth)
        e1 = b.error_report(snaps[1.0], N, 1.0)
        e2 = b.error_report(snaps[2.0], N, 2.0)
        base[(cfl, meth)] = (e1, e2)
        log(f'    {meth} cfl={cfl}: ({ns} steps) t=1 L2={e1["L2"]:.4e}, '
            f't=2 L2={e2["L2"]:.4e}')
    d1 = abs(base[(0.4, 'rk4')][0]['L2'] - base[(0.1, 'rk4')][0]['L2'])
    d2 = abs(base[(0.1, 'rk4')][0]['L2'] - base[(0.1, 'ssprk3')][0]['L2'])
    ref = base[(0.1, 'rk4')][0]['L2']
    check('RK4 temporal error negligible: |L2(cfl .4) - L2(cfl .1)| at t=1 < 2%',
          d1 < 0.02 * ref, f'(diff={d1:.2e}, error={ref:.2e})')
    check('RK4 vs SSP-RK3 agree once both temporally converged (cfl=0.1, t=1) < 2%',
          d2 < 0.02 * ref, f'(diff={d2:.2e})')

    # write CSV
    cols = ['N', 'L1_t1', 'L2_t1', 'Linf_t1', 'L1_t2', 'L2_t2', 'Linf_t2',
            'Linf_smooth_t2', 'shock_pos_err', 'nsteps', 'wall']
    with open(os.path.join(OUTDIR, 'convergence.csv'), 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            vals = [r.get('N'), r.get('L1_t1'), r.get('L2_t1'), r.get('Linf_t1'),
                    r.get('L1_t2'), r.get('L2_t2'), r.get('Linf_t2'),
                    r.get('Linf_smooth_t2'), r.get('shock_pos_err'),
                    r.get('nsteps'), r.get('wall')]
            f.write(','.join(f'{v:.6e}' if isinstance(v, float) else str(v)
                             for v in vals) + '\n')
    log(f"  wrote {os.path.join(OUTDIR, 'convergence.csv')}")
    return rows


# ----------------------------------------------------------------------

def test_conservation_invariance():
    log('\n=== 6. Conservation, anti-diagonal invariance, stability margin ===')
    N = 128
    h = b.L_DOMAIN / N
    x, u0 = b.initial_condition(N)
    m0 = b.mass(u0)
    dtmax = 0.4 * h / (2.0 * float(np.abs(u0).max()))
    uT, snaps, ns = b.integrate(u0, h, 2.0, dtmax, method='rk4')
    check('mass conserved to < 1e-12 (relative)',
          abs(b.mass(uT) - m0) < 1e-12, f'(drift={abs(b.mass(uT)-m0):.2e})')
    inv = float(np.max(np.abs(uT - np.roll(uT, (1, -1), axis=(0, 1)))))
    check('anti-diagonal invariance u(x,y)=u(x+dy,y-dy) < 1e-12',
          inv < 1e-12, f'(max dev={inv:.2e})')
    r0 = [float(u0.min()), float(u0.max())]
    rT = [float(uT.min()), float(uT.max())]
    check('no overshoot beyond [min u0, max u0] + 2e-3',
          rT[0] > r0[0] - 2e-3 and rT[1] < r0[1] + 2e-3,
          f'(u0 range {r0}, uT range {rT})')


# ----------------------------------------------------------------------

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else '/box/final'
os.makedirs(OUTDIR, exist_ok=True)

if __name__ == '__main__':
    t_start = time.time()
    log(f'Validation of 2D inviscid Burgers solver  ({time.ctime()})')
    log(f'numpy {np.__version__}')
    test_exact()
    test_advection()
    test_stationary()
    test_xonly()
    convergence_study()
    test_conservation_invariance()
    log(f'\nTotal wall time: {time.time()-t_start:.1f}s')
    log(f'\n{sum(PASS)}/{len(PASS)} checks passed.'
        + ('' if all(PASS) else '  FAILURES PRESENT'))
    with open(os.path.join(OUTDIR, 'validation.txt'), 'w') as f:
        f.write('\n'.join(OUT) + '\n')
    print('wrote', os.path.join(OUTDIR, 'validation.txt'))
    sys.exit(0 if all(PASS) else 1)
