"""Relative L2 error at t = 2 of every IB2D solution (ATHENA sensitivity runs and Claude Code / Opus 5.5 runs)
against the exact entropy solution. Writes ib2d_rl2.json, read by fig_first90_SI.py.
u_t + u u_x + u u_y = 0 on [0,4pi]^2, u0 = 1/3 + 2/3 sin((x+y)/2). With eta = (x+y)/2: v_t + v v_eta = 0.
In the frame xi = eta - t/3, w = v - 1/3 solves w_t + w w_xi = 0 with w0 = 2/3 sin xi (odd), so the shock
(formed at t = 1.5) stays at xi = pi; the entropy solution takes the characteristic from the near side of the shock.
Check: the L1 errors of the Opus runs reproduce their own reports (1.07e-3, 1.11e-3, 4.92e-4 at t = 2).
Run from Plots/:  python ib2d_rl2.py"""
import json, re
from pathlib import Path
import numpy as np

SENS = Path(__file__).resolve().parent.parent/'Results'/'Sensitivity'


def exact(x, y, t):
    xi = np.mod((x + y) / 2 - t / 3, 2 * np.pi)
    left = xi <= np.pi; z = np.where(left, xi, 2 * np.pi - xi)          # w(2pi - z) = -w(z)
    a = 2 * t / 3
    top = np.arccos(-1 / a) if a > 1 else np.pi                          # f(x0) = x0 + a sin x0 increasing on [0, top]
    lo, hi = np.zeros_like(z), np.full_like(z, top)
    for _ in range(80):
        mid = (lo + hi) / 2; f = mid + a * np.sin(mid)
        lo, hi = np.where(f < z, mid, lo), np.where(f < z, hi, mid)
    w = (2 / 3) * np.sin((lo + hi) / 2)
    w = np.where(left, w, -w)
    return 1 / 3 + np.where(np.isclose(xi, np.pi, atol=1e-14), 0.0, w)


def rl2(u, X, Y, t):
    ue = exact(X, Y, t); return float(np.linalg.norm(u - ue) / np.linalg.norm(ue)), float(np.abs(u - ue).mean())


def centered(N):
    h = 4 * np.pi / N; g = np.arange(N) * h + h / 2; return np.meshgrid(g, g, indexing='ij')


out = {'athena': {}, 'opus': {}}
# ATHENA: every scored trial (iteration = Run + 1), field saved in burgers.npz
for team in ('closed', 'open'):
    for r in range(1, 6):
        for rep in sorted((SENS/team/'IB'/f'run{r}').rglob('Experiment_Report.md')):
            if '/box/' in str(rep) or not (rep.parent/'burgers.npz').exists(): continue
            m = re.search(r'## Full Trial History\s*\n(.*?)(?=\n##|\n###|$)', rep.read_text(), re.DOTALL)
            rows = [c for c in ([x.strip().replace('**', '') for x in l.split('|')[1:-1]] for l in m.group(1).split('\n')) if c and c[0].isdigit()]
            if not rows: continue
            z = np.load(rep.parent/'burgers.npz'); k = int(np.argmin(np.abs(z['t'] - 2.0)))
            X, Y = np.meshgrid(z['x'], z['y'], indexing='ij')
            e, _ = rl2(z['usol'][k], X, Y, z['t'][k]); it = int(rows[-1][0]) + 1
            out['athena'][f'{team}/{r}/{it}'] = e
            print(f'ATHENA {team:6s} run{r} it{it} score {rows[-1][-1]}: RL2(t=2) {e:.3e}')
# Opus 5.5 (bare Claude Code): final fields
B = SENS/'bare_claude'/'IB_request_only'
f = lambda r: B/f'run{r}'/'box'/'final'
z = np.load(f(1)/'snapshots_N512.npz'); out['opus']['1'] = rl2(z['t2.00'], *centered(512), 2.0)
z = np.load(f(2)/'solution_N512.npz'); out['opus']['2'] = rl2(z['t2.0'], *centered(512), 2.0)
z = np.load(f(3)/'snapshots_N1024.npz'); out['opus']['3'] = rl2(z['u_t2.0'], *np.meshgrid(z['x'], z['y'], indexing='ij'), 2.0)
z = np.load(f(4)/'solution_N512.npz'); T = list(np.round(z['times'], 6))
out['opus']['4'] = rl2(z['u'][T.index(2.0)], *np.meshgrid(z['x'], z['y'], indexing='ij'), 2.0)
# run 5 saved only errors.json; its N=512 scheme and reported errors equal run 4's, so its RMS L2 / RMS(exact)
j5 = json.loads((f(5)/'errors.json').read_text())['512']['2.0']
ue = exact(*centered(512), 2.0); out['opus']['5'] = (j5['L2'] / float(np.sqrt((ue ** 2).mean())), j5['L1'])
for r, (e, l1) in out['opus'].items(): print(f'Opus run{r}: RL2(t=2) {e:.3e}  (L1 {l1:.2e})')
out['opus'] = {r: v[0] for r, v in out['opus'].items()}
json.dump(out, open(Path(__file__).parent/'ib2d_rl2.json', 'w'), indent=1)
print('wrote ib2d_rl2.json')
