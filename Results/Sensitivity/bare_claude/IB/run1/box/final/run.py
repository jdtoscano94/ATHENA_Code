import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from burgers2d import solve, exact, L
res = {}
# 1) convergence in smooth regime (t=1 < t*=1.5) and after shock (t=2)
for T in (1.0, 2.0):
    res[T] = []
    for N in (64, 128, 256, 512):
        X, Y, u, _, dm = solve(N, T)
        ue = exact(X, Y, T); e = u - ue
        th = np.mod((X+Y)/2 - T/3, 2*np.pi)          # shock at phi=pi -> th=0 mod 2pi
        dist = np.abs(th - np.pi)   # shock at phi=pi
        away = dist > 0.5
        r = dict(N=N, L1=float(np.mean(abs(e))*L*L/ (L*L)), L2=float(np.sqrt(np.mean(e**2))),
                 Linf=float(abs(e).max()), Linf_away=float(abs(e[away]).max()), mass_err=float(dm),
                 umax=float(u.max()), umin=float(u.min()))
        res[T].append(r); print(T, r, flush=True)
json.dump({str(k): v for k, v in res.items()}, open('errors.json', 'w'), indent=1)
# 2) production run with snapshots
N = 512; times = [0.0, 0.5, 1.0, 1.5, 1.75, 2.0]
X, Y, u, snaps, dm = solve(N, 2.0, save_times=times)
np.savez_compressed('solution_N512.npz', x=X[:, 0], times=np.array(times), u=np.array([snaps[t] for t in times]))
fig = plt.figure(figsize=(18, 10))
for i, t in enumerate(times):
    ax = fig.add_subplot(3, 6, i+1)
    im = ax.imshow(snaps[t].T, origin='lower', extent=[0, L, 0, L], cmap='viridis', vmin=-1/3, vmax=1)
    ax.set_title(f't = {t}'); ax.set_xlabel('x'); ax.set_ylabel('y')
fig.colorbar(im, ax=fig.axes[:6], shrink=0.8, label='u')
# diagonal cuts y=x... use line y=0 (u depends on x+y), compare with exact
for i, t in enumerate(times):
    ax = fig.add_subplot(3, 6, 7+i)
    ax.plot(X[:, 0], snaps[t][:, 0], 'b-', lw=1.5, label='WENO5 N=512')
    ax.plot(X[:, 0], exact(X[:, 0], Y[:, 0], t), 'r--', lw=1, label='exact')
    ax.set_title(f'cut y={Y[0,0]:.3f}, t={t}'); ax.set_xlabel('x'); ax.set_ylim(-0.45, 1.1)
    if i == 0: ax.legend(fontsize=7)
for j, T in enumerate((1.0, 2.0)):
    ax = fig.add_subplot(3, 3, 7+j)
    Ns = [r['N'] for r in res[T]]
    for key in ('L1', 'L2', 'Linf_away'):
        ax.loglog(Ns, [r[key] for r in res[T]], 'o-', label=key)
    ax.set_title(f'error vs exact, t={T}'); ax.set_xlabel('N'); ax.legend(fontsize=8); ax.grid(True, which='both', alpha=.3)
ax = fig.add_subplot(3, 3, 9)
ue = exact(X, Y, 2.0)
im = ax.imshow(np.abs(u - ue).T, origin='lower', extent=[0, L, 0, L], cmap='magma', norm=matplotlib.colors.LogNorm(1e-10, 1))
ax.set_title('|u - u_exact| at t=2, N=512'); fig.colorbar(im, ax=ax)
fig.tight_layout() if False else None
plt.savefig('summary_all.png', dpi=110, bbox_inches='tight')
print('mass err', dm)
