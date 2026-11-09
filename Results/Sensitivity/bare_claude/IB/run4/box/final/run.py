import numpy as np, json, time, sys
sys.path.insert(0, '/box/final')
from burgers2d import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

times = [0.0, 0.5, 1.0, 1.5, 1.75, 2.0]
res = {}
for N in [32, 64, 128, 256, 512]:
    t0 = time.time(); out = solve(N, 2.0, snaps=[0.5, 1.0, 1.5, 1.75]); el = time.time()-t0
    res[N] = {}
    for t in [1.0, 2.0]:
        xc = (np.arange(N)+0.5)*L/N; ue = exact(xc[:, None], xc[None, :], t); e = out[t]-ue
        res[N][t] = dict(L1=float(np.abs(e).mean()), L2=float(np.sqrt((e**2).mean())), Linf=float(np.abs(e).max()))
    xc = (np.arange(N)+0.5)*L/N; X, Y = xc[:, None], xc[None, :]
    zs = np.mod((X+Y)/2 - 2/3 + np.pi, 2*np.pi) - np.pi; dist = (np.pi-np.abs(zs))*np.sqrt(2)  # distance to shock line
    e = out[2.0]-exact(X, Y, 2.0)
    for D in [0.25, 1.0]:
        m = dist > D
        res[N][f'away{D}'] = dict(L1=float(np.abs(e[m]).sum()/e.size), Linf=float(np.abs(e[m]).max()))
    mass_err = float(abs(out[2.0].mean() - 1/3))
    res[N]['mass_err'] = mass_err; res[N]['umin'] = float(out[2.0].min()); res[N]['umax'] = float(out[2.0].max())
    print(N, f'{el:.1f}s', res[N], flush=True)
    if N == 512: keep = out
Ns = sorted(res)
for t in [1.0, 2.0]:
    for k in ['L1', 'L2', 'Linf']:
        rates = [np.log2(res[Ns[i]][t][k]/res[Ns[i+1]][t][k]) for i in range(len(Ns)-1)]
        print(f't={t} {k} rates:', np.round(rates, 2))
for D in [0.25, 1.0]:
    for k in ['L1', 'Linf']:
        rates = [np.log2(res[Ns[i]][f'away{D}'][k]/res[Ns[i+1]][f'away{D}'][k]) for i in range(len(Ns)-1)]
        print(f't=2 dist>{D} {k} rates:', np.round(rates, 2))
json.dump({str(k): {str(a): b for a, b in v.items()} for k, v in res.items()}, open('/box/final/errors.json', 'w'), indent=1)
np.savez_compressed('/box/final/solution_N512.npz', **{f't{t}': keep[t] for t in keep})

# ---- summary plot ----
N = 512; h = L/N; xc = (np.arange(N)+0.5)*h
fig, axs = plt.subplots(3, 3, figsize=(16, 15))
fields = {0.0: u0(xc[:, None], xc[None, :]), **keep}
for ax, t in zip(axs.flat[:6], times):
    im = ax.imshow(fields[t].T, origin='lower', extent=[0, L, 0, L], cmap='RdBu_r', vmin=-1/3, vmax=1)
    ax.set_title(f'u(x,y,t={t})  (N={N})'); ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.colorbar(im, ax=ax, fraction=0.046)
# diagonal cut x=y
ax = axs[2, 0]; xf = np.linspace(0, L, 2000)
for t in times:
    ax.plot(xc, np.diagonal(fields[t]), lw=1.5, label=f't={t}')
    ax.plot(xf, exact(xf, xf, t), 'k:', lw=0.8)
ax.set_title('Diagonal cut x=y: WENO5 (colour) vs exact (black dotted)'); ax.set_xlabel('x=y'); ax.legend(fontsize=8)
# error at t=2
ax = axs[2, 1]; e = keep[2.0] - exact(xc[:, None], xc[None, :], 2.0)
im = ax.imshow(np.log10(np.abs(e)+1e-16).T, origin='lower', extent=[0, L, 0, L], cmap='magma', vmin=-12, vmax=0)
ax.set_title('log10 |u - u_exact| at t=2'); fig.colorbar(im, ax=ax, fraction=0.046)
ax = axs[2, 2]
for t, m in [(1.0, 'o-'), (2.0, 's-')]:
    ax.loglog(Ns, [res[n][t]['L1'] for n in Ns], m, label=f'L1 error t={t}')
ax.loglog(Ns, [res[n]['away0.25']['L1'] for n in Ns], '^-', label='L1 t=2, dist to shock>0.25')
ax.loglog(Ns, [res[Ns[0]][1.0]['L1']*(Ns[0]/n)**5 for n in Ns], 'k--', label='O(h^5)')
ax.loglog(Ns, [res[Ns[0]][2.0]['L1']*(Ns[0]/n)**1 for n in Ns], 'k:', label='O(h)')
ax.set_xlabel('N'); ax.set_title('Grid convergence vs exact solution'); ax.legend(); ax.grid(True, which='both', alpha=.3)
plt.tight_layout(); plt.savefig('/box/final/summary_all.png', dpi=110)
