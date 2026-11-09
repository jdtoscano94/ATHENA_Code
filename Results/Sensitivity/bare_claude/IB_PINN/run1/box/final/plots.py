import sys, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
run, ref, out = sys.argv[1], sys.argv[2], sys.argv[3]
d = np.load(ref); t, x, u = d['t'], d['x'], d['usol']; p = np.load(f'{run}/upred.npy'); h = np.load(f'{run}/hist.npy')
INK, INK2, SURF, BLUE, ORANGE = '#0b0b0b', '#52514e', '#fcfcfb', '#2a78d6', '#eb6834'
div = LinearSegmentedColormap.from_list('div', ['#104281', '#2a78d6', '#f0efec', '#eb6834', '#9a3412'])
seq = LinearSegmentedColormap.from_list('seq', ['#f0efec', '#86b6ef', '#2a78d6', '#104281'])
plt.rcParams.update({'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
for a_, f, ttl, cm, vr in [(ax[0], u, 'Reference (exact entropy solution)', div, (-1, 1)), (ax[1], p, 'PINN prediction', div, (-1, 1)), (ax[2], np.abs(p-u), '|PINN − reference|', seq, (0, None))]:
    im = a_.pcolormesh(x, t, f, cmap=cm, vmin=vr[0], vmax=vr[1], shading='auto', rasterized=True)
    a_.set_title(ttl, color=INK); a_.set_xlabel('x'); a_.set_ylabel('t'); fig.colorbar(im, ax=a_)
fig.savefig(f'{out}/fields.png', dpi=150)
fig, ax = plt.subplots(1, 5, figsize=(17, 3.4), sharey=True, constrained_layout=True)
for a_, ti in zip(ax, [0, 50, 63, 120, 199]):
    a_.plot(x, u[ti], color=INK2, lw=2, label='Reference'); a_.plot(x, p[ti], color=BLUE, lw=2, ls='--', label='PINN')
    a_.set_title(f't = {t[ti]:.3f}', color=INK); a_.set_xlabel('x'); a_.grid(alpha=0.25)
ax[0].set_ylabel('u'); ax[0].legend(frameon=False)
fig.savefig(f'{out}/slices.png', dpi=150)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6), constrained_layout=True)
ax[0].semilogy(h[:, 0], h[:, 1], color=BLUE, lw=2); ax[0].set_title('Training loss (weighted, causal)', color=INK); ax[0].set_xlabel('iteration')
ax[1].semilogy(h[:, 0], h[:, 2], color=BLUE, lw=2); ax[1].set_title('Relative L2 error vs reference', color=INK); ax[1].set_xlabel('iteration')
et = np.linalg.norm(p-u, axis=1)/np.maximum(np.linalg.norm(u, axis=1), 1e-12)
ax[2].semilogy(t[1:], et[1:], color=BLUE, lw=2); ax[2].axvline(1/np.pi, color=INK2, ls=':', lw=1); ax[2].text(1/np.pi+0.01, et[1:].max(), 'shock forms t=1/π', color=INK2, va='top')
ax[2].set_title('Relative L2 error per time slice', color=INK); ax[2].set_xlabel('t')
for a_ in ax: a_.grid(alpha=0.25)
fig.savefig(f'{out}/training_and_error.png', dpi=150)
m = np.abs(x) > 0.05
print(f"relL2 {np.linalg.norm(p-u)/np.linalg.norm(u):.4e}  maxabs {np.abs(p-u).max():.4e}  MAE {np.abs(p-u).mean():.4e}  relL2(|x|>0.05) {np.linalg.norm((p-u)[:, m])/np.linalg.norm(u[:, m]):.4e}  worst-slice t={t[et.argmax()]:.3f} relL2 {et.max():.3e}")
