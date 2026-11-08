"""PIML benchmarks of Table 1 as bar plots, in the style of Figure 7 (fig7_sensitivity.py, row 3).

Row 1: relative L2 error and trainable parameters for Human SOTA, Claude Code (Opus 5.5) and ATHENA.
Row 2: residual MSE and training time for Opus 5.5 and ATHENA (not reported by the human papers).
Bars as in Figure 7 (light fill, coloured edge, marker on the value; Helmholtz Opus = best of its five runs, run2).
Run from Plots/:  python fig_pinn_bars.py  ->  pinn_comparison_bars.png

Sources
- Rel. L2: Tables D7/D8 of the paper (Human SOTA = lowest-error human row per case, full field).
- Human parameters: vRBA [38] states 2,011 (3x30, periodic encoding); Muller & Zeinhofer [54] state 257;
  Urban et al. [51] (no count given): Helmholtz 2x20 on 4 periodic inputs = 541, KdV 3x30 on (t,x) = 1,981;
  IB: DPINN (Lei, Exposito, Mao, JCP 2026), best full-field human result in Table D8, Table 1 of their paper: rel L2 0.88%, 771 params.
- Opus parameters: sums of the saved parameter arrays (Poisson and Helmholtz counted from their code).
- ATHENA parameters: model.init of each final script (match the total_params printed in the run logs).
- MSE: mean squared raw PDE residual on ATHENA's evaluation grids (Table 1; Opus via opus_residual_mse.py).
- Training time: Opus from the run's own logs (results json / train logs); ATHENA from the tqdm logs of the
  plotted runs (Adam + SSBroyden). Opus AC, VB and IB shared the GPU with a second job they launched.
- KdV Opus: replay of KdV/run1 with identical code plus an error-trajectory hook (KdV_traj_replay; rel L2 1.07e-4,
  MSE 1.16e-7, 1398 s on its own GPU). The original run1 gave 4.44e-4, MSE 9.45e-8, 2238 s on a shared GPU.
"""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator

# same font configuration as fig7_sensitivity.py
BASE_FONTSIZE = 16
TITLE_FONTSIZE = BASE_FONTSIZE + 2
LABEL_FONTSIZE = BASE_FONTSIZE
LEGEND_FONTSIZE = BASE_FONTSIZE - 4
TICK_FONTSIZE = BASE_FONTSIZE - 4
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = BASE_FONTSIZE
mpl.rcParams['figure.dpi'] = 100

CASES = ['AC', 'VB', 'H', 'P', 'KdV', 'IB']   # Allen-Cahn, viscous Burgers, Helmholtz, Poisson, KdV, inviscid Burgers
ARMS = {   # label, colour, marker  (ATHENA black circle and Opus green triangle as in Figure 7)
    'human':  ('Human SOTA', 'tab:orange', 'D'),
    'opus':   ('Opus 5.5', 'green', '^'),
    'athena': ('ATHENA', 'black', 'o'),
}
# one value per case, or a list of runs (bar = first entry, the run reported in Table 1; markers = all runs)
H_OPUS_RL2 = 2.80e-5          # best of the five Helmholtz runs (run2), as in Table 1
H_OPUS_TIME = 113.5 / 60       # run2
DATA = {
    'rl2': {'human':  [2.14e-6, 1.51e-8, 3.60e-7, 1.0e-7, 6.00e-6, 8.80e-3],
            'opus':   [2.24e-4, 3.50e-5, H_OPUS_RL2, 1.77e-4, 1.07e-4, 2.07e-3],
            'athena': [2.07e-6, 7.87e-9, 7.25e-9, 8.13e-9, 7.92e-8, 5.65e-4]},
    'params': {'human':  [2011, 2011, 541, 257, 1981, 771],
               'opus':   [214529, 280321, 12865, 66177, 665610, 79233],
               'athena': [2071, 2071, 2161, 2231, 2231, 3456]},
    'mse': {'opus':   [3.73e-7, 7.54e-8, 2.08e-5, 1.30e-1, 1.16e-7, 8.53e-5],
            'athena': [8.11e-11, 2.64e-14, 7.25e-13, 3.95e-13, 1.22e-13, 6.95e-7]},
    'time': {'opus':   [2161 / 60, 1909 / 60, H_OPUS_TIME, 152.8 / 60, 1398 / 60, 1514 / 60],
             'athena': [127 / 60, 237 / 60, 140 / 60, 1176 / 60, 1752 / 60, 477 / 60]},
}


def plot_bars(ax, metric, ylabel, show_legend=False):
    arms = [a for a in ARMS if a in DATA[metric]]
    width = 0.27 if len(arms) == 3 else 0.36
    offs = {a: (i - (len(arms) - 1) / 2) * width for i, a in enumerate(arms)}
    rng = np.random.default_rng(0)
    for a in arms:
        lab, col, mk = ARMS[a]
        for k, v in enumerate(DATA[metric][a]):
            x0 = k + offs[a]
            vs = v if isinstance(v, list) else [v]
            ax.bar(x0, vs[0], width=width * 0.9, color=col, alpha=0.25, edgecolor=col, linewidth=1.2)
            for j, u in enumerate(vs):
                jx = x0 + (rng.uniform(-width * 0.25, width * 0.25) if len(vs) > 1 else 0.0)
                ax.plot(jx, u, marker=mk, ms=7 if mk == '^' else 6, ls='none', color=col, alpha=0.9)
            print(f'{metric:6s} {CASES[k]:4s} {a:6s} {vs}')
    ax.set_yscale('log')
    ax.set_xticks(range(len(CASES)))
    ax.set_xticklabels(CASES, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    if metric == 'time':
        ax.set_ylim(bottom=1.0)   # keep the 10^0 tick labelled
    if show_legend:
        h = [Line2D([], [], color=ARMS[a][1], marker=ARMS[a][2], ls='none', ms=7, label=ARMS[a][0]) for a in arms]
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 40)   # headroom above the data so the legend covers no marker
        ax.legend(handles=h, fontsize=LEGEND_FONTSIZE, framealpha=0.95, loc='upper center', ncols=3,
                  columnspacing=0.4, handletextpad=0.1, handlelength=1.0, borderpad=0.3, borderaxespad=0.3)


fig, axes = plt.subplots(2, 2, figsize=(12, 7))
plot_bars(axes[0, 0], 'rl2', r'Relative $L^2$ error', show_legend=True)
plot_bars(axes[0, 1], 'params', 'Trainable parameters')
plot_bars(axes[1, 0], 'mse', 'Residual MSE', show_legend=True)
plot_bars(axes[1, 1], 'time', 'Training time (min)')
plt.tight_layout()
plt.savefig('pinn_comparison_bars.png', dpi=300, bbox_inches='tight')
print('saved pinn_comparison_bars.png')
