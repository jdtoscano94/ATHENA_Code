"""Figure 5(A) (burgers_comparison_final_v3.png) with the single-agent baseline added.

Code copied from Plots/PINNs.ipynb cell 3 (same data, style and size). The only addition is the
single-agent run (Claude Code + Claude Opus 5.5, Results/Sensitivity/bare_claude/IB_PINN/run1) as a
green line in the relative L2 error panel. It trains at nu = 0 throughout (no continuation), so it has no
line in the viscosity panel; it appears there only as a legend entry. Its error is measured against the
same reference as A and A+H (identical u_exact, checked: max difference 0).
Run from Plots/:  python fig5A_burgers_opus.py   ->  burgers_comparison_final_v3_opus.png
"""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ============================================================================
# 1. SETUP & STYLE
# ============================================================================
BASE_FONTSIZE = 14
TITLE_FONTSIZE = BASE_FONTSIZE
LABEL_FONTSIZE = BASE_FONTSIZE
LEGEND_FONTSIZE = BASE_FONTSIZE - 2
TICK_FONTSIZE = BASE_FONTSIZE - 2
COLORBAR_FONTSIZE = BASE_FONTSIZE - 2

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = BASE_FONTSIZE
mpl.rcParams['figure.dpi'] = 300

path_case_a = '../Results/PINN_codes/Burgers_Periodic_SinPiX/2025_12_04_1621_human_IB/burgers_pred_complete.npz'
path_case_ah = '../Results/PINN_codes/Burgers_Periodic_SinPiX/2025_11_25_0148_wavelets_lower/viscosity_results_complete.npz'
path_opus = '../Results/Sensitivity/bare_claude/IB_PINN/run1/box/final/model/hist.npy'   # columns: step, loss, relL2, nu, wmin

# ============================================================================
# 2. LOAD DATA
# ============================================================================
data_a = np.load(path_case_a)
a_iter = data_a['iter_history']
a_error = data_a['error_history']

data_ah = np.load(path_case_ah)
ah_iter = data_ah['iter_history']
ah_error = data_ah['error_history']
ah_viscosity = data_ah['viscosity_history']
ah_T = data_ah['T']
ah_X = data_ah['X']
ah_u_pred = data_ah['u_pred']
ah_u_exact = data_ah['u_exact']
ah_pointwise_error = np.abs(ah_u_exact - ah_u_pred)
ah_iter_visc = ah_iter[-len(ah_viscosity):]

opus = np.load(path_opus)
opus_iter, opus_error = opus[:, 0], opus[:, 2]

# ============================================================================
# 3. RECONSTRUCT VISCOSITY FOR CASE A
# ============================================================================
continuation_schedule = [
    {'nu': 0.005, 'dwell': 5000},
    {'nu': 0.002, 'dwell': 10000},
    {'nu': 0.001, 'dwell': 15000},
    {'nu': 0.0005, 'dwell': 20000},
    {'nu': 0.00035, 'dwell': 20000},
    {'nu': 0.0002, 'dwell': 30000},
    {'nu': 0.0001, 'dwell': 80000},
]
total_scheduled_iters = sum(stage['dwell'] for stage in continuation_schedule)
max_iter = max(a_iter.max(), total_scheduled_iters) + 1
full_visc_schedule = np.zeros(int(max_iter))
current_idx = 0
for stage in continuation_schedule:
    end_idx = current_idx + stage['dwell']
    full_visc_schedule[current_idx:end_idx] = stage['nu']
    current_idx = end_idx
if current_idx < len(full_visc_schedule):
    full_visc_schedule[current_idx:] = continuation_schedule[-1]['nu']
a_visc_history = full_visc_schedule[a_iter.astype(int)]

# ============================================================================
# 4. PLOTTING
# ============================================================================
color_a = 'blue'
color_ah = 'red'
color_opus = 'green'

fig = plt.figure(figsize=(12, 3))
gs = fig.add_gridspec(1, 4, wspace=0.35, top=0.85, bottom=0.15, left=0.05, right=0.96)

# --- COLUMN 1: RELATIVE ERROR ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(a_iter, a_error, color=color_a, linestyle='-', linewidth=1.5, label='A', alpha=0.8)
ax1.plot(ah_iter, ah_error, color=color_ah, linestyle='-', linewidth=1.5, label='A+H')
ax1.plot(opus_iter, opus_error, color=color_opus, linestyle='-', linewidth=1.5, label='Opus 5.5')
ax1.set_yscale('log')
ax1.set_ylim(top=1e2)   # headroom for the single-column legend
ax1.yaxis.set_major_locator(LogLocator(base=10, numticks=12))   # a label every decade
ax1.legend(fontsize=LEGEND_FONTSIZE, loc='upper right', ncols=1, handlelength=1.2, handletextpad=0.3, borderpad=0.3)
ax1.set_xlabel('Gradient steps', fontsize=LABEL_FONTSIZE)
ax1.set_title('Relative $L^2$ Error', fontsize=TITLE_FONTSIZE)
ax1.grid(True, which="both", ls="--", alpha=0.4)
ax1.tick_params(labelsize=TICK_FONTSIZE)

# --- COLUMN 2: VISCOSITY SCHEMES (LOG SCALE) ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(a_iter, a_visc_history, color=color_a, linestyle='-', linewidth=1.5, label='A')
ax2.plot(ah_iter_visc, ah_viscosity, color=color_ah, linestyle='-', linewidth=1.5, label='A+H')
ax2.set_xlabel('Gradient steps', fontsize=LABEL_FONTSIZE)
ax2.set_title(r'Viscosity $\nu$', fontsize=TITLE_FONTSIZE)
ax2.set_yscale('log')
ax2.grid(True, ls="--", alpha=0.4, which="both")
ax2.tick_params(labelsize=TICK_FONTSIZE)

# --- COLUMN 3: PREDICTION (Case A+H) ---
ax3 = fig.add_subplot(gs[0, 2])
levels_pred = np.linspace(ah_u_pred.min(), ah_u_pred.max(), 100)
contour3 = ax3.contourf(ah_T, ah_X, ah_u_pred, cmap='jet', levels=levels_pred, extend='both')
ax3.set_title('Prediction (A+H)', fontsize=TITLE_FONTSIZE)
ax3.set_xlabel('$t$', fontsize=LABEL_FONTSIZE)
ax3.set_ylabel('$x$', fontsize=LABEL_FONTSIZE)
ax3.set_xticks([])
ax3.set_yticks([])
divider3 = make_axes_locatable(ax3)
cax3 = divider3.append_axes("right", size="5%", pad=0.05)
cbar3 = fig.colorbar(contour3, cax=cax3)
cbar3.ax.tick_params(labelsize=COLORBAR_FONTSIZE)

# --- COLUMN 4: POINTWISE ERROR ---
ax4 = fig.add_subplot(gs[0, 3])
err_safe = np.maximum(ah_pointwise_error, 1e-16)
norm_err = colors.LogNorm(vmin=np.median(err_safe), vmax=err_safe.max())
mesh4 = ax4.pcolormesh(ah_T, ah_X, err_safe, cmap='jet', norm=norm_err, shading='gouraud')
ax4.set_title('Error (A+H)', fontsize=TITLE_FONTSIZE)
ax4.set_xlabel('$t$', fontsize=LABEL_FONTSIZE)
ax4.set_xticks([])
ax4.set_yticks([])
divider4 = make_axes_locatable(ax4)
cax4 = divider4.append_axes("right", size="5%", pad=0.05)
cbar4 = fig.colorbar(mesh4, cax=cax4)
cbar4.ax.tick_params(labelsize=COLORBAR_FONTSIZE)

# ============================================================================
# 5. SAVE
# ============================================================================
output_file = 'burgers_comparison_final_v3_opus.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Figure saved as '{output_file}'")
