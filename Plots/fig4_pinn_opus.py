"""Figure 4 (pde_comparison_5cases.png = paper Figures/PINN_bm.png) with the Opus 5.5 baseline added.

Code copied from Plots/PINNs.ipynb cell 2 (same data, style and size). Changes: ATHENA's error curve is
blue (as ATHENA in Figure 5A) and the Claude Code + Opus 5.5 run is a green line in column 1, as in Figures 5A and 7.
Opus histories (Results/Sensitivity/bare_claude/<case>/run*/box/final), iterations counted cumulatively over
Adam then L-BFGS:
  Allen-Cahn  Allen_Cahn/run1 final_s0_hist.json            (final 2.24e-4)
  Burgers     VB_PINN/run1 pinn_hist_B.json                 (final 3.51e-5)
  Helmholtz   Helmholtz/run2 results_w64_s0_m100.json       (seed 0, final 2.73e-5; best of the 5 runs)
  Poisson     Poisson/run1 results.json                     (final 1.77e-4)
  KdV         KdV_traj_replay/box/traj_A.npy (replay of KdV/run1, identical code plus an error-logging hook):
              time marching over 10 windows of 20k iterations; the curve is the relative L2 error over the full
              domain t in [0, 5] every 250 iterations, with windows not yet trained counted as u_pred = 0;
              final 1.07e-4 (original run1: 4.44e-4).
Run from Plots/:  python fig4_pinn_opus.py   ->  pde_comparison_5cases_opus.png
"""
import matplotlib as mpl
mpl.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors

# Set font and style parameters
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.dpi'] = 100

# ## Load Data from All Four Cases
print("Loading data...")

# Allen-Cahn
allen_cahn_path = '../Results/PINN_codes/AllenCahn_Periodic_X2CosPiX/2025_12_05_1208_human/complete_results.npz'
allen_cahn_data = np.load(allen_cahn_path)
print(f"Allen-Cahn variables: {list(allen_cahn_data.keys())}")

# Burgers
# Burgers: ATHENA autonomous (bare-problem) run. Min RL2 7.9e-9, final ~1e-8 (flat noisy band, loss ~1e-14).
# Matches the curated 1236_human run (min 7.9e-9) with a better final/pointwise at ~half the SSBroyden iters (~99k vs ~196k).
burgers_path = '../Results/PINN_codes/Burgers_Periodic_SinPi/2026_06_17_1749/solution_data.npz'
burgers_data = np.load(burgers_path)
print(f"Burgers variables: {list(burgers_data.keys())}")

# Helmholtz
helmholtz_path = '../Results/PINN_codes/Helmholtz_Periodic_SinPiA1SinPiA2/2025_12_05_1421/validation_data.npz'
helmholtz_data = np.load(helmholtz_path)
print(f"Helmholtz variables: {list(helmholtz_data.keys())}")

# Poisson
poisson_path = '../Results/PINN_codes/Poisson_Dirichlet_Sin4pixSin4piy/2026_01_10_2319_plots/Poisson.npz'
poisson_data = np.load(poisson_path)
print(f"Poisson variables: {list(poisson_data.keys())}")

# KdV
kdv_path = '../Results/PINN_codes/KdV_Mixed_DerivedIC/2026_01_10_2026_human/Kdv.npz'
kdv_data = np.load(kdv_path)
print(f"KdV variables: {list(kdv_data.keys())}")

# ## Extract Allen-Cahn Data
ac_T = allen_cahn_data['T']
ac_X = allen_cahn_data['X']
ac_u_pred = allen_cahn_data['u_pred']
ac_u_exact = allen_cahn_data['u_exact']
ac_residual = allen_cahn_data['pde_residual_grid']
ac_iter = allen_cahn_data['iter_history']
ac_loss = allen_cahn_data['loss_history']
ac_errors = allen_cahn_data['error_history']
ac_pointwise_error = np.abs(ac_u_exact - ac_u_pred)

print(f"Allen-Cahn shapes - T: {ac_T.shape}, u_pred: {ac_u_pred.shape}")

# ## Extract Burgers Data
b_T = burgers_data['T']
b_X = burgers_data['X']
b_u_pred = burgers_data['u_pred']
b_u_exact = burgers_data['u_exact']
b_residual = burgers_data['pde_residual_grid']
b_iter = burgers_data['iter_history']
b_loss = burgers_data['loss_history']
b_errors = burgers_data['error_history']
b_pointwise_error = np.abs(b_u_exact - b_u_pred)

print(f"Burgers shapes - T: {b_T.shape}, u_pred: {b_u_pred.shape}")

# ## Extract Helmholtz Data
h_X = helmholtz_data['X']
h_Y = helmholtz_data['Y']
h_U_pred = helmholtz_data['u_pred']
h_U_exact = helmholtz_data['u_exact']
h_residual = helmholtz_data['pde_residual_grid']
h_iter = helmholtz_data['iter_history']
h_loss = helmholtz_data['loss_history']
h_errors = helmholtz_data['error_history']
h_pointwise_error = helmholtz_data['pointwise_error']

print(f"Helmholtz shapes - X: {h_X.shape}, u_pred: {h_U_pred.shape}")

# ## Extract Poisson Data
p_X = poisson_data['X']
p_Y = poisson_data['Y']
p_u_pred = poisson_data['u_pred']
p_u_exact = poisson_data['u_exact']
p_residual = poisson_data['pde_residual_grid']
p_iter = poisson_data['iter_history']
p_loss = poisson_data['loss_history']
p_errors = poisson_data['error_history']
p_pointwise_error = np.abs(p_u_exact - p_u_pred)

print(f"Poisson shapes - X: {p_X.shape}, u_pred: {p_u_pred.shape}")

# ## Extract KdV Data
kdv_T = kdv_data['T']
kdv_X = kdv_data['X']
kdv_u_pred = kdv_data['u_pred']
kdv_u_exact = kdv_data['u_exact']
kdv_residual = kdv_data['pde_residual_grid'].T
kdv_iter = kdv_data['iter_history']
kdv_loss = kdv_data['loss_history']
kdv_errors = kdv_data['error_history']
kdv_pointwise_error = np.abs(kdv_u_exact - kdv_u_pred)

print(f"KdV shapes - T: {kdv_T.shape}, u_pred: {kdv_u_pred.shape}, residual: {kdv_residual.shape}")

# Check if transpose is needed for KdV
if kdv_T.shape != kdv_u_pred.shape:
    print(f"WARNING: KdV mesh shape mismatch. Transposing data...")
    kdv_u_pred = kdv_u_pred.T
    kdv_u_exact = kdv_u_exact.T
    kdv_residual = kdv_residual.T
    kdv_pointwise_error = kdv_pointwise_error.T
    print(f"After transpose - u_pred: {kdv_u_pred.shape}, residual: {kdv_residual.shape}")


# ## Opus 5.5 baseline error histories
import json
from matplotlib.lines import Line2D
_B = '../Results/Sensitivity/bare_claude/'
def _cum(stages_its):   # cumulative iteration over consecutive optimiser stages (Adam, then L-BFGS)
    out, off, prev_stage, last = [], 0, None, 0
    for st, it in stages_its:
        if prev_stage is not None and st != prev_stage:
            off += last
        out.append(off + it); prev_stage, last = st, it
    return np.array(out)
_h = json.load(open(_B + 'Allen_Cahn/run1/box/final/final_s0_hist.json'))['hist']
opus_ac = (np.array([r['it'] for r in _h]), np.array([r['rel_l2'] for r in _h]))
_h = json.load(open(_B + 'VB_PINN/run1/box/final/pinn_hist_B.json'))['hist']
opus_b = (np.array([r['it'] for r in _h]), np.array([r['relL2'] for r in _h]))
_h = json.load(open(_B + 'Helmholtz/run2/box/final/results_w64_s0_m100.json'))['history']   # [stage, it, loss, relL2, maxerr]
opus_h = (_cum([(r[0], r[1]) for r in _h]), np.array([r[3] for r in _h]))
_h = json.load(open(_B + 'Poisson/run1/box/final/results.json'))['history']
opus_p = (_cum([(r['stage'], r['it']) for r in _h]), np.array([r['rel_l2'] for r in _h]))
# KdV: full-field relative L2 error over the whole domain t in [0, 5] during training, from the replay of
# KdV/run1 (identical code + logging hook). traj_A.npy (every 250 iterations): global iter, window k,
# relL2 over the windows marched so far (j <= k), relL2 of window k. Same 401 x 26 grid per window as the log.
# Windows not trained yet (j > k) have no prediction and count as u_pred = 0.
import sys as _sys
_KB = _B + 'KdV_traj_replay/box'
_sys.path.insert(0, _KB)
import jax.numpy as _jnp
from lib import uex as _uex, L as _L
_U = []
for _w in range(10):
    _X, _TT = np.meshgrid(np.linspace(0, _L, 401), np.linspace(0.5 * _w, 0.5 * (_w + 1), 26))
    _ue = np.array(_uex(_jnp.array(_X.ravel(), _jnp.float32), _jnp.array(_TT.ravel(), _jnp.float32)))
    _U.append(float(np.sum(_ue ** 2)))
_U = np.array(_U); _C = np.cumsum(_U)
_tr = np.load(_KB + '/traj_A.npy'); _kw = _tr[:, 1].astype(int)
opus_kdv = (_tr[:, 0], np.sqrt((_tr[:, 2] ** 2 * _C[_kw] + (_U.sum() - _C[_kw])) / _U.sum()))
opus_color = 'green'
for _n, (_i, _v) in zip(['AC', 'B', 'H', 'P', 'KdV'], [opus_ac, opus_b, opus_h, opus_p, opus_kdv]):
    print(f"Opus {_n}: {len(_i)} points, last iter {_i[-1]}, final relL2 {_v[-1]:.3e}")

# ## Create Combined 5×4 Visualization (TRANSPOSED)
# Rows = Equations (Allen-Cahn, Burgers, Helmholtz, Poisson, KdV)
# Columns = Fields (Error History, Residuals, Prediction, Pointwise Error)

fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(5, 4, hspace=0.35, wspace=0.35, top=0.96, bottom=0.05, 
                      left=0.08, right=0.98)

# Define consistent styling
error_color = 'blue'

# Column labels (will be added as titles in first row)
col_labels = ['Relative $L^2$ Error', 'Residuals', 'Prediction', 'Pointwise Error']

# Row labels (equation names, will be added as y-labels in first column)
row_labels = ['Allen-Cahn', 'Burgers', 'Helmholtz', 'Poisson', 'KdV']

# ============================================================================
# ROW 1: ALLEN-CAHN
# ============================================================================

# Column 1: Error History
ax_ac_err = fig.add_subplot(gs[0, 0])
ax_ac_err.plot(ac_iter, ac_errors, color=error_color, linewidth=2)
ax_ac_err.plot(*opus_ac, color=opus_color, linewidth=1.5)
ax_ac_err.set_yscale('log')
ax_ac_err.set_title(col_labels[0], fontsize=14, pad=10)
#ax_ac_err.set_xlabel('Iterations')
ax_ac_err.set_ylabel(row_labels[0], fontsize=14, )
ax_ac_err.grid(True, which="both", ls="--", alpha=0.6)
ax_ac_err.legend(handles=[Line2D([], [], color=error_color, lw=2), Line2D([], [], color=opus_color, lw=1.5)],
                 labels=['ATHENA', 'Opus 5.5'], fontsize=10, loc='upper right', framealpha=0.9)

# Column 2: Residuals
ax_ac_res = fig.add_subplot(gs[0, 1])
ac_res_safe = np.maximum(np.abs(ac_residual), 1e-16)
ac_res_norm = colors.LogNorm(vmin=ac_res_safe.min(), vmax=ac_res_safe.max())
mesh_ac_res = ax_ac_res.pcolormesh(ac_T, ac_X, ac_res_safe, cmap='jet', 
                                    norm=ac_res_norm, shading='gouraud')
cbar_ac_res = fig.colorbar(mesh_ac_res, ax=ax_ac_res)
ax_ac_res.set_title(col_labels[1], fontsize=14, pad=10)
ax_ac_res.set_xlabel('')
ax_ac_res.set_ylabel('')
ax_ac_res.set_xticks([])
ax_ac_res.set_yticks([])

# Column 3: Prediction
ax_ac_pred = fig.add_subplot(gs[0, 2])
ac_vmin, ac_vmax = ac_u_pred.min(), ac_u_pred.max()
ac_levels = np.linspace(ac_vmin, ac_vmax, 100)
contour_ac_pred = ax_ac_pred.contourf(ac_T, ac_X, ac_u_pred, cmap='jet', 
                                       levels=ac_levels, extend='both')
cbar_ac_pred = fig.colorbar(contour_ac_pred, ax=ax_ac_pred)
ax_ac_pred.set_title(col_labels[2], fontsize=14, pad=10)
ax_ac_pred.set_xlabel('')
ax_ac_pred.set_ylabel('')
ax_ac_pred.set_xticks([])
ax_ac_pred.set_yticks([])

# Column 4: Pointwise Error
ax_ac_perr = fig.add_subplot(gs[0, 3])
ac_perr_safe = np.maximum(ac_pointwise_error, 1e-16)
ac_perr_norm = colors.LogNorm(vmin=np.median(ac_perr_safe), vmax=ac_perr_safe.max())
mesh_ac_perr = ax_ac_perr.pcolormesh(ac_T, ac_X, ac_perr_safe, cmap='jet', 
                                      norm=ac_perr_norm, shading='gouraud')
cbar_ac_perr = fig.colorbar(mesh_ac_perr, ax=ax_ac_perr)
ax_ac_perr.set_title(col_labels[3], fontsize=14, pad=10)
ax_ac_perr.set_xlabel('')
ax_ac_perr.set_ylabel('')
ax_ac_perr.set_xticks([])
ax_ac_perr.set_yticks([])

# ============================================================================
# ROW 2: BURGERS
# ============================================================================

# Column 1: Error History
ax_b_err = fig.add_subplot(gs[1, 0])
ax_b_err.plot(b_iter, b_errors, color=error_color, linewidth=2)
ax_b_err.plot(*opus_b, color=opus_color, linewidth=1.5)
ax_b_err.set_yscale('log')
#ax_b_err.set_xlabel('Iterations')
ax_b_err.set_ylabel(row_labels[1], fontsize=14, )
ax_b_err.grid(True, which="both", ls="--", alpha=0.6)

# Column 2: Residuals
ax_b_res = fig.add_subplot(gs[1, 1])
b_res_safe = np.maximum(np.abs(b_residual), 1e-16)
b_res_norm = colors.LogNorm(vmin=b_res_safe.min(), vmax=b_res_safe.max())
mesh_b_res = ax_b_res.pcolormesh(b_T, b_X, b_res_safe, cmap='jet', 
                                  norm=b_res_norm, shading='gouraud')
cbar_b_res = fig.colorbar(mesh_b_res, ax=ax_b_res)
ax_b_res.set_xlabel('')
ax_b_res.set_ylabel('')
ax_b_res.set_xticks([])
ax_b_res.set_yticks([])

# Column 3: Prediction
ax_b_pred = fig.add_subplot(gs[1, 2])
b_vmin, b_vmax = b_u_pred.min(), b_u_pred.max()
b_levels = np.linspace(b_vmin, b_vmax, 100)
contour_b_pred = ax_b_pred.contourf(b_T, b_X, b_u_pred, cmap='jet', 
                                     levels=b_levels, extend='both')
cbar_b_pred = fig.colorbar(contour_b_pred, ax=ax_b_pred)
ax_b_pred.set_xlabel('')
ax_b_pred.set_ylabel('')
ax_b_pred.set_xticks([])
ax_b_pred.set_yticks([])

# Column 4: Pointwise Error
ax_b_perr = fig.add_subplot(gs[1, 3])
b_perr_safe = np.maximum(b_pointwise_error, 1e-16)
b_perr_norm = colors.LogNorm(vmin=np.median(b_perr_safe), vmax=b_perr_safe.max())
mesh_b_perr = ax_b_perr.pcolormesh(b_T, b_X, b_perr_safe, cmap='jet', 
                                    norm=b_perr_norm, shading='gouraud')
cbar_b_perr = fig.colorbar(mesh_b_perr, ax=ax_b_perr)
ax_b_perr.set_xlabel('')
ax_b_perr.set_ylabel('')
ax_b_perr.set_xticks([])
ax_b_perr.set_yticks([])

# ============================================================================
# ROW 3: HELMHOLTZ
# ============================================================================

# Column 1: Error History
ax_h_err = fig.add_subplot(gs[2, 0])
ax_h_err.plot(h_iter, h_errors, color=error_color, linewidth=2)
ax_h_err.plot(*opus_h, color=opus_color, linewidth=1.5)
ax_h_err.set_yscale('log')
#ax_h_err.set_xlabel('Iterations')
ax_h_err.set_ylabel(row_labels[2], fontsize=14, )
ax_h_err.grid(True, which="both", ls="--", alpha=0.6)

# Column 2: Residuals
ax_h_res = fig.add_subplot(gs[2, 1])
h_res_safe = np.maximum(np.abs(h_residual), 1e-16)
h_res_norm = colors.LogNorm(vmin=h_res_safe.min(), vmax=h_res_safe.max())
mesh_h_res = ax_h_res.pcolormesh(h_X, h_Y, h_res_safe, cmap='jet', 
                                  norm=h_res_norm, shading='gouraud')
cbar_h_res = fig.colorbar(mesh_h_res, ax=ax_h_res)
ax_h_res.set_xlabel('')
ax_h_res.set_ylabel('')
ax_h_res.set_xticks([])
ax_h_res.set_yticks([])

# Column 3: Prediction
ax_h_pred = fig.add_subplot(gs[2, 2])
h_vmin, h_vmax = h_U_pred.min(), h_U_pred.max()
h_levels = np.linspace(h_vmin, h_vmax, 100)
contour_h_pred = ax_h_pred.contourf(h_X, h_Y, h_U_pred, cmap='jet', 
                                     levels=h_levels, extend='both')
cbar_h_pred = fig.colorbar(contour_h_pred, ax=ax_h_pred)
ax_h_pred.set_xlabel('')
ax_h_pred.set_ylabel('')
ax_h_pred.set_xticks([])
ax_h_pred.set_yticks([])

# Column 4: Pointwise Error
ax_h_perr = fig.add_subplot(gs[2, 3])
h_perr_safe = np.maximum(np.abs(h_pointwise_error), 1e-16)
h_perr_norm = colors.LogNorm(vmin=np.median(h_perr_safe), vmax=h_perr_safe.max())
mesh_h_perr = ax_h_perr.pcolormesh(h_X, h_Y, h_perr_safe, cmap='jet', 
                                    norm=h_perr_norm, shading='gouraud')
cbar_h_perr = fig.colorbar(mesh_h_perr, ax=ax_h_perr)
ax_h_perr.set_xlabel('')
ax_h_perr.set_ylabel('')
ax_h_perr.set_xticks([])
ax_h_perr.set_yticks([])

# ============================================================================
# ROW 4: POISSON
# ============================================================================

# Column 1: Error History
ax_p_err = fig.add_subplot(gs[3, 0])
ax_p_err.plot(p_iter, p_errors, color=error_color, linewidth=2)
ax_p_err.plot(*opus_p, color=opus_color, linewidth=1.5)
ax_p_err.set_yscale('log')
#ax_p_err.set_xlabel('Iterations')
ax_p_err.set_ylabel(row_labels[3], fontsize=14, )
ax_p_err.grid(True, which="both", ls="--", alpha=0.6)

# Column 2: Residuals
ax_p_res = fig.add_subplot(gs[3, 1])
p_res_safe = np.maximum(np.abs(p_residual), 1e-16)
p_res_norm = colors.LogNorm(vmin=p_res_safe.min(), vmax=p_res_safe.max())
mesh_p_res = ax_p_res.pcolormesh(p_X, p_Y, p_res_safe, cmap='jet', 
                                  norm=p_res_norm, shading='gouraud')
cbar_p_res = fig.colorbar(mesh_p_res, ax=ax_p_res)
ax_p_res.set_xlabel('')
ax_p_res.set_ylabel('')
ax_p_res.set_xticks([])
ax_p_res.set_yticks([])

# Column 3: Prediction
ax_p_pred = fig.add_subplot(gs[3, 2])
p_vmin, p_vmax = p_u_pred.min(), p_u_pred.max()
p_levels = np.linspace(p_vmin, p_vmax, 100)
contour_p_pred = ax_p_pred.contourf(p_X, p_Y, p_u_pred, cmap='jet', 
                                     levels=p_levels, extend='both')
cbar_p_pred = fig.colorbar(contour_p_pred, ax=ax_p_pred)
ax_p_pred.set_xlabel('')
ax_p_pred.set_ylabel('')
ax_p_pred.set_xticks([])
ax_p_pred.set_yticks([])

# Column 4: Pointwise Error
ax_p_perr = fig.add_subplot(gs[3, 3])
p_perr_safe = np.maximum(p_pointwise_error, 1e-16)
p_perr_norm = colors.LogNorm(vmin=np.median(p_perr_safe), vmax=p_perr_safe.max())
mesh_p_perr = ax_p_perr.pcolormesh(p_X, p_Y, p_perr_safe, cmap='jet', 
                                    norm=p_perr_norm, shading='gouraud')
cbar_p_perr = fig.colorbar(mesh_p_perr, ax=ax_p_perr)
ax_p_perr.set_xlabel('')
ax_p_perr.set_ylabel('')
ax_p_perr.set_xticks([])
ax_p_perr.set_yticks([])

# ============================================================================
# ROW 5: KDV
# ============================================================================

# Column 1: Error History
ax_kdv_err = fig.add_subplot(gs[4, 0])
ax_kdv_err.plot(kdv_iter, kdv_errors, color=error_color, linewidth=2)
ax_kdv_err.plot(*opus_kdv, color=opus_color, linewidth=1.5)
ax_kdv_err.set_yscale('log')
ax_kdv_err.set_xlabel('Iterations')
ax_kdv_err.set_ylabel(row_labels[4], fontsize=14, )
ax_kdv_err.grid(True, which="both", ls="--", alpha=0.6)

# Column 2: Residuals
ax_kdv_res = fig.add_subplot(gs[4, 1])
kdv_res_safe = np.maximum(np.abs(kdv_residual), 1e-16)
kdv_res_norm = colors.LogNorm(vmin=kdv_res_safe.min(), vmax=kdv_res_safe.max())
mesh_kdv_res = ax_kdv_res.pcolormesh(kdv_T, kdv_X, kdv_res_safe, cmap='jet', 
                                      norm=kdv_res_norm, shading='gouraud')
cbar_kdv_res = fig.colorbar(mesh_kdv_res, ax=ax_kdv_res)
ax_kdv_res.set_xlabel('')
ax_kdv_res.set_ylabel('')
ax_kdv_res.set_xticks([])
ax_kdv_res.set_yticks([])

# Column 3: Prediction
ax_kdv_pred = fig.add_subplot(gs[4, 2])
kdv_vmin, kdv_vmax = kdv_u_pred.min(), kdv_u_pred.max()
kdv_levels = np.linspace(kdv_vmin, kdv_vmax, 100)
contour_kdv_pred = ax_kdv_pred.contourf(kdv_T, kdv_X, kdv_u_pred, cmap='jet', 
                                         levels=kdv_levels, extend='both')
cbar_kdv_pred = fig.colorbar(contour_kdv_pred, ax=ax_kdv_pred)
ax_kdv_pred.set_xlabel('')
ax_kdv_pred.set_ylabel('')
ax_kdv_pred.set_xticks([])
ax_kdv_pred.set_yticks([])

# Column 4: Pointwise Error
ax_kdv_perr = fig.add_subplot(gs[4, 3])
kdv_perr_safe = np.maximum(kdv_pointwise_error, 1e-16)
kdv_perr_norm = colors.LogNorm(vmin=np.median(kdv_perr_safe), vmax=kdv_perr_safe.max())
mesh_kdv_perr = ax_kdv_perr.pcolormesh(kdv_T, kdv_X, kdv_perr_safe, cmap='jet', 
                                        norm=kdv_perr_norm, shading='gouraud')
cbar_kdv_perr = fig.colorbar(mesh_kdv_perr, ax=ax_kdv_perr)
ax_kdv_perr.set_xlabel('')
ax_kdv_perr.set_ylabel('')
ax_kdv_perr.set_xticks([])
ax_kdv_perr.set_yticks([])

# ============================================================================
# Final Layout and Save
# ============================================================================

plt.savefig('pde_comparison_5cases_opus.png', dpi=300, bbox_inches='tight')
print("\n✓ Figure saved as 'pde_comparison_5cases_opus.png'")

plt.close()

# ============================================================================
# Print Summary Statistics (All Metrics Included)
# ============================================================================

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

# Define a helper to print all metrics consistently
def print_metrics(name, errors, p_error, residual):
    mse_res = np.mean(np.square(residual))
    print(f"\n{name}:")
    print(f"  Final Relative L2:     {errors[-1]:.6e}")
    print(f"  Min Relative L2:       {errors.min():.6e}")
    print(f"  Max Pointwise Error:   {p_error.max():.6e}")
    print(f"  Mean Pointwise Error:  {p_error.mean():.6e}")
    print(f"  Max Residual:          {np.abs(residual).max():.6e}")
    print(f"  MSE of Residual:       {mse_res:.6e}")

print_metrics("Allen-Cahn", ac_errors, ac_pointwise_error, ac_residual)
print_metrics("Burgers", b_errors, b_pointwise_error, b_residual)
print_metrics("Helmholtz", h_errors, h_pointwise_error, h_residual)
print_metrics("Poisson", p_errors, p_pointwise_error, p_residual)
print_metrics("KdV", kdv_errors, kdv_pointwise_error, kdv_residual)

print("\n" + "="*70)
print("\n" + "="*70)