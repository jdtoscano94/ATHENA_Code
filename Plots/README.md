# Plots — paper figures

The notebooks here assemble the figures in the ATHENA paper from the per-run
results under [`../Results/`](../Results). They read result arrays through
**repo-relative paths** (`../Results/...`), so they work wherever the repo is
cloned — open any of them with the working directory set to this `Plots/` folder.

## Notebooks

| Notebook | Builds | Reads from |
|---|---|---|
| `Analytical.ipynb` | Analytical / exact-solution comparisons | `../Results/NM_codes/` |
| `NM.ipynb` | Classical numerical-method figures (family A) | `../Results/NM_codes/` |
| `PINNs.ipynb` | PINN figures (family B) | `../Results/PINN_codes/` |
| `Regret_analysis.ipynb` | Cross-family regret / score analysis | `../Results/{NM,PINN,DeepONet}_codes/` |

The standalone `*.png` files in this folder (ablation, comparison, total-score)
are committed final figures produced by these notebooks.

## Scripts (model sensitivity and the single-agent baseline)

The figures added with the model-sensitivity study and the Claude Code (Opus 5.5) baseline are
plain scripts. Run each one from this folder (`cd Plots && python <script>.py`); each writes its
PNG next to itself.

| Script | Builds | Reads from |
|---|---|---|
| `fig4_pinn_opus.py` | `pde_comparison_5cases_opus.png` (PIML benchmarks with the Opus 5.5 baseline) | `../Results/PINN_codes/`, `../Results/Sensitivity/bare_claude/` |
| `fig5A_burgers_opus.py` | `burgers_comparison_final_v3_opus.png` | `../Results/PINN_codes/`, `../Results/Sensitivity/bare_claude/IB_PINN/` |
| `fig7_sensitivity.py` | `athena_comparison_sensitivity.png`, `athena_sensitivity_regret_SI.png` (open and closed teams, Opus 5.5) | `../Results/Sensitivity/{open,closed,bare_claude}/`, `../run_logs/sensitivity/` |
| `fig_first90_SI.py` | `athena_first90_SI.png` (accepted vs first trial scoring at least 90) | same as `fig7_sensitivity.py`, plus `ib2d_rl2.json` |
| `fig_pinn_bars.py` | `pinn_comparison_bars.png` (Table 1 as bars) | values in the script |
| `opus_residual_mse.py` | prints the residual MSE of the Opus PINN runs (Table 1) | `../Results/Sensitivity/bare_claude/` |
| `ib2d_rl2.py` | `ib2d_rl2.json` (relative L2 error at t = 2 of every IB2D solution) | the full IB2D solutions (see below) |

## Data

The data these notebooks read is committed under `../Results/`, so the figures
reproduce directly from a clone — just run the notebooks. To keep the repo small,
the large time-resolved runs (compressible Euler / Navier–Stokes /
Kelvin–Helmholtz, and the 2-D anisotropic-diffusion field) are stored as only the
time slices the figures display; each colorbar is computed from those same
slices. The DeepONet file keeps the test set and training diagnostics it plots
and drops the unused training trajectories.

For the sensitivity runs, the repository keeps what the scripts read from each run (the
`Experiment_Report.md` trial histories, the usage logs, `job_info.txt`, the program outputs, and
the Claude Code event streams) together with every generated `.py`/`.jl`, but not the vector-DB
copies or the large solution arrays. The one exception is `ib2d_rl2.py`: it evaluates the full
IB2D solution files (about 120 MB, not committed), and its output `ib2d_rl2.json` is committed,
so `fig_first90_SI.py` runs from a clone without them.

The script (`.jl`/`.py`) that produced each run folder is also tracked, so you can
regenerate any result from scratch instead of using the stored data — run that
folder's script through the matching execution bridge:

```bash
# from the repo root

# Julia (family A): writes its .npz back into the same folder (uses @__DIR__)
bash Documents/Scripts/run_julia.sh  Results/NM_codes/<Project>/<timestamp>/<script>.jl  <script>.jl.err

# Python / PINN / operator (families B, C): run FROM the folder so np.savez lands there
cd Results/PINN_codes/<Project>/<timestamp> && \
  bash <repo>/Documents/Scripts/run_python.sh <script>.py <script>.py.err
```

- **Julia** solves are deterministic.
- **PINN / operator** scripts carry a fixed seed (`--SEED 9999`), so reruns
  reproduce the figures up to library / floating-point nondeterminism.

A regenerated solve writes the *full* time series, not the displayed slices, so
the figure reproduces but the folder grows back to its original size.

## Notes

- The plotting notebooks are tracked through a `!Plots/*.ipynb` exception to the
  repo-wide `*.ipynb` ignore (the example workflows under `Examples_*/` are tracked
  too); their cell **outputs are stripped** to keep diffs and size small, so a fresh
  checkout shows code only — run them to render.
- The scripts and notebooks run in the orchestrator env (`athena`), except
  `fig4_pinn_opus.py`, which imports JAX through the KdV reference helper and needs the
  `NABLA_JAX` env (or the container's `$NABLA_PY`).
- See the repo `README.md` → *Reproducing the paper figures* for the same
  workflow in context.
