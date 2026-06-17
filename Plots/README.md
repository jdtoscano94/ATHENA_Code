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

## Data

The data these notebooks read is committed under `../Results/`, so the figures
reproduce directly from a clone — just run the notebooks. To keep the repo small,
the large time-resolved runs (compressible Euler / Navier–Stokes /
Kelvin–Helmholtz, and the 2-D anisotropic-diffusion field) are stored as only the
time slices the figures display; each colorbar is computed from those same
slices. The DeepONet file keeps the test set and training diagnostics it plots
and drops the unused training trajectories.

The script (`.jl`/`.py`) that produced each run folder is also tracked, so you can
regenerate any result from scratch instead of using the stored data — run that
folder's script through the matching execution bridge:

```bash
# from the repo root

# Julia (family A) — writes its .npz back into the same folder (uses @__DIR__)
Documents/Scripts/run_julia.sh  Results/NM_codes/<Project>/<timestamp>/<script>.jl

# Python / PINN / operator (families B, C) — run FROM the folder so np.savez lands there
cd Results/PINN_codes/<Project>/<timestamp> && \
  /abs/path/to/Documents/Scripts/run_python.sh <script>.py
```

- **Julia** solves are deterministic.
- **PINN / operator** scripts carry a fixed seed (`--SEED 9999`), so reruns
  reproduce the figures up to library / floating-point nondeterminism.

A regenerated solve writes the *full* time series, not the displayed slices, so
the figure reproduces but the folder grows back to its original size.

## Notes

- These are the only notebooks tracked in git (via a `!Plots/*.ipynb` exception
  to the repo-wide `*.ipynb` ignore); their cell **outputs are stripped** to keep
  diffs and size small, so a fresh checkout shows code only — run them to render.
- See the repo `README.md` → *Reproducing the paper figures* for the same
  workflow in context.
