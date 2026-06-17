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

## Only the data is missing — regenerate it

Everything needed to rebuild the figures is in the repo **except the heavy data
outputs**. Concretely:

- ✅ **Committed:** the figure notebooks, the final `*.png` figures, and the
  **generated script** (`.jl`/`.py`) that produced every run folder's data.
- ❌ **Not committed (gitignored):** the bulky result outputs themselves —
  `.npz` arrays, intermediate `.png`, and reports under `../Results/` — across
  *all* families (PINN, DeepONet, and the Julia/NM solves). They are excluded
  only because of size, not because anything is lost; each is reproduced by
  rerunning its committed script.

So if a notebook cell can't find its `.npz`, regenerate it by running that
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

Once the `.npz` files are back in place, run the notebook to rebuild the figure.

## Notes

- These are the only notebooks tracked in git (via a `!Plots/*.ipynb` exception
  to the repo-wide `*.ipynb` ignore); their cell **outputs are stripped** to keep
  diffs and size small, so a fresh checkout shows code only — run them to render.
- See the repo `README.md` → *Reproducing the paper figures* for the same
  workflow in context.
