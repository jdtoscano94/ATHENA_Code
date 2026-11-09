
# Experiment Report
*Last Updated: 2026-11-06 12:57:04*

## User Request
Lets solve the 2D Helmholtz equation.

The equation is:
$$
u_{xx} + u_{yy} + k^2 u = g(x, y)
$$

where g(x, y)  is chosen to satisfy the  solution: u_{an}=sin(pi a1)sin (pi a2)

The spatio-temporal domain is $(x, y) \in [-1, 1] \times [-1, 1]$.
 
The boundary conditions are **periodic** in both $x$ and $y$. And let a_1=1 and a_2=4 and k=1.

---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
# Strategy: 2D Helmholtz — Advisor-Mandated Extension (Path 2)

**Triage:** [ADVISOR_REPORT] exists → **MANDATORY CURE**: Extend the final SSBroyden stage from 120,000 → **240,000 fixed iterations**, restarting from the current final weights with a **freshly initialized Hessian**. All other settings (Adam phase, physics, architecture, Fourier degree, vRBA) remain **unchanged**.

---

## 1. Data Source
No new data. Resume from the **final checkpoint weights of the previous run** (fallback: full re-run, Adam 10k → SSBroyden 240k).

## 2. Problem Statement
- **Governing Equation (Helmholtz, k=1):**
$$u_{xx} + u_{yy} + u = g(x,y), \quad (x,y)\in[-1,1]\times[-1,1]$$
- **PDE Residual:** $r = u_{xx} + u_{yy} + u - g$, with $g = (1-17\pi^2)\sin(\pi x)\sin(4\pi y)$
- **BCs:** Periodic in x and y — **enforced architecturally** (Fourier embedding), no BC loss.
- **Analytical Solution:** $u_{an} = \sin(\pi x)\sin(4\pi y)$ ($a_1=1, a_2=4$). Yes, admits analytical solution.
- **Data Role:** Validation/Error Metrics ONLY (forward problem, no data loss).
- **Problem Type:** Forward. Well-posed for PINNs: smooth strong-form solution, no resonance ($k^2=1 \neq 17\pi^2$).
- **Input aliasing note:** If template names inputs (t, x), both are spatial: first input → physical x, second → physical y. No time.

## 2.1/2.2 Dimensions
- Input: 2D spatial (x, y). Output: scalar u(x,y). Smooth, oscillatory (4π in y), periodic.

## 3. Base Model
**[MLP with Hybrid Adam and SSBroyden]** — 2D problem → width=30, depth=3, tanh.

## 4. Architecture (UNCHANGED)
`Input (raw x,y) → Periodic Fourier Embedding → MLP Body (30×3, tanh) → Identity Output → u`

- **Normalization:** NONE. Coordinates already in [-1,1] (nice range) — pass raw coordinates directly.
- **Periodic Fourier Features:** degree=4, **Lx = Ly = 2.0** (raw range [-1,1] → max−min = 2.0). Applied separately to each coordinate.
- **Input Composition:** MLP body receives **ONLY concatenated embeddings** — no bare coordinates.
- **Output Transformation:** None (direct scalar). Periodicity exact by construction.
- All operations differentiable, inside the model call function.

## 5. Constraint Strategy
Periodic BCs → architectural (Fourier embedding). Active loss count = **1** (L_pde only). No hard constraints needed.

## 6. Loss Assembly
$$L_{total} = L_{pde}$$

## 6.1 Iteration Schedule (THE ONLY CHANGE)
- **Resume from previous run's final weights.**
- **SSBroyden: 240,000 fixed iterations** (Nchange=100 → 2,400 outer loops), **Hessian freshly initialized (H0=I) at stage start**.
- Unweighted MSE during SSBroyden; vRBA lambdas for **sampling only**.
- Early stopping ONLY on NaN/Inf within the fixed budget. No open-ended stopping rules.
- **Fallback (no checkpoint):** Adam 10,000 iters (vRBA-weighted, exp-decay LR) → SSBroyden 240,000 iters.

## 7. Weighting
- **Local:** vRBA — lambdas for Weighting & Sampling during Adam, **Sampling ONLY during SSBroyden**.
- **Global:** 1 term → no balancing needed.
- **SSBroyden Integrity:** Do NOT modify internal weighting/sampling/batching of the template optimizer.

## 8. Visualization
Keep template callbacks unchanged.

## 9. Saving & Reporting
- Save fields as .npz (u_pred, u_an, pointwise error, residual) with named keys.
- Report **final-iteration RL2 and best-so-far RL2 separately** — never substitute the best dip.
- Post-run: verify **both** loss and error band centers horizontal over last 50% of the final stage.

---

**Main Model:** [MLP with Hybrid Adam and SSBrodyden]
**Enhancements:** [Periodic Fourier Features (degree=4)]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
APPROVED

## Section 1: Status Decision
* **Status:** **APPROVED**

The run is stable, solves the requested forward problem without data leakage, and reaches a final-iteration relative \(L^2\) error of **\(8.10991224\times10^{-9}\)**. Both training curves occupy effectively flat, noisy floors over the final portion of training.

## Section 2: Quantitative Scorecard
* **Score Integrity:** 35/35
* **Score Accuracy:** 35/35
* **Score Details:** 12/15
* **Score Optimality:** 14/15
* **TOTAL SCORE:** **96/100**

The details score retains a small deduction for the visible oscillations around the final loss/error floors. The optimality score reflects a strong standard MLP-PINN pipeline: periodic Fourier embeddings, vRBA, and SSBroyden are all appropriately used. Entropy regularization is not applicable to this smooth Helmholtz problem.

## Section 3: Forensic Analysis & Feedback

### Observation

**Loss and error histories**
- The loss plot spans early values around or above \(10^0\) down to a final floor near \(10^{-12}\). The error plot falls from order \(10^0\) to approximately \(10^{-8}\).
- There is an initial rapid Adam-phase decrease, a visible optimizer transition around 10,000 iterations, and then a steep SSBroyden-phase improvement. Subsequent training is much more gradual and noisy.
- The final stage is approximately the segment from 10,000 to 150,000 plotted iterations, placing its midpoint near 80,000.
- **Loss dual-point reading:**
  - \(A\), midpoint band center: approximately **\(2\times10^{-12}\)**.
  - \(B\), end band center: approximately **\(1\times10^{-12}\)**.
- **Relative-error dual-point reading:**
  - \(A\), midpoint band center: approximately **\(1.4\times10^{-8}\)**.
  - \(B\), end band center: approximately **\(8\times10^{-9}\)**.
- For both curves, these readings are within approximately a factor of two. Across the latter part of the final stage, the oscillation bands remain around the same floors rather than continuing a substantial average descent. They therefore pass the dual-curve plateau test, although neither floor is perfectly noise-free.

**Residual and pointwise-error fields**
- The PDE residual is mottled and broadly distributed across the domain. There is no coherent failure at a boundary, corner, shock, or other localized structure.
- The log confirms residual values from approximately **\(1.2264\times10^{-11}\)** to **\(1.8239\times10^{-6}\)**.
- Pointwise error has a weak banded/mottled structure consistent with the oscillatory manufactured solution, but no visually dominant localized defect.

**Reference versus prediction**
- The prediction reproduces the reference’s alternating sign pattern and its different oscillation frequencies in the two spatial directions.
- No meaningful phase error, amplitude error, or boundary mismatch is visible at the plotted resolution.

**Log confirmation**
- The earlier `RL2 error: 0.00663973` entries are warm-up-stage metrics and are not used for scoring.
- **Final-iteration RL2 error:** `8.10991224e-09`
- **Exact minimum achieved RL2 error:** `2.547440716997534e-09`
- Scoring uses the **final-iteration value**, not the earlier best dip. The final value is nevertheless exceptionally accurate and is far below even a stringent \(10^{-6}\) precision target for this smooth manufactured solution.

The implementation also solves the correct equation:
\[
g=(1-17\pi^2)\sin(\pi x)\sin(4\pi y),
\]
with \(k=1\), and uses the analytical solution only for validation. Periodic Fourier features enforce the requested periodic topology without introducing a data or boundary-loss term.

### Diagnosis
**Successful converged forward PINN with a low stochastic optimization noise floor.** There is no evidence of divergence, spectral failure, data leakage, incorrect forcing, or premature termination with materially descending curves.

### Targeted Improvements
* **For Integrity:** None.
* **For Accuracy:** None. Further training merely to chase the earlier \(2.55\times10^{-9}\) best dip is not justified.
* **For Details:** None required. Retain the current final-stage budget and continue reporting final-iteration and best-so-far RL2 separately.
* **For Optimality:** None required. Preserve the periodic Fourier embedding, vRBA sampling/weighting strategy, and SSBroyden fine-tuning. KANs, altered physics, and entropy penalties are not indicated for this problem.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The implementation matches the strategy. It uses a PINN with Periodic Fourier Features of degree 4 and Lx=Ly=2.0 applied to both spatial inputs; only the concatenated embeddings are passed to the MLP. The MLP has 3 hidden layers of width 30 with tanh activation and a direct scalar output, with no output transformation. The PDE residual is r = u_tt + u_xx + u - g with k^2=1 and g=(1-17π^2)sin(πx)sin(4πy), under the code's t/x aliasing for the two physical coordinates. The analytical solution is sin(πx)sin(4πy), used only for validation/error metrics. The loss contains only L_pde: during Adam it is vRBA-weighted MSE on the PDE residual; during SSBroyden it is unweighted MSE, while vRBA lambdas are updated and used for sampling only. The optimizer workflow is Adam for 10,000 iterations with exponential-decay learning rate and vRBA importance sampling (batch size 20,000), followed by SSBroyden for a fixed 10,000-step budget using Nchange=100 per outer loop (100 outer iterations). Early stopping is only on NaN/Inf. No BC or data loss terms are added. | 2.00e-06 | 35 | 35 | 7 | 14 | **91** |
| 2 | Implements the specified 2D Helmholtz PINN. The base model is an MLP with three hidden width-30 tanh layers and scalar output, using Periodic Fourier Features of degree 4 with Lx=Ly=2.0 applied separately to the two raw coordinates. The main body receives only the concatenated embedded coordinates, with no bare coordinates and no normalization. The loss is only L_pde: during Adam it is the vRBA-weighted mean squared Helmholtz residual; during SSBroyden it is the unweighted mean squared residual, with vRBA lambdas used for resampling only. The PDE residual is u_tt + u_xx + u - g with g=(1-17*pi^2)*sin(pi*t)*sin(4*pi*x). No BC/IC or data loss terms are included. Optimizer workflow matches the refinement plan: Adam for 10,000 iterations with exponential-decay LR and vRBA weighting/sampling, followed by SSBroyden for 20,000 fixed iterations (200 outer batches of 100) with vRBA resampling only and NaN/Inf early stopping. The analytical solution used for validation/error is sin(pi*x)*sin(4*pi*y). | 5.00e-07 | 35 | 35 | 7 | 14 | **91** |
| 4 | Implementation matches the advisor-mandated refinement strategy. Architecture: PINN using Periodic Fourier Features (degree 4, Lx=Ly=2.0) applied independently to both input coordinates, with only the concatenated embeddings passed to an MLP body (width=30, depth=3, tanh) and a linear identity output. No input normalization is performed; raw coordinates are passed directly into the embedding. Optimizer workflow: Phase 1 uses Adam for 10,000 iterations with vRBA-weighted MSE and exponential-decay learning rate; Phase 2 uses the SSBroyden optimizer for 40,000 fixed iterations (N_change=100, giving 400 outer loops), with unweighted MSE and vRBA lambdas used only for importance sampling. Early stopping is limited to non-finite loss. Loss assembly contains exactly one term, L_total = L_pde, where the Helmholtz residual is u_xx + u_yy + u - g with g=(1-17*pi^2)*sin(pi*x)*sin(4*pi*y). Periodic BCs are enforced architecturally through the Fourier embedding; no BC or data loss terms are present. The analytical solution u_an=sin(pi*x)*sin(4*pi*y) is used only for error metrics and visualization. | 1.00e-07 | 35 | 35 | 7 | 14 | **91** |
| 5 | Implemented 2D Helmholtz PINN with MLP (3 hidden layers of 30, tanh) and separate Periodic Fourier Features (degree 4, Lx=Ly=2.0) for each coordinate. No input normalization; raw coordinates are passed to the embeddings. PDE residual is r = u_tt + u_xx + u - g with g = (1 - 17π²) sin(π t) sin(4π x) (t=x_phys, x=y_phys), matching g = (1 - 17π²) sin(π x) sin(4π y). Analytical solution is u = sin(π x) sin(4π y). Loss is L_total = L_pde only; no L_data and no BC loss. Optimizer: Adam phase 10,000 iterations with exponential-decay LR, vRBA-weighted MSE and vRBA importance sampling; then SSBroyden phase 120,000 iterations via 1,200 outer loops of N_change=100, with Hessian reinitialized at start (H0=I), unweighted MSE loss, and vRBA used only for sampling. Early stopping only on NaN/Inf. Final-iteration RL2 and best-so-far RL2 are reported separately. | 3.46e-08 | 35 | 35 | 7 | 14 | **91** |
| 6 | Implements a PINN for the 2D Helmholtz problem with k=1. Architecture: MLP body (30,30,30,1) with tanh activation, preceded by separate Periodic Fourier Feature embeddings for each coordinate (degree=4, Lx=Ly=2.0); no raw coordinates are passed to the MLP and no external normalization is used. Output is a direct scalar u. PDE residual is r = u_tt + u_xx + u - g with g = (1 - 17*pi^2) * sin(pi*t) * sin(4*pi*x), where t and x are the aliased first and second spatial coordinates. Loss assembly is exactly L_total = L_pde only; there are no BC, IC, or data losses. vRBA is used for weighting and sampling during Adam; during SSBroyden, vRBA lambdas are updated for sampling only and the SSBroyden objective is unweighted MSE. Training schedule: fallback full re-run (no checkpoint loading) uses Adam for 10,000 iterations with exponential-decay LR and vRBA, followed by SSBroyden for 240,000 fixed iterations via 2,400 outer loops of Nchange=100, with H0=I freshly initialized at the start of the second stage. The analytical solution sin(pi*x)*sin(4*pi*y) is used only for validation and error metrics. Final-iteration RL2 and best-so-far RL2 are both reported separately. Visualization includes prediction, reference, pointwise error, residual, loss history, and error history. | 8.11e-09 | 35 | 35 | 12 | 14 | **96** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves `Helmholtz_results.npz` (NumPy .npz) with keys: `x` (100x100 float64 physical-x grid), `y` (100x100 float64 physical-y grid), `u_pred` (100x100 float64 predicted field), `u_analytical` (100x100 float64 exact field), and `relative_l2_error` (1D float64 error-history array; default length 655 = 555 Adam logged entries + 100 SSBroyden outer iterations). It also saves an extra non-required MATLAB file `Log_files_SSBroyden.mat` containing `model_name` plus history keys `all_errors`, `all_its`, `all_loss`, `all_gamma`, `all_lamB`, `all_max_RBA`, `all_L_infty`, and `all_L_2`; with default settings the shared Adam+SSBroyden histories are length 655, while Adam-only histories are length 555. |
| 2 | Saves Helmholtz_results.npz in NumPy .npz format with keys: 'x' (NC_TEST x NC_TEST meshgrid array of the first physical coordinate, default 100x100), 'y' (NC_TEST x NC_TEST meshgrid array of the second physical coordinate), 'u_pred' (same grid shape), 'u_analytical' (same grid shape), 'error' (same grid shape, absolute difference), and 'residual' (same grid shape, PDE residual field). Also saves Log_files_SSBroyden.mat in MAT format containing training logs: 'model_name', 'all_errors', 'all_its', 'all_loss', 'all_gamma', 'all_lamB', 'all_max_RBA', 'all_L_infty', and 'all_L_2'. |
| 4 | Saves Helmholtz_results.npz containing keys: 'x' (100x100), 'y' (100x100), 'u_pred' (100x100), 'u_analytical' (100x100), 'pointwise_error' (100x100), 'residual' (100x100), 'all_loss' (1D history), 'all_errors' (1D history), 'all_its' (1D history), 'all_L_infty' (1D history), 'all_L_2' (1D history). Also saves Log_files_SSBroyden.mat with model_name and training histories including all_errors, all_its, all_loss, all_gamma, all_lamB, all_max_RBA, all_L_infty, and all_L_2. |
| 5 | Saves Helmholtz_results.npz (NumPy .npz) with keys: 'x' (100×100), 'y' (100×100), 'u_pred' (100×100), 'u_analytical' (100×100), 'pointwise_error' (100×100), 'residual' (100×100), 'all_loss' (1D array), 'all_errors' (1D array), 'all_its' (1D array), 'all_L_infty' (1D array), 'all_L_2' (1D array). Also saves Log_files_SSBroyden.mat (MATLAB .mat) with keys: 'model_name' (string), 'all_errors' (1D array), 'all_its' (1D array), 'all_loss' (1D array), 'all_gamma' (1D array), 'all_lamB' (1D array), 'all_max_RBA' (1D array), 'all_L_infty' (1D array), 'all_L_2' (1D array). |
| 6 | Saves Helmholtz_results.npz with keys: x (100x100), y (100x100), u_pred (100x100), u_analytical (100x100, equivalent to the strategy's u_an), pointwise_error (100x100), residual (100x100), all_loss (1D), all_errors (1D), all_its (1D), all_L_infty (1D), all_L_2 (1D). Also saves Log_files_SSBroyden.mat with keys model_name, all_errors, all_its, all_loss, all_gamma, all_lamB, all_max_RBA, all_L_infty, all_L_2. The spatial field arrays are on the NC_TEST=100 validation grid, so x, y, u_pred, u_analytical, pointwise_error, and residual are 100x100. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run5/results//Helmholtz_2D_Periodic_Sin_pi_x_Sin_4pi_y/2026_11_06_1214/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run5/results//Helmholtz_2D_Periodic_Sin_pi_x_Sin_4pi_y/2026_11_06_1222/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run5/results//Helmholtz_2D_Periodic_Sin_pi_x_Sin_4pi_y/2026_11_06_1232/` |
| 5 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run5/results//Helmholtz_2D_Periodic_Sin_pi_x_Sin_4pi_y/2026_11_06_1239/` |
| 6 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run5/results//Helmholtz_2D_Periodic_Sin_pi_x_Sin_4pi_y/2026_11_06_1249/` |


---
