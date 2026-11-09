
# Experiment Report
*Last Updated: 2026-11-06 12:46:10*

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
# Strategy: 2D Helmholtz — Advisor-Mandated Budget Extension

## 0. Triage
**Path 2: Advisor Recommendation (MANDATORY).** Diagnosis: Budget-Limited/Premature Stop. Cure: **Extend ONLY the final SSBroyden stage from 40,000 → 120,000 fixed iterations.** All physics, architecture, embeddings, and Adam phase remain **unchanged** from Run 4.

## 1. Data Source
No new data. Analytical solution used for validation only.

## 2. Problem Statement
- **Governing Equation (2D Helmholtz):** $u_{xx} + u_{yy} + k^2 u = g(x,y)$, with $k=1$.
- **PDE Residual:** $r = u_{xx} + u_{yy} + u - g$, where $g = (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$.
- **BCs:** Periodic in both $x$ and $y$ on $[-1,1]\times[-1,1]$ — enforced **architecturally** (no loss term).
- **Analytical Solution:** $u_{an} = \sin(\pi x)\sin(4\pi y)$ (with $a_1=1$, $a_2=4$).
- **Data Role:** Validation/Error Metrics ONLY. **No L_data.**
- **Problem Type:** Forward. Well-posed for PINNs (smooth strong-form solution).

## 2.1 / 2.2 Input-Output
- Input: 2D $(x, y) \in [-1,1]^2$. Output: scalar $u(x,y)$. Smooth/oscillatory, no shocks.

## 3. Base Model Selection
**MLP with Hybrid Adam and SSBroyden** (2D problem). Width=30, depth=3, tanh.

## 4. Main Architecture
`Input (x,y) → Periodic Fourier Embedding (per-coordinate) → Concatenation → MLP Body (30×3, tanh) → Linear Output → u_pred`

- **Normalization:** NONE. Coordinates already in $[-1,1]$ — pass raw coordinates directly to embedding. All operations inside the model's call function (differentiable).
- **Input Embedding:** **Periodic Fourier Features, degree 4** (advisor: keep unchanged). Applied separately to $x$ and $y$. **Lx = Ly = 2.0** (raw range $[-1,1]$ → range = 2.0).
- **Input Composition Rule:** Main body receives **ONLY** the concatenated embeddings — no bare $x$, $y$.
- **Output Transformation:** None (identity). Periodicity is exact by construction.

## 5. Constraint Implementation
- Periodic BCs: satisfied by architecture → **no L_bc terms**.
- Soft loss count: `L_pde` only (1 term) → count < 5 → no hard constraints required.

## 6. Loss Function Assembly
$$L_{total} = L_{pde}$$
- Adam phase: vRBA-weighted MSE of residual.
- SSBroyden phase: unweighted MSE; vRBA lambdas used for **sampling only**.

## 6.1 Continuation
Not applicable (no Re/Péclet). No parameter annealing.

## 7. Iteration Schedule (Advisor-Mandated)
- **Phase 1 — Adam:** **10,000 iterations** (unchanged). Exponential-decay LR, vRBA weighting + sampling. STATIC.
- **Phase 2 — SSBroyden:** **120,000 fixed iterations** (3× extension from 40,000). N_change=100 → 1,200 outer loops. **CRITICAL: Reinitialize the SSBroyden Hessian at the start of this stage.**
- **Early stopping:** Only on NaN/Inf loss/error. No open-ended stopping rules.
- **Reporting:** Log final-iteration RL2 **separately** from best-so-far minimum.

## 8. Weighting Strategy
- **Local (lambdas/vRBA):** Weighting & Sampling during Adam; Sampling ONLY during SSBroyden. Never frozen.
- **Global:** 1 term → no balancing needed.
- **SSBroyden Integrity:** Do NOT modify internal SSBroyden weighting/sampling/batching.

## 9. Visualization
Do NOT modify template visualization callbacks. Ensure final-iteration RL2 is reported distinctly from the best-so-far value.

## 10. Saving
Save predicted field, analytical field, and error history as `.npz` with keys specifying variable names.

---

**Main Model:**
[MLP with Hybrid Adam and SSBroyden]

**Enhancements:**
[Periodic Fourier Features (degree 4, Lx=Ly=2.0)]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
REJECTED_FOR_REFINEMENT

## Section 1: Status Decision
* **Status:** **REJECTED_FOR_REFINEMENT**

Although the run is stable and highly accurate, the **loss curve is still decreasing on average during the final stage**. It therefore fails the mandatory dual-curve plateau test.

## Section 2: Quantitative Scorecard
* **Score Integrity:** 35
* **Score Accuracy:** 35
* **Score Details:** 7
* **Score Optimality:** 14
* **TOTAL SCORE:** **91**

Accuracy is scored using the **final-iteration RL2 error, \(3.45958476\times10^{-8}\)**, not the earlier best value. The implementation solves the requested forward Helmholtz problem with \(k=1\), \(a_1=1\), \(a_2=4\), architectural periodicity, and no data-loss term.

## Section 3: Forensic Analysis & Feedback

### Observation
* **Problem type:** Type A, forward simulation with a smooth manufactured solution and no noise. The analytical solution is used for validation, while training contains only \(L_{\mathrm{pde}}\); there is no evident data leakage.
* **Loss history:** The plot is readable and finite. The loss falls from roughly \(10^3\)–\(10^4\) initially to approximately \(10^{-10}\) at the right edge, with labeled ticks extending lower. There is an Adam phase followed by a clear transition near 10,000 iterations, a steep SSBroyden decrease, and then a long, noisy but still downward-sloping tail.
* **Error history:** The relative \(L^2\) error falls from order unity to a noisy band of a few times \(10^{-8}\). It becomes much flatter than the loss late in training, although individual dips reach substantially below the band center.
* **Mandatory final-region readings:** Reading the plotted post-Adam final stage as approximately 10,000–80,000 iterations, its midpoint is approximately 45,000:
  * **Loss:** \(A \approx 1.0\times10^{-9}\) at the midpoint band center; \(B \approx 1.0\times10^{-10}\) at the end band center. These are not approximately equal: the loss is still descending.
  * **Relative error:** \(A \approx 6\times10^{-8}\); \(B \approx 4\times10^{-8}\). This is approximately a flat, noisy band within the allowed factor-of-two reading.
  * Within the last-50% loss window, the band center drops from roughly \(5\times10^{-10}\) in its first half to roughly \(2\times10^{-10}\) in its second half. Thus, the loss gate fails even though the error is nearly flat.
* **PDE residual heatmap:** The residual is speckled and distributed throughout the domain rather than concentrated at a boundary or coherent structure. The log reports a range from \(1.0320\times10^{-10}\) to \(2.1641\times10^{-6}\).
* **Solution fields:** Reference and prediction are visually almost indistinguishable, including the high-frequency \(y\)-oscillations and the left/right sign structure.
* **Pointwise error:** Errors are of order \(10^{-8}\), with smooth red hotspots mainly in the upper-central and right-middle portions of the domain. There is no strong periodic-boundary artifact or unresolved oscillatory pattern.
* **Log confirmation:** Because SSBroyden was used, the earlier `RL2 error: 0.00891681` is treated as the Adam warm-up result. The bottom of the log reports:
  * **Final-iteration RL2:** `3.45958476e-08`
  * **Minimum achieved RL2:** `3.826902096948086e-09`
  
  The right edge of the plotted error curve is consistent with approximately \(3.5\times10^{-8}\). The best-so-far minimum is about nine times smaller, but it is not the final model error and is not used for scoring.

### Diagnosis
**Budget-Limited / Premature Stop.** There is no instability, explosion, or evidence of an inaccurate physical formulation. The remaining bottleneck is that the loss band has not reached a horizontal floor. The excellent total score does not override the mandatory plateau check.

### Targeted Improvements
* **For Integrity:** None.
* **For Accuracy:** **Increase only the final SSBroyden stage from the reported 120,000 to 240,000 fixed iterations.** Restart/extend that stage from the current final weights with a freshly initialized Hessian. Keep the Adam phase, physics, architecture, Fourier degree, and vRBA settings unchanged.
* **For Details:** After the extended run, verify that **both** the loss and error band centers are horizontal over the last 50% of the final stage. Continue reporting final-iteration and best-so-far RL2 separately; do not substitute the best dip for the final value.
* **For Optimality:** None. Retain the periodic Fourier input embedding, vRBA, and SSBroyden. Do not feed raw coordinates directly to the MLP or remove periodic embeddings. No shock entropy penalty is needed for this smooth Helmholtz problem.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The implementation matches the strategy. It uses a PINN with Periodic Fourier Features of degree 4 and Lx=Ly=2.0 applied to both spatial inputs; only the concatenated embeddings are passed to the MLP. The MLP has 3 hidden layers of width 30 with tanh activation and a direct scalar output, with no output transformation. The PDE residual is r = u_tt + u_xx + u - g with k^2=1 and g=(1-17π^2)sin(πx)sin(4πy), under the code's t/x aliasing for the two physical coordinates. The analytical solution is sin(πx)sin(4πy), used only for validation/error metrics. The loss contains only L_pde: during Adam it is vRBA-weighted MSE on the PDE residual; during SSBroyden it is unweighted MSE, while vRBA lambdas are updated and used for sampling only. The optimizer workflow is Adam for 10,000 iterations with exponential-decay learning rate and vRBA importance sampling (batch size 20,000), followed by SSBroyden for a fixed 10,000-step budget using Nchange=100 per outer loop (100 outer iterations). Early stopping is only on NaN/Inf. No BC or data loss terms are added. | 2.00e-06 | 35 | 35 | 7 | 14 | **91** |
| 2 | Implements the specified 2D Helmholtz PINN. The base model is an MLP with three hidden width-30 tanh layers and scalar output, using Periodic Fourier Features of degree 4 with Lx=Ly=2.0 applied separately to the two raw coordinates. The main body receives only the concatenated embedded coordinates, with no bare coordinates and no normalization. The loss is only L_pde: during Adam it is the vRBA-weighted mean squared Helmholtz residual; during SSBroyden it is the unweighted mean squared residual, with vRBA lambdas used for resampling only. The PDE residual is u_tt + u_xx + u - g with g=(1-17*pi^2)*sin(pi*t)*sin(4*pi*x). No BC/IC or data loss terms are included. Optimizer workflow matches the refinement plan: Adam for 10,000 iterations with exponential-decay LR and vRBA weighting/sampling, followed by SSBroyden for 20,000 fixed iterations (200 outer batches of 100) with vRBA resampling only and NaN/Inf early stopping. The analytical solution used for validation/error is sin(pi*x)*sin(4*pi*y). | 5.00e-07 | 35 | 35 | 7 | 14 | **91** |
| 4 | Implementation matches the advisor-mandated refinement strategy. Architecture: PINN using Periodic Fourier Features (degree 4, Lx=Ly=2.0) applied independently to both input coordinates, with only the concatenated embeddings passed to an MLP body (width=30, depth=3, tanh) and a linear identity output. No input normalization is performed; raw coordinates are passed directly into the embedding. Optimizer workflow: Phase 1 uses Adam for 10,000 iterations with vRBA-weighted MSE and exponential-decay learning rate; Phase 2 uses the SSBroyden optimizer for 40,000 fixed iterations (N_change=100, giving 400 outer loops), with unweighted MSE and vRBA lambdas used only for importance sampling. Early stopping is limited to non-finite loss. Loss assembly contains exactly one term, L_total = L_pde, where the Helmholtz residual is u_xx + u_yy + u - g with g=(1-17*pi^2)*sin(pi*x)*sin(4*pi*y). Periodic BCs are enforced architecturally through the Fourier embedding; no BC or data loss terms are present. The analytical solution u_an=sin(pi*x)*sin(4*pi*y) is used only for error metrics and visualization. | 1.00e-07 | 35 | 35 | 7 | 14 | **91** |
| 5 | Implemented 2D Helmholtz PINN with MLP (3 hidden layers of 30, tanh) and separate Periodic Fourier Features (degree 4, Lx=Ly=2.0) for each coordinate. No input normalization; raw coordinates are passed to the embeddings. PDE residual is r = u_tt + u_xx + u - g with g = (1 - 17π²) sin(π t) sin(4π x) (t=x_phys, x=y_phys), matching g = (1 - 17π²) sin(π x) sin(4π y). Analytical solution is u = sin(π x) sin(4π y). Loss is L_total = L_pde only; no L_data and no BC loss. Optimizer: Adam phase 10,000 iterations with exponential-decay LR, vRBA-weighted MSE and vRBA importance sampling; then SSBroyden phase 120,000 iterations via 1,200 outer loops of N_change=100, with Hessian reinitialized at start (H0=I), unweighted MSE loss, and vRBA used only for sampling. Early stopping only on NaN/Inf. Final-iteration RL2 and best-so-far RL2 are reported separately. | 3.46e-08 | 35 | 35 | 7 | 14 | **91** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves `Helmholtz_results.npz` (NumPy .npz) with keys: `x` (100x100 float64 physical-x grid), `y` (100x100 float64 physical-y grid), `u_pred` (100x100 float64 predicted field), `u_analytical` (100x100 float64 exact field), and `relative_l2_error` (1D float64 error-history array; default length 655 = 555 Adam logged entries + 100 SSBroyden outer iterations). It also saves an extra non-required MATLAB file `Log_files_SSBroyden.mat` containing `model_name` plus history keys `all_errors`, `all_its`, `all_loss`, `all_gamma`, `all_lamB`, `all_max_RBA`, `all_L_infty`, and `all_L_2`; with default settings the shared Adam+SSBroyden histories are length 655, while Adam-only histories are length 555. |
| 2 | Saves Helmholtz_results.npz in NumPy .npz format with keys: 'x' (NC_TEST x NC_TEST meshgrid array of the first physical coordinate, default 100x100), 'y' (NC_TEST x NC_TEST meshgrid array of the second physical coordinate), 'u_pred' (same grid shape), 'u_analytical' (same grid shape), 'error' (same grid shape, absolute difference), and 'residual' (same grid shape, PDE residual field). Also saves Log_files_SSBroyden.mat in MAT format containing training logs: 'model_name', 'all_errors', 'all_its', 'all_loss', 'all_gamma', 'all_lamB', 'all_max_RBA', 'all_L_infty', and 'all_L_2'. |
| 4 | Saves Helmholtz_results.npz containing keys: 'x' (100x100), 'y' (100x100), 'u_pred' (100x100), 'u_analytical' (100x100), 'pointwise_error' (100x100), 'residual' (100x100), 'all_loss' (1D history), 'all_errors' (1D history), 'all_its' (1D history), 'all_L_infty' (1D history), 'all_L_2' (1D history). Also saves Log_files_SSBroyden.mat with model_name and training histories including all_errors, all_its, all_loss, all_gamma, all_lamB, all_max_RBA, all_L_infty, and all_L_2. |
| 5 | Saves Helmholtz_results.npz (NumPy .npz) with keys: 'x' (100×100), 'y' (100×100), 'u_pred' (100×100), 'u_analytical' (100×100), 'pointwise_error' (100×100), 'residual' (100×100), 'all_loss' (1D array), 'all_errors' (1D array), 'all_its' (1D array), 'all_L_infty' (1D array), 'all_L_2' (1D array). Also saves Log_files_SSBroyden.mat (MATLAB .mat) with keys: 'model_name' (string), 'all_errors' (1D array), 'all_its' (1D array), 'all_loss' (1D array), 'all_gamma' (1D array), 'all_lamB' (1D array), 'all_max_RBA' (1D array), 'all_L_infty' (1D array), 'all_L_2' (1D array). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run5/results//Helmholtz_2D_Periodic_Sin_pi_x_Sin_4pi_y/2026_11_06_1214/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run5/results//Helmholtz_2D_Periodic_Sin_pi_x_Sin_4pi_y/2026_11_06_1222/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run5/results//Helmholtz_2D_Periodic_Sin_pi_x_Sin_4pi_y/2026_11_06_1232/` |
| 5 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run5/results//Helmholtz_2D_Periodic_Sin_pi_x_Sin_4pi_y/2026_11_06_1239/` |


---
