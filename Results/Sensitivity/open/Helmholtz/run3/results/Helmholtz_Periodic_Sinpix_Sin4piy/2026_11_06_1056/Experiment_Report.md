
# Experiment Report
*Last Updated: 2026-11-06 11:04:44*

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
# Strategy: 2D Helmholtz (Refinement Run — Budget Extension Only)

**Path 2 (Advisor Recommendation):** Architecture/physics validated (score 91). Advisor diagnosis: **Budget-Limited / Premature Stop**. Cure: **extend final SSBroyden stage 40,000 → 120,000 iterations** (Adam 10,000 unchanged; 130,000 total). All other settings frozen.

## 1. Data Role
**Role: Validation/Error Metrics ONLY.** Forward problem; analytical solution is reference only. No L_data.

## 2. Problem Statement
- **Governing Equation:** $u_{xx} + u_{yy} + k^2 u = g(x,y)$, with $k=1$.
- **PDE Residual:** $r = u_{xx} + u_{yy} + u - g(x,y)$
- **Source term (derived from analytical solution):** $g(x,y) = (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$
- **Analytical Solution:** $u_{an} = \sin(\pi x)\sin(4\pi y)$ (from $a_1=1, a_2=4$)
- **BCs:** Periodic in both $x$ and $y$ on $[-1,1]\times[-1,1]$. No IC (steady-state).
- **Problem Type:** Forward. Well-posed for PINNs — linear elliptic, smooth strong solution exists.
- **Has Data Loss:** No.

## 3. Base Model Selection
**[MLP with Hybrid Adam and SSBroyden]** (2D problem, Case A).

## 4. Architecture (UNCHANGED from prior runs)
**Input → Periodic Fourier Embedding → MLP → Output**

- **Input (d=2):** $(x,y)$, both already in $[-1,1]$ → **NO normalization** (pass raw coordinates; any scaler is forbidden).
- **Embedding:** Periodic Fourier Features on **both** $x$ and $y$: **degree=5, $L_x = L_y = 2.0$** (range of coordinates entering embedding = 2.0). Computed *inside* the model call (differentiable).
- **Input Composition Rule:** MLP body receives **ONLY** the 20 embedding features — no raw $x,y$.
- **Main Body:** MLP, width=30, depth=3.
- **Output (q=1):** Scalar $u$, identity output transformation. Periodicity enforced by construction (embedding) — no output ansatz needed.

## 5. Constraint Strategy
Periodic BCs satisfied **architecturally** by the periodic embedding. Soft loss count = 1 ($L_{pde}$ only) → **Loss-Based, no hard constraints**.

## 6. Loss Function
$$L_{total} = L_{pde}$$
No $L_{bc}$ (periodic by architecture), no $L_{ic}$, no $L_{data}$.

## 7. Weighting & Training Schedule
- **Local weights:** vRBA `lambdas` — **weighting & sampling during Adam; sampling ONLY during SSBroyden.**
- **Global weights:** None (single loss term; no balancing).
- **Schedule (hard-capped, finite budget):**
  - **Phase 1 — Adam:** exactly **10,000 iterations**. Static. vRBA weighting active.
  - **Phase 2 — SSBroyden:** exactly **120,000 iterations** (extended 3× per Advisor). **Reinitialize the Hessian at the Adam→SSBroyden transition** (and again if restarting from trained weights). Do NOT interfere with SSBroyden internals.
  - **Total: 130,000 iterations.** Stop early only on NaN/Inf. No open-ended stopping rules.
- **No continuation** (no Reynolds/Peclet).

## 8. Visualization
Do NOT modify template visualization callbacks.

## 9. Saving
Save fields as `.npz` with keys: `u_pred`, `u_an`, `pde_residual`, plus coordinates. Report best-ever and final-iteration RL2 error separately.

## Modules for Librarian
**Main Model:** [MLP with Hybrid Adam and SSBroyden]
**Enhancements:** [Periodic Fourier Features (degree=5, Lx=Ly=2.0)]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
APPROVED

## Section 1: Status Decision
* **Status:** **APPROVED**
* **Problem type:** **Type A — forward, noise-free simulation.** The analytical solution is used exclusively for validation/error metrics; there is no data-loss leakage.
* The extended 130,000-iteration run completed with finite values and readable plots. Both final-stage curves have reached horizontal noise floors, resolving the previous budget-limited/premature-stop diagnosis.
* **History check:** Runs 1, 2, and 4 previously scored 91. Fewer than five attempts since the first success are recorded, so the automatic “exploration complete” stop is not invoked. This approval is based on the current run independently passing the score and plateau gates.

## Section 2: Quantitative Scorecard
* **Score Integrity:** **35/35** — completed successfully; finite losses, errors, residuals, and readable fields.
* **Score Accuracy:** **35/35**
  * **Numerical Precision:** 20/20. Using \(\epsilon_{\text{target}}=10^{-6}\), consistent with the prior accuracy assessments, the final-iteration RL2 error \(1.651519570730467\times10^{-8}\) comfortably meets the target.
  * **Consistency:** 15/15. The source term, residual, periodic geometry, \(k=1\), and analytical reference are correct. No training data loss or inappropriate continuation was introduced.
* **Score Details:** **12/15** — periodic boundaries are enforced architecturally and both curves are flat on average, but the final error/loss bands remain visibly oscillatory rather than perfectly steady.
* **Score Optimality:** **14/15** — correct periodic-Fourier \(\rightarrow\) MLP pipeline, vRBA, and low-dimensional SSBroyden fine-tuning. No entropy penalty is relevant for this linear, shock-free Helmholtz problem; no KAN is needed at this accuracy.
* **TOTAL SCORE:** **96/100**

## Section 3: Forensic Analysis & Feedback

### Observation
**1. Total Loss History**
* The displayed logarithmic axis spans approximately \(10^{3}\) down to \(10^{-12}\).
* The loss drops rapidly during Adam, shows the expected transition feature near 10,000 iterations, and then decreases sharply under SSBroyden before settling near \(10^{-12}\).
* The final SSBroyden stage is approximately **10,000–130,000 iterations**, whose midpoint is **70,000**:
  * **Loss \(A\), midpoint band center:** approximately **\(1\times10^{-12}\)**.
  * **Loss \(B\), end band center:** approximately **\(1\times10^{-12}\)**.
* Over the last 50% window, the 70,000–100,000 and 100,000–130,000 portions have essentially the same average level. Thus, the loss is **flat on average**, with noise around a fixed floor rather than continued descent.

**2. Relative \(L^2\) Error History**
* The error begins around \(10^{-1}\)–\(10^{0}\), passes through a small transition bump, and falls steeply to roughly \(10^{-7}\) by about 35,000–40,000 iterations. It subsequently oscillates in the low-\(10^{-8}\) region.
* For the same final stage:
  * **Error \(A\), midpoint band center at 70,000:** approximately **\(4\times10^{-8}\)**.
  * **Error \(B\), end band center at 130,000:** approximately **\(4\times10^{-8}\)**.
* The first and second halves of the last-50% window therefore have approximately equal band centers. Individual dips and spikes do not represent further average convergence. The rightmost point is in the low-\(10^{-8}\) range and is consistent with the logged final-iteration value.

**3. Residual and Solution Fields**
* The PDE-residual heatmap covers roughly \(10^{-11}\)–\(10^{-6}\). Its larger orange/red residuals are distributed as fine speckles throughout the domain, without a dominant shock front, corner failure, or systematic boundary layer.
* The log reports residual extrema of **\(7.178257988016412\times10^{-12}\)** and **\(1.413523847304532\times10^{-6}\)**.
* Reference and prediction are visually indistinguishable: the same oscillatory bands, signs, zero contours, and amplitudes are reproduced.
* Pointwise error consists only of very small, broad patches. There is no visible phase or amplitude mismatch; the cyan/blue patches are lower-error regions, not solution defects.

**4. Log Extraction**
* There is no literal `Minimum achieved average RL2 error:` line in the supplied tail. The corresponding bottom-of-log **best-ever/minimum** value is:
  * **Best-ever RL2:** `3.4092910280689148e-09`
  * **Final-iteration RL2:** `1.651519570730467e-08`
* The earlier repeated `RL2 error: 0.00435141` is a warm-up/transition report superseded by the final SSBroyden metrics.
* **Scoring uses the final-iteration value \(1.6515\times10^{-8}\), not the earlier best-ever dip.**

### Diagnosis
**Converged residual/error noise floor with mild sampling-induced oscillations.** The previous premature-stop problem has been corrected by the budget extension. There is no evidence of divergence, stagnation at high error, incorrect physics, boundary failure, or data leakage.

### Targeted Improvements
* **For Integrity:** None.
* **For Accuracy:** None required. Further iteration increases are not justified by the now-flat average curves.
* **For Details:** None required. The remaining oscillations are minor; continue reporting best-ever and final-iteration errors separately.
* **For Optimality:** Freeze the present formulation and architecture. Retain periodic Fourier embeddings, vRBA, and SSBroyden. No entropy regularization, continuation, RAR, or KAN conversion is warranted for this problem.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | The code implements a 2D forward Helmholtz PINN for u_xx + u_yy + u = g with k=1 and source g=(1-17π²) sin(πx) sin(4πy). The network uses an MLP with width 30 and three hidden layers, preceded by periodic Fourier features on both x and y with degree 5 and Lx=Ly=2.0; the MLP body receives only the 20 embedding features, with no raw coordinates and no external normalization. The PDE residual is exactly r = u_xx + u_yy + u - g. The total loss is L_pde only; no boundary, data, or gauge loss is added. vRBA lambdas are used for weighting and importance sampling in Adam, and for sampling only in SSBroyden. Training uses Adam for exactly 10,000 iterations followed by SSBroyden for exactly 10,000 iterations, for a hard-capped total of 20,000. The analytical solution u_an=sin(πx)sin(4πy) is used only for validation and error metrics. | 8.00e-07 | 35 | 35 | 7 | 14 | **91** |
| 2 | The implementation is compliant with the scientific strategy. It defines a PINN for the 2D Helmholtz equation using periodic Fourier feature embeddings on both x and y (degree=5, Lx=Ly=2.0), followed by an MLP with width 30 and depth 3, and identity output. No external normalization is applied; raw coordinates enter the model and are embedded inside the model call. The loss is exactly L_pde for the residual u_xx + u_yy + u - g, with g(x,y) = (1 - 17*pi^2) sin(pi x) sin(4 pi y); no data loss, no BC loss, and no extra gauge/Neumann terms are present. Training uses the specified hybrid schedule: Adam for 10,000 iterations with vRBA weighting and sampling, followed by SSBroyden for 20,000 iterations with vRBA sampling only and a fresh Hessian initialized at the transition, for a total hard-capped budget of 30,000 iterations. The analytical solution u_an = sin(pi x) sin(4 pi y) is used only for validation/error metrics, not training. | 1.00e-06 | 35 | 35 | 7 | 14 | **91** |
| 4 | Implemented a 2D forward Helmholtz PINN. Inputs (x,y) are passed raw; the model applies independent periodic Fourier embeddings for x and y with degree=5 and Lx=Ly=2.0, producing 20 embedding features that feed an MLP with three hidden layers of width 30 and a scalar output. The PDE residual is u_xx + u_yy + u - g with g=(1-17π²)sin(πx)sin(4πy), and L_total = L_pde only, with no BC loss or data loss. Training uses Adam for exactly 10,000 iterations with vRBA weighting and sampling, followed by SSBroyden for exactly 40,000 iterations (50,000 total), with vRBA used only for sampling in SSBroyden and the Hessian reinitialized to identity at the transition. The analytical solution sin(πx)sin(4πy) is used only for validation and error metrics. Best-ever and final-iteration RL2 errors are reported separately. | 1.40e-08 | 35 | 35 | 7 | 14 | **91** |
| 5 | Implements a PINN for the 2D Helmholtz forward problem. Architecture: raw (x,y) inputs pass through separate Periodic Fourier Features per coordinate (degree=5, Lx=Ly=2.0; 10 features each, 20 total, with no raw coordinates concatenated into the MLP body), followed by an MLP with three hidden layers of width 30 and a scalar output. The PDE residual is u_xx + u_yy + u - g with k=1 and g=(1-17π²) sin(πx) sin(4πy); the analytical reference u_an=sin(πx) sin(4πy) is used only for error metrics. Loss is L_total = L_pde only; no L_data, no L_bc, no L_ic. Optimizer workflow: hybrid Adam for exactly 10,000 iterations with vRBA weighting and sampling active; then SSBroyden for exactly 120,000 effective iterations, with the Hessian reinitialized to identity at the transition and vRBA sampling only during SSBroyden (unweighted PDE loss in that stage). Periodic BCs are enforced architecturally by the Fourier embedding. Training reports best-ever and final-iteration RL2 error separately. | 1.65e-08 | 35 | 35 | 12 | 14 | **96** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves the required file Helmholtz_results.npz with keys: 'u_pred' (100x100), 'u_an' (100x100), 'error' (100x100), 'x' (100x100), and 'y' (100x100). It also saves Log_files_SSBroyden.mat containing training logs such as all_errors, all_its, all_loss, all_gamma, all_lamB, all_max_RBA, all_L_infty, and all_L_2. |
| 2 | The code saves results in Helmholtz_results.npz. The .npz file contains keys: 'u_pred' (100x100 predicted field), 'u_an' (100x100 analytical reference field), 'error' (100x100 pointwise absolute error), 'x' (100x100 meshgrid x-coordinates), and 'y' (100x100 meshgrid y-coordinates). It also saves a separate log file Log_files_SSBroyden.mat containing keys: 'model_name', 'all_errors', 'all_its', 'all_loss', 'all_gamma', 'all_lamB', 'all_max_RBA', 'all_L_infty', and 'all_L_2'. |
| 4 | Saves Helmholtz_results.npz containing keys: 'u_pred' (100x100), 'u_an' (100x100), 'error' (100x100), 'residual' (100x100), 'x' (100x100), and 'y' (100x100). It also writes Log_files_SSBroyden.mat with training logs: model_name, all_errors, all_its, all_loss, all_gamma, all_lamB, all_max_RBA, all_L_infty, and all_L_2. |
| 5 | Saves Helmholtz_results.npz in NumPy .npz format with keys: u_pred (100×100), u_an (100×100), error (100×100), pde_residual (100×100), x (100×100), y (100×100). Also saves Log_files_SSBroyden.mat in MATLAB .mat format containing training histories: model_name, all_errors, all_its, all_loss, all_gamma, all_lamB, all_max_RBA, all_L_infty, all_L_2. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run3/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1029/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run3/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1035/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run3/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1046/` |
| 5 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run3/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1056/` |


---
