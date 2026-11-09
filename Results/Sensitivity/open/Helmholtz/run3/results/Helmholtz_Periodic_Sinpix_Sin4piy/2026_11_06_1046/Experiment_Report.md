
# Experiment Report
*Last Updated: 2026-11-06 10:51:03*

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
# STRATEGY: 2D Helmholtz (Refinement Run — Budget Extension)

## 0. Triage
**Path 2 (Advisor — MANDATORY):** Architecture, loss, embeddings, and capacity are all correct and stable. The run is **budget-limited** (curves still descending). **ONLY change: extend final SSBroyden stage from 20,000 → 40,000 iterations** (total 50,000). Everything else stays identical to Run 2.

## 1. Data Source
No new data. Analytical solution is **Role: Validation/Error Metrics ONLY** (forward problem).

## 2. Problem Statement
- **Governing Equation (Helmholtz):** $u_{xx} + u_{yy} + k^2 u = g(x,y)$, with $k=1$.
- **PDE Residual:** $r(x,y) = u_{xx} + u_{yy} + u - g(x,y)$, where $g(x,y) = (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$ (from $k^2 - \pi^2(a_1^2+a_2^2)$, $a_1=1$, $a_2=4$).
- **BCs:** Periodic in both $x$ and $y$ on $[-1,1]\times[-1,1]$ — enforced **architecturally** (periodic embedding), no BC loss.
- **Analytical Solution:** Yes. $u_{an}(x,y) = \sin(\pi x)\sin(4\pi y)$.
- **Data Loss Term:** None.
- **Problem Type:** Forward. Well-posed for PINNs (linear elliptic, smooth strong solution).

## 2.1 / 2.2 Input & Output
- Input: 2D spatial $(x,y)$, no time. Already in $[-1,1]^2$ — **NO normalization** (identity, pass raw coordinates).
- Output: scalar $u(x,y)$. Smooth, periodic, oscillatory (1 mode in $x$, 4 modes in $y$).

## 3. Base Model
**MLP with Hybrid Adam and SSBroyden** (2D forward, low-dim).

## 4. Main Architecture
Flow: **Input $(x,y)$ → Periodic Fourier Embedding → MLP body → identity → $u$**
- **Input Transformation:** Periodic Fourier features on BOTH $x$ and $y$: **degree=5, $L_x = L_y = 2.0$** (range of coordinates entering the embedding). Computed **inside the model call**, differentiable. **Advisor mandates keeping degree=5.**
- **Input Composition Rule:** MLP body receives **ONLY** the 20 embedding features — no raw coordinates.
- **Main Body:** MLP, width=30, depth=3 (unchanged).
- **Output Transformation:** Identity. No hard constraints needed.

## 5. Constraint Strategy
Periodic BCs satisfied by architecture → **exclude** $L_{bc}$. Loss term count = 1 ($L_{pde}$) < 5 → loss-based, no hard constraints.

## 6. Loss Function
$$L_{total} = L_{pde}$$

## 6.1 Continuation
Not applicable (no Re/Peclet). No parameter annealing.

## Training Schedule (Advisor-Mandated — the ONLY change)
- **Phase 1 (Adam):** Exactly **10,000 iterations**. vRBA lambdas for weighting + sampling. Static.
- **Phase 2 (SSBroyden):** Exactly **40,000 iterations** (extended from 20,000). vRBA for **sampling only**. **Reinitialize the SSBroyden Hessian at the transition** (mandatory; also if restarting from trained weights).
- **Total hard cap: 50,000 iterations.** No open-ended stopping; halt early only on NaN/Inf loss/error.

## 7. Weighting Strategy
- **Local:** vRBA — lambdas for Weighting & Sampling during Adam, **Sampling ONLY** during SSBroyden.
- **Global:** 1 loss term → no balancing needed.
- **SSBroyden Integrity:** Do NOT modify internal weighting/sampling/batching of the SSBroyden template.

## 8. Visualization
Do NOT modify legacy visualization callbacks.

## 9. Saving
Save predicted field, analytical field, pointwise error, and PDE residual on a validation grid as `.npz` with keys specifying variable names (e.g., `u_pred`, `u_an`, `error`, `residual`). Report best-ever and final-iteration RL2 errors **separately**.

---
**Main Model:**
[MLP with Hybrid Adam and SSBroyden]

**Enhancements:**
[Periodic Fourier Features (degree=5, Lx=Ly=2.0)]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
REJECTED_FOR_REFINEMENT

## Section 1: Status Decision
* **Status:** **REJECTED_FOR_REFINEMENT**
* This is a stable, highly accurate **forward, noise-free Helmholtz problem**, but the final training stage remains budget-limited. Despite the excellent score, the loss and error bands have not both flattened.

## Section 2: Quantitative Scorecard
* **Score Integrity:** 35
* **Score Accuracy:** 35
* **Score Details:** 7
* **Score Optimality:** 14
* **TOTAL SCORE:** **91**

The score exceeds 90, but the mandatory plateau check fails: the final-stage curves are still decreasing on average.

## Section 3: Forensic Analysis & Feedback

### Observation
**Visual integrity and stability**
- The summary plot is readable, all fields are populated, and no NaN, Inf, or loss explosion is visible.
- The Adam-to-SSBroyden transition occurs at approximately iteration 10,000. The final SSBroyden region therefore spans iterations 10,000–50,000, with its midpoint at 30,000.

**Loss history**
- The vertical axis spans approximately \(10^{1}\) down to \(10^{-11}\).
- Adam produces a rapid initial decrease, followed by a small transition disturbance near 10,000. SSBroyden then produces a long, mostly smooth descent with small late-time fluctuations.
- Explicit final-region readings, using the center of the oscillation band:
  - **Loss \(A\) at iteration 30,000:** approximately **\(3\times10^{-10}\)**.
  - **Loss \(B\) at iteration 50,000:** approximately **\(1\times10^{-11}\)**.
- Since \(B\ll A\), the loss is **still decreasing on average**, not plateaued. The average band in 40,000–50,000 is also visibly below that in 30,000–40,000.

**Relative \(L^2\) error history**
- The error starts around \(10^{-1}\) or higher, decreases to roughly \(10^{-2}\) during Adam, and then drops sharply after the SSBroyden transition.
- The later curve is noisy but continues trending downward toward \(10^{-8}\).
- Explicit final-region band readings:
  - **Error \(A\) at iteration 30,000:** approximately **\(7\times10^{-8}\)**.
  - **Error \(B\) at iteration 50,000:** approximately **\(2\times10^{-8}\)**.
- Thus \(B<A\) by more than the approximate 2× flatness allowance. This is a noisy descending band, not a horizontal floor.

**Log confirmation**
- Because SSBroyden was used, the repeated `RL2 error: 0.00435141` is treated as the Adam warm-up endpoint and is not used for final scoring.
- **Minimum/best-ever RL2 error:** `9.376621500361907e-09`
- **Final-iteration RL2 error:** `1.3996374508498242e-08`
- The final value agrees with the right edge of the plotted error curve and is the value used for scoring. The slightly smaller best-ever value is not substituted for it.

**Residual and solution fields**
- Reference and prediction are visually indistinguishable: both reproduce the same periodic modal pattern, band locations, signs, and amplitudes.
- The PDE residual is spatially distributed and speckled rather than concentrated at a shock, corner, or boundary. The log reports residuals from approximately \(1.65\times10^{-10}\) to \(2.92\times10^{-6}\).
- Pointwise error is very small overall, with a plotted range extending to approximately \(10^{-8}\). Its spatial structure follows curvilinear/modal features in the interior; there is no visible large-scale boundary failure.
- The implementation solves the requested equation and source term exactly, enforces periodicity through the input embedding, and uses the analytical solution only for validation. There is no data leakage.

### Diagnosis
**Budget-Limited / Premature Stop.** The physics, architecture, and optimization pipeline are working exceptionally well, but both final-stage average bands—especially the loss—are still descending. Late oscillations are secondary; they do not justify changing the architecture or physics.

### Targeted Improvements
* **For Integrity:** None. Execution, plots, and numerical values are valid.
* **For Accuracy:** Make only a fixed budget extension: increase the final SSBroyden stage from **40,000 to 120,000 iterations** (3×), while retaining the 10,000-iteration Adam stage, for **130,000 total iterations**. Reinitialize the SSBroyden Hessian at the transition, and also if restarting from trained weights. Keep the embeddings, network, residual, and vRBA settings unchanged.
* **For Details:** The present run has no genuine final dual-curve plateau. After this fixed extension, assess convergence using the average band centers over the last 50% of the final stage; both curves must oscillate around essentially unchanged horizontal floors.
* **For Optimality:** None. Retain `Input → Periodic Fourier Embedding → MLP → Output`, vRBA, and SSBroyden. Entropy regularization and conservative shock formulations are not applicable to this smooth linear Helmholtz problem.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | The code implements a 2D forward Helmholtz PINN for u_xx + u_yy + u = g with k=1 and source g=(1-17π²) sin(πx) sin(4πy). The network uses an MLP with width 30 and three hidden layers, preceded by periodic Fourier features on both x and y with degree 5 and Lx=Ly=2.0; the MLP body receives only the 20 embedding features, with no raw coordinates and no external normalization. The PDE residual is exactly r = u_xx + u_yy + u - g. The total loss is L_pde only; no boundary, data, or gauge loss is added. vRBA lambdas are used for weighting and importance sampling in Adam, and for sampling only in SSBroyden. Training uses Adam for exactly 10,000 iterations followed by SSBroyden for exactly 10,000 iterations, for a hard-capped total of 20,000. The analytical solution u_an=sin(πx)sin(4πy) is used only for validation and error metrics. | 8.00e-07 | 35 | 35 | 7 | 14 | **91** |
| 2 | The implementation is compliant with the scientific strategy. It defines a PINN for the 2D Helmholtz equation using periodic Fourier feature embeddings on both x and y (degree=5, Lx=Ly=2.0), followed by an MLP with width 30 and depth 3, and identity output. No external normalization is applied; raw coordinates enter the model and are embedded inside the model call. The loss is exactly L_pde for the residual u_xx + u_yy + u - g, with g(x,y) = (1 - 17*pi^2) sin(pi x) sin(4 pi y); no data loss, no BC loss, and no extra gauge/Neumann terms are present. Training uses the specified hybrid schedule: Adam for 10,000 iterations with vRBA weighting and sampling, followed by SSBroyden for 20,000 iterations with vRBA sampling only and a fresh Hessian initialized at the transition, for a total hard-capped budget of 30,000 iterations. The analytical solution u_an = sin(pi x) sin(4 pi y) is used only for validation/error metrics, not training. | 1.00e-06 | 35 | 35 | 7 | 14 | **91** |
| 4 | Implemented a 2D forward Helmholtz PINN. Inputs (x,y) are passed raw; the model applies independent periodic Fourier embeddings for x and y with degree=5 and Lx=Ly=2.0, producing 20 embedding features that feed an MLP with three hidden layers of width 30 and a scalar output. The PDE residual is u_xx + u_yy + u - g with g=(1-17π²)sin(πx)sin(4πy), and L_total = L_pde only, with no BC loss or data loss. Training uses Adam for exactly 10,000 iterations with vRBA weighting and sampling, followed by SSBroyden for exactly 40,000 iterations (50,000 total), with vRBA used only for sampling in SSBroyden and the Hessian reinitialized to identity at the transition. The analytical solution sin(πx)sin(4πy) is used only for validation and error metrics. Best-ever and final-iteration RL2 errors are reported separately. | 1.40e-08 | 35 | 35 | 7 | 14 | **91** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves the required file Helmholtz_results.npz with keys: 'u_pred' (100x100), 'u_an' (100x100), 'error' (100x100), 'x' (100x100), and 'y' (100x100). It also saves Log_files_SSBroyden.mat containing training logs such as all_errors, all_its, all_loss, all_gamma, all_lamB, all_max_RBA, all_L_infty, and all_L_2. |
| 2 | The code saves results in Helmholtz_results.npz. The .npz file contains keys: 'u_pred' (100x100 predicted field), 'u_an' (100x100 analytical reference field), 'error' (100x100 pointwise absolute error), 'x' (100x100 meshgrid x-coordinates), and 'y' (100x100 meshgrid y-coordinates). It also saves a separate log file Log_files_SSBroyden.mat containing keys: 'model_name', 'all_errors', 'all_its', 'all_loss', 'all_gamma', 'all_lamB', 'all_max_RBA', 'all_L_infty', and 'all_L_2'. |
| 4 | Saves Helmholtz_results.npz containing keys: 'u_pred' (100x100), 'u_an' (100x100), 'error' (100x100), 'residual' (100x100), 'x' (100x100), and 'y' (100x100). It also writes Log_files_SSBroyden.mat with training logs: model_name, all_errors, all_its, all_loss, all_gamma, all_lamB, all_max_RBA, all_L_infty, and all_L_2. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run3/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1029/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run3/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1035/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run3/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1046/` |


---
