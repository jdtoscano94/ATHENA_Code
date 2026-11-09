
# Experiment Report
*Last Updated: 2026-11-06 10:40:42*

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
# Strategy: 2D Helmholtz (Refinement Run — Extended SSBroyden Budget)

## 0. Triage
**Path 2: Advisor Recommendation (MANDATORY).** Diagnosis: Budget-limited premature stop — loss still decreasing at right edge. **Cure: Double ONLY the final SSBroyden stage 10,000 → 20,000 iterations (30,000 total), keep 10,000-iteration Adam stage. Keep physics, Fourier degree, capacity, and vRBA settings UNCHANGED.** If implemented as a restart from trained weights, reinitialize the SSBroyden Hessian.

## 1. Data Source
No new data. Analytical solution used for validation only.
**Role: Validation/Error Metrics ONLY.**

## 2. Problem Statement
- **Governing Equation (Helmholtz):** $u_{xx} + u_{yy} + k^2 u = g(x,y)$, with $k=1$.
- **PDE Residual:** $r = u_{xx} + u_{yy} + u - g(x,y)$, where the manufactured source is $g(x,y) = (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$ (since $a_1=1, a_2=4$: $-\pi^2 a_1^2 - \pi^2 a_2^2 + k^2 = 1 - 17\pi^2$).
- **BCs:** Periodic in both $x$ and $y$ on $[-1,1]\times[-1,1]$.
- **Analytical Solution:** YES — $u_{an}(x,y) = \sin(\pi x)\sin(4\pi y)$.
- **Data Loss Term:** No.
- **Problem Type:** Forward.
- **Well-posedness:** Yes — smooth elliptic problem admitting a strong solution; ideal for PINNs.

### 2.1 Input
- 2D: $(x,y) \in [-1,1]\times[-1,1]$. No time.

### 2.2 Output
- Scalar $u(x,y)$. Periodic, smooth, oscillatory (1 period in $x$, 4 periods in $y$).

## 3. Base Model Selection
2D forward (low-dim) → **MLP with Hybrid Adam and SSBroyden**. Base MLP: `width=30, depth=3`.

## 4. Main Architecture
Pipeline: **Input → Input Transformation → Main Body → Output Transformation → Output**

- **Normalization:** NONE. Both $x,y$ already in $[-1,1]$ (nice range) — pass raw coordinates; adding redundant scalers introduces `Lx` mismatch bugs.
- **Input Transformation:** Periodic Fourier Features on BOTH $x$ and $y$: **degree=5** (unchanged per Advisor), **`Lx = Ly = 2.0`** (range of raw coordinates entering the embedding: $1-(-1)=2$). All transformations differentiable, inside the model's call function.
- **Input Composition Rule:** Main body receives **ONLY** the 20 embedding features (10 per coordinate). NO bare $x,y$ passed to the MLP.
- **Main Body:** MLP, width=30, depth=3 (unchanged).
- **Output Transformation:** None (identity). Periodicity is enforced by construction; no Dirichlet ansatz needed.

## 5. Constraint Implementation Strategy
- Periodic BCs: **architecturally satisfied** by the periodic embedding → `L_bc_periodic` EXCLUDED.
- Soft loss count: `L_pde` = 1 term → count < 5 → **Loss-Based, no hard constraints**.

## 6. Loss Function Assembly
$$L_{total} = L_{pde}$$

## 6.1 Continuation Strategy
Not applicable (no Reynolds/Peclet).

**Training Schedule (per Advisor — the ONLY change):**
- **Phase 1 (Adam):** 10,000 iterations, static. No parameter changes.
- **Phase 2 (SSBroyden):** **20,000 iterations (DOUBLED from 10,000)**. If restarted from the trained weights, **REINITIALIZE the SSBroyden Hessian** at the stage transition. Total budget: **30,000 iterations, hard-capped.**
- Keep physics, Fourier degree=5, network capacity, and vRBA settings unchanged. Stop if loss/error becomes NaN/Inf.

## 7. Weighting Strategy
- **Local (`lambdas`):** vRBA — used for **Weighting & Sampling during Adam, Sampling ONLY during SSBroyden** (never frozen, used differently).
- **Global (`lam`):** Total terms = 1 → no balancing needed.
- **SSBroyden Integrity Rule:** Do NOT interfere with SSBroyden's internal weighting, sampling, or batching — it is tuned for stability.

## 8. Visualization Strategy
Do NOT modify, delete, or improve the template visualization callbacks.

## 9. Saving Data
Save predicted field and reference solution as `.npz` arrays with keys specifying variable names (e.g., `u_pred`, `u_an`, `x`, `y`) for visualization.

---

## Modules for Librarian
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
* **Status:** REJECTED_FOR_REFINEMENT

This is a stable, accurate **Type A forward problem**, but both final-stage curves are still decreasing. The run therefore remains **budget-limited**, despite its excellent score and very low best-ever error.

## Section 2: Quantitative Scorecard
* **Score Integrity:** 35
* **Score Accuracy:** 35
* **Score Details:** 7
* **Score Optimality:** 14
* **TOTAL SCORE:** **91**

The score exceeds 90, but the mandatory final-stage plateau check fails. Consequently, the run cannot be approved.

## Section 3: Forensic Analysis & Feedback

### Observation
* **Visual integrity:** All six plots are readable and finite. There is no blank plot, NaN, Inf, or explosive loss magnitude.
* **Loss history:** The y-axis spans approximately \(10^3\) down to nearly \(10^{-10}\). There are two clear stages:
  * Adam from 0–10,000 iterations, with rapid initial descent followed by slower, slightly irregular improvement.
  * SSBroyden from 10,000–30,000 iterations, beginning with a transition spike and then a smooth, sustained descent.
* **Final-stage loss dual-point check:** The final stage is 10,000–30,000 iterations, so its midpoint is 20,000:
  * **A, loss at the midpoint:** approximately **\(2\times10^{-8}\)**.
  * **B, loss at the end:** approximately **\(2\times10^{-10}\)**.
  * Since \(B\ll A\), the loss is **still decreasing on average**, not flat.
* **Relative \(L^2\) error history:** The error decreases from order \(10^0\) to a few \(10^{-3}\) during Adam. After the SSBroyden transition, it drops sharply through \(10^{-4}\), then continues downward with substantial oscillations.
* **Final-stage error dual-point check:**
  * **A, average error band at 20,000:** approximately **\(8\times10^{-6}\)**, roughly the \(10^{-5}\) level.
  * **B, average error band at 30,000:** approximately **\(1\times10^{-6}\)**.
  * The terminal band is therefore still lower than the midpoint band. Noise does not alter this conclusion: the error is **descending on average**, not plateaued.
* **Final error used for scoring:** The rightmost plotted error is approximately **\(1\times10^{-6}\)**. This final plotted level—not an earlier lucky minimum—is used for scoring.
* **Residual heatmap:** Residuals are spatially distributed orange/yellow patches with greener and bluer speckles. There is no localized shock-like structure or dominant boundary failure. The log reports residual extrema of approximately \(1.34\times10^{-10}\) and \(1.56\times10^{-5}\).
* **Reference versus prediction:** The fields are visually very similar. Both show the correct single sinusoidal modulation in \(x\), four oscillations in \(y\), and matching amplitude and phase. No visible boundary artifacts are apparent.
* **Pointwise error:** The error is broadly distributed. The blue/cyan oval near \(x\approx0.1,\ y\approx-0.35\) is a **low-error region**, while the larger errors form the diffuse red/orange background rather than a localized defect.
* **Log extraction:** The exact best-ever value is:
  * **`Minimum achieved RL2 error: 3.501324611136703e-08`**
  * This is the minimum attained at some point during training, not the final sustained error. The earlier `RL2 error: 0.00263673` lines are not used as the final SSBroyden error under the dual-phase extraction protocol.
* **Scientific consistency:** The source term \(g=(1-17\pi^2)\sin(\pi x)\sin(4\pi y)\), periodic Fourier embeddings, PDE-only training loss, and validation-only use of the analytical solution are all consistent with the request. There is no data leakage.

### Diagnosis
**Budget-Limited / Premature Stop.** The method is stable and accurate, but neither the loss nor the error has established a horizontal average band over the last 50% of the final stage. The excellent minimum \(3.5013\times10^{-8}\) does not establish convergence because the model did not remain at that level.

### Targeted Improvements
* **For Integrity:** None.
* **For Accuracy:** Increase **only the final SSBroyden stage from 20,000 to 40,000 iterations**, giving a fixed total budget of **50,000 iterations** with the existing 10,000-iteration Adam stage. Keep the PDE, source term, Fourier degree 5, MLP capacity, and vRBA settings unchanged. If this is implemented as a restart from trained weights, reinitialize the SSBroyden Hessian.
* **For Details:** After that fixed-budget run, reassess both curves using their average bands over the last 50% of the final stage. Continue to report best-ever and final-iteration errors separately.
* **For Optimality:** None. Retain the periodic Fourier input embeddings and low-dimensional SSBroyden fine-tuning. Entropy regularization, stream-function formulations, and positivity transforms are not relevant to this scalar linear Helmholtz problem.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | The code implements a 2D forward Helmholtz PINN for u_xx + u_yy + u = g with k=1 and source g=(1-17π²) sin(πx) sin(4πy). The network uses an MLP with width 30 and three hidden layers, preceded by periodic Fourier features on both x and y with degree 5 and Lx=Ly=2.0; the MLP body receives only the 20 embedding features, with no raw coordinates and no external normalization. The PDE residual is exactly r = u_xx + u_yy + u - g. The total loss is L_pde only; no boundary, data, or gauge loss is added. vRBA lambdas are used for weighting and importance sampling in Adam, and for sampling only in SSBroyden. Training uses Adam for exactly 10,000 iterations followed by SSBroyden for exactly 10,000 iterations, for a hard-capped total of 20,000. The analytical solution u_an=sin(πx)sin(4πy) is used only for validation and error metrics. | 8.00e-07 | 35 | 35 | 7 | 14 | **91** |
| 2 | The implementation is compliant with the scientific strategy. It defines a PINN for the 2D Helmholtz equation using periodic Fourier feature embeddings on both x and y (degree=5, Lx=Ly=2.0), followed by an MLP with width 30 and depth 3, and identity output. No external normalization is applied; raw coordinates enter the model and are embedded inside the model call. The loss is exactly L_pde for the residual u_xx + u_yy + u - g, with g(x,y) = (1 - 17*pi^2) sin(pi x) sin(4 pi y); no data loss, no BC loss, and no extra gauge/Neumann terms are present. Training uses the specified hybrid schedule: Adam for 10,000 iterations with vRBA weighting and sampling, followed by SSBroyden for 20,000 iterations with vRBA sampling only and a fresh Hessian initialized at the transition, for a total hard-capped budget of 30,000 iterations. The analytical solution u_an = sin(pi x) sin(4 pi y) is used only for validation/error metrics, not training. | 1.00e-06 | 35 | 35 | 7 | 14 | **91** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves the required file Helmholtz_results.npz with keys: 'u_pred' (100x100), 'u_an' (100x100), 'error' (100x100), 'x' (100x100), and 'y' (100x100). It also saves Log_files_SSBroyden.mat containing training logs such as all_errors, all_its, all_loss, all_gamma, all_lamB, all_max_RBA, all_L_infty, and all_L_2. |
| 2 | The code saves results in Helmholtz_results.npz. The .npz file contains keys: 'u_pred' (100x100 predicted field), 'u_an' (100x100 analytical reference field), 'error' (100x100 pointwise absolute error), 'x' (100x100 meshgrid x-coordinates), and 'y' (100x100 meshgrid y-coordinates). It also saves a separate log file Log_files_SSBroyden.mat containing keys: 'model_name', 'all_errors', 'all_its', 'all_loss', 'all_gamma', 'all_lamB', 'all_max_RBA', 'all_L_infty', and 'all_L_2'. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run3/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1029/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run3/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1035/` |


---
