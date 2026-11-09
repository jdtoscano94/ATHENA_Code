
# Experiment Report
*Last Updated: 2026-11-06 03:06:29*

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
# Strategy: 2D Helmholtz — Extended Convergence Run

## 0. Triage
**Path 2: Advisor Recommendation (MANDATORY).** Diagnosis = **Budget-Limited / Premature Stop**. Architecture is validated and working — **NO architectural changes**. The ONLY modification: **extend the SSBroyden final stage by 2–5×** and enforce a genuine plateau before termination.

## 1. Data Source
No new data. Analytical solution used for **validation/error metrics ONLY**.

**Role: Validation Data (NO L_data).**

## 2. Problem Statement
- **Governing Equation (Helmholtz, k=1):**
$$u_{xx} + u_{yy} + k^2 u = g(x,y), \quad k=1$$
- **PDE Residual:** $r(x,y) = u_{xx} + u_{yy} + u - g(x,y)$
- **Manufactured Source (derived from analytical solution):**
$$g(x,y) = (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$$
(since $u_{xx}=-\pi^2 u$, $u_{yy}=-16\pi^2 u$, $k^2u=u$)
- **BCs:** Periodic in both $x$ and $y$ on $[-1,1]^2$ — enforced **architecturally**.
- **Analytical Solution:** $u_{an}(x,y) = \sin(\pi x)\sin(4\pi y)$ ($a_1=1$, $a_2=4$)
- **Problem Type:** Forward. Well-posed for PINNs (smooth, strong-form solution exists).
- **Data Loss Term:** None.

## 2.1 / 2.2 Input-Output
- **Input:** 2D $(x,y) \in [-1,1]^2$. Both coordinates already in nice range → **NO normalization** (identity).
- **Output:** Scalar $u(x,y)$. Oscillatory ($a_2=4$), smooth, periodic.

## 3. Base Model Selection
2D steady problem → **MLP with Hybrid Adam and SSBroyden** (Case A: width=30, depth=3).

## 4. Main Architecture (UNCHANGED from Run 0 — Advisor mandate)
**Pipeline: Input → Input Transformation → Main Body → Output Transformation → Output**

- **Input Transformation:** Periodic Fourier Features on **BOTH** $x$ and $y$: **degree=5, Lx=2.0** (raw range $[-1,1]$ → range = 2.0; consistent). Computed **inside the model call function** (differentiable). **DO NOT** pass raw coordinates to the body — body receives ONLY the 20 concatenated embedding features.
- **Main Body:** MLP, width=30, depth=3, tanh.
- **Output Transformation:** Identity (periodicity handled by embedding).

## 5. Constraint Implementation
- Periodic BCs: **architecturally exact** via Periodic Fourier Features → **NO** $L_{bc}$ terms.
- Loss term count = 1 ($L_{pde}$ only) → count < 5 → loss-based, no hard constraints needed.

## 6. Loss Function Assembly
$$L_{total} = L_{pde}$$
(Single term — no global balancing needed. No continuation: $k=1$ fixed.)

## 6.1 Training Schedule (THE ONLY CHANGE)
- **Phase 1 — Adam:** 10,000 iterations. vRBA for **weighting + sampling**. Static parameters.
- **Phase 2 — SSBroyden: EXTEND from 12,500 → 50,000 iterations** (4×, within Advisor's 2–5× range). vRBA for **sampling ONLY**.
- **CRITICAL:** **Reinitialize the Hessian** at the Adam→SSBroyden transition (and on any restart from saved weights).
- **PLATEAU GATE (acceptance criterion):** Run is accepted ONLY if both loss and RL2 error **band centers are approximately horizontal over the last 50% of the final stage**. Do NOT accept best-ever dips or optimizer termination as convergence evidence. If still descending at 50,000 iters, continue up to 62,500 (5×).

## 7. Weighting Strategy
- **Local:** vRBA — `lambdas` for weighting & sampling during Adam, **sampling ONLY** during SSBroyden.
- **Global:** Single loss term → no balancing required.
- **SSBroyden Integrity:** Do NOT modify internal weighting/sampling/batching of the SSBroyden template.

## 8. Visualization
Do NOT modify visualization callbacks (already validated).

## 9. Saving Data
Save fields as .npz arrays with keys: `u_pred`, `u_analytical`, `pointwise_error`, `pde_residual`, plus loss/error histories.

---

**Main Model:**
[MLP with Hybrid Adam and SSBroyden]

**Enhancements:**
[Periodic Fourier Features (degree=5, Lx=2.0)]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
REJECTED_FOR_REFINEMENT

## Section 1: Status Decision
* **Status:** **REJECTED_FOR_REFINEMENT**
* **Problem type:** Type A — noise-free forward Helmholtz problem. The analytical solution is used only for validation, with no \(L_{\text{data}}\) or data leakage.
* **Decision rationale:** The run is stable and highly accurate, but it fails the mandatory dual-curve plateau check. Both the loss and RL2 error band centers continue decreasing through the final region, indicating a **Budget-Limited / Premature Stop**, rather than convergence.

## Section 2: Quantitative Scorecard
* **Score Integrity:** 35
* **Score Accuracy:** 35  
  * Numerical Precision: 20/20. Using \(\epsilon_{\text{target}}=10^{-6}\), the final plotted RL2 error of approximately \(10^{-7}\) comfortably meets the target.
  * Consistency: 15/15. The implementation solves the requested \(a_1=1,\ a_2=4,\ k=1\) periodic problem with the correct manufactured source and no training-data loss.
* **Score Details:** 6
* **Score Optimality:** 14
* **TOTAL SCORE:** **90**

Although the score reaches 90, approval is blocked by the still-descending final-stage curves.

## Section 3: Forensic Analysis & Feedback

### Observation
**Loss and error histories**

The final region is the segment after the Adam-to-SSBroyden transition near iteration 10,000, extending to the visible end near iteration 40,000. Its midpoint is therefore approximately iteration 25,000.

* **Total Loss:**
  * The visible values span roughly \(10^2\) initially to \(10^{-10}\) at the end.
  * Adam produces a rapid initial decrease, followed by a small transition feature near 10,000. SSBroyden then produces a sharp drop and a subsequent smooth, continuing decline.
  * **Midpoint band level:** \(A \approx 10^{-8}\) at iteration 25,000.
  * **End band level:** \(B \approx 10^{-10}\) at iteration 40,000.
  * Thus \(B\) is approximately **100 times lower** than \(A\): the loss is still decreasing on average, not plateaued.

* **Relative \(L^2\) Error:**
  * The error begins around a few \(10^{-1}\), hovers around several \(10^{-3}\) before the transition, and then drops sharply when SSBroyden begins.
  * The final segment is noisy, but the center of the oscillation band moves downward.
  * **Midpoint band level:** \(A \approx 10^{-6}\) at iteration 25,000.
  * **End band level:** \(B \approx 10^{-7}\) at iteration 40,000, with the rightmost point approximately \((0.5\text{–}1)\times10^{-7}\).
  * Thus \(B\) is approximately **10 times lower** than \(A\). This is a noisy descent, not a horizontal floor.

For the last-50% window, approximately iterations 25,000–40,000, the first-half versus second-half band centers also shift downward: roughly \(3\times10^{-9}\rightarrow3\times10^{-10}\) for loss and \(5\times10^{-7}\rightarrow1.5\times10^{-7}\) for RL2 error. Both exceed the allowed approximately \(2\times\) flatness tolerance.

**Residual and solution fields**

* The PDE residual heatmap is finite and broadly mottled across the domain, without a localized shock-like structure. The log reports residuals from approximately \(4.42\times10^{-10}\) to \(5.35\times10^{-6}\).
* Reference and prediction are visually almost indistinguishable: the oscillation pattern, phase, and amplitude agree at the displayed scale.
* Pointwise errors are largest mainly near the lower \(x\) region and portions of the outer edges, while dark-blue interior patches have substantially smaller errors. There is no visual evidence of a large boundary-condition failure.

**Log confirmation**

* The repeated `RL2 error: 0.00435141` is the Adam warm-up metric and is not used as the final SSBroyden result.
* The exact best-ever value at the bottom of the log is:
  \[
  \texttt{Minimum achieved RL2 error}=
  1.1189920837495984\times10^{-8}.
  \]
* This minimum is a best dip, not necessarily the final sustained value. For scoring, the final plotted value of approximately \(10^{-7}\) is used instead.

### Diagnosis
**Budget-Limited / Premature Stop.** The formulation, periodic embedding, and optimization pipeline are working very well. There is no divergence, missing-physics failure, or need for increased model capacity. The remaining bottleneck is simply that training ends while both curves are still improving.

### Targeted Improvements
* **For Integrity:** None.
* **For Accuracy:** Extend only the final SSBroyden stage—approximately double its present budget as the next step. If continuing from saved weights, reinitialize the Hessian on restart. Keep the PDE, degree-5 periodic Fourier embeddings, MLP, and vRBA strategy unchanged. Stop only after both loss and RL2 band centers remain on approximately the same horizontal floor over the last 50% of the final stage.
* **For Details:** Report final-iteration RL2 separately from the best-ever minimum, and do not use an isolated dip or optimizer termination as evidence of convergence.
* **For Optimality:** None. Retain the periodic embedding pipeline and SSBroyden fine-tuning. No entropy penalty is relevant to this smooth Helmholtz problem, and no KAN or architecture change is warranted.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The code implements a forward 2D Helmholtz PINN. Architecture: an MLP body with 3 hidden layers of width 30 and tanh activation; both input coordinates pass through Periodic_Fourier_Features with degree=5 and Lx=2.0, producing 20 embedding features that are concatenated before the MLP. No raw coordinates are passed to the body, no normalization is applied, and the output transformation is identity. The PDE residual is u_tt + u_xx + u - g with g=(1-17π²) sin(π t) sin(4π x), matching the manufactured solution sin(π x) sin(4π y) under the coordinate mapping used in the code. The loss contains only L_pde; no L_bc, L_ic, or L_data is used. Training uses Adam for 10,000 epochs with vRBA weighting and importance sampling, followed by SSBroyden/BFGS-style second-order optimization for 12,500 iterations in 100-step chunks with resampling. vRBA is used for weighting+sampling during Adam and for sampling only during SSBroyden. There is no PDE parameter continuation; k=1 is fixed. The analytical solution is used only for validation/error metrics. | 1.00e-06 | 35 | 35 | 6 | 14 | **90** |
| 1 | The implementation matches the strategy. It uses a PINN with Periodic Fourier Features (degree=5, Lx=2.0) applied to both coordinates, producing 20 embedding features that are concatenated and passed into an MLP body with width=30 and depth=3 using tanh activations and identity output. The PDE residual is the Helmholtz residual u_tt + u_xx + u - g with k=1 and g=(1-17*pi^2) sin(pi t) sin(4*pi x); the analytical reference is sin(pi t) sin(4*pi x). The loss is only L_pde: during Adam it is vRBA-weighted and vRBA-sampled, and during SSBroyden the PDE loss is unweighted while vRBA is used for sampling only. Adam runs for 10,000 iterations, then SSBroyden runs for 50,000 iterations with the Hessian reinitialized to identity at the transition. No L_data and no L_bc are included. | 1.00e-07 | 35 | 35 | 6 | 14 | **90** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves 'Helmholtz_predictions.npz' in NumPy .npz format. The npz contains keys: 'u_pred' with shape (100, 100) float64, 'u_analytical' with shape (100, 100) float64, 'X' with shape (100, 100) float64 representing the first coordinate grid, and 'Y' with shape (100, 100) float64 representing the second coordinate grid. It also saves 'Log_files_SSBroyden.mat' with training log arrays. |
| 1 | Saves Helmholtz_predictions.npz containing keys: 'u_pred' (100x100), 'u_analytical' (100x100), 'pointwise_error' (100x100), 'pde_residual' (100x100), 'X' (100x100), 'Y' (100x100), 'all_errors' (1055,), 'all_its' (1055,), and 'all_loss' (1055,). Also saves Log_files_SSBroyden.mat with model_name and histories: all_errors/all_its/all_loss/all_L_infty/all_L_2 length 1055; all_gamma/all_lamB/all_max_RBA length 555. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run1//Helmholtz2D_Periodic_SinPiX_Sin4PiY/2026_11_06_0252/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run1//Helmholtz2D_Periodic_SinPiX_Sin4PiY/2026_11_06_0300/` |


---
