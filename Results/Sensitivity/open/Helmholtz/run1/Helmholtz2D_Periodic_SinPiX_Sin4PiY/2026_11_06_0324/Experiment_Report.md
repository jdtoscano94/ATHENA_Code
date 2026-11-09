
# Experiment Report
*Last Updated: 2026-11-06 03:31:03*

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
# Strategy: EXPLORATION RUN 3 of 5 (2D Helmholtz)

**Triage:** Path 2 (Advisor). Verdict: Run 2 (96) is numerically converged and **preserved as current best**. Rejection is purely the exploration protocol (2/5 complete). Advisor mandates: preserve verified pipeline (periodic Fourier embeddings on both coords, no raw-coordinate bypass, vRBA, SSBroyden fine-tuning); **no KAN, no stream-function, no entropy regularization**. Do not chase the isolated 3.2e-9 dip.

**Exploration variation (single knob):** Periodic embedding degree **5 → 6**. Everything else identical to Run 2.

## 1. Data Source
No new data. Analytical solution used for validation only.

## 2. Problem Statement
* **Governing PDE (Helmholtz, k=1):** $u_{xx} + u_{yy} + k^2 u = g(x,y)$
* **Residual:** $r = u_{xx} + u_{yy} + u - g$, with $g(x,y) = (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$
* **BCs:** Periodic in x and y — enforced **architecturally** (Fourier embedding), no loss term.
* **Analytical solution (manufactured):** $u_{an}(x,y) = \sin(\pi x)\sin(4\pi y)$ (a1=1, a2=4, k=1)
* **Data Role:** Condition B — **Validation/Error Metrics ONLY.** No L_data.
* **Problem Type:** Forward. Well-posed: smooth strong-form solution exists.
* **Note:** Template may name inputs (t,x); map t→x, x→y as in verified Runs 0–2.

## 2.1 / 2.2 Dimensions
* Input: 2D spatial, $(x,y) \in [-1,1]^2$. Output: scalar $u$. Smooth, oscillatory (mode 4 in y), periodic.

## 3. Base Model
**MLP with Hybrid Adam and SSBroyden** (Case A: 2D → width=30, depth=3).

## 4. Architecture
**Flow:** $(x,y)$ → Periodic Fourier Features (each coord) → concat → MLP body → identity output → $u$
* **Normalization:** NONE — both coords already in $[-1,1]$ (nice range; skip scaler).
* **Input transformation:** Periodic Fourier Features on **BOTH** coordinates, **degree=6** (exploration knob), **Lx = 2.0** (raw range $[-1,1]$ entering embedding). Concatenate 24 embedding features. **FORBIDDEN: raw coordinates to the MLP body — embeddings only.** All ops inside the model call function.
* **Main body:** MLP, width=30, depth=3, tanh.
* **Output transformation:** Identity (no Dirichlet to impose; periodicity is architectural).

## 5. Constraint Strategy
Loss-term count: **1** (L_pde only; periodic BC excluded — architectural). Count < 5 → **Loss-Based, no hard constraints.**

## 6. Loss Assembly
$$L_{total} = L_{pde}$$

## 6.1 Continuation
N/A (k=1 fixed, no high Re).

**Optimization schedule (identical to Run 2):**
* **Phase 1 — Adam:** 10,000 iterations, vRBA weighting + sampling.
* **Phase 2 — SSBroyden:** 100,000 iterations, Nchange=100 inner chunks, **Hessian reinitialized to identity at transition**, vRBA for sampling only.
* **Plateau stopping test (retain — Details scorer):** dual-curve test on loss AND RL2 over trailing 50% of SSBroyden stage (early/late band centers, both change factors ≤ 2); extend beyond budget if not flat.
* **Report final-iteration RL2 and best-ever RL2 separately** (best-ever explicitly not convergence evidence).

## 7. Weighting
* **Local:** vRBA (`lambdas`) — weighting & sampling during Adam, **sampling ONLY** during SSBroyden.
* **Global:** Total terms = 1 → no balancing needed.
* **SSBroyden Integrity:** Do NOT modify internal weighting/sampling/batching of the SSBroyden template.

## 8. Visualization
Do NOT modify template visualization callbacks.

## 9. Saving
Save fields as .npz arrays with keys specifying variable names (u_pred, u_analytical, residual, loss/RL2 histories).

---
**Main Model:** MLP with Hybrid Adam and SSBroyden
**Enhancements:** Periodic Fourier Features (degree=6, Lx=2.0)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
REJECTED_FOR_REFINEMENT

## Section 1: Status Decision
* **Status:** **REJECTED_FOR_REFINEMENT**

The run is numerically successful and converged. Rejection is **only for the exploration protocol**: this is exploration run **3/5**, with three attempts since the first successful Run 0. Two further exploration runs are required before stopping and selecting the best model.

## Section 2: Quantitative Scorecard
* **Score Integrity:** 35
* **Score Accuracy:** 35
* **Score Details:** 12
* **Score Optimality:** 14
* **TOTAL SCORE:** **96**

## Section 3: Forensic Analysis & Feedback

### Observation
* **Integrity:** The plot is readable, training completed, and all reported values are finite. There is no explosion, NaN, or Inf behavior.
* **Loss history:** The loss falls from approximately \(O(10^1)\) at initialization to a noisy floor around \(O(10^{-13})\). The Adam-to-SSBroyden transition near iteration 10,000 is visible, followed by a steep decrease and then a noisy, approximately horizontal final region.
* **RL2 history:** The error decreases from approximately \(O(10^{-1}\text{–}10^0)\) to \(O(10^{-8})\). The Adam warm-up value of `0.00342719` is not the final SSBroyden result.
* **Final-stage dual-curve test:** The SSBroyden stage runs approximately from iteration 10,000 to 110,000, putting its midpoint near 60,000:
  * **Loss:** midpoint band \(A \approx 2.0\times10^{-13}\); end band \(B \approx 1.2\times10^{-13}\).
  * **RL2:** midpoint band \(A \approx 2.1\times10^{-8}\); end band \(B \approx 1.7\times10^{-8}\).
  
  These are the same horizontal noise floors within the required factor-of-two criterion. The log confirms this with trailing-window loss centers of `2.061098e-13 / 1.229417e-13` and RL2 centers of `2.143004e-08 / 1.658889e-08`, giving change factors of **1.676** and **1.292**. Thus, neither curve is still meaningfully descending on average.
* **Residual heatmap:** The residual is mottled and broadly distributed, without a localized shock-like structure or systematic boundary failure. The reported residual range is approximately \(7.94\times10^{-11}\) to \(1.63\times10^{-6}\).
* **Solution fields:** Reference and prediction are visually indistinguishable at the plotted scale. The oscillatory bands and sign reversal across the domain are reproduced correctly.
* **Pointwise error:** The error is extremely small and shows no coherent boundary artifact. A small cyan/green lower-error patch appears near \(t\approx-0.5,\ x\approx-0.5\); otherwise, the error is broadly distributed.
* **Log extraction:**
  * **Best-ever minimum RL2:** `3.5860967299230598e-09` — an isolated dip, not used for scoring.
  * **Final-iteration RL2:** **`1.4160796322438660e-08`** — used for accuracy scoring.
* **Consistency:** The implementation solves the requested Helmholtz problem with \(k=1\), the correct manufactured source, and periodic Fourier embeddings on both coordinates. The analytical solution is used only for validation, so there is no data leakage.

### Diagnosis
**Converged, high-accuracy forward PINN; exploration protocol incomplete.** There is no budget-limited premature stop, instability, stagnation, or physics mismatch. The remaining visible final-stage noise prevents a perfect Details score, but it does not invalidate the plateau.

This run ties Run 2’s score of 96 and improves the **final-iteration** RL2 relative to Run 2’s \(3.51\times10^{-8}\). It should be preserved as a leading candidate.

### Targeted Improvements
* **For Integrity:** None.
* **For Accuracy:** No additional iterations or physics changes are indicated for this run. Complete exploration runs **4/5 and 5/5** using controlled single-knob variations, then stop and select the best preserved run.
* **For Details:** None. Retain the dual-curve plateau test and the separate reporting of final-iteration and best-ever RL2.
* **For Optimality:** Preserve the verified pipeline: periodic Fourier embeddings on both coordinates with no raw-coordinate bypass, vRBA, and SSBroyden fine-tuning. KANs, entropy penalties, stream-function formulations, and positivity transforms are not warranted for this smooth scalar Helmholtz problem.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The code implements a forward 2D Helmholtz PINN. Architecture: an MLP body with 3 hidden layers of width 30 and tanh activation; both input coordinates pass through Periodic_Fourier_Features with degree=5 and Lx=2.0, producing 20 embedding features that are concatenated before the MLP. No raw coordinates are passed to the body, no normalization is applied, and the output transformation is identity. The PDE residual is u_tt + u_xx + u - g with g=(1-17π²) sin(π t) sin(4π x), matching the manufactured solution sin(π x) sin(4π y) under the coordinate mapping used in the code. The loss contains only L_pde; no L_bc, L_ic, or L_data is used. Training uses Adam for 10,000 epochs with vRBA weighting and importance sampling, followed by SSBroyden/BFGS-style second-order optimization for 12,500 iterations in 100-step chunks with resampling. vRBA is used for weighting+sampling during Adam and for sampling only during SSBroyden. There is no PDE parameter continuation; k=1 is fixed. The analytical solution is used only for validation/error metrics. | 1.00e-06 | 35 | 35 | 6 | 14 | **90** |
| 1 | The implementation matches the strategy. It uses a PINN with Periodic Fourier Features (degree=5, Lx=2.0) applied to both coordinates, producing 20 embedding features that are concatenated and passed into an MLP body with width=30 and depth=3 using tanh activations and identity output. The PDE residual is the Helmholtz residual u_tt + u_xx + u - g with k=1 and g=(1-17*pi^2) sin(pi t) sin(4*pi x); the analytical reference is sin(pi t) sin(4*pi x). The loss is only L_pde: during Adam it is vRBA-weighted and vRBA-sampled, and during SSBroyden the PDE loss is unweighted while vRBA is used for sampling only. Adam runs for 10,000 iterations, then SSBroyden runs for 50,000 iterations with the Hessian reinitialized to identity at the transition. No L_data and no L_bc are included. | 1.00e-07 | 35 | 35 | 6 | 14 | **90** |
| 2 | Implementation matches the strategy. Architecture: MLP PINN with width 30 and three hidden layers (features [30,30,30,1]) using tanh activation and identity output. Input transformation uses Periodic Fourier Features on BOTH coordinates only, degree=5 with Lx=2.0, concatenated before the MLP body; no raw coordinates and no external normalization. PDE is Helmholtz with k=1: residual u_tt + u_xx + u - g, with g=(1-17*pi^2)sin(pi*t)sin(4*pi*x), matching the strategy under the t->x, x->y mapping. Analytical solution sin(pi*T)sin(4*pi*X) is used for validation/RL2 only. Loss assembly contains exactly one PDE loss term: during Adam it is vRBA-weighted mean squared residual with vRBA weighting and sampling; during SSBroyden it is unweighted mean squared residual with vRBA used for sampling only. No data, BC, IC, Neumann, gauge, or other ghost loss terms are included. Optimizer workflow: Adam for 10,000 iterations with vRBA, followed by SSBroyden for Nbfgs=100,000 using Nchange=100 inner iterations and Hessian reinitialized to identity at the Adam-to-SSBroyden transition. The plateau stopping criterion is implemented: after the 100,000 budget, it checks loss and RL2 band centers over the trailing 50% of the SSBroyden stage split into early/late halves, requires both change factors <= 2, and continues beyond budget if not flat. Final-iteration RL2 and best-ever minimum RL2 are reported separately, with the text noting that best-ever is not convergence evidence. | 3.51e-08 | 35 | 35 | 12 | 14 | **96** |
| 3 | Implements the specified 2D Helmholtz forward PINN. Architecture: MLP width 30, depth 3, tanh activation. Inputs (t,x) are mapped to (x,y); both coordinates pass through Periodic Fourier Features with degree=6 and Lx=2.0, producing 12 features per coordinate and 24 concatenated features. No raw-coordinate bypass is used. The sole loss is L_pde; residual is u_tt + u_xx + u - g with g=(1-17π²)sin(πt)sin(4πx). During Adam, vRBA weighting and sampling are used; during SSBroyden, vRBA is used for sampling only and the PDE loss is unweighted mean(residual²). Optimizer workflow matches the strategy: Adam for 10,000 iterations, then SSBroyden for 100,000 iterations with Nchange=100, Hessian initialized to identity at transition, and template-internal SSBroyden mechanics preserved. A dual-curve plateau test is implemented over the trailing 50% of the SSBroyden stage using loss and RL2 band centers with change factors ≤2, extending beyond budget if not flat. Final-iteration RL2 and best-ever RL2 are reported separately, with best-ever explicitly not treated as convergence evidence. Analytical solution matches u_an=sin(πt)sin(4πx). | 1.42e-08 | 35 | 35 | 12 | 14 | **96** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves 'Helmholtz_predictions.npz' in NumPy .npz format. The npz contains keys: 'u_pred' with shape (100, 100) float64, 'u_analytical' with shape (100, 100) float64, 'X' with shape (100, 100) float64 representing the first coordinate grid, and 'Y' with shape (100, 100) float64 representing the second coordinate grid. It also saves 'Log_files_SSBroyden.mat' with training log arrays. |
| 1 | Saves Helmholtz_predictions.npz containing keys: 'u_pred' (100x100), 'u_analytical' (100x100), 'pointwise_error' (100x100), 'pde_residual' (100x100), 'X' (100x100), 'Y' (100x100), 'all_errors' (1055,), 'all_its' (1055,), and 'all_loss' (1055,). Also saves Log_files_SSBroyden.mat with model_name and histories: all_errors/all_its/all_loss/all_L_infty/all_L_2 length 1055; all_gamma/all_lamB/all_max_RBA length 555. |
| 2 | Saves Helmholtz_predictions.npz containing required keys: 'u_pred' (100x100), 'u_exact' (100x100), 'error' (100x100 pointwise absolute error), 'X' (100x100 coordinate grid, from T meshgrid), and 'Y' (100x100 coordinate grid, from X meshgrid). It also saves extra keys: 'pde_residual' (100x100), 'all_errors' (1D variable length log), 'all_its' (1D variable length log), and 'all_loss' (1D variable length log). Additionally saves Log_files_SSBroyden.mat with training histories and metadata. Required .npz fields are present. |
| 3 | Saves 'Helmholtz_predictions.npz' (NPZ format) containing keys: 'u_pred' (100x100 prediction field), 'u_exact' (100x100 analytical/reference field, equivalent to the strategy's u_analytical), 'error' (100x100 absolute pointwise error), 'pde_residual' (100x100 signed PDE residual), 'X' (100x100 first-coordinate meshgrid), 'Y' (100x100 second-coordinate meshgrid), 'all_errors' (1D logged RL2 history), 'all_its' (1D logged iteration history), and 'all_loss' (1D logged loss history). Also saves 'Log_files_SSBroyden.mat' with keys: model_name, all_errors, all_its, all_loss, all_gamma, all_lamB, all_max_RBA, all_L_infty, all_L_2. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run1//Helmholtz2D_Periodic_SinPiX_Sin4PiY/2026_11_06_0252/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run1//Helmholtz2D_Periodic_SinPiX_Sin4PiY/2026_11_06_0300/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run1//Helmholtz2D_Periodic_SinPiX_Sin4PiY/2026_11_06_0312/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run1//Helmholtz2D_Periodic_SinPiX_Sin4PiY/2026_11_06_0324/` |


---
