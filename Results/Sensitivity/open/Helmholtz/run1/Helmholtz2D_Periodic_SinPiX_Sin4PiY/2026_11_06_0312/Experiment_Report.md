
# Experiment Report
*Last Updated: 2026-11-06 03:20:57*

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
# STRATEGY: Advisor-Driven Refinement (Path 2)

**Triage:** Advisor diagnosis = **Budget-Limited / Premature Stop**. Formulation, embedding, and optimizer pipeline are correct. **Mandatory cure: extend the SSBroyden stage ~2×, reinitialize Hessian on restart, add a plateau stopping criterion.** Architecture and loss stay UNCHANGED.

---

## 1. Data Source
* No new data. **Role: Validation/Error Metrics ONLY** (forward problem; analytical solution used solely for RL2).

## 2. Problem Statement
* **Governing Equation (Helmholtz, k=1):**
$$u_{xx} + u_{yy} + k^2 u = g(x,y), \quad k=1$$
* **PDE Residual:**
$$r(x,y) = u_{xx} + u_{yy} + u - g(x,y), \quad g(x,y) = (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$$
* **BCs:** Periodic in both $x$ and $y$ on $[-1,1]\times[-1,1]$.
* **Analytical Solution (EXISTS — use for validation):**
$$u_{an}(x,y) = \sin(\pi a_1 x)\sin(\pi a_2 y) = \sin(\pi x)\sin(4\pi y), \quad a_1=1,\ a_2=4$$
* **Problem Type:** Forward. No data loss term. Well-posed for PINNs (smooth strong-form solution).
* **Note:** If the template uses $(t,x)$ coordinates, map $t\to x$, $x\to y$ exactly as in prior accepted runs.

## 3. Base Model Selection
* 2D problem → **MLP with Hybrid Adam and SSBroyden** (Case A: width=30, depth=3).

## 4. Main Architecture (UNCHANGED from Run 1)
* **Input Transformation:** $(x,y)\in[-1,1]^2$ — already in nice range, **NO normalization** (identity). **Periodic Fourier Features on BOTH coordinates: degree=5, Lx=2.0** (range $=1-(-1)=2$). Feed ONLY the embeddings to the body — no raw coordinates. All operations inside the model's call function.
* **Main Body:** MLP, width=30, depth=3, tanh.
* **Output Transformation:** Identity. Periodicity is exact by construction → no boundary ansatz needed.

## 5. Constraint Strategy
* Periodic BCs → enforced architecturally (embedding). No IC (steady problem). Loss term count = 1 → loss-based, no hard constraints.

## 6. Loss Assembly
$$L_{total} = L_{pde}$$

## 6.1 Continuation
* None. $k=1$ is fixed and benign — no parameter continuation.

## 7. Weighting
* **Local:** vRBA — weighting + sampling during Adam; **sampling ONLY during SSBroyden** (never frozen, role switches).
* **Global:** Single loss term → no balancing needed.
* **SSBroyden Integrity:** Do NOT modify internal SSBroyden weighting/batching.

## 7.1 Training Schedule (THE ONLY CHANGE)
* **Phase 1 — Adam:** 10,000 iterations, static $k=1$. Unchanged.
* **Phase 2 — SSBroyden: EXTENDED to ~100,000 iterations** (double the previous 50,000). **Reinitialize the Hessian at the Adam→SSBroyden transition; if resuming from saved weights, reinitialize the Hessian at restart.**
* **Stopping Criterion (mandatory):** Stop only when BOTH loss and RL2 error band centers remain flat (≤ ~2× change) over the last 50% of the final stage. If still descending at budget end, extend further.
* **Reporting (mandatory):** Report final-iteration RL2 **separately** from the best-ever minimum. Convergence must be judged on the sustained plateau, never on an isolated dip.

## 8. Visualization
* Do NOT modify template visualization callbacks.

## 9. Saving Data
* Save fields as `.npz` with keys: `u_pred`, `u_exact`, `X`, `Y` (plus pointwise `error`) for visualization.

---

**Main Model:**
* MLP with Hybrid Adam and SSBroyden

**Enhancements:**
* Periodic Fourier Features (degree=5, Lx=2.0, applied to both coordinates)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
REJECTED_FOR_REFINEMENT

## Section 1: Status Decision
* **Status:** `REJECTED_FOR_REFINEMENT`
* **Numerical verdict:** The current run, effectively **Run 2**, is stable, accurate, and genuinely plateaued. It passes the numerical approval gates and is the **best model so far**.
* **Campaign verdict:** Rejection is solely for the mandatory exploration protocol. Run 0 was the first run scoring ≥90; only Run 1 and the current Run 2 have followed it, giving **2 of the required 5 post-success exploration runs**. Three further exploration runs are required before declaring exploration complete and stopping.

## Section 2: Quantitative Scorecard
* **Score Integrity:** 35
* **Score Accuracy:** 35  
  * Numerical Precision: 20/20
  * Consistency and Strategy: 15/15
* **Score Details:** 12
* **Score Optimality:** 14
* **TOTAL SCORE:** **96**

## Section 3: Forensic Analysis & Feedback

### Observation
* **Problem type:** Type A, forward simulation of the requested periodic 2D Helmholtz problem. The implemented source \((1-17\pi^2)\sin(\pi x)\sin(4\pi y)\), manufactured solution, and \(k=1\) are consistent with the request. The analytical solution is used only for validation/RL2, so there is no data-loss leakage.
* **Visual integrity:** All six plots are readable and finite. There is no blank-plot failure, NaN, or divergence.

#### Loss and error histories
* The curves show two stages: Adam through approximately iteration 10,000, followed by SSBroyden through approximately iteration 110,000. The transition is followed by several orders of magnitude of improvement.
* **Total Loss:** plotted values span roughly \(10^{1}\) initially to a final noise floor around \(10^{-13}\). For the final SSBroyden stage:
  * Midpoint, approximately iteration 60,000: **A ≈ \(2.0\times10^{-13}\)**.
  * End-band center, approximately iteration 110,000: **B ≈ \(1.2\times10^{-13}\)**.
  * These are within a factor of two and represent the same horizontal noise floor, rather than continued average descent.
* **Relative \(L^2\) error:** values fall from approximately \(5\times10^{-1}\) initially to a final oscillation band around \(1.5\times10^{-8}\), with isolated dips near a few \(10^{-9}\).
  * Final-stage midpoint: **A ≈ \(1.8\times10^{-8}\)**.
  * End-band center: **B ≈ \(1.5\times10^{-8}\)**.
  * The rightmost individual point is an upward fluctuation, approximately \(3.5\times10^{-8}\), not a collapse of the plateau.
* Thus, **both curves pass the dual-curve plateau test**. The remaining oscillations justify 12 rather than 15 details points, but they are bounded oscillations around fixed floors—not a budget-limited descent.

The log confirms the visual reading with trailing-window diagnostics:
* Loss early/late band centers: \(1.723659\times10^{-13}\) / \(1.175205\times10^{-13}\), change factor **1.467**.
* RL2 early/late band centers: \(1.731721\times10^{-8}\) / \(1.525053\times10^{-8}\), change factor **1.136**.

#### Error reporting
* **Final-iteration RL2 used for scoring:**  
  \[
  \boxed{3.5052900529578033\times10^{-8}}
  \]
* **Exact best-ever/minimum achieved RL2 reported in the log:**  
  \[
  \boxed{3.2395880966560680\times10^{-9}}
  \]
  This is explicitly an isolated dip and is **not** used as the final score or as plateau evidence.
* The earlier `RL2 error: 0.00263673` belongs to the Adam warm-up reporting and is not the SSBroyden final result.

#### Residual and solution fields
* Reference and prediction are visually indistinguishable: both reproduce the same oscillatory stripe pattern, signs, and amplitudes.
* The PDE residual is small and broadly scattered, with no dominant corner, shock-like front, or boundary-localized failure. Its reported maximum is approximately \(1.5452\times10^{-6}\).
* Pointwise errors form smooth, low-amplitude patches, up to roughly \(2.2\times10^{-8}\) on the displayed color scale, without a systematic boundary-error structure.
* Periodic boundaries are enforced architecturally by the Fourier embedding rather than by a ghost boundary-loss term.

### Diagnosis
**Numerically converged forward Helmholtz solve with a stable noisy floor.** There is no instability, premature stop, spectral-bias failure, or formulation inconsistency. The only outstanding issue is the campaign-level requirement for five exploration runs after the first score-≥90 run.

### Targeted Improvements
* **For Integrity:** None.
* **For Accuracy:** No further training or physics modification is required for Run 2. Preserve it as the current best. Do not extend it merely to chase the isolated \(3.24\times10^{-9}\) dip. Complete **three additional exploration trials**, then stop and select the best-scoring model.
* **For Details:** None for this run. Retain the dual-curve plateau stopping test and the separate reporting of final-iteration and best-ever RL2.
* **For Optimality:** No mandatory change. Exploration trials must preserve the verified pipeline: periodic Fourier embeddings on both coordinates, no raw-coordinate bypass, vRBA, and SSBroyden fine-tuning. Entropy regularization is inapplicable because this smooth linear Helmholtz problem has no shocks; a KAN or stream-function formulation is not needed.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The code implements a forward 2D Helmholtz PINN. Architecture: an MLP body with 3 hidden layers of width 30 and tanh activation; both input coordinates pass through Periodic_Fourier_Features with degree=5 and Lx=2.0, producing 20 embedding features that are concatenated before the MLP. No raw coordinates are passed to the body, no normalization is applied, and the output transformation is identity. The PDE residual is u_tt + u_xx + u - g with g=(1-17π²) sin(π t) sin(4π x), matching the manufactured solution sin(π x) sin(4π y) under the coordinate mapping used in the code. The loss contains only L_pde; no L_bc, L_ic, or L_data is used. Training uses Adam for 10,000 epochs with vRBA weighting and importance sampling, followed by SSBroyden/BFGS-style second-order optimization for 12,500 iterations in 100-step chunks with resampling. vRBA is used for weighting+sampling during Adam and for sampling only during SSBroyden. There is no PDE parameter continuation; k=1 is fixed. The analytical solution is used only for validation/error metrics. | 1.00e-06 | 35 | 35 | 6 | 14 | **90** |
| 1 | The implementation matches the strategy. It uses a PINN with Periodic Fourier Features (degree=5, Lx=2.0) applied to both coordinates, producing 20 embedding features that are concatenated and passed into an MLP body with width=30 and depth=3 using tanh activations and identity output. The PDE residual is the Helmholtz residual u_tt + u_xx + u - g with k=1 and g=(1-17*pi^2) sin(pi t) sin(4*pi x); the analytical reference is sin(pi t) sin(4*pi x). The loss is only L_pde: during Adam it is vRBA-weighted and vRBA-sampled, and during SSBroyden the PDE loss is unweighted while vRBA is used for sampling only. Adam runs for 10,000 iterations, then SSBroyden runs for 50,000 iterations with the Hessian reinitialized to identity at the transition. No L_data and no L_bc are included. | 1.00e-07 | 35 | 35 | 6 | 14 | **90** |
| 2 | Implementation matches the strategy. Architecture: MLP PINN with width 30 and three hidden layers (features [30,30,30,1]) using tanh activation and identity output. Input transformation uses Periodic Fourier Features on BOTH coordinates only, degree=5 with Lx=2.0, concatenated before the MLP body; no raw coordinates and no external normalization. PDE is Helmholtz with k=1: residual u_tt + u_xx + u - g, with g=(1-17*pi^2)sin(pi*t)sin(4*pi*x), matching the strategy under the t->x, x->y mapping. Analytical solution sin(pi*T)sin(4*pi*X) is used for validation/RL2 only. Loss assembly contains exactly one PDE loss term: during Adam it is vRBA-weighted mean squared residual with vRBA weighting and sampling; during SSBroyden it is unweighted mean squared residual with vRBA used for sampling only. No data, BC, IC, Neumann, gauge, or other ghost loss terms are included. Optimizer workflow: Adam for 10,000 iterations with vRBA, followed by SSBroyden for Nbfgs=100,000 using Nchange=100 inner iterations and Hessian reinitialized to identity at the Adam-to-SSBroyden transition. The plateau stopping criterion is implemented: after the 100,000 budget, it checks loss and RL2 band centers over the trailing 50% of the SSBroyden stage split into early/late halves, requires both change factors <= 2, and continues beyond budget if not flat. Final-iteration RL2 and best-ever minimum RL2 are reported separately, with the text noting that best-ever is not convergence evidence. | 3.51e-08 | 35 | 35 | 12 | 14 | **96** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves 'Helmholtz_predictions.npz' in NumPy .npz format. The npz contains keys: 'u_pred' with shape (100, 100) float64, 'u_analytical' with shape (100, 100) float64, 'X' with shape (100, 100) float64 representing the first coordinate grid, and 'Y' with shape (100, 100) float64 representing the second coordinate grid. It also saves 'Log_files_SSBroyden.mat' with training log arrays. |
| 1 | Saves Helmholtz_predictions.npz containing keys: 'u_pred' (100x100), 'u_analytical' (100x100), 'pointwise_error' (100x100), 'pde_residual' (100x100), 'X' (100x100), 'Y' (100x100), 'all_errors' (1055,), 'all_its' (1055,), and 'all_loss' (1055,). Also saves Log_files_SSBroyden.mat with model_name and histories: all_errors/all_its/all_loss/all_L_infty/all_L_2 length 1055; all_gamma/all_lamB/all_max_RBA length 555. |
| 2 | Saves Helmholtz_predictions.npz containing required keys: 'u_pred' (100x100), 'u_exact' (100x100), 'error' (100x100 pointwise absolute error), 'X' (100x100 coordinate grid, from T meshgrid), and 'Y' (100x100 coordinate grid, from X meshgrid). It also saves extra keys: 'pde_residual' (100x100), 'all_errors' (1D variable length log), 'all_its' (1D variable length log), and 'all_loss' (1D variable length log). Additionally saves Log_files_SSBroyden.mat with training histories and metadata. Required .npz fields are present. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run1//Helmholtz2D_Periodic_SinPiX_Sin4PiY/2026_11_06_0252/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run1//Helmholtz2D_Periodic_SinPiX_Sin4PiY/2026_11_06_0300/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run1//Helmholtz2D_Periodic_SinPiX_Sin4PiY/2026_11_06_0312/` |


---
