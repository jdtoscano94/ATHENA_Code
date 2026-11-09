
# Experiment Report
*Last Updated: 2026-11-06 03:51:23*

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
# STRATEGY — Run 5/5 (Final Exploration Run)

## 0. Triage
**Path 2 (Advisor Mandatory):** Protocol-only rejection. Run 5/5 = **single width-only variation: width 30→40**. ALL other controls identical to Run 4 (degree=7, depth=3, loss assembly, vRBA, optimizer workflow, plateau test). Compare via **sustained final-iteration RL2**, not best-ever dips.

## 1. Data Source
No new data. Analytical solution = validation only.

## 2. Problem Statement
- **Governing Equation (2D Helmholtz, k=1):** $u_{xx} + u_{yy} + k^2 u = g(x,y)$, $k=1$
- **PDE Residual:** $r = u_{xx} + u_{yy} + u - g(x,y)$
- **Source (manufactured):** $g(x,y) = (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$ [from $g = (k^2 - \pi^2 a_1^2 - \pi^2 a_2^2)u_{an}$, $a_1=1$, $a_2=4$]
- **BCs:** Periodic in $x$ and $y$ on $[-1,1]^2$ — enforced **architecturally** (Fourier embedding), no $L_{bc}$.
- **Analytical Solution:** YES. $u_{an} = \sin(\pi x)\sin(4\pi y)$
- **Data Loss:** None. **Role: Validation/Error Metrics ONLY.**
- **Problem Type:** Forward. Well-posed; smooth strong-form solution exists.
- **Legacy naming (keep consistent with Runs 0–4):** template inputs $(t,x)$ map $t\to x$, $x\to y$; residual $u_{tt}+u_{xx}+u-g$ with $g=(1-17\pi^2)\sin(\pi t)\sin(4\pi x)$.

## 2.1 / 2.2 Input & Output
- **Input:** 2D spatial $(x,y)\in[-1,1]^2$. No time.
- **Output:** scalar $u(x,y)$. Smooth, periodic, oscillatory (4 half-waves in $y$).

## 3. Base Model
**MLP with Hybrid Adam and SSBroyden** (2D problem class).

## 4. Main Architecture
`Input (x,y) raw → Periodic Fourier Features → MLP body → identity → u`
- **Normalization:** NONE — coordinates already in $[-1,1]$ (nice range, pass through unchanged).
- **Embedding:** Periodic Fourier Features on **BOTH** coordinates, **degree=7, Lx=2.0 each** (raw range $[-1,1]$ → $L_x = 2.0$). 28 features total. **No raw coordinates to body — embedding only.**
- **Main Body:** MLP, **width=40** (Run 5 variation), depth=3, tanh.
- **Output Transformation:** Identity. All ops differentiable, inside call function.

## 5. Constraint Strategy
Count = 1 ($L_{pde}$ only; periodic BCs architectural, no IC — steady-state). **Loss-based, no hard constraints.**

## 6. Loss Assembly
$$L_{total} = L_{pde}$$

**6.1 Continuation:** None. $k=1$ fixed.

**Optimizer Workflow (unchanged from Run 4):**
- **Phase 1 (Adam):** 10,000 iterations. vRBA weighting + sampling.
- **Phase 2 (SSBroyden):** 100,000 iterations, Nchange=100, Hessian reinitialized to identity at transition, vRBA **sampling only**, unweighted PDE loss.
- **Dual-curve plateau test:** trailing 50% of SSBroyden stage, early/late halves, loss & RL2 band centers, change factors ≤2; extend beyond budget if not flat.
- Report **final-iteration RL2 and best-ever RL2 separately** (best-ever ≠ convergence evidence).

## 7. Weighting
- Total terms = 1 → **no global balancing needed.**
- **vRBA:** `lambdas` for Weighting & Sampling during Adam; **Sampling ONLY during SSBroyden.**
- **SSBroyden Integrity:** Do NOT modify internal weighting/sampling/batching mechanics.

## 8. Visualization
Do NOT modify template visualization callbacks.

## 9. Saving
Save $u_{pred}$, $u_{an}$, residual field, and coordinate grids as `.npz` with keys specifying variable names.

## 10. Run 5 Selection Note
After Run 5, select best **sustained final-iteration RL2** across all runs. Run 4 currently leads.

---
**Main Model:** [MLP with Hybrid Adam and SSBrodyden]
**Enhancements:** [Periodic Fourier Features]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
APPROVED

## Section 1: Status Decision
* **Status:** **APPROVED**
* **Reason:** The run is stable, accurate, and satisfies the dual-curve plateau test. This is also the fifth exploration run since the first successful Run 0, completing the prescribed exploration budget. **Run 5 is the best model**, with the lowest sustained final-iteration RL2 error: **4.8720731501001599e-09**.

## Section 2: Quantitative Scorecard
* **Score Integrity:** **35/35**
* **Score Accuracy:** **35/35**
  * Numerical Precision: **20/20** — final RL2 is approximately \(4.87\times10^{-9}\), well below the campaign’s implicit \(10^{-6}\) working tolerance.
  * Consistency: **15/15** — correct forward Helmholtz problem, manufactured source, \(k=1\), periodic Fourier treatment, and no data-loss leakage.
* **Score Details:** **12/15** — both curves reach a genuine noisy plateau; minor final-segment fluctuations remain.
* **Score Optimality:** **14/15**
* **TOTAL SCORE:** **96/100**

## Section 3: Forensic Analysis & Feedback

### Observation
This is a **Type A, noise-free forward problem**. The analytical solution is used only for validation and error measurement.

**Loss and error histories:**
* Both plots are readable and finite, with no NaN, Inf, or divergent excursion.
* The loss falls from approximately \(10^{0}\) to a final noise floor near \(10^{-13}\text{–}10^{-14}\). There is an initial rapid Adam decrease, a small transition feature near 10,000 iterations, a long smooth SSBroyden decrease, and a noisy final floor.
* The RL2 curve decreases from approximately \(10^{0}\) through \(10^{-2}\), \(10^{-4}\), and \(10^{-6}\), before settling around \(10^{-8}\).
* **Explicit final-stage band readings**, judging oscillation-band centers rather than isolated dips:
  * **Loss:** \(A \approx 6.381202\times10^{-14}\) at the final-stage midpoint/early trailing window; \(B \approx 4.057348\times10^{-14}\) at the end/late trailing window. The change factor is **1.573**, within the allowed factor of two: **flat on average**.
  * **RL2:** \(A \approx 9.041793\times10^{-9}\); \(B \approx 8.236900\times10^{-9}\). The change factor is **1.098**: **flat on average**.
* Thus, **both** curves pass the dual-curve plateau test. The small decrease in band centers is within the noise-band criterion and does not indicate a still-descending training run.

**Residual and pointwise-error fields:**
* The PDE residual is broadly distributed as fine orange/red speckling, without a localized shock, corner failure, or systematic boundary strip. The log reports residual values from **1.5447199075424578e-11** to **8.007721703506832e-07**.
* Pointwise error uses approximately a \(10^{-13}\text{–}10^{-9}\) color scale. Its largest red-background values are broadly distributed through the interior, with thin yellow/green filamentary variations. There is no visible boundary-localized or shock-like error structure.

**Reference versus prediction:**
* The fields are visually almost indistinguishable.
* The alternating horizontal oscillations, left/right sign reversal, amplitudes, phases, and periodic structure are reproduced without visible blurring or phase shift.

**Log extraction and final-error selection:**
* The initial **RL2 error: 0.00168009** belongs to the Adam warm-up and is ignored because SSBroyden subsequently ran.
* Exact best-ever minimum:
  * **Best-ever minimum RL2 error: 1.3307463260074361e-09**
  * This is explicitly an isolated dip and is **not** used for scoring.
* Exact sustained final value:
  * **Final-iteration RL2 error: 4.8720731501001599e-09**
  * This is the value used for scoring and model selection.
* The final value improves on Run 4’s \(8.20\times10^{-9}\).

### Diagnosis
**Successful, stable convergence with low-amplitude noise around a very low error floor.** There is no instability, premature stop, stagnation, spectral-bias failure, or physics/formulation mismatch.

The architecture and implementation are consistent: periodic Fourier features on both coordinates, degree 7, width-40/depth-3 tanh MLP, PDE-only loss, vRBA, and Adam followed by SSBroyden. Entropy regularization is not applicable because this smooth Helmholtz problem contains no shocks. A KAN is unnecessary given the achieved accuracy.

**History decision:** Runs 1–5 constitute five attempts since the first successful Run 0. Exploration is therefore complete. Run 5 is selected because it ties the best score and has the best sustained final-iteration RL2.

### Targeted Improvements
* **For Integrity:** None.
* **For Accuracy:** None. Stop training and retain Run 5; additional iterations or architecture changes are not justified.
* **For Details:** None. The final loss and RL2 bands satisfy the plateau criterion despite minor oscillations.
* **For Optimality:** None. Preserve the periodic Fourier input pipeline and current optimizer workflow; do not feed raw coordinates directly to the MLP.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The code implements a forward 2D Helmholtz PINN. Architecture: an MLP body with 3 hidden layers of width 30 and tanh activation; both input coordinates pass through Periodic_Fourier_Features with degree=5 and Lx=2.0, producing 20 embedding features that are concatenated before the MLP. No raw coordinates are passed to the body, no normalization is applied, and the output transformation is identity. The PDE residual is u_tt + u_xx + u - g with g=(1-17π²) sin(π t) sin(4π x), matching the manufactured solution sin(π x) sin(4π y) under the coordinate mapping used in the code. The loss contains only L_pde; no L_bc, L_ic, or L_data is used. Training uses Adam for 10,000 epochs with vRBA weighting and importance sampling, followed by SSBroyden/BFGS-style second-order optimization for 12,500 iterations in 100-step chunks with resampling. vRBA is used for weighting+sampling during Adam and for sampling only during SSBroyden. There is no PDE parameter continuation; k=1 is fixed. The analytical solution is used only for validation/error metrics. | 1.00e-06 | 35 | 35 | 6 | 14 | **90** |
| 1 | The implementation matches the strategy. It uses a PINN with Periodic Fourier Features (degree=5, Lx=2.0) applied to both coordinates, producing 20 embedding features that are concatenated and passed into an MLP body with width=30 and depth=3 using tanh activations and identity output. The PDE residual is the Helmholtz residual u_tt + u_xx + u - g with k=1 and g=(1-17*pi^2) sin(pi t) sin(4*pi x); the analytical reference is sin(pi t) sin(4*pi x). The loss is only L_pde: during Adam it is vRBA-weighted and vRBA-sampled, and during SSBroyden the PDE loss is unweighted while vRBA is used for sampling only. Adam runs for 10,000 iterations, then SSBroyden runs for 50,000 iterations with the Hessian reinitialized to identity at the transition. No L_data and no L_bc are included. | 1.00e-07 | 35 | 35 | 6 | 14 | **90** |
| 2 | Implementation matches the strategy. Architecture: MLP PINN with width 30 and three hidden layers (features [30,30,30,1]) using tanh activation and identity output. Input transformation uses Periodic Fourier Features on BOTH coordinates only, degree=5 with Lx=2.0, concatenated before the MLP body; no raw coordinates and no external normalization. PDE is Helmholtz with k=1: residual u_tt + u_xx + u - g, with g=(1-17*pi^2)sin(pi*t)sin(4*pi*x), matching the strategy under the t->x, x->y mapping. Analytical solution sin(pi*T)sin(4*pi*X) is used for validation/RL2 only. Loss assembly contains exactly one PDE loss term: during Adam it is vRBA-weighted mean squared residual with vRBA weighting and sampling; during SSBroyden it is unweighted mean squared residual with vRBA used for sampling only. No data, BC, IC, Neumann, gauge, or other ghost loss terms are included. Optimizer workflow: Adam for 10,000 iterations with vRBA, followed by SSBroyden for Nbfgs=100,000 using Nchange=100 inner iterations and Hessian reinitialized to identity at the Adam-to-SSBroyden transition. The plateau stopping criterion is implemented: after the 100,000 budget, it checks loss and RL2 band centers over the trailing 50% of the SSBroyden stage split into early/late halves, requires both change factors <= 2, and continues beyond budget if not flat. Final-iteration RL2 and best-ever minimum RL2 are reported separately, with the text noting that best-ever is not convergence evidence. | 3.51e-08 | 35 | 35 | 12 | 14 | **96** |
| 3 | Implements the specified 2D Helmholtz forward PINN. Architecture: MLP width 30, depth 3, tanh activation. Inputs (t,x) are mapped to (x,y); both coordinates pass through Periodic Fourier Features with degree=6 and Lx=2.0, producing 12 features per coordinate and 24 concatenated features. No raw-coordinate bypass is used. The sole loss is L_pde; residual is u_tt + u_xx + u - g with g=(1-17π²)sin(πt)sin(4πx). During Adam, vRBA weighting and sampling are used; during SSBroyden, vRBA is used for sampling only and the PDE loss is unweighted mean(residual²). Optimizer workflow matches the strategy: Adam for 10,000 iterations, then SSBroyden for 100,000 iterations with Nchange=100, Hessian initialized to identity at transition, and template-internal SSBroyden mechanics preserved. A dual-curve plateau test is implemented over the trailing 50% of the SSBroyden stage using loss and RL2 band centers with change factors ≤2, extending beyond budget if not flat. Final-iteration RL2 and best-ever RL2 are reported separately, with best-ever explicitly not treated as convergence evidence. Analytical solution matches u_an=sin(πt)sin(4πx). | 1.42e-08 | 35 | 35 | 12 | 14 | **96** |
| 4 | The implementation follows the Run 4 strategy: a PINN for the 2D Helmholtz problem with k=1 and manufactured solution u=sin(pi*x)*sin(4*pi*y). The model applies Periodic Fourier Features with degree 7 and Lx=2.0 to both input coordinates, yielding 28 embedded features, which are passed exclusively into an MLP body of width 30 and depth 3 with tanh activations; no raw coordinates are passed to the MLP. The PDE residual is u_tt + u_xx + u - g with g=(1-17*pi^2)*sin(pi*t)*sin(4*pi*x), corresponding to the strategy's Helmholtz equation under the legacy (t,x) naming. Training uses Adam for 10,000 iterations with vRBA weighting and importance sampling, followed by SSBroyden for 100,000 iterations with Nchange=100, Hessian reinitialization to identity, vRBA sampling only, and an unweighted PDE loss during SSBroyden. The loss assembly contains only L_pde; no L_bc, L_ic, or L_data is used, and the analytical solution is used solely for error metrics. The dual-curve plateau test is implemented over the trailing 50% of the SSBroyden stage, split into early/late halves with change factors <= 2, and the loop extends beyond budget if the plateau is not reached. | 8.20e-09 | 35 | 35 | 12 | 14 | **96** |
| 5 | The code implements a 2D Helmholtz PINN (k=1) matching the strategy. Architecture: a PINN module that applies Periodic Fourier Features (degree=7, Lx=2.0) to BOTH coordinates (t→x and x→y per legacy naming), producing 28 features, followed by an MLP body (width=40 per Run 5, depth=3, tanh, identity output). No raw coordinates enter the body and no normalization is applied. The PDE residual is computed as u_tt + u_xx + u - g with g=(1-17π²)sin(πt)sin(4πx), matching the manufactured source; the analytical validation solution is sin(πt)sin(4πx). The total loss is L_pde only (no BC, IC, or data/leakage terms; u_gt is reserved for error metrics). Optimizer workflow follows the prescribed hybrid: Phase 1 Adam for 10,000 iterations with vRBA used for BOTH weighting (lambdas multiplied inside the PDE loss) and sampling (lambda-weighted random choice); Phase 2 SSBroyden for 100,000 iterations with Nchange=100, Hessian initialized to identity at transition, and vRBA used for SAMPLING ONLY with an unweighted PDE loss (mean(residuals²)). A dual-curve plateau test is implemented using the trailing 50% of the SSBroyden stage split into early/late halves with band centers and change factors ≤2, extending the budget until flat; final-iteration RL2 and best-ever RL2 are reported separately. | 4.87e-09 | 35 | 35 | 12 | 14 | **96** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves 'Helmholtz_predictions.npz' in NumPy .npz format. The npz contains keys: 'u_pred' with shape (100, 100) float64, 'u_analytical' with shape (100, 100) float64, 'X' with shape (100, 100) float64 representing the first coordinate grid, and 'Y' with shape (100, 100) float64 representing the second coordinate grid. It also saves 'Log_files_SSBroyden.mat' with training log arrays. |
| 1 | Saves Helmholtz_predictions.npz containing keys: 'u_pred' (100x100), 'u_analytical' (100x100), 'pointwise_error' (100x100), 'pde_residual' (100x100), 'X' (100x100), 'Y' (100x100), 'all_errors' (1055,), 'all_its' (1055,), and 'all_loss' (1055,). Also saves Log_files_SSBroyden.mat with model_name and histories: all_errors/all_its/all_loss/all_L_infty/all_L_2 length 1055; all_gamma/all_lamB/all_max_RBA length 555. |
| 2 | Saves Helmholtz_predictions.npz containing required keys: 'u_pred' (100x100), 'u_exact' (100x100), 'error' (100x100 pointwise absolute error), 'X' (100x100 coordinate grid, from T meshgrid), and 'Y' (100x100 coordinate grid, from X meshgrid). It also saves extra keys: 'pde_residual' (100x100), 'all_errors' (1D variable length log), 'all_its' (1D variable length log), and 'all_loss' (1D variable length log). Additionally saves Log_files_SSBroyden.mat with training histories and metadata. Required .npz fields are present. |
| 3 | Saves 'Helmholtz_predictions.npz' (NPZ format) containing keys: 'u_pred' (100x100 prediction field), 'u_exact' (100x100 analytical/reference field, equivalent to the strategy's u_analytical), 'error' (100x100 absolute pointwise error), 'pde_residual' (100x100 signed PDE residual), 'X' (100x100 first-coordinate meshgrid), 'Y' (100x100 second-coordinate meshgrid), 'all_errors' (1D logged RL2 history), 'all_its' (1D logged iteration history), and 'all_loss' (1D logged loss history). Also saves 'Log_files_SSBroyden.mat' with keys: model_name, all_errors, all_its, all_loss, all_gamma, all_lamB, all_max_RBA, all_L_infty, all_L_2. |
| 4 | Two artifacts are saved. (1) 'Log_files_SSBroyden.mat' saved via scipy.io.savemat contains keys: 'model_name' (string), 'all_errors' (1D array length N), 'all_its' (1D array length N), 'all_loss' (1D array length N), 'all_gamma' (1D array length N), 'all_lamB' (1D array length N), 'all_max_RBA' (1D array length N), 'all_L_infty' (1D array length N), 'all_L_2' (1D array length N). (2) 'Helmholtz_predictions.npz' saved via np.savez contains keys: 'u_pred' (100x100), 'u_exact' (100x100, the analytical reference), 'error' (100x100, pointwise absolute error), 'pde_residual' (100x100), 'X' (100x100 grid first coordinate), 'Y' (100x100 grid second coordinate), 'all_errors' (1D array length N), 'all_its' (1D array length N), 'all_loss' (1D array length N). N is the variable number of logged training iterations. |
| 5 | Saves 'Helmholtz_predictions.npz' (NumPy .npz) containing keys: 'u_pred' (100x100 prediction grid), 'u_exact' (100x100 reference), 'error' (100x100 pointwise absolute error), 'pde_residual' (100x100 residual field), 'X' (100x100 grid, the t/x meshgrid), 'Y' (100x100 grid, the x/y meshgrid), 'all_errors' (1D loss/RL2 history), 'all_its' (1D iteration array), 'all_loss' (1D loss history). Additionally saves 'Log_files_SSBroyden.mat' via scipy.io.savemat with keys model_name, all_errors, all_its, all_loss, all_gamma, all_lamB, all_max_RBA, all_L_infty, all_L_2. The four required artifacts (u_pred, u_an/u_exact, residual field, coordinate grids) are all present. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run1//Helmholtz2D_Periodic_SinPiX_Sin4PiY/2026_11_06_0252/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run1//Helmholtz2D_Periodic_SinPiX_Sin4PiY/2026_11_06_0300/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run1//Helmholtz2D_Periodic_SinPiX_Sin4PiY/2026_11_06_0312/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run1//Helmholtz2D_Periodic_SinPiX_Sin4PiY/2026_11_06_0324/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run1//Helmholtz2D_Periodic_SinPiX_Sin4PiY/2026_11_06_0335/` |
| 5 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run1//Helmholtz2D_Periodic_SinPiX_Sin4PiY/2026_11_06_0344/` |


---
