
# Experiment Report
*Last Updated: 2026-11-06 12:36:15*

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
# STRATEGY: 2D Helmholtz — Refinement Run (Advisor-Mandated)

## 0. Triage
**Path 2: Advisor Recommendation (MANDATORY).** Status = REJECTED_FOR_REFINEMENT. Diagnosis: *Budget-limited premature stop* — final SSBroyden stage ends while loss/error still descend.
**Cure:** Increase **only** the final SSBroyden stage: **20,000 → 40,000 fixed iterations**. All other components (embeddings, MLP, vRBA, Adam warm-up, forcing, physics) remain **UNCHANGED** from Run 2.

## 1. Data Source
No new data. Analytical solution used for **validation/error metrics ONLY**.

## 2. Problem Statement
* **Governing Equation (Helmholtz):** $u_{xx} + u_{yy} + k^2 u = g(x,y)$, with $k=1$.
* **PDE Residual:** $r(x,y) = u_{xx} + u_{yy} + u - g(x,y)$
* **Forcing (derived from $a_1=1, a_2=4, k=1$):** $g(x,y) = (1 - 17\pi^2)\,\sin(\pi x)\sin(4\pi y)$
* **BCs:** Periodic in both $x$ and $y$ on $[-1,1]\times[-1,1]$.
* **Analytical Solution:** $u_{an}(x,y) = \sin(\pi x)\sin(4\pi y)$ — **validation only**.
* **Data Loss Term:** None. **Role: Validation/Error Metrics ONLY.**
* **Problem Type:** Forward. Well-posed for PINNs (smooth strong-form solution).

## 2.1 / 2.2 Dimensions
* **Input:** 2D $(x,y) \in [-1,1]^2$. **Output:** scalar $u$; smooth, oscillatory, periodic.

## 3. Base Model
**[MLP with Hybrid Adam and SSBroyden]** (2D = low-dim case).

## 4. Main Architecture
`Input (x,y) → Periodic Fourier Embedding → MLP Body → Identity Output → u_pred`
All operations differentiable, inside the model call function.

* **4.1 Input Transformation:** **NO normalization** (coordinates already in $[-1,1]$ — pass through unchanged). **Periodic Fourier Features: degree 4, Lx = Ly = 2.0** (range of raw coordinates entering embedding). Main body receives **ONLY the concatenated embeddings** — no bare coordinates.
* **4.2 Main Body:** MLP, width=30, depth=3, tanh.
* **4.3 Output Transformation:** None (identity scalar). Periodicity enforced exactly by embedding.

## 5. Constraint Strategy
Loss term count = **1** ($L_{pde}$ only; periodic BCs satisfied architecturally → $L_{bc}$ excluded). Count < 5 → **Loss-Based**, no hard constraints.

## 6. Loss Assembly
$$L_{total} = L_{pde}$$

## 6.1 Optimization Schedule (THE ONLY CHANGE)
* **Phase 1 — Adam (STATIC):** 10,000 iterations, vRBA-weighted MSE, exponential-decay LR. Unchanged.
* **Phase 2 — SSBroyden:** **40,000 FIXED iterations** (Nchange=100 → 400 outer loops). Unweighted MSE; vRBA lambdas for **sampling only**. Early stopping only on NaN/Inf. No continuation (no Re).

## 7. Weighting
* **Local (vRBA):** lambdas for **Weighting & Sampling during Adam; Sampling ONLY during SSBroyden**.
* **Global:** 1 term → no balancing needed.
* **SSBroyden Integrity:** Do NOT modify internal SSBroyden weighting/sampling/batching.

## 8. Visualization
Do NOT modify template visualization callbacks.

## 9. Saving
Save fields as .npz with named keys: `u_pred`, `u_analytical`, `pointwise_error`, `residual` (plus loss/error histories).

---
**Main Model:** [MLP with Hybrid Adam and SSBroyden]
**Enhancements:** [Periodic Fourier Features (degree 4, Lx=Ly=2.0)]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
REJECTED_FOR_REFINEMENT

## Section 1: Status Decision
* **Status:** **REJECTED_FOR_REFINEMENT**
* **Problem type:** **Type A — forward problem**, with no observational noise. The analytical solution is used only for validation and plotting, so there is no data-loss leakage.
* **Decision rationale:** The run is stable and highly accurate, and its numerical score exceeds 90. However, it fails the **mandatory plateau check**: the final-stage loss is still descending substantially, and the error band also remains lower at the end than at the midpoint. This is another budget-limited stop, not a converged optimum.

## Section 2: Quantitative Scorecard
* **Score Integrity:** **35/35** — completed successfully; readable plots and finite values.
* **Score Accuracy:** **35/35**
  * **Numerical precision:** 20/20. The final plotted RL2 error is approximately \(10^{-7}\), below even a stringent \(\epsilon_{\text{target}}=10^{-6}\).
  * **Consistency:** 15/15. The forcing \(g=(1-17\pi^2)\sin(\pi x)\sin(4\pi y)\), \(k=1\), periodic formulation, and validation-only use of the analytical solution all match the request.
* **Score Details:** **7/15** — excellent field agreement and architectural periodicity, but no genuine final loss/error plateau.
* **Score Optimality:** **14/15** — appropriate periodic Fourier embedding, MLP, vRBA, and low-dimensional SSBroyden fine-tuning. Entropy regularization is not applicable to this smooth Helmholtz problem.
* **TOTAL SCORE:** **91/100**

## Section 3: Forensic Analysis & Feedback

### Observation
**Loss and error histories:**
* The curves show two principal stages: Adam through approximately iteration 10,000, followed by SSBroyden after the sharp transition.
* The loss falls from roughly order \(10^0\)–\(10^1\) initially to approximately \(10^{-10}\)–\(10^{-11}\) at the right edge. The RL2 error falls from roughly \(10^{-1}\)–\(10^0\) to a noisy band near \(10^{-7}\).
* The final stage is approximately iterations 10,000 to the right edge, giving a midpoint near 22,000–23,000. Reading the **centers of the oscillation bands**, not isolated dips:
  * **Loss:** \(A \approx 1\times10^{-8}\) at the final-stage midpoint; \(B \approx 5\times10^{-11}\) at the end. Thus \(B\ll A\), by roughly two orders of magnitude: **still decreasing, not flat**.
  * **RL2 error:** \(A \approx 4\times10^{-7}\) at the midpoint; \(B \approx 1\times10^{-7}\) at the end. The noisy band is still lower at the end: **not a demonstrated horizontal plateau**.
* The unambiguously descending loss alone fails the loss gate, even if the late error curve were interpreted as merely noisy.

**Residual and pointwise-error plots:**
* The PDE residual is mottled and broadly distributed over the domain, without a coherent shock-like or boundary-dominated failure structure.
* The pointwise error is mostly very small, with relatively larger thin central cell-like structures and a small localized patch near the right side. These features do not produce a visible discrepancy in the predicted field.
* The logged residual range is finite: approximately \(9.04\times10^{-11}\) to \(6.34\times10^{-6}\).

**Solution fields:**
* Reference and prediction are visually almost indistinguishable: the oscillation pattern, phase, signs, and amplitude agree at the plotted scale.
* No obvious periodic-boundary seam or large boundary artifact is visible.

**Log confirmation:**
* The earlier `RL2 error: 0.00891681` belongs to the Adam warm-up and is not the final SSBroyden result.
* The exact logged best value is:
  **`Minimum achieved RL2 error: 2.0881010534791152e-08`**
* That is a best-so-far dip, not the final model error. The right-edge plotted RL2 band, approximately \(10^{-7}\), is used for scoring.

### Diagnosis
**Budget-Limited / Premature Stop — still improving.** The physics, architecture, and accuracy are excellent; the remaining defect is that training ends before the final-stage curves flatten. This is not stagnation and does not justify changing the physics or embeddings.

### Targeted Improvements
* **For Integrity:** None.
* **For Accuracy:** Increase **only the final SSBroyden stage from 40,000 to 120,000 fixed iterations** (a \(3\times\) budget). Keep the 10,000-iteration Adam warm-up, Fourier degree 4, MLP dimensions, vRBA behavior, forcing, and all physics unchanged. If restarting from the current weights, reinitialize the final-stage SSBroyden Hessian.
* **For Details:** After the extended run, verify that **both** average bands are genuinely horizontal over the last 50% of the final stage. Report final-iteration RL2 separately from the best-so-far minimum.
* **For Optimality:** None. Retain the periodic Fourier input pipeline and SSBroyden/vRBA setup; do not replace the periodic embedding with raw coordinates or add irrelevant shock/entropy machinery.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The implementation matches the strategy. It uses a PINN with Periodic Fourier Features of degree 4 and Lx=Ly=2.0 applied to both spatial inputs; only the concatenated embeddings are passed to the MLP. The MLP has 3 hidden layers of width 30 with tanh activation and a direct scalar output, with no output transformation. The PDE residual is r = u_tt + u_xx + u - g with k^2=1 and g=(1-17π^2)sin(πx)sin(4πy), under the code's t/x aliasing for the two physical coordinates. The analytical solution is sin(πx)sin(4πy), used only for validation/error metrics. The loss contains only L_pde: during Adam it is vRBA-weighted MSE on the PDE residual; during SSBroyden it is unweighted MSE, while vRBA lambdas are updated and used for sampling only. The optimizer workflow is Adam for 10,000 iterations with exponential-decay learning rate and vRBA importance sampling (batch size 20,000), followed by SSBroyden for a fixed 10,000-step budget using Nchange=100 per outer loop (100 outer iterations). Early stopping is only on NaN/Inf. No BC or data loss terms are added. | 2.00e-06 | 35 | 35 | 7 | 14 | **91** |
| 2 | Implements the specified 2D Helmholtz PINN. The base model is an MLP with three hidden width-30 tanh layers and scalar output, using Periodic Fourier Features of degree 4 with Lx=Ly=2.0 applied separately to the two raw coordinates. The main body receives only the concatenated embedded coordinates, with no bare coordinates and no normalization. The loss is only L_pde: during Adam it is the vRBA-weighted mean squared Helmholtz residual; during SSBroyden it is the unweighted mean squared residual, with vRBA lambdas used for resampling only. The PDE residual is u_tt + u_xx + u - g with g=(1-17*pi^2)*sin(pi*t)*sin(4*pi*x). No BC/IC or data loss terms are included. Optimizer workflow matches the refinement plan: Adam for 10,000 iterations with exponential-decay LR and vRBA weighting/sampling, followed by SSBroyden for 20,000 fixed iterations (200 outer batches of 100) with vRBA resampling only and NaN/Inf early stopping. The analytical solution used for validation/error is sin(pi*x)*sin(4*pi*y). | 5.00e-07 | 35 | 35 | 7 | 14 | **91** |
| 4 | Implementation matches the advisor-mandated refinement strategy. Architecture: PINN using Periodic Fourier Features (degree 4, Lx=Ly=2.0) applied independently to both input coordinates, with only the concatenated embeddings passed to an MLP body (width=30, depth=3, tanh) and a linear identity output. No input normalization is performed; raw coordinates are passed directly into the embedding. Optimizer workflow: Phase 1 uses Adam for 10,000 iterations with vRBA-weighted MSE and exponential-decay learning rate; Phase 2 uses the SSBroyden optimizer for 40,000 fixed iterations (N_change=100, giving 400 outer loops), with unweighted MSE and vRBA lambdas used only for importance sampling. Early stopping is limited to non-finite loss. Loss assembly contains exactly one term, L_total = L_pde, where the Helmholtz residual is u_xx + u_yy + u - g with g=(1-17*pi^2)*sin(pi*x)*sin(4*pi*y). Periodic BCs are enforced architecturally through the Fourier embedding; no BC or data loss terms are present. The analytical solution u_an=sin(pi*x)*sin(4*pi*y) is used only for error metrics and visualization. | 1.00e-07 | 35 | 35 | 7 | 14 | **91** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves `Helmholtz_results.npz` (NumPy .npz) with keys: `x` (100x100 float64 physical-x grid), `y` (100x100 float64 physical-y grid), `u_pred` (100x100 float64 predicted field), `u_analytical` (100x100 float64 exact field), and `relative_l2_error` (1D float64 error-history array; default length 655 = 555 Adam logged entries + 100 SSBroyden outer iterations). It also saves an extra non-required MATLAB file `Log_files_SSBroyden.mat` containing `model_name` plus history keys `all_errors`, `all_its`, `all_loss`, `all_gamma`, `all_lamB`, `all_max_RBA`, `all_L_infty`, and `all_L_2`; with default settings the shared Adam+SSBroyden histories are length 655, while Adam-only histories are length 555. |
| 2 | Saves Helmholtz_results.npz in NumPy .npz format with keys: 'x' (NC_TEST x NC_TEST meshgrid array of the first physical coordinate, default 100x100), 'y' (NC_TEST x NC_TEST meshgrid array of the second physical coordinate), 'u_pred' (same grid shape), 'u_analytical' (same grid shape), 'error' (same grid shape, absolute difference), and 'residual' (same grid shape, PDE residual field). Also saves Log_files_SSBroyden.mat in MAT format containing training logs: 'model_name', 'all_errors', 'all_its', 'all_loss', 'all_gamma', 'all_lamB', 'all_max_RBA', 'all_L_infty', and 'all_L_2'. |
| 4 | Saves Helmholtz_results.npz containing keys: 'x' (100x100), 'y' (100x100), 'u_pred' (100x100), 'u_analytical' (100x100), 'pointwise_error' (100x100), 'residual' (100x100), 'all_loss' (1D history), 'all_errors' (1D history), 'all_its' (1D history), 'all_L_infty' (1D history), 'all_L_2' (1D history). Also saves Log_files_SSBroyden.mat with model_name and training histories including all_errors, all_its, all_loss, all_gamma, all_lamB, all_max_RBA, all_L_infty, and all_L_2. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run5/results//Helmholtz_2D_Periodic_Sin_pi_x_Sin_4pi_y/2026_11_06_1214/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run5/results//Helmholtz_2D_Periodic_Sin_pi_x_Sin_4pi_y/2026_11_06_1222/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run5/results//Helmholtz_2D_Periodic_Sin_pi_x_Sin_4pi_y/2026_11_06_1232/` |


---
