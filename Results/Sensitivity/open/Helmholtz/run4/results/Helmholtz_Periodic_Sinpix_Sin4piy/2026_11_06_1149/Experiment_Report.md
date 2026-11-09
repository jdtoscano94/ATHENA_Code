
# Experiment Report
*Last Updated: 2026-11-06 11:54:28*

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
# Strategy: Run 4 — Extended SSBroyden Convergence (Advisor-Mandated)

**Triage:** Advisor says **REJECTED_FOR_REFINEMENT** — Budget-Limited/Premature Stop. Cure: warm-start from checkpoint, reinit Hessian, double SSBroyden budget to 80,000 iterations. **No architecture/physics changes.** I follow this exactly.

## 1. Data Role
**Role: Validation/Error Metrics ONLY.** Forward problem; analytical solution is reference only. No data loss.

## 2. Problem Statement
- **Governing Equation (Helmholtz, k=1):**
  $$u_{xx} + u_{yy} + u = g(x,y), \quad g(x,y) = (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$$
- **PDE Residual:** $r = u_{xx} + u_{yy} + u - (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$
- **BCs:** Periodic in x and y — **enforced architecturally** (Fourier embedding), no BC loss.
- **Analytical Solution:** $u_{an} = \sin(\pi x)\sin(4\pi y)$ (a₁=1, a₂=4, k=1). Smooth, strong-form solution exists → well-posed for PINNs.
- **Problem Type:** Forward. No data loss term.

## 3. Input / Output
- **Input:** d=2, raw $(x,y) \in [-1,1]^2$. Already in nice range → **NO normalization** (identity), pass raw coordinates directly.
- **Output:** q=1 scalar $u(x,y)$. Smooth, oscillatory (4π frequency in y), periodic.

## 4. Architecture (UNCHANGED from Run 3)
**Pipeline:** Raw (x,y) → Per-coordinate Periodic Fourier Embedding (degree=4, **Lx=Ly=2.0**, matching raw range [-1,1]) → concat embeddings ONLY (no bare x,y to body) → MLP body (width=30, depth=3, tanh) → identity output → u(x,y).

All operations inside the model call function, differentiable.

## 5. Constraints
Periodicity: hard-enforced by embeddings. No Dirichlet/IC → no output transformation needed. Loss term count = 1 (< 5) → loss-based, no hard constraints.

## 6. Loss Function
$$L_{total} = L_{pde}$$
Periodic BC loss **excluded** (architecturally exact). No L_bc, no L_ic, no L_data.

## 7. Training Schedule (Advisor Cure)
- **Warm-start:** Load `final_model_checkpoint.npz` from previous run. If available, **skip Adam**; if unavailable, fallback: Adam 10,000 iterations.
- **SSBroyden Phase:** **Reinitialize Hessian to identity** at stage start. Run **80,000 fixed iterations = 800 outer iterations × maxiter=100**.
- **vRBA:** `lambdas` for Weighting & Sampling during Adam; **Sampling ONLY** during SSBroyden. SSBroyden loss unweighted. Do NOT modify SSBroyden internals.
- **Global weights:** 1 term → no balancing needed.
- **Finite budget:** Fixed iteration counts; stop only on NaN/Inf. No open-ended stopping rules.

## 8. Visualization
Do NOT modify template visualization callbacks.

## 9. Saving
Save predicted field, analytical field, and error history as `.npz` with keys `u_pred`, `u_ref`, plus final/best relative L² errors.

## Modules for Librarian
**Main Model:** [MLP with Hybrid Adam and SSBroyden]
**Enhancements:** [Per-coordinate Periodic Fourier Features (degree=4, Lx=Ly=2.0)]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
REJECTED_FOR_REFINEMENT

## Section 1: Status Decision
* **Status:** **REJECTED_FOR_REFINEMENT**
* The run is stable and highly accurate, but it fails the mandatory dual-curve plateau test: the residual loss is still drifting downward in the final region. This is a premature stop, not an optimal converged run.

## Section 2: Quantitative Scorecard
* **Score Integrity:** **35/35** — Completed successfully with finite values and readable plots; no NaN, Inf, or divergence.
* **Score Accuracy:** **35/35**
  * **Numerical Precision:** 20/20. Using \(\epsilon_{\text{target}}=10^{-5}\), consistent with the earlier full-accuracy trials, the final error is far below target.
  * **Consistency:** 15/15. The correct forward Helmholtz problem, manufactured solution, forcing, and periodic topology are used. The reference solution is used only for validation, so there is no data leakage.
* **Score Details:** **5/15** — Excellent field agreement, but the loss has not reached a genuine final plateau and the error remains strongly oscillatory.
* **Score Optimality:** **15/15** — Appropriate periodic Fourier embeddings, vRBA, low-dimensional SSBroyden fine-tuning, and unweighted PDE loss. Entropy regularization is not applicable to this smooth Helmholtz problem.
* **TOTAL SCORE:** **90/100**

**Final error used for scoring:** \(\boxed{2.83295651\times10^{-8}}\).

The exact log entry **“Minimum achieved RL2 error”** is:
\[
\boxed{3.587388161437383\times10^{-9}}.
\]
That minimum was a transient best dip, approximately eight times smaller than the final error, and is therefore not used as the final-model score.

## Section 3: Forensic Analysis & Feedback

### Observation

**Problem type:** Type A, forward manufactured-solution problem, with no noise. The analytical field is validation data only.

**Loss and error histories:**
* Both histories span approximately **0–44,000/45,000 iterations**. There is no clear later continuation jump in the displayed curves, so the entire displayed warm-started SSBroyden history is treated as the final stage.
* **Total Loss:** The logarithmic axis extends roughly from \(8\times10^{-14}\) to \(2\times10^{-12}\). Loss begins around \(10^{-12}\), drops rapidly during the first approximately 10,000 iterations, and then remains noisy while slowly drifting toward \(10^{-13}\).
  * \(A_{\text{loss}}\), the typical band level near the final-stage midpoint around iteration 22,000: **\(\approx1.9\times10^{-13}\)**.
  * \(B_{\text{loss}}\), the typical band level near the right edge: **\(\approx1.5\times10^{-13}\)**, with the lower envelope producing dips below \(10^{-13}\).
  * Thus \(B_{\text{loss}}<A_{\text{loss}}\), and the lower envelope continues to improve. Although the drop is modest, this is still a downward-drifting band rather than noise around one genuinely horizontal floor.
* **Relative \(L^2\) Error:** The axis spans approximately \(3\times10^{-9}\) to \(10^{-7}\). The green curve oscillates strongly throughout, mostly between several \(10^{-9}\) and several \(10^{-8}\), with spikes approaching \(10^{-7}\).
  * \(A_{\text{error}}\approx\mathbf{1.8\times10^{-8}}\).
  * \(B_{\text{error}}\approx\mathbf{1.8\times10^{-8}}\).
  * The error is therefore **noisy but approximately flat on average**, rather than steadily decreasing.
* The log confirms this interpretation:
  * Loss last-50% band centers: \(1.91744632\times10^{-13}\rightarrow1.50568018\times10^{-13}\), reported `horizontal=False`.
  * Error band centers: \(1.85452904\times10^{-8}\rightarrow1.71673498\times10^{-8}\), reported `horizontal=True`.
  * Dual-curve plateau verification: `False`.

**PDE residual heatmap:**
* Residuals are finite, with logged extrema approximately \(3.94\times10^{-11}\) and \(1.83\times10^{-6}\).
* The larger residuals appear as broadly distributed orange/red speckles rather than one dominant shock, corner, or boundary layer. There is no visual indication of a localized physical instability.

**Reference and prediction:**
* The reference and predicted fields are visually almost indistinguishable. Both reproduce the alternating horizontal bands, the sign reversal across \(x=0\), and the same amplitude scale.
* No visible phase shift or substantial amplitude error is present.

**Pointwise error:**
* Errors are very small, with the color scale topping out around \(2\times10^{-8}\).
* The largest pointwise errors form smooth red regions predominantly on the **left side**, especially toward the upper-left, while much of the center-right is cyan/blue. This is a smooth error topology, not a shock artifact.

### Diagnosis
**Budget-Limited / Premature Stop — residual loss not yet plateaued.**

This is not divergence, data leakage, incorrect physics, or an architecture-capacity failure. The final accuracy is excellent, and the error curve is approximately flat, but the loss gate fails: the residual-loss band and its lower envelope are still improving near the right edge. Consequently, the score of 90 cannot be approved as converged.

### Targeted Improvements

* **For Integrity:** None.
* **For Accuracy:** **Extend only the final optimization stage.** Warm-start from the Run 4 final checkpoint, reinitialize the SSBroyden Hessian to identity, and increase the final SSBroyden budget from **80,000 to 160,000 fixed iterations**. Keep the PDE, forcing, periodic embeddings, network architecture, vRBA settings, and loss formulation unchanged.
* **For Details:** After that fixed-budget run, repeat the last-50% dual-curve assessment. Both curves must oscillate around horizontal average floors before convergence is claimed. Continue reporting the **final-model** RL2 error separately from the transient best minimum.
* **For Optimality:** None. Do not switch to KAN, alter the Fourier degree, or remove the periodic embedding. No entropy penalty, positivity transform, or fluid-mechanics stream-function formulation is relevant here.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | The code implements the 2D Helmholtz manufactured-solution PINN. Architecture: inputs (x,y) pass through separate periodic Fourier embeddings (degree=4, Lx=Ly=2.0) on each coordinate; only the concatenated embeddings enter an MLP body with 3 hidden layers of width 30 and identity output. The PDE residual is uxx + uyy + u - (1 - 17*pi^2)*sin(pi*x)*sin(4*pi*y), and the only loss is L_total = L_pde = mean(residual^2). No BC/IC or data loss is used; periodicity is enforced by the embeddings. Training uses Phase 1 Adam for 10,000 iterations with vRBA weighting and sampling, followed by Phase 2 SSBroyden for 10,000 iterations (100 outer iterations of maxiter=100) where vRBA is used for sampling only and the SSBroyden loss is unweighted. NaN/Inf checks are present. The analytical solution sin(pi*x)*sin(4*pi*y) is used for validation/error metrics. | 3.54e-06 | 35 | 35 | 5 | 15 | **90** |
| 2 | The implementation matches the strategy. It uses a PINN with separate per-coordinate Periodic Fourier Features (degree=4, Lx=Ly=2.0) feeding a 30-wide, 3-hidden-layer MLP with tanh activation and identity scalar output. Inputs are raw coordinates; no external normalization is applied. The PDE residual is u_xx + u_yy + u - (1 - 17*pi^2)*sin(pi*x)*sin(4*pi*y), with analytical solution u_ref = sin(pi*x)*sin(4*pi*y). Only the PDE loss is used; no BC/IC/data losses. Optimizer workflow: Adam phase for 10,000 iterations unless warm-starting from a saved checkpoint; then SSBroyden phase for 20,000 iterations implemented as 200 outer iterations of maxiter=100. The SSBroyden Hessian is reinitialized to identity at the start of the extended stage. vRBA is used for sampling during SSBroyden, and the SSBroyden loss is unweighted. The code also attempts warm-start from final_model_checkpoint.npz and falls back to full retraining if unavailable. It includes plateau checks over the last 50% of the final stage for both loss and relative L2 error, and reports final-model error separately from the best-so-far minimum. | 2.32e-07 | 35 | 35 | 5 | 15 | **90** |
| 3 | The implementation uses a PINN with raw (x,y) passed directly into per-coordinate Periodic Fourier Features (degree=4, Lx=Ly=2.0); embeddings are concatenated and fed to an MLP with 3 hidden layers of width 30 and tanh activation. The PDE is Helmholtz k=1 with manufactured solution u_an = sin(pi x)sin(4 pi y); the loss is only the PDE residual, with no BC/IC/data terms. Training attempts to warm-start from final_model_checkpoint.npz in the run directory; if available it skips Adam, otherwise it falls back to Adam 10,000 iterations followed by SSBroyden. The SSBroyden stage reinitializes H0 to identity, runs 40,000 iterations as 400 outer iterations x maxiter=100, uses vRBA sampling for batch selection while keeping the SSBroyden loss unweighted, and checks NaN/Inf. Final evaluation reports best-so-far and final relative L2 error and includes a post-run last-50% plateau check. | 3.95e-08 | 35 | 35 | 5 | 15 | **90** |
| 4 | The code implements the Helmholtz k=1 forward problem with analytical reference u=sin(pi x)sin(4 pi y). Architecture matches the strategy: per-coordinate Periodic Fourier embedding with degree=4 and Lx=Ly=2.0, concatenated embeddings only (no bare x,y), followed by an MLP body of width 30 and 3 hidden layers with tanh activation and identity output. No input normalization is performed; raw coordinates are passed to the embeddings. The PDE residual is uxx+uyy+u-(1-17 pi^2)sin(pi x)sin(4 pi y), and the only loss is L_pde. Adam phase uses vRBA for both residual weighting and importance sampling; SSBroyden phase uses vRBA sampling only with an unweighted PDE loss. Training follows the advisor cure: warm-start from final_model_checkpoint.npz when present and skip Adam, otherwise run 10,000 Adam iterations; then reinitialize Hessian to identity and run 80,000 SSBroyden iterations as 800 outer iterations with maxiter=100, stopping only on non-finite values. | 2.83e-08 | 35 | 35 | 5 | 15 | **90** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves 'Helmholtz_results.npz' via np.savez with keys: 'x' (10000x1 float64), 'y' (10000x1 float64), 'u_pred' (10000x1 float64), 'u_an' (10000x1 float64), 'rel_l2_error' (scalar float64), and 'loss_history' (655-element 1D float64 array for the default 10,000 Adam + 10,000 SSBroyden schedule). |
| 2 | Saves 'Helmholtz_results.npz' containing keys: 'x' with shape (10000, 1), 'y' with shape (10000, 1), 'u_pred' with shape (10000, 1), 'u_ref' with shape (10000, 1), 'pointwise_error' with shape (10000, 1), 'rel_l2_error' scalar, and 'loss_history' with shape (N,). Also saves the final model checkpoint to '<args.Name>/final_model_checkpoint.npz' with key 'weights' containing a flattened parameter vector. |
| 3 | Saves Helmholtz_results.npz in the working directory with keys: x (10000x1), y (10000x1), u_pred (10000x1), u_ref (10000x1), residual (10000x1), pointwise_error (10000x1), rel_l2_error (scalar), loss_history (1D, length N_logged; 400 in warm-start mode). Also saves args.Name/final_model_checkpoint.npz containing key weights (flattened 1D parameter vector). With default NC_TEST=100, the 100x100 meshgrid is flattened to 10000 points. |
| 4 | Two npz artifacts are written. Helmholtz_results.npz (np.savez, relative current directory) contains: x (10000,1) float64; y (10000,1) float64; u_pred (10000,1) float64; u_ref (10000,1) float64; residual (10000,1) float64; pointwise_error (10000,1) float64; rel_l2_error scalar float64; best_rel_l2_error scalar float64; error_history 1D float64 of length N_logged; loss_history 1D float64 of length N_logged. For the default full Adam fallback plus 800 SSBroyden outer iterations, N_logged=1355 (555 logged Adam + 800 SSBroyden); if warm-start skips Adam, N_logged=800. final_model_checkpoint.npz (np.savez, in args.Name directory) contains: weights (2401,) float64 flattened parameter vector for the default 16-dimensional embedded input and 30x3 MLP. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run4/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1122/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run4/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1134/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run4/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1142/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run4/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1149/` |


---
