
# Experiment Report
*Last Updated: 2026-11-06 12:03:17*

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
# STRATEGY: Helmholtz 2D — Run 5 (Extended Convergence Stage)

## 0. Triage
**Path 2: Advisor Recommendation (MANDATORY).** Cure: warm-start from Run 4 checkpoint, reinitialize SSBroyden Hessian, extend final SSBroyden stage from 80,000 → **160,000 fixed iterations**. Keep architecture, embeddings, PDE, and loss formulation **unchanged**.

## 1. Data Source
No new data. Analytical solution = validation only.

## 2. Problem Statement
- **Governing Equation:** Helmholtz: $u_{xx} + u_{yy} + k^2 u = g(x,y)$, $k=1$.
- **PDE Residual:** $r = u_{xx} + u_{yy} + u - (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$
  (since $u_{xx}=-\pi^2 u$, $u_{yy}=-16\pi^2 u$, $k^2u=u$ → $g=(1-17\pi^2)\sin(\pi x)\sin(4\pi y)$).
- **BCs:** Periodic in $x$ and $y$ on $[-1,1]\times[-1,1]$ — enforced **by architecture** (no BC loss).
- **Analytical Solution:** Yes: $u_{an}(x,y) = \sin(\pi x)\sin(4\pi y)$.
- **Data Loss Term:** None. **Role: Validation/Error Metrics ONLY.**
- **Problem Type:** Forward, well-posed, smooth strong-form solution. PINN-suitable.

## 2.1 / 2.2 Input-Output
- Input: 2D $(x,y)$, no time. Output: scalar $u(x,y)$. Smooth, oscillatory ($4\pi$ in $y$), periodic.

## 3. Base Model Selection
Low-dim (2D) → **MLP with Hybrid Adam and SSBroyden**. Width=30, depth=3, tanh, identity output.

## 4. Main Architecture
`Input (x,y) → Per-coordinate Periodic Fourier Embedding (degree=4, Lx=Ly=2.0) → Concatenation → MLP(30×3, tanh) → Identity → u(x,y)`

- **Normalization:** NONE. $x,y \in [-1,1]$ already in nice range — pass raw coordinates directly to embeddings. Any normalization must be differentiable and inside the call function (here: identity).
- **Embedding:** Periodic Fourier Features, **degree=4** (advisor forbids changing), $L_x = L_y = 2.0$ (range of $[-1,1]$). Separate embedding per coordinate; **only embeddings enter the MLP body — no bare $x,y$**.
- **Output Transformation:** None (no Dirichlet constraints; periodicity is architectural).

## 5. Constraint Implementation
Loss term count: $L_{pde}$ only (periodic BCs satisfied by embedding, no IC, no data) → count = 1 < 5 → **Loss-Based, no hard constraints**.

## 6. Loss Function Assembly
$$L_{total} = L_{pde}$$

## 6.1 Training Schedule (Advisor Cure — MANDATORY)
- **Phase 1 (Adam):** Warm-start from Run 4 `final_model_checkpoint.npz` → **skip Adam**. Fallback: if checkpoint unavailable, 10,000 Adam iterations with vRBA weighting + sampling.
- **Phase 2 (SSBroyden):** **Reinitialize Hessian to identity** at stage start. Run **160,000 fixed iterations** (1600 outer × maxiter=100). vRBA for **sampling only**; SSBroyden loss unweighted. Stop only on NaN/Inf within the fixed budget.
- **Post-run:** Repeat last-50% dual-curve plateau assessment on **both** loss and RL2 error. Report **final-model RL2 error separately** from transient best minimum.
- No continuation (no Re/Peclet).

## 7. Weighting Strategy
- **Local:** vRBA — `lambdas` for Weighting & Sampling during Adam, Sampling ONLY during SSBroyden.
- **Global:** 1 term → no balancing needed.
- **SSBroyden Integrity:** Do NOT modify internal SSBroyden weighting/sampling/batching.

## 8. Visualization
Do NOT modify template visualization callbacks.

## 9. Saving Data
Save predicted field, reference field, and grid as `.npz` with named keys; save final checkpoint (`final_model_checkpoint.npz`) for future warm-start.

---

**Main Model:** [MLP with Hybrid Adam and SSBroyden]
**Enhancements:** [Periodic Fourier Features (degree=4, Lx=Ly=2.0)]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
APPROVED

## Section 1: Status Decision
* **Status:** **APPROVED**
* **Problem type:** **Type A — forward, well-posed Helmholtz problem**, with no observational noise. The analytical solution is used only for validation, not training.
* **Decision basis:** The run completed successfully, solved the requested problem, and both curves satisfy the final-region band-center plateau test. Approval uses the **final-model error**, not the substantially lower transient minimum.

## Section 2: Quantitative Scorecard
* **Score Integrity:** **35/35**
* **Score Accuracy:** **35/35**
  * Numerical Precision: **20/20** — final RL2 error is approximately \(3.0\times10^{-8}\).
  * Consistency: **15/15** — correct PDE, source term, domain, parameters, periodic formulation, and validation-only use of the reference solution.
* **Score Details:** **12/15** — both curves are flat on average, but remain visibly noisy/spiky around their respective floors.
* **Score Optimality:** **15/15** — periodic Fourier embeddings, MLP, vRBA, and low-dimensional SSBroyden fine-tuning are appropriate. Entropy regularization is not applicable to this smooth, non-shock problem.
* **TOTAL SCORE:** **97/100**

## Section 3: Forensic Analysis & Feedback

### Observation

**1. Loss and error histories**

* The plot is readable and all displayed values are finite. There is no NaN/Inf event or divergence.
* **Total Loss:** The labeled vertical range is approximately \(4\times10^{-14}\) to \(2\times10^{-13}\). The loss initially decreases, then becomes highly oscillatory around a low floor. There is no visible final continuation/optimizer jump, so the plotted SSBroyden history is treated as the final region.
  * Final-region midpoint band center: **\(A \approx 6.8\times10^{-14}\)**.
  * End-of-region band center: **\(B \approx 6.1\times10^{-14}\)**.
  * These are essentially the same horizontal floor; the small decrease is far below the 2× threshold. Individual spikes and dips do not indicate a continuing average descent.
* **Relative \(L^2\) error:** The green curve is jagged around the \(10^{-8}\) scale, with excursions roughly in the \(10^{-9}\)–\(10^{-7}\) envelope, but shows no runaway growth.
  * Final-region midpoint band center: **\(A \approx 1.11\times10^{-8}\)**.
  * End-of-region band center: **\(B \approx 1.07\times10^{-8}\)**.
  * Thus, the error curve is also **flat on average**, rather than still decreasing.
* The rightmost RL2 point is approximately **\(3.0\times10^{-8}\)**. This final point lies above the center of the noise band, but it does not overturn the plateau assessment.

**2. PDE residual heatmap**

* The residual is spatially heterogeneous and speckled, with the largest red/orange values scattered through the interior rather than concentrated at a boundary, corner, or shock-like structure.
* The color scale spans approximately \(10^{-10}\)–\(10^{-6}\). The log confirms:
  * Minimum residual: \(6.877779256453528\times10^{-11}\)
  * Maximum residual: \(1.018765342308825\times10^{-6}\)
* There is no visible systematic periodic-boundary failure.

**3. Reference, prediction, and pointwise error**

* Reference and prediction are visually almost indistinguishable: the oscillation phase, stripe pattern, amplitude, and periodic structure all match.
* The pointwise error consists of smooth, small-amplitude patches, approximately \(10^{-8}\)–\(1.8\times10^{-8}\) on the displayed color scale. The larger errors are scattered interior patches, with no dominant boundary artifact.

### Log confirmation and final-error selection

* **Exact minimum achieved RL2 error:**
  \[
  \boxed{1.8550587515561003\times10^{-9}}
  \]
* **Final relative \(L^2\) error:**
  \[
  \boxed{2.99569003\times10^{-8}}
  \]
* The **final value \(2.99569003\times10^{-8}\)** is used for scoring. The transient minimum is not treated as the final-model accuracy.
* The log’s independent last-50% calculations confirm the visual assessment:
  * Loss band centers: \(6.84504689\times10^{-14}\rightarrow6.10513144\times10^{-14}\), reported horizontal.
  * RL2 band centers: \(1.11076653\times10^{-8}\rightarrow1.07102398\times10^{-8}\), reported horizontal.
  * Dual-curve plateau verification: **True**.

### Diagnosis

**Successful convergence to a noisy but stationary, very-low-error floor.** The earlier budget-limited behavior has been resolved: neither curve is still descending on average. There is no evidence of instability, spectral bias, data leakage, incorrect physics, or periodic-boundary violation.

Runs 1–4 previously scored 90; the present run improves the convergence-details assessment and passes the mandatory dual-curve plateau gate on its own merits.

### Targeted Improvements

* **For Integrity:** None.
* **For Accuracy:** None. Further training is not required merely to chase the transient minimum.
* **For Details:** None required. The residual oscillations justify a small score deduction, but not rejection.
* **For Optimality:** None. Retain the periodic embedding pipeline and SSBroyden/vRBA setup; shock entropy penalties and fluid-specific stream-function constraints are irrelevant here.

**Recommendation:** Approve the current run and retain `final_model_checkpoint.npz` as the converged final model.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | The code implements the 2D Helmholtz manufactured-solution PINN. Architecture: inputs (x,y) pass through separate periodic Fourier embeddings (degree=4, Lx=Ly=2.0) on each coordinate; only the concatenated embeddings enter an MLP body with 3 hidden layers of width 30 and identity output. The PDE residual is uxx + uyy + u - (1 - 17*pi^2)*sin(pi*x)*sin(4*pi*y), and the only loss is L_total = L_pde = mean(residual^2). No BC/IC or data loss is used; periodicity is enforced by the embeddings. Training uses Phase 1 Adam for 10,000 iterations with vRBA weighting and sampling, followed by Phase 2 SSBroyden for 10,000 iterations (100 outer iterations of maxiter=100) where vRBA is used for sampling only and the SSBroyden loss is unweighted. NaN/Inf checks are present. The analytical solution sin(pi*x)*sin(4*pi*y) is used for validation/error metrics. | 3.54e-06 | 35 | 35 | 5 | 15 | **90** |
| 2 | The implementation matches the strategy. It uses a PINN with separate per-coordinate Periodic Fourier Features (degree=4, Lx=Ly=2.0) feeding a 30-wide, 3-hidden-layer MLP with tanh activation and identity scalar output. Inputs are raw coordinates; no external normalization is applied. The PDE residual is u_xx + u_yy + u - (1 - 17*pi^2)*sin(pi*x)*sin(4*pi*y), with analytical solution u_ref = sin(pi*x)*sin(4*pi*y). Only the PDE loss is used; no BC/IC/data losses. Optimizer workflow: Adam phase for 10,000 iterations unless warm-starting from a saved checkpoint; then SSBroyden phase for 20,000 iterations implemented as 200 outer iterations of maxiter=100. The SSBroyden Hessian is reinitialized to identity at the start of the extended stage. vRBA is used for sampling during SSBroyden, and the SSBroyden loss is unweighted. The code also attempts warm-start from final_model_checkpoint.npz and falls back to full retraining if unavailable. It includes plateau checks over the last 50% of the final stage for both loss and relative L2 error, and reports final-model error separately from the best-so-far minimum. | 2.32e-07 | 35 | 35 | 5 | 15 | **90** |
| 3 | The implementation uses a PINN with raw (x,y) passed directly into per-coordinate Periodic Fourier Features (degree=4, Lx=Ly=2.0); embeddings are concatenated and fed to an MLP with 3 hidden layers of width 30 and tanh activation. The PDE is Helmholtz k=1 with manufactured solution u_an = sin(pi x)sin(4 pi y); the loss is only the PDE residual, with no BC/IC/data terms. Training attempts to warm-start from final_model_checkpoint.npz in the run directory; if available it skips Adam, otherwise it falls back to Adam 10,000 iterations followed by SSBroyden. The SSBroyden stage reinitializes H0 to identity, runs 40,000 iterations as 400 outer iterations x maxiter=100, uses vRBA sampling for batch selection while keeping the SSBroyden loss unweighted, and checks NaN/Inf. Final evaluation reports best-so-far and final relative L2 error and includes a post-run last-50% plateau check. | 3.95e-08 | 35 | 35 | 5 | 15 | **90** |
| 4 | The code implements the Helmholtz k=1 forward problem with analytical reference u=sin(pi x)sin(4 pi y). Architecture matches the strategy: per-coordinate Periodic Fourier embedding with degree=4 and Lx=Ly=2.0, concatenated embeddings only (no bare x,y), followed by an MLP body of width 30 and 3 hidden layers with tanh activation and identity output. No input normalization is performed; raw coordinates are passed to the embeddings. The PDE residual is uxx+uyy+u-(1-17 pi^2)sin(pi x)sin(4 pi y), and the only loss is L_pde. Adam phase uses vRBA for both residual weighting and importance sampling; SSBroyden phase uses vRBA sampling only with an unweighted PDE loss. Training follows the advisor cure: warm-start from final_model_checkpoint.npz when present and skip Adam, otherwise run 10,000 Adam iterations; then reinitialize Hessian to identity and run 80,000 SSBroyden iterations as 800 outer iterations with maxiter=100, stopping only on non-finite values. | 2.83e-08 | 35 | 35 | 5 | 15 | **90** |
| 5 | Implemented a 2D Helmholtz PINN. Architecture: per-coordinate Periodic Fourier Features (degree=4, Lx=Ly=2.0) are concatenated and passed to an MLP with 3 hidden layers of width 30, tanh activation, and identity output; no bare x,y enter the MLP and no normalization is applied. The PDE residual is u_xx + u_yy + u - (1 - 17π²) sin(πx) sin(4πy), with k=1. The total loss contains exactly L_pde only; no data, IC, or BC losses are included. Training attempts to warm-start from final_model_checkpoint.npz and skips Adam; if the checkpoint is unavailable, it falls back to 10,000 Adam iterations with vRBA weighting and sampling. The SSBroyden stage reinitializes the Hessian to identity and runs 160,000 fixed iterations as 1,600 outer iterations × maxiter=100. During SSBroyden, vRBA is used for sampling only and the loss remains unweighted. Post-run performs last-50% plateau checks on both loss and RL2 error and reports final-model RL2 separately from the transient best minimum. | 3.00e-08 | 35 | 35 | 12 | 15 | **97** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves 'Helmholtz_results.npz' via np.savez with keys: 'x' (10000x1 float64), 'y' (10000x1 float64), 'u_pred' (10000x1 float64), 'u_an' (10000x1 float64), 'rel_l2_error' (scalar float64), and 'loss_history' (655-element 1D float64 array for the default 10,000 Adam + 10,000 SSBroyden schedule). |
| 2 | Saves 'Helmholtz_results.npz' containing keys: 'x' with shape (10000, 1), 'y' with shape (10000, 1), 'u_pred' with shape (10000, 1), 'u_ref' with shape (10000, 1), 'pointwise_error' with shape (10000, 1), 'rel_l2_error' scalar, and 'loss_history' with shape (N,). Also saves the final model checkpoint to '<args.Name>/final_model_checkpoint.npz' with key 'weights' containing a flattened parameter vector. |
| 3 | Saves Helmholtz_results.npz in the working directory with keys: x (10000x1), y (10000x1), u_pred (10000x1), u_ref (10000x1), residual (10000x1), pointwise_error (10000x1), rel_l2_error (scalar), loss_history (1D, length N_logged; 400 in warm-start mode). Also saves args.Name/final_model_checkpoint.npz containing key weights (flattened 1D parameter vector). With default NC_TEST=100, the 100x100 meshgrid is flattened to 10000 points. |
| 4 | Two npz artifacts are written. Helmholtz_results.npz (np.savez, relative current directory) contains: x (10000,1) float64; y (10000,1) float64; u_pred (10000,1) float64; u_ref (10000,1) float64; residual (10000,1) float64; pointwise_error (10000,1) float64; rel_l2_error scalar float64; best_rel_l2_error scalar float64; error_history 1D float64 of length N_logged; loss_history 1D float64 of length N_logged. For the default full Adam fallback plus 800 SSBroyden outer iterations, N_logged=1355 (555 logged Adam + 800 SSBroyden); if warm-start skips Adam, N_logged=800. final_model_checkpoint.npz (np.savez, in args.Name directory) contains: weights (2401,) float64 flattened parameter vector for the default 16-dimensional embedded input and 30x3 MLP. |
| 5 | Writes Helmholtz_results.npz containing keys: 'x' (10000×1), 'y' (10000×1) flattened grid coordinates for the default 100×100 mesh, 'u_pred' (10000×1), 'u_ref' (10000×1), 'residual' (10000×1), 'pointwise_error' (10000×1), 'rel_l2_error' scalar, 'best_rel_l2_error' scalar, 'error_history' (1D array; length 1600 for a warm-start SSBroyden-only run, or longer if Adam fallback logs additional entries), and 'loss_history' (same length as error_history). Also writes final_model_checkpoint.npz with key 'weights' containing the flat parameter vector of shape (2401,). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run4/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1122/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run4/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1134/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run4/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1142/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run4/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1149/` |
| 5 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run4/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1157/` |


---
