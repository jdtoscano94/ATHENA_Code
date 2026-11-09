
# Experiment Report
*Last Updated: 2026-11-06 11:46:36*

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
# STRATEGY: Run 3 — Warm-Start Refinement (Advisor-Mandated)

**Triage:** Path 2 (Advisor Recommendation). The Advisor diagnosed **Budget-Limited / Premature Stop** — no architecture, physics, or weighting changes are warranted. The ONLY change: **warm-start from the saved checkpoint, reinitialize the SSBroyden Hessian, and extend the final SSBroyden stage to 40,000 iterations (fixed budget)**. Everything else stays EXACTLY as Run 2.

---

## 1. Data Role
**Role: Validation/Error Metrics ONLY.** Forward problem; analytical solution is the reference. No data loss.

## 2. Problem Statement
- **Governing Equation (Helmholtz, k=1):** $u_{xx} + u_{yy} + k^2 u = g(x,y)$, with manufactured solution $u_{an} = \sin(\pi x)\sin(4\pi y)$ ($a_1=1, a_2=4$).
- **PDE Residual:** $r = u_{xx} + u_{yy} + u - (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$
- **BCs:** Periodic in $x$ and $y$ on $[-1,1]\times[-1,1]$ — **enforced by architecture (no BC loss).** No IC (steady-state).
- **Analytical Solution:** Yes — $u_{an} = \sin(\pi x)\sin(4\pi y)$. Smooth, oscillatory (4 periods in $y$), strong-form well-posed for PINNs.
- **Problem Type:** Forward. No data loss term.

## 3. Input/Output
- **Input (d=2):** $(x, y)$, raw coordinates already in $[-1,1]$ — **NO normalization (identity), pass through unchanged.**
- **Output (q=1):** scalar $u(x,y)$.

## 4. Architecture (UNCHANGED from Run 2)
**Main Model: MLP with Hybrid Adam and SSBroyden** (Case A: 2D).

Pipeline: `Input (raw x,y) → Per-coordinate Periodic Fourier Embedding (degree=4, Lx=Ly=2.0, computed inside call) → concat embeddings ONLY (no bare x,y to body) → MLP (width=30, depth=3, tanh) → identity → u(x,y)`

- **Embedding:** Periodic Fourier Features, degree=4, `Lx = Ly = 2.0` (raw range $[-1,1]$ → range 2.0; consistent, no mismatch).
- **Output transformation:** None (identity); periodicity is architectural.

## 5. Constraints
Periodic BCs satisfied exactly by the periodic embeddings — no `L_bc_periodic`, no hard-constraint ansatz needed.

## 6. Loss Assembly
$$L_{total} = L_{pde}$$
Single term. No BC/IC/data losses.

## 7. Training Schedule (THE ONLY CHANGE)
- **Warm-start:** Load `final_model_checkpoint.npz` from Run 2. **Skip Adam** (model already in the correct basin). Fallback if checkpoint unavailable: full retrain — Adam 10,000 iters, then SSBroyden 40,000 iters.
- **SSBroyden Phase:** **REINITIALIZE the Hessian to identity** at stage start (mandatory for stability after transition). Run **40,000 iterations fixed budget** (e.g., 400 outer × maxiter=100).
- **vRBA:** Sampling ONLY during SSBroyden; SSBroyden loss unweighted. Do NOT interfere with SSBroyden internals.
- **Global weights:** Single loss term — no balancing needed.
- **NaN/Inf guard:** Stop immediately if loss or error becomes NaN/Inf. Fixed budget only; no open-ended stopping rules. Plateau is judged post-run by the Advisor (verify dual-curve plateau over last 50% of the final stage in the report).

## 8. Visualization
**Do NOT modify** existing visualization callbacks. Keep loss/error history plots and field heatmaps (prediction, reference, pointwise error, residual).

## 9. Saving
Save final model checkpoint (for potential future warm-start) and fields as `.npz` with keys: `x`, `y`, `u_pred`, `u_ref`, `residual`. Report final-model relative $L^2$ error separately from best-so-far minimum.

---

**Main Model:** MLP with Hybrid Adam and SSBroyden
**Enhancements:** Periodic Fourier Features (per-coordinate, degree=4, Lx=Ly=2.0)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
REJECTED_FOR_REFINEMENT

## Section 1: Status Decision
* **Status:** **REJECTED_FOR_REFINEMENT**

The run is stable and highly accurate, but it fails the mandatory dual-curve plateau check. Both the loss and error band centers are still decreasing in the final region, so this is a **Budget-Limited / Premature Stop**, not a converged optimum.

## Section 2: Quantitative Scorecard
* **Score Integrity:** **35/35** — Completed successfully; all plots are readable and values are finite. No NaN, Inf, or divergence is visible.
* **Score Accuracy:** **35/35**
  * **Numerical Precision:** 20/20. The final-model relative \(L^2\) error is \(3.94694536\times10^{-8}\). No tolerance was explicitly requested; even using \(\epsilon_{\text{target}}=10^{-5}\) for this smooth manufactured forward problem, the final error easily qualifies for full credit.
  * **Consistency:** 15/15. This is a **Type A forward problem without noise**. The implementation solves the requested Helmholtz equation with \(k=1\), \(a_1=1\), \(a_2=4\), and periodicity in both directions. The analytical solution is used only for validation, with no data-loss leakage.
* **Score Details:** **5/15** — Periodicity is enforced architecturally and the solution fields agree closely, but the final loss/error curves have not plateaued. The error history remains strongly oscillatory while its band center drifts downward.
* **Score Optimality:** **15/15** — Appropriate pipeline: raw coordinates \(\rightarrow\) per-coordinate periodic Fourier embeddings \(\rightarrow\) MLP \(\rightarrow\) scalar output. vRBA sampling and low-dimensional SSBroyden fine-tuning are present. No entropy penalty is needed because this problem has no shocks.
* **TOTAL SCORE:** **90/100**

**Decision gate:** Although the score reaches 90, approval is prohibited because the final-stage curves are still decreasing on average.

## Section 3: Forensic Analysis & Feedback

### Observation

**1. Total Loss History**
* The curve is finite and stable. It begins around the \(10^{-10}\)–\(10^{-9}\) region, has a brief initial rise, and then decreases mostly smoothly to approximately \(10^{-12}\), with small spikes superimposed later in training.
* No later continuation jump is visible, so the displayed curve is treated as the final stage.
* Explicit midpoint-versus-end reading:
  * **A, midpoint band center:** approximately **\(1\times10^{-11}\)** around iterations 10,000–11,000.
  * **B, ending band center:** approximately **\(1.2\times10^{-12}\)** near the right edge.
  * Thus \(B\) is roughly eight times lower than \(A\): the loss is **still decreasing**, not flat.
* The log’s independent last-50% half-window calculation confirms this: **\(3.83617533\times10^{-12}\rightarrow1.38321569\times10^{-12}\)**, drift \(=-0.4430\) dex, `horizontal=False`.

**2. Relative \(L^2\) Error History**
* The error is jagged throughout, with oscillations extending roughly from \(10^{-8}\) toward several \(10^{-7}\), but its typical band moves downward.
* Explicit midpoint-versus-end reading:
  * **A, midpoint band center:** approximately **\(8\times10^{-8}\)**.
  * **B, ending band center:** approximately **\(4\times10^{-8}\)**.
  * Individual dips approach \(10^{-8}\), but those lucky minima do not define the band center or the final-model error.
* The log confirms the last-50% half-window decrease: **\(7.76537873\times10^{-8}\rightarrow3.82227893\times10^{-8}\)**, drift \(=-0.3078\) dex, `horizontal=False`.
* Even if this approximately factor-of-two error decrease were considered borderline, the **loss gate unambiguously fails**, so the run cannot be classified as converged.

**3. Residual and Solution Fields**
* The PDE-residual heatmap is a spatially distributed orange/red speckled field, without a localized shock front or one dominant corner failure. The log reports residuals from \(9.1864\times10^{-12}\) to \(3.5325\times10^{-6}\).
* Reference and prediction are visually very close: both show the same alternating horizontal bands, left/right sign reversal, phase, and amplitude. No substantial difference is visible at the shared solution color scale.
* The pointwise-error field is nonuniform. Red/orange higher-error regions occur mainly around outer portions of the domain, while the smallest errors form a blue pocket slightly left of center around \(y\approx-0.4\). This is an accuracy/convergence issue, not evidence of a periodic-boundary formulation failure.

**4. Log Extraction**
* **Minimum achieved RL2 error:** \(\boxed{9.91763707216313\times10^{-9}}\).
* **Final relative \(L^2\) error:** \(\boxed{3.94694536\times10^{-8}}\).
* The rightmost plot value, approximately \(4\times10^{-8}\), agrees with the reported final error. **Scoring uses the final value, not the earlier best dip.**
* The log explicitly reports: `Dual-curve plateau verified over last 50% of SSBroyden: False`.

### Diagnosis
**Budget-Limited / Premature Stop (Still Improving).** The physics, architecture, and optimization pipeline are working very well. The remaining bottleneck is that training ended while the average loss band—and also the noisy error band—was still moving downward. This is neither instability nor capacity-limited stagnation.

### Targeted Improvements
* **For Integrity:** None.
* **For Accuracy:** **Warm-start from this run’s final checkpoint, reinitialize the SSBroyden Hessian to identity, and double the final SSBroyden budget from 40,000 to 80,000 fixed iterations**—for example, 800 outer iterations with `maxiter=100`. Keep the PDE, Fourier embeddings, MLP, vRBA settings, and loss formulation unchanged.
* **For Details:** After the extended run, reassess both curves using their average band centers over the last 50% of the final stage. Do not substitute the best-so-far error dip or optimizer termination for evidence of a plateau.
* **For Optimality:** None. Retain the periodic Fourier input embedding and SSBroyden/vRBA pipeline; no KAN, architecture, physics, or entropy modification is warranted for this run.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | The code implements the 2D Helmholtz manufactured-solution PINN. Architecture: inputs (x,y) pass through separate periodic Fourier embeddings (degree=4, Lx=Ly=2.0) on each coordinate; only the concatenated embeddings enter an MLP body with 3 hidden layers of width 30 and identity output. The PDE residual is uxx + uyy + u - (1 - 17*pi^2)*sin(pi*x)*sin(4*pi*y), and the only loss is L_total = L_pde = mean(residual^2). No BC/IC or data loss is used; periodicity is enforced by the embeddings. Training uses Phase 1 Adam for 10,000 iterations with vRBA weighting and sampling, followed by Phase 2 SSBroyden for 10,000 iterations (100 outer iterations of maxiter=100) where vRBA is used for sampling only and the SSBroyden loss is unweighted. NaN/Inf checks are present. The analytical solution sin(pi*x)*sin(4*pi*y) is used for validation/error metrics. | 3.54e-06 | 35 | 35 | 5 | 15 | **90** |
| 2 | The implementation matches the strategy. It uses a PINN with separate per-coordinate Periodic Fourier Features (degree=4, Lx=Ly=2.0) feeding a 30-wide, 3-hidden-layer MLP with tanh activation and identity scalar output. Inputs are raw coordinates; no external normalization is applied. The PDE residual is u_xx + u_yy + u - (1 - 17*pi^2)*sin(pi*x)*sin(4*pi*y), with analytical solution u_ref = sin(pi*x)*sin(4*pi*y). Only the PDE loss is used; no BC/IC/data losses. Optimizer workflow: Adam phase for 10,000 iterations unless warm-starting from a saved checkpoint; then SSBroyden phase for 20,000 iterations implemented as 200 outer iterations of maxiter=100. The SSBroyden Hessian is reinitialized to identity at the start of the extended stage. vRBA is used for sampling during SSBroyden, and the SSBroyden loss is unweighted. The code also attempts warm-start from final_model_checkpoint.npz and falls back to full retraining if unavailable. It includes plateau checks over the last 50% of the final stage for both loss and relative L2 error, and reports final-model error separately from the best-so-far minimum. | 2.32e-07 | 35 | 35 | 5 | 15 | **90** |
| 3 | The implementation uses a PINN with raw (x,y) passed directly into per-coordinate Periodic Fourier Features (degree=4, Lx=Ly=2.0); embeddings are concatenated and fed to an MLP with 3 hidden layers of width 30 and tanh activation. The PDE is Helmholtz k=1 with manufactured solution u_an = sin(pi x)sin(4 pi y); the loss is only the PDE residual, with no BC/IC/data terms. Training attempts to warm-start from final_model_checkpoint.npz in the run directory; if available it skips Adam, otherwise it falls back to Adam 10,000 iterations followed by SSBroyden. The SSBroyden stage reinitializes H0 to identity, runs 40,000 iterations as 400 outer iterations x maxiter=100, uses vRBA sampling for batch selection while keeping the SSBroyden loss unweighted, and checks NaN/Inf. Final evaluation reports best-so-far and final relative L2 error and includes a post-run last-50% plateau check. | 3.95e-08 | 35 | 35 | 5 | 15 | **90** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves 'Helmholtz_results.npz' via np.savez with keys: 'x' (10000x1 float64), 'y' (10000x1 float64), 'u_pred' (10000x1 float64), 'u_an' (10000x1 float64), 'rel_l2_error' (scalar float64), and 'loss_history' (655-element 1D float64 array for the default 10,000 Adam + 10,000 SSBroyden schedule). |
| 2 | Saves 'Helmholtz_results.npz' containing keys: 'x' with shape (10000, 1), 'y' with shape (10000, 1), 'u_pred' with shape (10000, 1), 'u_ref' with shape (10000, 1), 'pointwise_error' with shape (10000, 1), 'rel_l2_error' scalar, and 'loss_history' with shape (N,). Also saves the final model checkpoint to '<args.Name>/final_model_checkpoint.npz' with key 'weights' containing a flattened parameter vector. |
| 3 | Saves Helmholtz_results.npz in the working directory with keys: x (10000x1), y (10000x1), u_pred (10000x1), u_ref (10000x1), residual (10000x1), pointwise_error (10000x1), rel_l2_error (scalar), loss_history (1D, length N_logged; 400 in warm-start mode). Also saves args.Name/final_model_checkpoint.npz containing key weights (flattened 1D parameter vector). With default NC_TEST=100, the 100x100 meshgrid is flattened to 10000 points. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run4/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1122/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run4/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1134/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run4/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1142/` |


---
