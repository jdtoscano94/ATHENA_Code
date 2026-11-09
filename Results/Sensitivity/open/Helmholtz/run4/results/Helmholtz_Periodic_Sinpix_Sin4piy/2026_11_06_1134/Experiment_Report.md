
# Experiment Report
*Last Updated: 2026-11-06 11:37:53*

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
# STRATEGY: Advisor-Mandated Refinement (Extended Convergence)

**Triage:** Advisor says REJECTED_FOR_REFINEMENT — cure is **NOT architectural**. Keep Run 1's model exactly; **extend the final SSBroyden stage from 10,000 → 20,000 iterations**, warm-starting from Run 1's final weights with a **reinitialized Hessian**. Goal: genuine dual-curve plateau (loss AND error).

---

## 1. Problem Statement (UNCHANGED from Run 1)
- **Governing Equation (2D Helmholtz, k=1):**
$$u_{xx} + u_{yy} + u = g(x,y), \quad (x,y) \in [-1,1]\times[-1,1]$$
- **PDE Residual:**
$$r = u_{xx} + u_{yy} + u - (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$$
- **BCs:** Periodic in x AND y (enforced architecturally — no BC loss).
- **Analytical Solution (validation ONLY):** $u_{an} = \sin(\pi x)\sin(4\pi y)$ ($a_1=1, a_2=4$).
- **Problem Type:** Forward. Well-posed, smooth, strong-form solution exists. No continuation parameter (k=1 fixed).
- **Data Role:** Validation/Error Metrics ONLY. No L_data.

## 2. Architecture (UNCHANGED — Advisor mandate)
Pipeline: **Input (x,y) → Periodic Fourier Embeddings → MLP → Identity → u**
- **Input:** Raw (x,y), already in $[-1,1]$ → **NO normalization** (identity; do not add scalers).
- **Embeddings:** Separate periodic Fourier embeddings per coordinate, **degree=4**, **Lx = Ly = 2.0** (range of raw coords entering embedding). Main body receives **ONLY the concatenated embeddings** (16 features) — never raw x,y.
- **Main Body:** MLP, width=30, depth=3 (3 hidden layers).
- **Output:** Identity, scalar u. No output ansatz needed (no Dirichlet BCs).

## 3. Loss Function
$$L_{total} = L_{pde}$$
(Single term. Periodicity is exact by construction — no BC/IC/data terms.)

## 4. Training Schedule (THE ONLY CHANGE)
- **Warm-start:** Resume from Run 1's final saved model weights. If checkpoint unavailable, fallback = full retrain (Adam 10,000 iters → SSBroyden 20,000 iters).
- **Phase 2 (SSBroyden) — EXTENDED:**
  - **REINITIALIZE the SSBroyden Hessian** at the start of this extended stage (mandatory for stability after transition).
  - Run **20,000 iterations** (200 outer × maxiter=100). Finite budget; stop only on NaN/Inf.
  - vRBA `lambdas`: **Sampling ONLY** during SSBroyden; SSBroyden loss unweighted. Do NOT modify SSBroyden internals.
- **No continuation/annealing.** All physics parameters static.

## 5. Weighting
- **Global:** 1 term → no balancing, no global weights.
- **Local:** vRBA as stated above (never frozen, role changes by phase).

## 6. Reporting & Verification (Advisor mandate)
- Verify **both** loss and relative L2 error form horizontal average bands over the **last 50%** of the final stage.
- Report **final-model error separately** from best-so-far minimum (score uses final value).
- Keep all Run 1 visualizations unchanged (residual heatmap, ref/pred/error fields).

## 7. Saving
- Save predicted field, analytical field, and pointwise error as .npz (keys: `u_pred`, `u_ref`, `x`, `y`).
- **Save the final model checkpoint** to enable future continuation.

---

**Main Model:** [MLP with Hybrid Adam and SSBroyden]
**Enhancements:** [Periodic Fourier Features (degree=4, per-coordinate, Lx=Ly=2.0)]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
REJECTED_FOR_REFINEMENT

## Section 1: Status Decision
* **Status:** REJECTED_FOR_REFINEMENT

The run is stable and highly accurate, but **both the loss and error are still decreasing on average in the final stage**. The required dual-curve plateau is therefore absent.

## Section 2: Quantitative Scorecard
* **Score Integrity:** 35/35
* **Score Accuracy:** 35/35
* **Score Details:** 5/15
* **Score Optimality:** 15/15
* **TOTAL SCORE:** **90/100**

Accuracy uses the **final-model relative \(L^2\) error, \(2.31852002\times10^{-7}\)**, rather than the earlier best-so-far minimum. The implementation solves the requested forward Helmholtz problem with the correct manufactured solution, forcing, periodic formulation, and validation-only use of reference data.

## Section 3: Forensic Analysis & Feedback

### Observation

**Loss and error histories:**
- The plots are readable and all displayed values are finite. There is no explosion or NaN/Inf divergence.
- The total loss begins around \(O(1)\), decreases during the initial Adam stage, and shows a clear optimizer-transition jump near iteration 10,000. During the final SSBroyden stage it decreases smoothly from roughly \(10^{-2}\)–\(10^{-3}\) toward approximately \(10^{-10}\), below the lowest labeled \(10^{-9}\) tick.
- **Final-stage loss midpoint/end test:** the visual band level is approximately  
  \[
  A_{\text{loss}}\approx 1\times10^{-9},\qquad
  B_{\text{loss}}\approx 1\times10^{-10}.
  \]
  Thus \(B_{\text{loss}}\) is roughly an order of magnitude below \(A_{\text{loss}}\): the loss is **still decreasing, not flat**.
- The relative \(L^2\) error begins around \(O(1)\), reaches the \(10^{-2}\) regime during Adam, and then drops sharply after the optimizer transition. Its final stage is jagged but trends downward toward a rightmost value of approximately \(2.3\times10^{-7}\).
- **Final-stage error midpoint/end test:** the oscillation-band centers are approximately  
  \[
  A_{\text{error}}\approx 1\times10^{-6},\qquad
  B_{\text{error}}\approx 3\times10^{-7}.
  \]
  The end band is about three times lower, so the error is also **still decreasing on average**, despite the noise.

The log’s explicit last-50% band calculation confirms these visual readings:
- Loss: \(6.34852433\times10^{-10}\rightarrow1.16583309\times10^{-10}\), a decrease by approximately \(5.4\times\).
- Relative \(L^2\) error: \(8.04727393\times10^{-7}\rightarrow3.20747801\times10^{-7}\), a decrease by approximately \(2.5\times\).
- Both are reported as `horizontal=False`, and the dual-curve plateau verification is `False`.

**Residual and solution fields:**
- The PDE-residual heatmap is mottled and broadly distributed, without a localized shock front or singular corner structure. The log reports residuals from \(1.9683454866026295\times10^{-10}\) to \(3.102341690919275\times10^{-5}\).
- Reference and prediction fields are visually very similar: the periodic bands, sign changes, locations, and amplitudes agree well.
- Pointwise error is smooth and broadly distributed. Larger values occur in reddish central/right patches, while smaller cyan/blue values appear near parts of the left and lower edges. There is no visible shock-like error structure or obvious periodic-boundary mismatch.

**Log error extraction:**
- The earlier `RL2 error: 0.00891681` belongs to the Adam warm-up reporting and is not the final SSBroyden result.
- **Minimum achieved RL2 error:**  
  \[
  \boxed{9.811584296199231\times10^{-8}}
  \]
- **Final relative \(L^2\) error:**  
  \[
  \boxed{2.31852002\times10^{-7}}
  \]
  The final value is used for scoring because the model did not remain at its earlier minimum.

### Diagnosis
**Budget-Limited / Premature Stop — stable and accurate, but still improving.**

This is not instability, stagnation, spectral-bias failure, or a physics-formulation error. The architecture and physics are working extremely well; the fixed training budget ended before either curve established a horizontal final-stage floor. The score of 90 does not override the mandatory plateau requirement.

### Targeted Improvements
* **For Integrity:** None.
* **For Accuracy:** Warm-start from the newly saved final checkpoint, reinitialize the SSBroyden Hessian, and **extend the final SSBroyden stage from 20,000 to 40,000 iterations** using that fixed budget. Keep the PDE, periodic Fourier embeddings, MLP, and vRBA configuration unchanged. Assess the dual-curve plateau from the resulting plots after completion.
* **For Details:** None beyond the required post-run verification that both final-stage average bands become horizontal.
* **For Optimality:** None; no KAN, architecture change, altered physics, or entropy regularization is warranted for this smooth, non-shock problem.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | The code implements the 2D Helmholtz manufactured-solution PINN. Architecture: inputs (x,y) pass through separate periodic Fourier embeddings (degree=4, Lx=Ly=2.0) on each coordinate; only the concatenated embeddings enter an MLP body with 3 hidden layers of width 30 and identity output. The PDE residual is uxx + uyy + u - (1 - 17*pi^2)*sin(pi*x)*sin(4*pi*y), and the only loss is L_total = L_pde = mean(residual^2). No BC/IC or data loss is used; periodicity is enforced by the embeddings. Training uses Phase 1 Adam for 10,000 iterations with vRBA weighting and sampling, followed by Phase 2 SSBroyden for 10,000 iterations (100 outer iterations of maxiter=100) where vRBA is used for sampling only and the SSBroyden loss is unweighted. NaN/Inf checks are present. The analytical solution sin(pi*x)*sin(4*pi*y) is used for validation/error metrics. | 3.54e-06 | 35 | 35 | 5 | 15 | **90** |
| 2 | The implementation matches the strategy. It uses a PINN with separate per-coordinate Periodic Fourier Features (degree=4, Lx=Ly=2.0) feeding a 30-wide, 3-hidden-layer MLP with tanh activation and identity scalar output. Inputs are raw coordinates; no external normalization is applied. The PDE residual is u_xx + u_yy + u - (1 - 17*pi^2)*sin(pi*x)*sin(4*pi*y), with analytical solution u_ref = sin(pi*x)*sin(4*pi*y). Only the PDE loss is used; no BC/IC/data losses. Optimizer workflow: Adam phase for 10,000 iterations unless warm-starting from a saved checkpoint; then SSBroyden phase for 20,000 iterations implemented as 200 outer iterations of maxiter=100. The SSBroyden Hessian is reinitialized to identity at the start of the extended stage. vRBA is used for sampling during SSBroyden, and the SSBroyden loss is unweighted. The code also attempts warm-start from final_model_checkpoint.npz and falls back to full retraining if unavailable. It includes plateau checks over the last 50% of the final stage for both loss and relative L2 error, and reports final-model error separately from the best-so-far minimum. | 2.32e-07 | 35 | 35 | 5 | 15 | **90** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves 'Helmholtz_results.npz' via np.savez with keys: 'x' (10000x1 float64), 'y' (10000x1 float64), 'u_pred' (10000x1 float64), 'u_an' (10000x1 float64), 'rel_l2_error' (scalar float64), and 'loss_history' (655-element 1D float64 array for the default 10,000 Adam + 10,000 SSBroyden schedule). |
| 2 | Saves 'Helmholtz_results.npz' containing keys: 'x' with shape (10000, 1), 'y' with shape (10000, 1), 'u_pred' with shape (10000, 1), 'u_ref' with shape (10000, 1), 'pointwise_error' with shape (10000, 1), 'rel_l2_error' scalar, and 'loss_history' with shape (N,). Also saves the final model checkpoint to '<args.Name>/final_model_checkpoint.npz' with key 'weights' containing a flattened parameter vector. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run4/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1122/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run4/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1134/` |


---
