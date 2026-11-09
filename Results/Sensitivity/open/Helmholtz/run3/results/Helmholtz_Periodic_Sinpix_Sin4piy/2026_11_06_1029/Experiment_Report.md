
# Experiment Report
*Last Updated: 2026-11-06 10:33:19*

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
# SciML Strategy: 2D Helmholtz (Manufactured Solution) — Rev. 2

**0. Triage:** Initial proposal, Critic correction accepted → fixed finite budgets. No user architecture requests, no Advisor report.

**1. Data Source:** None. Analytical solution exists → **Role: Validation/Error Metrics ONLY.**

**2. Problem Statement**
- **Governing Equation (Helmholtz, k=1):**
$$u_{xx} + u_{yy} + k^2 u = g(x,y), \quad k=1$$
- **PDE Residual:**
$$r(x,y) = u_{xx} + u_{yy} + u - g(x,y)$$
- **Source term (derived from manufactured solution):**
$$g(x,y) = (1 - 17\pi^2)\,\sin(\pi x)\sin(4\pi y) \approx -166.8\,\sin(\pi x)\sin(4\pi y)$$
(since $u_{xx}=-\pi^2 u$, $u_{yy}=-16\pi^2 u$ for $u_{an}$).
- **BCs:** Periodic in both x and y on $[-1,1]^2$: $u(-1,y)=u(1,y)$, $u(x,-1)=u(x,1)$. No IC (steady problem — no time dimension despite "spatio-temporal" wording).
- **Analytical Solution (YES):** $u_{an}(x,y) = \sin(\pi a_1 x)\sin(\pi a_2 y) = \sin(\pi x)\sin(4\pi y)$, with $a_1=1$, $a_2=4$, $k=1$.
- **Data Loss Term:** None. **Problem Type:** Forward.
- **Well-posedness:** Yes. $k=1$ is non-resonant for the periodic Laplacian (eigenvalues $\pi^2(m^2+n^2) \geq \pi^2 > 1$). Smooth strong solution → PINN-suitable.

**2.1 Input:** d=2, $(x,y)\in[-1,1]^2$. No time.
**2.2 Output:** Scalar $u(x,y)$. Smooth, oscillatory (4 periods in y), periodic in both directions.
**2.3 Analytical Solution:** $u_{an}(x,y)=\sin(\pi x)\sin(4\pi y)$ (use for L2 error metrics).

**3. Base Model:** 2D Forward → **MLP with Hybrid Adam and SSBroyden**. Base MLP: width=30, depth=3 (template defaults).

**4. Main Architecture**
Pipeline: **Input → Input Transformation (Embedding) → Main Body (MLP) → Output (identity) → u**

- **4.1 Input Transformation:**
  - **Normalization: NONE.** x, y already in $[-1,1]$ (nice range) — pass raw coordinates, do NOT add scalers.
  - **Periodic Fourier embedding on BOTH x and y** (periodic BCs): $\{\cos(2\pi k x/L_x), \sin(2\pi k x/L_x)\}$, same for y.
  - **Lx = Ly = 2.0** (raw range $[-1,1]$ → max−min = 2; arguments become $\pi k x$, $\pi k y$).
  - **Degree = 5** — physics-driven: solution spectrum requires harmonics k=1 (x) and k=4 (y); degree=2 would omit the k=4 harmonic. (5 < 20 stability bound.)
  - **Input composition:** Main body receives ONLY the 20 embedding features (10 per coordinate). NO raw x, y to the body.
  - Embedding MUST be differentiable and implemented INSIDE the model call function.

- **4.2 Main Body:** MLP, width=30, depth=3 (template default activation/initializer).
- **4.3 Output Transformation:** None — raw scalar output. Periodicity is exact by construction (embedding). No Dirichlet BCs → no ansatz/distance function needed.

**5. Constraint Strategy:** Count = $L_{pde}$ only (1). Periodic BC excluded (architecturally enforced by embedding — do NOT add $L_{bc\_periodic}$). $1 < 5$ → loss-based, no hard constraints.

**6. Loss Assembly:**
$$L_{total} = L_{pde}$$

**6.1 Continuation:** Not applicable (no Re/Peclet; k=1 fixed and mild).

**Schedule (EXPLICIT FINITE BUDGETS):**
- **Phase 1 — Adam:** **exactly 10,000 iterations** (STATIC parameters).
- **Phase 2 — SSBroyden:** **exactly 10,000 iterations**.
- **Total: 20,000 iterations, hard-capped.** Early stopping permitted only strictly within these budgets; training halts immediately on NaN/Inf loss or error.

**7. Weighting:**
- **Local:** vRBA on the PDE residual. **"lambdas are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden."**
- **Global:** 1 term → no balancing needed (no `lam` required).
- Do NOT modify SSBroyden's internal weighting/sampling/batching.

**8. Visualization:** Keep template callbacks unchanged.

**9. Saving:** Save `u_pred`, `u_an`, pointwise error, and the (x,y) grid as .npz with keys: `'u_pred'`, `'u_an'`, `'error'`, `'x'`, `'y'`.

---

**Modules for Librarian:**
Main Model: **MLP with Hybrid Adam and SSBroyden**
Enhancements: **Periodic Fourier Features (degree=5, Lx=Ly=2.0)**

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
REJECTED_FOR_REFINEMENT

## Section 1: Status Decision
* **Status:** `REJECTED_FOR_REFINEMENT`
* **Problem type:** **Type A — forward problem**, with no noisy data. The analytical solution is used only for validation, so there is no data-leakage violation.
* Although the numerical score exceeds 90, the run fails the mandatory final-stage plateau check: the loss is still decreasing at the right edge.

## Section 2: Quantitative Scorecard
* **Score Integrity:** 35/35 — completed successfully; finite values and readable plots.
* **Score Accuracy:** 35/35 — using a scoring target of \(\epsilon_{\text{target}}=10^{-3}\) relative \(L^2\), since no tolerance was specified, the final plotted error of order \(10^{-6}\) earns 20/20 for precision. The requested equation, source term, parameters, domain, and periodic topology are all correct, earning 15/15 for consistency.
* **Score Details:** 7/15 — periodic boundaries are enforced architecturally and the fields agree visually, but there is no genuine final loss/error plateau.
* **Score Optimality:** 14/15 — appropriate periodic Fourier embedding, vRBA, low-dimensional SSBroyden fine-tuning, and correct residual formulation. No entropy penalty is needed for this smooth elliptic problem.
* **TOTAL SCORE:** **91/100**

## Section 3: Forensic Analysis & Feedback

### Observation
* **Total Loss History:** The y-axis spans approximately \(10^{2}\) down to a few \(10^{-9}\). There are two clear stages:
  * Adam, iterations 0–10,000: rapid initial decrease followed by a slow descent around \(10^{-2}\).
  * SSBroyden, iterations 10,000–20,000: a brief transition spike, then a smooth, sustained decrease over several orders of magnitude.
  * Final-stage midpoint and end readings:
    * **Loss \(A\), at iteration 15,000:** approximately \(\mathbf{10^{-7}}\).
    * **Loss \(B\), at iteration 20,000:** approximately \(\mathbf{2\times10^{-9}}\).
  * Since \(B\ll A\), the loss is unambiguously **still decreasing**, not flat.

* **Relative \(L^2\) Error History:** The error begins near \(10^{0}\), decreases to roughly \(10^{-2}\)–\(10^{-3}\) during Adam, and then becomes noisy while trending downward during SSBroyden.
  * **Error \(A\), band center near iteration 15,000:** approximately \(\mathbf{10^{-5}}\).
  * **Error \(B\), band center near iteration 20,000:** approximately \(\mathbf{10^{-6}}\).
  * Thus the error band also remains lower at the end than at the midpoint. The rightmost plotted error is approximately **\(8\times10^{-7}\), i.e. order \(10^{-6}\)**, which is used for scoring.
  * The exact bottom-of-log best-ever value is:
    **“Minimum achieved RL2 error: 3.6982635991314445e-07.”**
    This is a best dip, not evidence that the final curve plateaued. The repeated `RL2 error: 0.00435141` printout does not represent the right edge of the final plotted curve.

* **PDE Residual Heatmap:** The residual is spatially speckled and broadly distributed, without a coherent boundary, corner, or shock-like failure region. The log confirms finite residuals ranging from approximately \(2.26\times10^{-9}\) to \(8.27\times10^{-5}\).

* **Reference versus Prediction:** The fields are visually very similar. The prediction reproduces the four oscillations in \(y\), the sinusoidal modulation in \(x\), the signs, and the apparent amplitude without a visible phase shift.

* **Pointwise Error:** Errors are broadly distributed rather than concentrated at the periodic boundaries. The conspicuous blue/cyan pocket near \((x,y)\approx(-0.75,-0.55)\) is a localized **low-error** region. No shock or boundary-layer artifact is visible.

### Diagnosis
**Budget-Limited / Premature Stop.** The formulation is stable and already highly accurate, but the final SSBroyden stage ended while the loss—and on average the error—was still improving. The best-ever minimum must not be mistaken for convergence at the final model.

### Targeted Improvements
* **For Integrity:** None.
* **For Accuracy:** **Double only the final SSBroyden stage from 10,000 to 20,000 iterations**, giving 30,000 total iterations while retaining the 10,000-iteration Adam stage. Start from the trained weights; if implemented as a restarted stage, reinitialize the SSBroyden Hessian. Keep the physics, Fourier degree, network capacity, and vRBA settings unchanged.
* **For Details:** None beyond the iteration extension. In the next assessment, compare the average band levels of both final-stage curves and report final-iteration error separately from the best-ever minimum.
* **For Optimality:** None. Do not switch to KAN, remove periodic embeddings, or add entropy regularization; none addresses the observed bottleneck.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | The code implements a 2D forward Helmholtz PINN for u_xx + u_yy + u = g with k=1 and source g=(1-17π²) sin(πx) sin(4πy). The network uses an MLP with width 30 and three hidden layers, preceded by periodic Fourier features on both x and y with degree 5 and Lx=Ly=2.0; the MLP body receives only the 20 embedding features, with no raw coordinates and no external normalization. The PDE residual is exactly r = u_xx + u_yy + u - g. The total loss is L_pde only; no boundary, data, or gauge loss is added. vRBA lambdas are used for weighting and importance sampling in Adam, and for sampling only in SSBroyden. Training uses Adam for exactly 10,000 iterations followed by SSBroyden for exactly 10,000 iterations, for a hard-capped total of 20,000. The analytical solution u_an=sin(πx)sin(4πy) is used only for validation and error metrics. | 8.00e-07 | 35 | 35 | 7 | 14 | **91** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves the required file Helmholtz_results.npz with keys: 'u_pred' (100x100), 'u_an' (100x100), 'error' (100x100), 'x' (100x100), and 'y' (100x100). It also saves Log_files_SSBroyden.mat containing training logs such as all_errors, all_its, all_loss, all_gamma, all_lamB, all_max_RBA, all_L_infty, and all_L_2. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run3/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1029/` |


---
