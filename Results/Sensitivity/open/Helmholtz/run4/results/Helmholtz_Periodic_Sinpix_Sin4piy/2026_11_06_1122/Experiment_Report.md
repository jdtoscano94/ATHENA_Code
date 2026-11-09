
# Experiment Report
*Last Updated: 2026-11-06 11:26:16*

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
# STRATEGIC PLAN: 2D Helmholtz (Manufactured Solution)

**0. Triage:** Initial proposal. No user architecture requests, no Advisor. Standard strategy.

**1. Data Source:** None. No external dataset. The analytical solution is computed in-code.
**Role: Validation/Error Metrics ONLY.**

**2. Problem Statement**
- **Governing Equation:** Helmholtz: $u_{xx} + u_{yy} + k^2 u = g(x,y)$, with $k=1$.
- **PDE Residual:** $r(x,y) = u_{xx} + u_{yy} + u - (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$
- **Source term (derived from manufactured solution):** $g(x,y) = (k^2 - \pi^2 a_1^2 - \pi^2 a_2^2)\sin(\pi a_1 x)\sin(\pi a_2 y) = (1-17\pi^2)\sin(\pi x)\sin(4\pi y) \approx -166.78\,\sin(\pi x)\sin(4\pi y)$
- **BCs:** Periodic in BOTH $x$ and $y$ on $[-1,1]\times[-1,1]$.
- **Analytical Solution:** YES. $u_{an}(x,y) = \sin(\pi a_1 x)\sin(\pi a_2 y) = \sin(\pi x)\sin(4\pi y)$, $a_1=1$, $a_2=4$. (Verified compatible with periodicity: $u$ and normal derivatives match at $x=\pm1$, $y=\pm1$.)
- **Data Loss Term:** No.
- **Problem Type:** Forward.
- **Well-posedness:** Yes. Smooth strong solution exists; no resonance ($k^2=1$ is not a Laplacian eigenvalue $\pi^2(m^2+n^2)$). Strong-form PINN is valid.

**2.1 Input:** d=2, purely spatial $(x,y)\in[-1,1]^2$. No time.

**2.2 Output:** Scalar $u(x,y)$ (q=1). Smooth, oscillatory (4 periods in $y$), periodic in both axes.

**2.3 Analytical Solution:** $u_{an} = \sin(\pi x)\sin(4\pi y)$ — use for relative L2 error and error-field plots.

**3. Base Model:** [MLP with Hybrid Adam and SSBroyden] — 2D problem → Base MLP: width=30, depth=3.

**4. Main Architecture**
Pipeline: $(x,y)$ → Periodic Fourier Embedding → MLP Body → Identity → $u$.

- **Input Transformation:** NO normalization — coordinates already in $[-1,1]$ (pass raw; any scaler is forbidden here). **Periodic Fourier Features on BOTH $x$ and $y$:** degree=4, $L_x = 2.0$, $L_y = 2.0$ (ranges of raw coordinates entering the embedding). Degree=4 is physics-driven: the solution spectrum requires harmonics $k=1$ ($x$) and $k=4$ ($y$); degree=2 cannot represent the $4\pi y$-frequency. Well below the <20 stability bound.
- **Input Composition Rule:** Main body receives ONLY the concatenated embeddings — never raw $x$ or $y$.
- **Main Body:** MLP, width=30, depth=3 (template defaults).
- **Output Transformation:** None (identity). Periodicity is enforced by architecture; no Dirichlet constraints exist.

**5. Constraint Strategy:** Count = 1 ($L_{pde}$ only; $L_{bc,periodic}$ REMOVED — periodic embedding enforces it exactly). Count < 5 → Loss-based, no hard constraints.

**6. Loss Function Assembly**
$$L_{total} = L_{pde}$$

**6.1 Continuation:** N/A — no Reynolds/Peclet; $k=1$ is fixed and low.

**Training Schedule (finite budgets):**
- Phase 1 (Adam): 10,000 iterations, static parameters.
- Phase 2 (SSBroyden): 10,000 iterations, static parameters.
- Halt immediately on NaN/Inf loss or error.

**7. Weighting Strategy**
- **Local:** vRBA. State: "**`lambdas` are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden.**"
- **Global:** Total terms = 1 → no balancing needed.
- **SSBroyden Integrity:** Do NOT modify internal weighting/sampling/batching of the SSBroyden template.

**8. Visualization:** Do NOT modify template callbacks. Ensure plots compare $u_{pred}$ vs $u_{an}$ plus pointwise error field.

**9. Saving:** Save .npz with keys: `x`, `y`, `u_pred`, `u_an`, `rel_l2_error`, `loss_history`. Report final relative L2 error.

---

**Modules for Librarian:**
Main Model: [MLP with Hybrid Adam and SSBroyden]
Enhancements: [Periodic Fourier Features (degree=4, Lx=Ly=2.0)]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
REJECTED_FOR_REFINEMENT

## Section 1: Status Decision
* **Status:** **REJECTED_FOR_REFINEMENT**
* The run is stable, physically consistent, and highly accurate. However, it fails the mandatory dual-curve convergence check: both the loss and error bands are still decreasing during the final SSBroyden stage. The excellent score therefore cannot justify approval.

## Section 2: Quantitative Scorecard
* **Score Integrity:** 35/35
* **Score Accuracy:** 35/35  
  * Numerical Precision: 20/20. No tolerance was specified; using a conservative manufactured-solution benchmark of \(\epsilon_{\text{target}}=10^{-3}\), the final error of approximately \(3.54\times10^{-6}\) easily qualifies.
  * Consistency: 15/15. The correct source term, parameters, periodic domain, and solution are used. The analytical solution is used only for validation, so there is no data leakage.
* **Score Details:** 5/15
* **Score Optimality:** 15/15
* **TOTAL SCORE:** **90/100**

## Section 3: Forensic Analysis & Feedback

### Observation
This is a **Type A forward problem**, with no observational noise or external training data.

* **Total Loss History:**
  * The vertical axis spans approximately \(10^{1}\) down to below \(10^{-7}\), ending at a few times \(10^{-8}\).
  * During Adam, the loss decreases rapidly at first, followed by a slower, slightly irregular descent. A transition/spike occurs near iteration 10,000.
  * The final SSBroyden region runs approximately from iteration 10,000 to 16,000. Its loss decreases steeply and smoothly rather than forming a horizontal floor.
  * Explicit midpoint/end readings:
    * \(A_{\text{loss}}\approx 1\times10^{-6}\) near the final-stage midpoint, iteration 13,000.
    * \(B_{\text{loss}}\approx 3\times10^{-8}\) at the end.
  * Since \(B_{\text{loss}}<A_{\text{loss}}\) by roughly a factor of 30, the loss is **still decreasing**, not plateaued.

* **Relative \(L^2\) Error History:**
  * The axis spans approximately \(10^{0}\) to \(10^{-6}\), with some best dips below \(10^{-6}\).
  * The Adam error is nonmonotonic: it decreases initially, then rises and fluctuates before the optimizer transition.
  * During SSBroyden, the error becomes jagged but its oscillation band moves downward overall.
  * Explicit band-center readings:
    * \(A_{\text{error}}\approx 1\times10^{-5}\) near iteration 13,000.
    * \(B_{\text{error}}\approx 3\times10^{-6}\) at the end, with the rightmost value consistent with the logged final error.
  * Thus, the error is also **still decreasing on average**, with superimposed oscillations. Neither curve passes the plateau test.

* **PDE Residual Heatmap:**
  * The residual is mottled and broadly distributed across the domain, without a dominant shock-like structure or a clear periodic-boundary failure.
  * The log confirms finite residuals ranging from \(3.050999453080294\times10^{-10}\) to \(2.4482302036688\times10^{-4}\).

* **Reference, Prediction, and Pointwise Error:**
  * Reference and prediction are visually very similar: the four oscillations in \(y\), the sign change in \(x\), and the amplitudes all match at the displayed scale.
  * Pointwise errors are of order \(10^{-6}\), broadly distributed, with lower-error blue/green patches near the left and right sides. There is no visually significant boundary discontinuity.

* **Log Confirmation:**
  * Because SSBroyden was used, the earlier `RL2 error: 0.00891681` is treated as the Adam warm-up result and ignored for final scoring.
  * **Minimum achieved RL2 error:** `8.77793725525075e-07`.
  * **Final relative L2 error:** `3.54080509e-06`.
  * The final value, not the earlier best dip, is used for scoring. The model did not remain at its best-so-far error.

### Diagnosis
**Budget-Limited / Premature Stop.** The formulation is working and the final accuracy is excellent, but the final-stage loss gate fails: the loss remains on a clear descending trend, and the noisy error band also continues downward. This is neither divergence nor converged stagnation.

### Targeted Improvements
* **For Integrity:** None. Execution, plots, and numerical values are valid.
* **For Accuracy:** **Increase the final SSBroyden stage from 10,000 to 20,000 iterations**, continuing from the current weights and reinitializing the Hessian for the extended final stage. Keep the PDE, source term, architecture, and embeddings unchanged.
* **For Details:** After the extended run, verify that **both** the loss and error have genuinely horizontal average bands over the last 50% of the final stage. Continue reporting the final-model error separately from the best-so-far minimum.
* **For Optimality:** None. Retain the periodic Fourier embedding \(\rightarrow\) MLP pipeline, vRBA, and SSBroyden. No entropy penalty is needed for this smooth Helmholtz problem without shocks.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | The code implements the 2D Helmholtz manufactured-solution PINN. Architecture: inputs (x,y) pass through separate periodic Fourier embeddings (degree=4, Lx=Ly=2.0) on each coordinate; only the concatenated embeddings enter an MLP body with 3 hidden layers of width 30 and identity output. The PDE residual is uxx + uyy + u - (1 - 17*pi^2)*sin(pi*x)*sin(4*pi*y), and the only loss is L_total = L_pde = mean(residual^2). No BC/IC or data loss is used; periodicity is enforced by the embeddings. Training uses Phase 1 Adam for 10,000 iterations with vRBA weighting and sampling, followed by Phase 2 SSBroyden for 10,000 iterations (100 outer iterations of maxiter=100) where vRBA is used for sampling only and the SSBroyden loss is unweighted. NaN/Inf checks are present. The analytical solution sin(pi*x)*sin(4*pi*y) is used for validation/error metrics. | 3.54e-06 | 35 | 35 | 5 | 15 | **90** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves 'Helmholtz_results.npz' via np.savez with keys: 'x' (10000x1 float64), 'y' (10000x1 float64), 'u_pred' (10000x1 float64), 'u_an' (10000x1 float64), 'rel_l2_error' (scalar float64), and 'loss_history' (655-element 1D float64 array for the default 10,000 Adam + 10,000 SSBroyden schedule). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run4/results//Helmholtz_Periodic_Sinpix_Sin4piy/2026_11_06_1122/` |


---
