
# Experiment Report
*Last Updated: 2026-11-06 04:23:28*

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
# Strategy: Run 1 — Extended SSBroyden Budget (Advisor-Mandated Refinement)

## 0. Triage
**Path 2: Advisor Recommendation (MANDATORY).** Diagnosis: *Budget-Limited / Premature Stop*. Cure: **Keep ALL Run-0 settings unchanged (PDE, k=1, Fourier degree=2, MLP, vRBA); increase ONLY the final SSBroyden budget by 2–5×; reinitialize Hessian if resuming from checkpoint; stop only on a dual-curve (loss + error) plateau over the last 50% of the final stage.** No KAN, no degree increase, no physics changes.

## 1. Data Source
No new data. Analytical solution used for validation only.

## 2. Problem Statement
- **Governing Equation (Helmholtz, k=1):**
$$u_{xx} + u_{yy} + u = g(x,y), \quad (x,y)\in[-1,1]\times[-1,1]$$
- **PDE Residual:** $r = u_{xx} + u_{yy} + u - g(x,y)$, with manufactured source:
$$g(x,y) = (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$$
- **BCs:** Periodic in x and y: $u(-1,y)=u(1,y)$, $u(x,-1)=u(x,1)$ (and derivatives). **Enforced by architecture — no BC loss.**
- **Analytical Solution:** $u_{an} = \sin(\pi x)\sin(4\pi y)$ (a₁=1, a₂=4, k=1).
- **Data Loss Term:** None. **Role: Validation/Error Metrics ONLY.**
- **Problem Type:** Forward.
- **Well-posedness:** Yes — smooth strong-form solution; ideal for PINNs.

## 2.1 / 2.2. Input & Output
- Input: 2D (x, y), no time. Output: scalar u(x,y). Smooth, oscillatory (4 periods in y), exactly periodic.

## 3. Base Model Selection
Low-dim (2D) → **[MLP with Hybrid Adam and SSBroyden]**, width=30, depth=3. **Unchanged from Run 0.**

## 4. Main Architecture
`Input (x,y) → Periodic Fourier Embedding → MLP Body → Output u`

- **Normalization:** x,y ∈ [-1,1] already in nice range → **skip normalization (identity), pass raw coordinates unchanged.**
- **Input Transformation:** Periodic Fourier Features, **degree=2 on BOTH x and y** (unchanged). **Lx = Ly = 2.0** (range of coordinate entering embedding = 2). All differentiable, inside the model call only.
- **Input Composition:** MLP body receives **ONLY the concatenated embeddings** — no raw x, y.
- **Output Transformation:** None (no Dirichlet; periodicity exact by construction). Direct scalar output.

## 5. Constraint Strategy
Loss term count: L_pde = 1 (periodic BC removed — satisfied by embedding). Count < 5 → **Loss-Based, no hard constraints.**

## 6. Loss Assembly
$$L_{total} = L_{pde}$$

## 6.1 Continuation
Not applicable (no Re/Peclet; k=1 fixed).

## Training Schedule (ONLY change from Run 0)
- **Phase 1 — Adam:** 10,000 iterations (unchanged). vRBA weighting + sampling active.
- **Phase 2 — SSBroyden:** **Extend from 12,500 → 50,000 effective iterations (4×, within advisor's 2–5× range)**, outer blocks Nchange=100.
  - **If resuming from Run-0 checkpoint: REINITIALIZE the SSBroyden Hessian approximation.**
  - **Plateau gate (mandatory):** terminate only when both loss and RL2-error band centers are approximately horizontal over the last 50% of the final stage. Do not stop on a still-descending loss.

## 7. Weighting Strategy
- **Local:** vRBA (`lambdas`) — Weighting & Sampling during Adam, **Sampling ONLY during SSBroyden**.
- **Global:** 1 term → no balancing needed.
- **SSBroyden Integrity:** Do NOT modify internal SSBroyden weighting/sampling/batching.

## 8. Visualization
Do NOT modify template visualization callbacks.

## 9. Saving
Save u_pred, u_an, and residual fields as .npz arrays with keys specifying variable names.

---
**Main Model:** [MLP with Hybrid Adam and SSBrodyden]
**Enhancements:** [Periodic Fourier Features (degree=2, Lx=Ly=2.0)]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
REJECTED_FOR_REFINEMENT

## Section 1: Status Decision
* **Status:** **REJECTED_FOR_REFINEMENT**
* **Reason:** The run is stable and highly accurate, but the mandatory dual-curve plateau test fails. Both the loss and relative-error band centers remain lower at the end of the final stage than at its midpoint. The excellent score therefore does **not** justify approval.

## Section 2: Quantitative Scorecard
* **Score Integrity:** 35/35
* **Score Accuracy:** 35/35
* **Score Details:** 7/15
* **Score Optimality:** 14/15
* **TOTAL SCORE:** **91/100**

The final plotted RL2 error is approximately **\(5\times10^{-8}\)**, within a noisy final band of roughly \(10^{-8}\)–\(10^{-7}\). Using \(\epsilon_{\text{target}}=10^{-6}\), consistent with the previous run’s accuracy scoring, this earns full numerical-precision credit. The implementation solves the requested forward problem with the correct source, \(k=1\), periodic architecture, and no analytical-data leakage.

## Section 3: Forensic Analysis & Feedback

### Observation
This is a **forward, noise-free Helmholtz problem**. The analytical solution is used only for validation. The plots are readable, the fields are finite, and there is no visual explosion or NaN failure.

**Loss and error histories:**
- The loss spans approximately \(10^{1}\) down to the \(10^{-11}\)–\(10^{-12}\) region. The Adam portion, up to about iteration 10,000, is jagged and decreases slowly. A sharp transition then marks the SSBroyden stage, followed by a long, mostly smooth descent and small late-time spikes.
- The RL2 curve begins near order unity, remains noisy during Adam, drops sharply near iteration 10,000, and then continues downward with substantial small-scale oscillations.
- The final SSBroyden stage runs approximately from iteration 10,000 to 42,000, putting its midpoint near **26,000**. Reading oscillation-band centers rather than lucky dips:

| Curve | A: midpoint band center | B: end band center | Conclusion |
|---|---:|---:|---|
| Total loss | \(\approx 10^{-10}\) | \(\approx 10^{-12}\) | Still decreasing |
| Relative \(L^2\) error | \(\approx 3\times10^{-7}\) | \(\approx 5\times10^{-8}\) | Still decreasing on average |

Both decreases exceed the allowed approximately \(2\times\) plateau tolerance. Consequently, **neither curve qualifies as flat under the prescribed midpoint-to-end test**, and the dual-curve convergence gate fails. The visually flatter last few thousand iterations do not override the decrease across the full final-half window.

**Residual and solution fields:**
- The PDE-residual heatmap is broadly distributed and speckled, without a localized shock or coherent failure front. The log confirms residuals from **\(6.7697\times10^{-12}\)** to **\(2.42145\times10^{-6}\)**.
- Prediction and analytical solution are visually very close: both reproduce the same periodic alternating stripe pattern, phase, and amplitude.
- Pointwise error is largest in broad orange/red regions toward the outer portions of the domain, especially near the top and bottom. The smallest errors form two blue/green lobes near the center. This does not indicate a shock or an obvious periodic-boundary contradiction.

**Log reconciliation:**
- The required bottom-line entry is exactly:
  **`Minimum achieved RL2 error: 8.96072439052614e-09`**.
- That is the **best-ever dip**, not the final sustained error.
- The repeated `RL2 error: 0.00196387` phase-summary printout is not consistent with the right edge of the SSBroyden error curve and is not used for final scoring.
- Following the plot-first rule, scoring uses the final plotted value, approximately **\(5\times10^{-8}\)**, rather than either log number.

### Diagnosis
**Budget-Limited / Premature Stop — dual-curve plateau not achieved.**

This is not divergence, stagnation, spectral bias, or a physics-formulation failure. The model is still improving on average, but training ends before both curves establish horizontal floors. The very low minimum error should not be mistaken for sustained convergence.

### Targeted Improvements
* **For Integrity:** None required for execution. For clearer auditing, label Adam and SSBroyden metric printouts explicitly and print the **final-iteration RL2 error** alongside the minimum achieved value.
* **For Accuracy:** **Duplicate the final SSBroyden stage’s iteration budget**—use a \(2\times\) extension initially, with up to \(2\)–\(5\times\) if necessary. Continue the same final stage and reinitialize its Hessian approximation if resuming requires a restart. Keep the PDE, \(k=1\), MLP, Fourier degree 2, and vRBA settings unchanged. Stop only after both band centers are genuinely horizontal over the last 50% of that stage.
* **For Details:** Require a sustained dual-curve plateau rather than accepting a best dip or an apparently flat short tail. No boundary-condition reformulation is indicated; periodicity is already enforced by construction.
* **For Optimality:** Retain `Input → Periodic Fourier Embeddings → MLP → Output`, vRBA, and low-dimensional SSBroyden fine-tuning. Do not remove the periodic embedding or feed raw coordinates directly to the MLP. No KAN, degree increase, Reynolds continuation, or entropy penalty is warranted for this smooth, non-shock problem.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Implements a PINN for the 2D Helmholtz equation using an MLP body with 3 hidden layers of width 30 and Periodic Fourier Features of degree 2 on both input coordinates (Lx=Ly=2.0). The raw coordinates are passed without external normalization; only differentiable Fourier embeddings inside the model call are used, and the MLP receives only the concatenated embeddings. The PDE residual is exactly u_xx + u_yy + u - g with g=(1-17π^2)sin(πx)sin(4πy), and the analytical solution u_an=sin(πx)sin(4πy) is used only for validation and error metrics. The loss assembly is a single PDE loss with no data, BC, or IC losses; periodic boundaries are enforced by the Fourier embedding. Training uses Adam for 10,000 epochs with vRBA weighting and importance sampling on the PDE residual, followed by a hybrid SSBroyden phase for 12,500 effective iterations in outer blocks of Nchange=100; during SSBroyden, vRBA lambdas are updated and used for sampling only, while the minimized PDE loss remains unweighted. | 1.00e-06 | 35 | 35 | 7 | 14 | **91** |
| 2 | The implementation matches the strategy. It defines a Helmholtz PINN for u_xx + u_yy + u = g(x,y) with k=1 and manufactured source g=(1-17π²)sin(πx)sin(4πy). The architecture is an MLP with width 30 and depth 3, using Periodic Fourier Features with degree=2 and Lx=Ly=2.0 on both input coordinates. Only the concatenated Fourier embeddings enter the MLP, and the raw x,y are not passed. Periodicity is enforced by the embedding, so no BC loss is added. The total loss is only the PDE residual loss. During Adam, vRBA weighting and sampling are active; during SSBroyden, vRBA is used for sampling only and the loss remains unweighted PDE residual. Training uses 10,000 Adam iterations followed by up to 50,000 effective SSBroyden iterations in outer blocks of Nchange=100, with a dual-curve plateau gate over loss and relative L2 error. Analytical solution is used only for validation/error metrics, not as a data loss. Saved outputs include prediction, analytical solution, and residual fields. | 5.00e-08 | 35 | 35 | 7 | 14 | **91** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves Helmholtz_results.npz with the required keys: 'x' (100x100), 'y' (100x100), 'u_pred' (100x100), and 'u_an' (100x100). It additionally stores 'all_its' (1D iteration history) and 'all_errors' (1D relative L2 error history). |
| 2 | Saves the file 'Helmholtz_results.npz' in NPZ format. Keys and shapes for default NC_TEST=100 are: 'x' (100x100), 'y' (100x100), 'u_pred' (100x100), 'u_an' (100x100), 'residual' (100x100, absolute PDE residual), 'all_its' (1D variable-length history array), and 'all_errors' (1D variable-length history array). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run2//Helmholtz_Periodic_Sin_pi_a1_Sin_pi_a2/2026_11_06_0403/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run2//Helmholtz_Periodic_Sin_pi_a1_Sin_pi_a2/2026_11_06_0417/` |


---
