
# Experiment Report
*Last Updated: 2026-11-06 04:39:50*

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
# Strategy: 2D Helmholtz (Refinement Run — Budget Extension)

## 0. Triage
**Path 2: Advisor Recommendation (MANDATORY).** Diagnosis: *Budget-Limited / Premature Stop* — dual-curve plateau not achieved. Cure: **Keep the entire architecture unchanged; extend the final SSBroyden stage budget 2× (up to 5× if needed) with a strict dual-curve plateau gate.** No KAN, no degree increase, no reformulation.

## 1. Data Source
No new data. Analytical solution used for **validation only**.

## 2. Problem Statement
- **Governing Equation (Forward):** $u_{xx} + u_{yy} + k^2 u = g(x,y)$, with $k=1$.
- **PDE Residual:** $r = u_{xx} + u_{yy} + u - g(x,y)$, where the manufactured source is
$$g(x,y) = (1 - 17\pi^2)\,\sin(\pi x)\sin(4\pi y)$$
(since $a_1=1, a_2=4$: $u_{xx}+u_{yy} = -\pi^2(1+16)\,u_{an}$).
- **BCs:** Periodic in $x$ and $y$ on $[-1,1]\times[-1,1]$ — **enforced by construction** (periodic Fourier embeddings), no BC loss.
- **Analytical Solution:** $u_{an}(x,y) = \sin(\pi x)\sin(4\pi y)$.
- **Data Role:** Validation/Error Metrics ONLY (no $L_{data}$).
- **Problem Type:** Forward.
- **Well-posedness:** Yes. Smooth strong solution; no resonance ($k^2=1 \neq 17\pi^2$). Ideal for PINNs.

## 2.1 / 2.2 Input & Output
- **Input:** 2D $(x,y)$, both already in $[-1,1]$ → **no normalization** (pass raw coordinates; any scaler is forbidden).
- **Output:** scalar $u(x,y)$. Smooth, periodic, moderately oscillatory in $y$ ($a_2=4$).

## 3. Base Model Selection
**MLP with Hybrid Adam and SSBroyden** (2D, low-dim): width=30, depth=3.

## 4. Main Architecture
`Input (raw x,y) → Periodic Fourier Embedding (degree=2, Lx=2.0 for x; degree=2, Ly=2.0 for y) → concat → MLP(30×3) → u_pred`

- All embeddings differentiable, **inside the model call only**.
- MLP receives **ONLY** the concatenated embeddings (no raw coordinates).
- $L_x = L_y = 2.0$ (range of raw coordinates entering the embedding — consistent, no normalization).
- No output transformation (no Dirichlet/IC; periodicity is architectural).

## 5. Constraint Strategy
Periodic BCs satisfied by architecture → **no BC loss terms**. Count = 1 ($L_{pde}$ only) < 5 → loss-based, no hard constraints.

## 6. Loss Assembly
$$L_{total} = L_{pde}$$

## 6.1 Continuation
N/A (no Reynolds/Peclet).

## 7. Training Schedule (Per Advisor — THE KEY CHANGE)
- **Phase 1 — Adam (STATIC):** 10,000 iterations minimum. vRBA `lambdas` for **Weighting & Sampling**.
- **Phase 2 — SSBroyden (EXTENDED):** **2× the previous budget** → 100,000 effective iterations (outer blocks Nchange=100), with authorization to extend **up to 5× (250,000)** if the plateau gate is not met. vRBA for **Sampling ONLY**; unweighted PDE loss.
- **Dual-Curve Plateau Gate (MANDATORY STOP CRITERION):** Both total-loss AND RL2-error **band centers** must be horizontal over the **last 50% of the final stage** (midpoint-to-end decrease within ~2× tolerance). Best-ever dips or short flat tails do NOT qualify.
- **Hessian:** Reinitialize the SSBroyden Hessian approximation at the start of the phase (fresh run) — do not recycle across any restart/extension.
- **Integrity/logging:** Explicitly label Adam vs. SSBroyden metric printouts; print the **final-iteration RL2 error** alongside the minimum achieved RL2 in the bottom-line log.
- **Weighting:** Single loss term → no global balancing. Do NOT modify SSBroyden internal weighting/sampling/batching.

## 8. Visualization
Keep template visualization callbacks **unchanged**.

## 9. Saving
Save fields as `.npz`: `u_pred`, `u_analytical`, `residual` (plus loss/RL2 histories for the plateau audit).

---
**Main Model:** [MLP with Hybrid Adam and SSBroyden]
**Enhancements:** [Periodic Fourier Features (degree=2, Lx=Ly=2.0)]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
APPROVED

## Section 1: Status Decision
* **Status:** **APPROVED**

This is a noiseless **Type A forward problem**. The manufactured analytical solution is used only for validation, not training. The budget extension has resolved the previous premature-stop behavior: both curves now fluctuate around stationary final-stage floors rather than continuing to descend.

The current run passes independently with a score above 90 and a valid dual-curve plateau. For completeness, Run 0 was the first historical run scoring at least 90; only Run 2 and the current run follow it in the supplied history. Therefore, this approval is **not** based on the separate “five exploration runs completed” stopping clause.

## Section 2: Quantitative Scorecard
* **Score Integrity:** **35/35**
* **Score Accuracy:** **35/35**
  * Numerical Precision: **20/20** — using the established practical target \(\epsilon_{\text{target}}=10^{-6}\), the final RL2 error of \(1.70105769\times10^{-8}\) is comfortably below target.
  * Consistency: **15/15** — correct Helmholtz operator, source term, \(k=1\), manufactured solution, domain, and periodic topology. No data-loss leakage.
* **Score Details:** **12/15** — periodic boundaries are enforced architecturally and both curves are flat on average, but the final floors remain visibly noisy.
* **Score Optimality:** **14/15** — periodic Fourier embeddings, vRBA, low-dimensional SSBroyden fine-tuning, fresh Hessian, and appropriate raw dimensionless coordinates are used. No shock entropy term is needed for this smooth elliptic problem.
* **TOTAL SCORE:** **96/100**

## Section 3: Forensic Analysis & Feedback

### Observation

**1. Loss and error histories**

- The **Total Loss** axis spans approximately \(10^1\) at the beginning to below the labeled \(10^{-11}\) region late in training.
- The **Relative \(L^2\) Error** begins around several \(10^{-1}\), passes through the Adam handoff error of approximately \(2.22\times10^{-3}\), and ultimately reaches a noisy band around a few \(10^{-8}\), with isolated dips toward \(10^{-9}\)–\(10^{-8}\).
- Both plots show the expected two-stage structure:
  1. Adam and the transition near 10,000 iterations.
  2. A rapid SSBroyden decrease until roughly 30,000–40,000 iterations, followed by a stationary oscillation band.
- There are no NaNs, divergent spikes, or unrecoverable jumps.

**Mandatory final-stage band-center comparison:**  
The final SSBroyden stage runs from approximately 10,000 to 100,000 plotted iterations, placing its midpoint near 55,000.

| Curve | \(A\): average band at final-stage midpoint | \(B\): average band at final-stage end | Conclusion |
|---|---:|---:|---|
| Total Loss | \(\approx 2.0\times10^{-11}\) | \(\approx 1.5\times10^{-11}\) | Same floor; only about a 25% band-center change |
| Relative \(L^2\) Error | \(\approx 2.5\times10^{-8}\) | \(\approx 2.0\times10^{-8}\) | Same floor; only about a 20% band-center change |

These are band-center estimates, not lucky minimum dips. Across the last 50% of the final stage, neither curve’s first-half versus second-half average band drops by anywhere near a factor of two. Thus, **both curves satisfy the plateau test**. The residual fluctuations are noise around a fixed floor, not continued average descent.

**2. Residual and pointwise-error fields**

- The PDE residual is spatially mottled and broadly distributed, without a coherent shock front, boundary layer, or dominant corner failure.
- The log confirms residuals ranging from \(7.007283642224138\times10^{-12}\) to \(2.068265885668552\times10^{-6}\).
- Pointwise error is very small on its logarithmic scale. It is spatially nonuniform, with noticeable small structures near the top and bottom around \(x\approx-0.5\), but there is no visible large periodic-boundary seam or systematic amplitude failure.

**3. Analytical solution versus prediction**

- The prediction reproduces the analytical field’s alternating horizontal bands, left/right sign structure, phase, and amplitude.
- No visually meaningful phase shift, blurry mode, missing oscillation, or boundary mismatch is apparent.

**4. Log confirmation and scoring error**

Because SSBroyden was used, the Adam “Solution” metrics are only warm-up/handoff values and are not treated as final results.

- **Exact minimum achieved RL2 error:**  
  `4.521796461541044e-09`
- **Final-iteration SSBroyden RL2 error:**  
  `1.70105769e-08`

The minimum is the best transient dip, but **scoring uses the final/rightmost value, approximately \(1.70\times10^{-8}\)**, consistent with the plot. The logged plateau-gate changes—\(-0.1214\) log10 decades for loss and \(-0.0978\) for RL2—also confirm the small, sub-factor-of-two changes visible in the plot; they are not being used as a substitute for the visual plateau.

### Diagnosis
**Successful budget extension; previous Budget-Limited/Premature Stop is resolved.** The run has reached a stable, noisy dual-curve floor with excellent final accuracy. There is no evidence of instability, spectral-bias failure, periodic-boundary violation, or continuing final-stage descent.

### Targeted Improvements
* **For Integrity:** None. Execution, plots, and reported values are valid and finite.
* **For Accuracy:** None. Retain the final model; additional iterations or architecture changes are not required by the curves.
* **For Details:** None required. Preserve the final-iterate checkpoint and histories. The remaining oscillations are stationary noise rather than a convergence defect.
* **For Optimality:** Keep the current periodic-Fourier \(\rightarrow\) MLP \(\rightarrow\) SSBroyden pipeline and vRBA strategy. Do not remove the periodic embeddings or switch to KAN merely to chase the earlier best dip. Entropy regularization and conservative shock treatment are not applicable to this smooth Helmholtz problem.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Implements a PINN for the 2D Helmholtz equation using an MLP body with 3 hidden layers of width 30 and Periodic Fourier Features of degree 2 on both input coordinates (Lx=Ly=2.0). The raw coordinates are passed without external normalization; only differentiable Fourier embeddings inside the model call are used, and the MLP receives only the concatenated embeddings. The PDE residual is exactly u_xx + u_yy + u - g with g=(1-17π^2)sin(πx)sin(4πy), and the analytical solution u_an=sin(πx)sin(4πy) is used only for validation and error metrics. The loss assembly is a single PDE loss with no data, BC, or IC losses; periodic boundaries are enforced by the Fourier embedding. Training uses Adam for 10,000 epochs with vRBA weighting and importance sampling on the PDE residual, followed by a hybrid SSBroyden phase for 12,500 effective iterations in outer blocks of Nchange=100; during SSBroyden, vRBA lambdas are updated and used for sampling only, while the minimized PDE loss remains unweighted. | 1.00e-06 | 35 | 35 | 7 | 14 | **91** |
| 2 | The implementation matches the strategy. It defines a Helmholtz PINN for u_xx + u_yy + u = g(x,y) with k=1 and manufactured source g=(1-17π²)sin(πx)sin(4πy). The architecture is an MLP with width 30 and depth 3, using Periodic Fourier Features with degree=2 and Lx=Ly=2.0 on both input coordinates. Only the concatenated Fourier embeddings enter the MLP, and the raw x,y are not passed. Periodicity is enforced by the embedding, so no BC loss is added. The total loss is only the PDE residual loss. During Adam, vRBA weighting and sampling are active; during SSBroyden, vRBA is used for sampling only and the loss remains unweighted PDE residual. Training uses 10,000 Adam iterations followed by up to 50,000 effective SSBroyden iterations in outer blocks of Nchange=100, with a dual-curve plateau gate over loss and relative L2 error. Analytical solution is used only for validation/error metrics, not as a data loss. Saved outputs include prediction, analytical solution, and residual fields. | 5.00e-08 | 35 | 35 | 7 | 14 | **91** |
| 3 | The implementation matches the strategy. It uses a PINN with separate periodic Fourier feature embeddings for the two raw inputs, degree=2 and Lx=Ly=2.0, concatenated before an MLP of width 30 and three hidden layers with tanh activation; no external normalization is applied. The PDE residual is implemented as u_xx + u_yy + u - g with g = (1 - 17*pi^2)*sin(pi*x)*sin(4*pi*y), and the analytical solution sin(pi*x)*sin(4*pi*y) is used only for validation/error metrics. The loss is exactly L_pde with no BC, IC, or data terms. Adam runs for 10,000 iterations with vRBA weighting and importance sampling. SSBroyden then runs with a fresh Hessian, Nchange=100, an initial budget of 100,000 effective iterations, and an authorized cap of 250,000 if the dual-curve plateau gate is not met; the SSBroyden PDE loss is unweighted while vRBA is used for sampling only. The plateau gate compares log10 band centers of both total loss and RL2 error over the last 50% of the SSBroyden stage with a factor-of-two tolerance. Logging explicitly labels Adam and SSBroyden phases and reports final-iteration RL2 alongside minimum RL2. | 1.70e-08 | 35 | 35 | 12 | 14 | **96** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves Helmholtz_results.npz with the required keys: 'x' (100x100), 'y' (100x100), 'u_pred' (100x100), and 'u_an' (100x100). It additionally stores 'all_its' (1D iteration history) and 'all_errors' (1D relative L2 error history). |
| 2 | Saves the file 'Helmholtz_results.npz' in NPZ format. Keys and shapes for default NC_TEST=100 are: 'x' (100x100), 'y' (100x100), 'u_pred' (100x100), 'u_an' (100x100), 'residual' (100x100, absolute PDE residual), 'all_its' (1D variable-length history array), and 'all_errors' (1D variable-length history array). |
| 3 | Saves 'Helmholtz_results.npz' in NPZ format. Keys and shapes: 'x' (100x100), 'y' (100x100), 'u_pred' (100x100), 'u_analytical' (100x100), 'residual' (100x100, saved as absolute PDE residual magnitude on the same validation grid), 'all_its' (1D variable-length), 'all_loss' (1D variable-length), 'all_errors' (1D variable-length), 'ssbroyden_loss_history' (1D variable-length), and 'ssbroyden_rl2_history' (1D variable-length). The 100x100 fields come from the NC_TEST=100 validation mesh; all_its/all_loss/all_errors are combined Adam+SSBroyden logs, while ssbroyden_loss_history and ssbroyden_rl2_history are the SSBroyden-only plateau-audit histories. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run2//Helmholtz_Periodic_Sin_pi_a1_Sin_pi_a2/2026_11_06_0403/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run2//Helmholtz_Periodic_Sin_pi_a1_Sin_pi_a2/2026_11_06_0417/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run2//Helmholtz_Periodic_Sin_pi_a1_Sin_pi_a2/2026_11_06_0432/` |


---
