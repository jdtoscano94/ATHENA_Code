
# Experiment Report
*Last Updated: 2026-11-06 04:08:57*

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
# Strategy: 2D Helmholtz (Periodic, Manufactured Solution)

## 0. Triage
Initial proposal. No user architecture requests, no Advisor report → follow defaults.

## 1. Data Source
None. No external dataset. Analytical solution is computed in-code for validation only.

## 2. Problem Statement
- **Governing Equation (2D Helmholtz, steady, no time):**
$$u_{xx} + u_{yy} + k^2 u = g(x,y), \quad k=1$$
- **PDE Residual:** $r(x,y) = u_{xx} + u_{yy} + 1\cdot u - g(x,y)$
- **Source term (manufactured, derived from** $u_{an}$**):**
$$g(x,y) = \big(1 - \pi^2(a_1^2+a_2^2)\big)\sin(\pi a_1 x)\sin(\pi a_2 y) = (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$$
- **BCs (Periodic, all four boundaries):**
  - $u(-1,y)=u(1,y)$, $\partial_x u(-1,y)=\partial_x u(1,y)$
  - $u(x,-1)=u(x,1)$, $\partial_y u(x,-1)=\partial_y u(x,1)$
- **Analytical Solution: YES.** $u_{an}(x,y)=\sin(\pi x)\sin(4\pi y)$, with $a_1=1$, $a_2=4$, $k=1$.
- **Data Loss Term:** No.
- **Problem Type:** Forward.
- **Well-posedness:** Yes. Strong solution exists; $k=1$ is non-resonant with the periodic Laplacian spectrum $\pi^2(m^2+n^2)$ → unique solution. Suitable for strong-form PINN.

**Data Role:** Condition B → **"Role: Validation/Error Metrics ONLY."**

## 2.1 Input
d = 2 (x, y). No time dimension.

## 2.2 Output
$u(x,y)$: scalar (q=1). Oscillatory: mode 1 in x, mode 4 in y; smooth; periodic.

## 2.3 Analytical Solution
$$u_{an}(x,y) = \sin(\pi x)\sin(4\pi y), \quad a_1=1,\ a_2=4,\ k=1$$

## 3. Base Model Selection
2D problem → Case A → **[MLP with Hybrid Adam and SSBroyden]**, width=30, depth=3.

## 4. Main Architecture
**Pipeline:** Input (x,y) → Periodic Fourier Embedding → MLP Body → Identity Output → u(x,y)

- **Normalization:** NONE. (x,y) already in $[-1,1]$ — pass raw coordinates unchanged. (Any transformation used must be differentiable and inside the call function.)
- **Embedding:** Periodic Fourier Features on BOTH x and y. **degree=2** (mandated start). **$L_x = 2.0$, $L_y = 2.0$** (range of coordinates entering embedding: $1-(-1)=2$).
- **Input Composition:** Main body receives ONLY the embeddings — no raw x, y.
- **Physics note:** Target contains harmonic k=4 in y; at degree=2 the MLP body must synthesize it nonlinearly. If the Advisor flags underfitting, escalate degree sequentially (→4).
- **Output Transformation:** Identity. No Dirichlet/IC to impose; periodicity is exact by construction.

## 5. Constraint Implementation
Soft-term count: $L_{pde}$ = 1 (periodic BC excluded — enforced by embedding; no IC; no data). Count < 5 → **Loss-Based, no hard constraints.**

## 6. Loss Function Assembly
$$L_{total} = L_{pde}$$

## 6.1 Continuation Strategy
N/A — no Reynolds/Peclet; $k=1$ fixed. No continuation.
- **Phase 1 (Adam):** ≥10,000 iterations, static $k=1$.
- **Phase 2 (SSBroyden):** No parameter transitions. Do NOT interfere with SSBroyden internals.

## 7. Weighting Strategy
- **Global:** 1 term → no balancing needed.
- **Local:** vRBA on the PDE residual. **"lambdas are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden."**
- **SSBroyden Integrity:** Do not modify internal weighting/sampling/batching.

## 8. Visualization
Do NOT modify template callbacks. Add a $u_{an}$ comparison panel only if absent.

## 9. Saving Data
Save as .npz with keys: `x`, `y`, `u_pred`, `u_an`. Report relative L2 error vs. $u_{an}$ history.

---
**Main Model:** [MLP with Hybrid Adam and SSBroyden]
**Enhancements:** [Periodic Fourier Features]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
REJECTED_FOR_REFINEMENT

## Section 1: Status Decision
* **Status:** **REJECTED_FOR_REFINEMENT**
* **Problem type:** Type A — forward, noise-free Helmholtz problem. The manufactured solution is used only for validation, so there is no data leakage.
* **Decision rationale:** The run is stable and highly accurate, but the final SSBroyden stage is still improving. It fails the mandatory loss/error plateau test and should not be approved as converged.

## Section 2: Quantitative Scorecard
* **Score Integrity:** **35/35** — completed successfully; readable plots and finite values.
* **Score Accuracy:** **35/35**
  * **Numerical Precision:** 20/20. No tolerance was explicitly supplied; using \(\epsilon_{\text{target}}=10^{-3}\), the plotted final error of approximately \(10^{-6}\) easily qualifies.
  * **Consistency:** 15/15. The implementation solves the requested PDE with \(a_1=1\), \(a_2=4\), \(k=1\), the correct manufactured source, and periodic boundaries.
* **Score Details:** **7/15** — boundaries are enforced and the final stage is stable, but the loss and error have not established a genuine terminal plateau.
* **Score Optimality:** **14/15** — appropriate periodic Fourier embedding, MLP, vRBA, and SSBroyden fine-tuning. No entropy penalty is needed because this problem has no shocks.
* **TOTAL SCORE:** **91/100**

Although the score exceeds 90, the mandatory plateau gate fails, so the run is rejected for refinement.

## Section 3: Forensic Analysis & Feedback

### Observation
**Loss and error histories**

* The loss plot has visible ticks from approximately \(10^{1}\) down to \(10^{-9}\). It shows two distinct stages:
  * **Adam, iterations 0–10,000:** rapid initial decrease, followed by slow improvement around \(10^{-3}\), with small spikes.
  * **Final SSBroyden stage, after iteration 10,000:** a transition jump followed by a smooth, sustained decrease over several orders of magnitude.
* Reading the post-10,000 final region:
  * **Loss at the final-stage midpoint:** \(A_{\text{loss}}\approx 10^{-8}\).
  * **Loss at the right edge:** \(B_{\text{loss}}\approx 10^{-9}\).
  * Thus \(B_{\text{loss}}<A_{\text{loss}}\) by roughly an order of magnitude. The loss is **still descending**, not flat.
* The relative-\(L^2\) plot falls from \(O(1)\) to a noisy Adam band around \(10^{-2}\)–\(10^{-3}\), then drops sharply during SSBroyden.
  * **Error band center at the final-stage midpoint:** \(A_{\text{error}}\approx 3\times10^{-6}\).
  * **Error band center at the end:** \(B_{\text{error}}\approx 8\times10^{-7}\).
  * This is also a downward shift in the oscillation band, although the error is noisier than the loss.
* The rightmost plotted error is approximately \(10^{-6}\); this is the value used for scoring. The dual-curve test fails, and the still-descending loss alone is sufficient to establish a premature stop.

**Residual and pointwise-error fields**

* The PDE residual is spatially speckled and broadly distributed, without a concentrated boundary failure or shock-like structure. Its logged range is \(1.1650058695522603\times10^{-10}\) to \(3.8237737328472576\times10^{-5}\).
* The largest pointwise errors form diffuse orange/red interior patches, particularly in left/central regions, rather than a systematic periodic-boundary error.

**Solution fields**

* Prediction and analytical solution are visually very similar: both reproduce the single \(x\)-mode and the higher-frequency horizontal band structure associated with \(a_2=4\).
* No conspicuous phase shift or missing oscillation is visible. Periodic boundaries are enforced by construction through the Fourier embedding.

**Log confirmation**

* The exact requested minimum is:
  **`Minimum achieved RL2 error: 2.147192372070571e-07`**
* This is the best error ever reached, not the error at the end of the plotted curve.
* The repeated `RL2 error: 0.00221988` entries are consistent with the Adam warm-up reporting and are not used as the final SSBroyden scoring value. The plot-based final value remains approximately \(10^{-6}\).

### Diagnosis
**Budget-Limited / Premature Stop.** The formulation is stable and accurate, but training ended while the final-stage loss—and, more weakly, the error band—was still decreasing. This is not instability, spectral stagnation, or evidence that a larger/different network is required.

### Targeted Improvements
* **For Integrity:** None.
* **For Accuracy:** **Increase only the final SSBroyden stage’s iteration budget by at least \(2\times\), preferably \(2\)–\(5\times\).** If continuing from saved weights, reinitialize the Hessian approximation. Keep the PDE, \(k=1\), Fourier degree, MLP, and vRBA settings unchanged. Stop only after both the loss and error band centers remain approximately horizontal over the last 50% of the final stage.
* **For Details:** None beyond requiring the terminal dual-curve plateau.
* **For Optimality:** None. Do not switch to KAN, increase the embedding degree, or alter the physics merely to address this iteration-budget limitation.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Implements a PINN for the 2D Helmholtz equation using an MLP body with 3 hidden layers of width 30 and Periodic Fourier Features of degree 2 on both input coordinates (Lx=Ly=2.0). The raw coordinates are passed without external normalization; only differentiable Fourier embeddings inside the model call are used, and the MLP receives only the concatenated embeddings. The PDE residual is exactly u_xx + u_yy + u - g with g=(1-17π^2)sin(πx)sin(4πy), and the analytical solution u_an=sin(πx)sin(4πy) is used only for validation and error metrics. The loss assembly is a single PDE loss with no data, BC, or IC losses; periodic boundaries are enforced by the Fourier embedding. Training uses Adam for 10,000 epochs with vRBA weighting and importance sampling on the PDE residual, followed by a hybrid SSBroyden phase for 12,500 effective iterations in outer blocks of Nchange=100; during SSBroyden, vRBA lambdas are updated and used for sampling only, while the minimized PDE loss remains unweighted. | 1.00e-06 | 35 | 35 | 7 | 14 | **91** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves Helmholtz_results.npz with the required keys: 'x' (100x100), 'y' (100x100), 'u_pred' (100x100), and 'u_an' (100x100). It additionally stores 'all_its' (1D iteration history) and 'all_errors' (1D relative L2 error history). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run2//Helmholtz_Periodic_Sin_pi_a1_Sin_pi_a2/2026_11_06_0403/` |


---
