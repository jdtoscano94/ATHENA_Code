
# Experiment Report
*Last Updated: 2026-11-06 12:18:42*

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
# Strategy: 2D Helmholtz (Forward, Periodic)

## 0. Triage
No specific user requests. No Advisor report. Initial strategy from physics.

## 1. Data Source
No external data. Analytical solution is computed from the closed-form formula for validation.

**Data Role: Validation/Error Metrics ONLY** (Forward problem).

## 2. Problem Statement
- **Governing Equation:** $u_{xx} + u_{yy} + k^2 u = g(x,y)$, with $k=1$.
- **PDE Residual:** $r(x,y) = u_{xx} + u_{yy} + k^2 u - g(x,y)$
- **Source term (derived from analytical solution):** $g(x,y) = (k^2 - \pi^2(a_1^2 + a_2^2))\sin(\pi a_1 x)\sin(\pi a_2 y) = (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$
- **BCs:** Periodic in both $x$ and $y$ on $[-1,1]\times[-1,1]$. No IC (steady problem).
- **Analytical Solution:** $u_{an}(x,y) = \sin(\pi x)\sin(4\pi y)$
- **Problem Type:** Forward.
- **Well-posedness:** Yes. Smooth strong solution exists; $k^2=1$ is not a resonant eigenvalue ($k^2 \neq \pi^2(m^2+n^2)$). Ideal for PINNs.

## 2.1 / 2.2 Input & Output
- **Input:** 2D spatial $(x,y)$, both already in $[-1,1]$.
- **Output:** Scalar $u(x,y)$. Smooth, oscillatory ($\sin(4\pi y)$ = 4 periods in $y$), periodic in both coordinates.

## 3. Base Model Selection
2D Forward → **MLP with Hybrid Adam and SSBroyden**. Base MLP: `width=30, depth=3`.

## 4. Main Architecture
**Pipeline:** Input $(x,y)$ → Input Transformation → MLP Body → Output $u(x,y)$ (direct, no output transformation).

**Input Transformation:**
- **Normalization: NONE.** $x,y \in [-1,1]$ already — pass raw coordinates through unchanged. Any normalization must be differentiable and inside the call function (here: identity).
- **Embedding: Periodic Fourier Features on BOTH $x$ and $y$** (periodic BCs).
  - **Degree = 4** (justified by physics: solution requires harmonics $k=1$ in $x$ and $k=4$ in $y$; degree=2 cannot represent $\sin(4\pi y)$). Within stability limit (<20).
  - **$L_x = 2.0$, $L_y = 2.0$** (range of raw coordinates entering the embedding).
- **Input Composition:** Main body receives **ONLY the embeddings** — no bare $x$ or $y$ passed to the MLP.

**Output Transformation:** None required. Periodicity is exact by construction; no Dirichlet/IC constraints exist.

## 5. Constraint Implementation
- Periodic BCs: **enforced by architecture** (Periodic Fourier Features). No $L_{bc}$ term.
- Soft loss term count: **1** ($L_{pde}$ only). No hard constraints needed.

## 6. Loss Function
$$L_{total} = L_{pde}$$

## 6.1 Continuation
Not applicable (no Reynolds/Peclet). Fixed parameters: $k=1$.

**Iteration Budget (fixed, finite):**
- Phase 1 (Adam): **10,000 iterations** minimum, static parameters.
- Phase 2 (SSBroyden): **10,000 iterations** fixed budget.
- Stop early only on NaN/Inf loss.

## 7. Weighting Strategy
- **Local:** vRBA on PDE residual. "`lambdas` are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden."
- **Global:** Total terms = 1 → **no balancing needed**.
- **SSBroyden Integrity:** Do NOT modify internal weighting/sampling/batching of the SSBroyden template.

## 8. Visualization
Do NOT modify template visualization callbacks. Ensure plots show $u_{pred}$ vs $u_{an}$ fields and pointwise error.

## 9. Saving Data
Save as `.npz` with keys: `x`, `y`, `u_pred`, `u_analytical`, plus relative L2 error history.

---

**Main Model:**
[MLP with Hybrid Adam and SSBroyden]

**Enhancements:**
[Periodic Fourier Features (degree=4, Lx=Ly=2.0, applied to both x and y)]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
REJECTED_FOR_REFINEMENT

## Section 1: Status Decision
* **Status:** **REJECTED_FOR_REFINEMENT**
* **Problem type:** Type A — forward, noise-free Helmholtz problem with a manufactured solution used only for validation.
* **Reason:** The run is stable and highly accurate, but both the loss and error continue decreasing during the final SSBroyden stage. It therefore fails the mandatory dual-curve plateau check despite its high numerical score.

## Section 2: Quantitative Scorecard
* **Score Integrity:** 35/35
* **Score Accuracy:** 35/35
  * Numerical Precision: 20/20. No tolerance was explicitly supplied; using \(\epsilon_{\text{target}}=10^{-2}\), the final plotted error of order \(10^{-6}\) easily qualifies.
  * Consistency: 15/15. The correct source term, parameters, periodic geometry, and analytical solution were used. There is no data-loss leakage.
* **Score Details:** 7/15
* **Score Optimality:** 14/15
* **TOTAL SCORE:** **91/100**

The score exceeds 90, but approval is blocked because the final-stage curves are still descending rather than plateauing.

## Section 3: Forensic Analysis & Feedback

### Observation
* **Loss history:** The plot is readable and finite, spanning approximately \(10^0\) down to nearly \(10^{-8}\). During Adam, the loss drops rapidly and then varies around a slow, shallow descent. A sharp transition feature occurs near iteration 10,000. Afterward, SSBroyden produces a strong, sustained descent without divergence.
  * The final stage is approximately iterations 10,000–17,000, with midpoint near 13,500.
  * **Loss midpoint band:** \(A_{\text{loss}}\approx 2\times10^{-7}\).
  * **Loss end band:** \(B_{\text{loss}}\approx 1\times10^{-8}\).
  * Since \(B_{\text{loss}}<A_{\text{loss}}\) by roughly an order of magnitude or more, the loss is **still decreasing**, not flat.
* **Relative \(L^2\) error history:** The error falls from order \(10^0\) to a few \(10^{-3}\)–\(10^{-2}\) during Adam. Following the optimizer transition, it drops sharply and then becomes jagged while its overall band continues downward.
  * **Error midpoint band:** \(A_{\text{error}}\approx 1\times10^{-5}\).
  * **Error end band:** \(B_{\text{error}}\approx 2\times10^{-6}\), with lower individual dips.
  * Thus, the error is also **still decreasing on average**. Noise around the trend does not constitute a plateau.
* **Final error used for scoring:** Approximately **\(2\times10^{-6}\)**, read from the right edge of the plotted error curve.
* **Log extraction:** The exact reported minimum is:
  * **`Minimum achieved RL2 error: 3.9123398217137915e-07`**
  * This is the best dipped value, not the final plotted value. The repeated `RL2 error: 0.00891681` entries correspond to the earlier warm-up/solution printout and do not represent the right edge of the SSBroyden curve.
* **PDE residual heatmap:** Residuals are small and broadly scattered across the domain, with no dominant shock, corner, or boundary-localized failure. The log reports residuals from \(2.2178\times10^{-9}\) to \(1.5603\times10^{-4}\).
* **Reference versus prediction:** The fields are visually very similar. The prediction reproduces the \(x\)-modulation and the four \(y\)-oscillations without an obvious phase or amplitude discrepancy.
* **Pointwise error:** The error is small but structured rather than perfectly uniform. Arc/ring-like contours are visible near the upper-central and lower-left/central portions of the domain. The warm background regions contain the larger errors; there is no visible boundary-only failure.

### Diagnosis
**Budget-Limited / Premature Stop — still improving.**

This is not instability, spectral stagnation, or a physics-formulation failure. The manufactured solution, periodic Fourier embedding, vRBA weighting/sampling, and SSBroyden fine-tuning are working effectively. However, both final-stage curves continue improving, so the run cannot be certified as converged or optimal merely because its best error is already very small.

### Targeted Improvements
* **For Integrity:** None. The run completed with finite values and readable plots.
* **For Accuracy:** **Increase only the final SSBroyden stage from 10,000 to 20,000 fixed iterations (2×).** Keep the PDE, source term, periodic embeddings, network, sampling, and Adam warm-up unchanged. After the rerun, assess whether both average band levels are genuinely horizontal over the last 50% of the final stage.
* **For Details:** No separate boundary-condition fix is needed; periodicity is already enforced architecturally. Do not use the minimum dipped error as evidence of final convergence.
* **For Optimality:** Retain the degree-4 periodic Fourier features, vRBA, and SSBroyden. No entropy penalty is relevant to this smooth Helmholtz problem, and no architecture change is presently justified.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The implementation matches the strategy. It uses a PINN with Periodic Fourier Features of degree 4 and Lx=Ly=2.0 applied to both spatial inputs; only the concatenated embeddings are passed to the MLP. The MLP has 3 hidden layers of width 30 with tanh activation and a direct scalar output, with no output transformation. The PDE residual is r = u_tt + u_xx + u - g with k^2=1 and g=(1-17π^2)sin(πx)sin(4πy), under the code's t/x aliasing for the two physical coordinates. The analytical solution is sin(πx)sin(4πy), used only for validation/error metrics. The loss contains only L_pde: during Adam it is vRBA-weighted MSE on the PDE residual; during SSBroyden it is unweighted MSE, while vRBA lambdas are updated and used for sampling only. The optimizer workflow is Adam for 10,000 iterations with exponential-decay learning rate and vRBA importance sampling (batch size 20,000), followed by SSBroyden for a fixed 10,000-step budget using Nchange=100 per outer loop (100 outer iterations). Early stopping is only on NaN/Inf. No BC or data loss terms are added. | 2.00e-06 | 35 | 35 | 7 | 14 | **91** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves `Helmholtz_results.npz` (NumPy .npz) with keys: `x` (100x100 float64 physical-x grid), `y` (100x100 float64 physical-y grid), `u_pred` (100x100 float64 predicted field), `u_analytical` (100x100 float64 exact field), and `relative_l2_error` (1D float64 error-history array; default length 655 = 555 Adam logged entries + 100 SSBroyden outer iterations). It also saves an extra non-required MATLAB file `Log_files_SSBroyden.mat` containing `model_name` plus history keys `all_errors`, `all_its`, `all_loss`, `all_gamma`, `all_lamB`, `all_max_RBA`, `all_L_infty`, and `all_L_2`; with default settings the shared Adam+SSBroyden histories are length 655, while Adam-only histories are length 555. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run5/results//Helmholtz_2D_Periodic_Sin_pi_x_Sin_4pi_y/2026_11_06_1214/` |


---
