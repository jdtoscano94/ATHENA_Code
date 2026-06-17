
# Experiment Report
*Last Updated: 2026-01-10 20:37:16*

## User Request

We are solving a forward problem for the Korteweg-De Vries (KdV) equation:

$$
u_t + 6uu_x + u_{xxx} = 0
$$

**Domain:**
- Spatial: $x \in [0, 20]$
- Temporal: $t \in [0, 5]$

**Boundary Conditions:**

$$
u(0,x) = g_0(x) \quad 	ext{(Initial Condition)}
$$

$$
u(t,0) = g_1(t) \quad 	ext{(BC at } x=0	ext{)}
$$

$$
u(t,20) = g_2(t) \quad 	ext{(BC at } x=20	ext{)}
$$

$$
u_x(t,20) = g_3(t) \quad 	ext{(Neumann BC at } x=20	ext{)}
$$

**Analytical Solution:**

The solution is given by a two-soliton interaction formula:

$$
u(x,t) = rac{2(c_1 - c_2) \left[ c_1 \cosh^2\left(rac{\sqrt{c_2}\zeta_2}{2}ight) + c_2 \sinh^2\left(rac{\sqrt{c_1}\zeta_1}{2}ight) ight]}{\left[ \left(\sqrt{c_1} - \sqrt{c_2}ight) \cosh\left(rac{\sqrt{c_1}\zeta_1 + \sqrt{c_2}\zeta_2}{2}ight) + \left(\sqrt{c_1} + \sqrt{c_2}ight) \cosh\left(rac{\sqrt{c_1}\zeta_1 - \sqrt{c_2}\zeta_2}{2}ight) ight]^2}
$$


where:
- $\zeta_1 = x - c_1 t - x_1$
- $\zeta_2 = x - c_2 t - x_2$

**Parameters:**
- $c_1 = 6.0$ (velocity of first soliton)
- $c_2 = 2.0$ (velocity of second soliton)
- $x_1 = -2.0$ (initial position of first soliton)
- $x_2 = 2.0$ (initial position of second soliton)

The boundary conditions $g_0, g_1, g_2, g_3$ are derived by evaluating the analytical solution at the appropriate boundaries.


---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
### Analysis Steps

**0. Triage - Determine Strategy**
- **User Request:** Forward problem for KdV equation (1D+t).
- **Advisor Report:** **REJECTED_FOR_REFINEMENT**.
- **Recommendation:** The previous run (Run 7) was excellent (Error ~1.7e-7) but exhibited oscillations in the final SSBroyden phase.
- **Mandatory Action:** **Increase the SSBroyden resampling interval (`N_change`) from 100 to 200** to allow the optimizer to stabilize on the loss landscape before collocation points are moved.
- **Critic Response:** The Critic noted that changing `N_change` alters the sampling dynamics. I justify retaining the total iteration count (200,000) because the Advisor's specific goal is to *reduce the frequency* of resampling events to dampen oscillations. Halving the number of resampling events (by doubling the dwell time) is the direct mathematical implementation of this recommendation.

**1. Data Source**
- **Role:** Validation/Error Metrics ONLY.
- **Instruction:** Use the provided analytical solution for validation.

**2. Problem Statement**
- **Governing Equation:** Korteweg-De Vries (KdV)
  $$ u_t + 6uu_x + u_{xxx} = 0 $$
  - Residual: $R = u_t + 6 \cdot u \cdot u_x + u_{xxx}$
- **Boundary/Initial Conditions:**
  - IC: $u(0,x) = g_0(x)$ (Analytical solution at $t=0$)
  - BC Left (Dirichlet): $u(t,0) = g_1(t)$ (Analytical at $x=0$)
  - BC Right (Dirichlet): $u(t,20) = g_2(t)$ (Analytical at $x=20$)
  - BC Right (Neumann): $u_x(t,20) = g_3(t)$ (Analytical derivative at $x=20$)
- **Analytical Solution:**
  $$ u(x,t) = \frac{2(c_1 - c_2) [ c_1 \cosh^2(\frac{\sqrt{c_2}\zeta_2}{2}) + c_2 \sinh^2(\frac{\sqrt{c_1}\zeta_1}{2}) ]}{ [ (\sqrt{c_1} - \sqrt{c_2}) \cosh(\frac{\sqrt{c_1}\zeta_1 + \sqrt{c_2}\zeta_2}{2}) + (\sqrt{c_1} + \sqrt{c_2}) \cosh(\frac{\sqrt{c_1}\zeta_1 - \sqrt{c_2}\zeta_2}{2}) ]^2} $$
  where $\zeta_1 = x - 6.0t - (-2.0)$ and $\zeta_2 = x - 2.0t - 2.0$.
- **Problem Type:** Forward.
- **Input:** 2D ($t, x$). Domain: $t \in [0, 5], x \in [0, 20]$.
- **Output:** Scalar $u$.

**3. Base Model Selection**
- **Type:** Low Dim (1D+t).
- **Optimizer:** Hybrid Adam + SSBroyden.
- **Base MLP Specs:** `width=30`, `depth=3`.

**4. Main Architecture**
- **Structure:** Input $\to$ Normalization $\to$ Embedding $\to$ Main Body $\to$ Output.
- **Input Transformation:**
  - **Normalization:** Normalize $t \in [0, 5] \to [0, 1]$ and $x \in [0, 20] \to [0, 1]$ **inside** the model's `forward` method.
  - **Feature Embeddings:** **Option B: Random Fourier Features**.
    - **Specs:** `degree=5`, `scale=3.0`.
    - **Composition:** Pass **ONLY** the embeddings to the network (do not concatenate raw inputs).
- **Main Body:**
  - MLP with `width=30`, `depth=3`. Activation: `tanh`.
- **Output Transformation:**
  - Direct output `u`.

**5. Constraint Implementation Strategy**
- **Count:** $L_{pde}$ (1) + $L_{ic}$ (1) + $L_{bc\_dirichlet}$ (2) + $L_{bc\_neumann}$ (1) = 5 terms.
- **Strategy:** **Loss-Based (Soft Constraints)**.
  - *Reasoning:* While the count is > 4, Run 7 proved that soft constraints with high weighting work perfectly for this problem. We will maintain the successful strategy.

**6. Loss Function Assembly**
- **Symbolic Equation:**
  $$ L_{total} = L_{pde} + L_{ic} + L_{bc\_dirichlet} + L_{bc\_neumann} $$
- **Components:**
  - $L_{pde}$: MSE of Residual.
  - $L_{ic}$: MSE against $g_0(x)$.
  - $L_{bc\_dirichlet}$: MSE against $g_1(t)$ and $g_2(t)$.
  - $L_{bc\_neumann}$: MSE against $g_3(t)$.

**6.1 Continuation Strategy**
- **None.** Direct solve is effective.

**7. Weighting Strategy**
- **Local Weights (`lambdas`):** Use **vRBA**.
  - **Adam Phase:** Use for **Weighting & Sampling**.
  - **SSBroyden Phase:** Use for **Sampling ONLY**.
- **Global Weights (`lam`):**
  - **Strategy:** Fixed Global Weights.
  - **Values:** $L_{pde} = 1.0$, All BCs/ICs = $100.0$.
  - **Instruction:** **FREEZE** global weights. Do not update them dynamically.

**Optimization Schedule (Specifics):**
- **Phase 1 (Adam):** 5,000 iterations.
- **Phase 2 (SSBroyden):** 200,000 iterations.
  - **CRITICAL ADVISOR INSTRUCTION:** Set the resampling interval **`N_change = 200`** (increase from 100).
  - **Justification:** We maintain the total iteration count (200k) to preserve the training budget. Increasing `N_change` to 200 reduces the *frequency* of resampling events (halving the total number of resamples), which directly addresses the Advisor's requirement to allow the optimizer more time to stabilize on a fixed landscape between shifts.
  - **Reinitialization:** Reinitialize Hessian on every resampling step.

**8. Visualization Strategy**
- Standard plots (Solution, Error, Residuals).

**9. Saving Data**
- Save `u_pred`, `u_exact`, `x`, `t` as `.npz`.

Main Model:
[MLP with Hybrid Adam and SSBrodyden]

Enhancements:
[Random Fourier Features]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
APPROVED

**Section 1: Status Decision**
* **Status:** APPROVED

**Section 2: Quantitative Scorecard**
* **Score Integrity:** 35
* **Score Accuracy:** 35
* **Score Details:** 12
* **Score Optimality:** 15
* **TOTAL SCORE:** 97

**Section 3: Forensic Analysis & Feedback**
* **Observation:** The model training was highly successful. The predicted solution field is visually indistinguishable from the reference, accurately capturing the complex two-soliton interaction dynamics. The final relative L2 error is excellent, reaching approximately 1.88e-7. The loss and error curves show a rapid decrease during the initial Adam phase, followed by significant refinement in the SSBroyden phase. However, the error curve during the final SSBroyden stage exhibits persistent, low-amplitude, high-frequency oscillations. The strategist's goal of dampening these oscillations by increasing the resampling interval (`N_change=200`) was not fully achieved, although the final accuracy remains state-of-the-art.

* **Diagnosis:** Exploration Complete. The model performance has saturated at an extremely high level of accuracy across multiple recent runs (Runs 3-7 and the current run). The minor oscillations in the second-order optimization phase are a persistent artifact but do not detract from the exceptional quality of the final solution.

* **Targeted Improvements:**
    * **For Integrity:** None
    * **For Accuracy:** Exploration is complete. The best performance in the trial history was achieved in Run 7 (Error: 1.72e-7), with the current run being nearly identical in quality. The chosen methodology is robust and highly effective for this problem. No further improvements are required.
    * **For Details:** None
    * **For Optimality:** None

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 3 | The implementation uses a PINN architecture consisting of input normalization to [0,1] inside the model forward pass, followed by Random Fourier Features embedding (degree=5, scale=3.0, passing only embeddings without original coordinates), then an MLP main body (width=30, depth=3, tanh activation, output=1). Optimizer workflow is Hybrid Adam (5000 epochs, exponential decay LR schedule starting at 1e-3) with vRBA local weighting inside L_pde and importance sampling for collocation batches, transitioning to SSBroyden (12500 total iterations in blocks of 100 with resampling using vRBA lambdas for sampling only, fixed global weights). Loss terms exactly match: L_total = 1*L_pde (MSE on KdV residual u_t + 6uu_x + u_xxx with vRBA EMA-updated local lambdas via exponential potential) + 100*(L_ic + L_bc_left_Dirichlet + L_bc_right_Dirichlet + L_bc_right_Neumann), no additional terms or data leakage. Fixed global weights with no continuation/curriculum schedule beyond Adam LR decay. | 1.29e-05 | 35 | 35 | 13 | 15 | **98** |
| 4 | The implementation uses a PINN architecture: inputs (t,x) normalized to [0,1] inside the model forward pass, followed by Random Fourier Features embedding (degree=5, scale=3.0, sin/cos projections only, no raw concatenation), then MLP with layers [30,30,30,1] and Tanh activation. Optimizer workflow is Hybrid: Phase 1 Adam for 5000 iterations with vRBA (exponential potential, phi=1.0, c_log=1.0) for both weighting inside L_pde MSE and importance sampling of collocation batches (size=20000 from NC=200000 points); global weights lam_pde=1, lam_bc=100 applied to sum L_ic + L_bc_left + L_bc_right_D + L_bc_right_N. Phase 2 SSBroyden for 25000 total iterations (outer loops with Nchange=100 inner steps each), using vRBA-updated lambdas for sampling only (no weighting inside loss). Loss terms exactly match: L_total = L_pde (MSE of u_t + 6 u u_x + u_xxx with local vRBA) + 100 * (L_ic + L_bc_left + L_bc_right_D + L_bc_right_N). No curriculum; direct solve. All BC/IC from analytical two-soliton formula. | 3.19e-06 | 35 | 35 | 13 | 15 | **98** |
| 5 | The implementation uses a PINN architecture: Input (t,x) normalized to [0,1] inside forward pass, followed by Random Fourier Features embedding (degree=5, scale=3.0, ONLY embeddings without raw coordinates concatenation), then MLP (width=30, depth=3, Tanh activation), direct output u. Optimizer workflow: Phase 1 Adam for 5000 iterations with vRBA (exponential potential, phi=1.0, c_log=1.0) for local weighting in L_pde MSE and importance sampling (batch_size=20000 from NC=200000 collocation points); global weights fixed (L_pde:1.0, BCs/ICs:100.0). Phase 2 SSBroyden for 50000 iterations (outer loops with Nchange=100 inner steps, Hessian reinitialized to identity at start), using vRBA-updated lambdas for sampling only (frozen globals), full loss recomputed each outer step. Loss terms exactly match: L_total = L_pde (MSE residuals with vRBA lambdas) + 100*(L_ic + L_bc_left + L_bc_right_D + L_bc_right_N). Curriculum via vRBA adaptive weighting/sampling; no other schedules. | 9.05e-07 | 35 | 35 | 13 | 15 | **98** |
| 6 | The implementation uses a PINN architecture with Random Fourier Features (degree=5, scale=3.0) applied to normalized (t,x) inputs [0,1], passing only embeddings (no raw coordinates) to an MLP main body (width=30, depth=3, tanh activation, 1 output). Normalization occurs differentiably inside the model's forward method. Optimizer workflow is Hybrid Adam (5000 iterations with exponential LR decay from 1e-3, batch_size=20000 via vRBA importance sampling/resampling) followed by SSBroyden (100,000 total iterations across outer loops with Nchange=100 inner steps per resample, Hessian reinitialized/recycled). Loss terms match exactly: L_total = 1.0 * L_pde + 100.0 * (L_ic + L_bc_left + L_bc_right_D + L_bc_right_N), where L_pde is MSE(residual) with vRBA local weighting (exponential potential, EMA update) and sampling in Adam phase, and sampling-only (no local weighting in loss) in SSBroyden phase; BC/IC are MSE against analytical g0/g1/g2/g3 (Neumann via autodiff). Fixed global weights throughout; vRBA adaptive local lambdas for PDE. KdV physics (u_t + 6uu_x + u_xxx=0), IC/BCs, and analytical solution match precisely. Curriculum uses fixed global weights with vRBA schedule as specified. | 2.88e-07 | 35 | 35 | 10 | 15 | **95** |
| 7 | The implementation uses an MLP architecture (width=30, depth=3, tanh activation) with input normalization to [0,1] inside the forward pass, followed by Random Fourier Features embedding (degree=5, scale=3.0) passed exclusively to the MLP body without raw coordinates. Optimizer workflow is Hybrid Adam (5000 iterations with standard exponential decay LR, vRBA for local weighting in PDE loss and importance sampling) followed by SSBroyden (200,000 total iterations in outer loops of 100 steps each, vRBA lambdas updated for sampling only, Hessian reinitialization on resampling). Loss terms exactly match L_total = L_pde (MSE on u_t + 6uu_x + u_xxx) + 100*(L_ic + L_bc_left_D + L_bc_right_D + L_bc_right_N), with fixed global weights (1.0 PDE, 100.0 constraints) and no dynamic global updates. Direct full-physics training without continuation/curriculum phases. | 1.72e-07 | 35 | 35 | 12 | 15 | **97** |
| 8 | The implementation uses a PINN architecture consisting of input normalization to [0,1] inside the forward method, followed by Random Fourier Features embedding (degree=5, scale=3.0) passed exclusively to an MLP main body (width=30, depth=3, tanh activation), outputting scalar u. The optimizer follows a hybrid schedule: Phase 1 Adam for 5000 iterations with vRBA for both weighting (local lambdas via exponential potential, EMA with gamma=0.999, eta=0.01, phi=1.0) and importance sampling (batch_size=20000, k_samp=1.0); Phase 2 SSBroyden for 200000 iterations (N_change=200 resampling interval, Hessian reinitialization per resample via symmetrized cholesky-stabilized inverse) with vRBA for sampling only (k_samp_bfgs=1.0, c_samp_bfgs=0.0). Loss terms match exactly: L_total = 1.0 * L_pde (MSE of u_t + 6uu_x + u_xxx residual) + 100.0 * (L_ic + L_bc_left_dirichlet + L_bc_right_dirichlet + L_bc_right_neumann), with fixed global weights (no dynamic updates). No continuation/curriculum training. KdV physics, analytical IC/BCs/g3(t)=u_x exact, and domain [0,5]x[0,20] are correctly implemented. | 1.88e-07 | 35 | 35 | 12 | 15 | **97** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 3 | Saves 'kdv_solution.npz' containing keys: 'u_pred' (201x101), 'u_exact' (201x101), 'x' (201,), 't' (101,). |
| 4 | Saves 'kdv_solution.npz' containing keys: 'u_pred' (201x101), 'u_exact' (201x101), 'x' (201,), 't' (101,). |
| 5 | Saves 'kdv_solution.npz' containing keys: 'u_pred' (201x101), 'u_exact' (201x101), 'x' (201,), 't' (101,). |
| 6 | Saves 'kdv_solution.npz' containing keys: 'u_pred' (101x201), 'u_exact' (101x201), 'x' (201,), 't' (101,). Arrays represent 2D spatiotemporal grids (rows: t from linspace[0,5,101]; cols: x from linspace[0,20,201]) and 1D coordinate vectors. |
| 7 | Saves 'kdv_solution.npz' containing keys: 'u_pred' (201x101), 'u_exact' (201x101), 'x' (201,), 't' (101,). |
| 8 | Saves 'kdv_solution.npz' containing keys: 'u_pred' (201x101), 'u_exact' (201x101), 'x' (201,), 't' (101,). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 3 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//KdV_Mixed_DerivedIC/2026_01_10_1949/` |
| 4 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//KdV_Mixed_DerivedIC/2026_01_10_1953/` |
| 5 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//KdV_Mixed_DerivedIC/2026_01_10_1958/` |
| 6 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//KdV_Mixed_DerivedIC/2026_01_10_2004/` |
| 7 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//KdV_Mixed_DerivedIC/2026_01_10_2012/` |
| 8 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//KdV_Mixed_DerivedIC/2026_01_10_2026/` |


---
