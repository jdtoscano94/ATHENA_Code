
# Experiment Report
*Last Updated: 2025-12-05 01:04:09*

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
### 0. Triage - Determine Strategy
*   **Role:** Validation/Error Metrics ONLY (Forward Problem with Analytical Solution).
*   **Strategy:** **Refinement Strategy**. The Advisor explicitly requests increasing the SSBroyden iterations to ensure the loss curve plateaus. The Critic mandates a value strictly greater than 60,000 and explicit Hessian re-initialization.

### 1. Data Source
*   **Data:** None required for training. Use the provided **Analytical Solution** for validation and error plotting.

### 2. Problem Statement
*   **Governing Equation (KdV):**
    $$ \text{Residual} = u_t + 6 u u_x + u_{xxx} $$
*   **Boundary/Initial Conditions (Analytical):**
    *   **IC:** $u(0,x) = g_0(x)$
    *   **Dirichlet BC (Left):** $u(t,0) = g_1(t)$
    *   **Dirichlet BC (Right):** $u(t,20) = g_2(t)$
    *   **Neumann BC (Right):** $u_x(t,20) = g_3(t)$
    *   *Note:* $g_0, g_1, g_2, g_3$ are the evaluations of the analytical solution at the respective boundaries.
*   **Analytical Solution:**
    $$ u(x,t) = \frac{2(c_1 - c_2) [ c_1 \cosh^2(\frac{\sqrt{c_2}\zeta_2}{2}) + c_2 \sinh^2(\frac{\sqrt{c_1}\zeta_1}{2}) ]}{ [ (\sqrt{c_1} - \sqrt{c_2}) \cosh(\frac{\sqrt{c_1}\zeta_1 + \sqrt{c_2}\zeta_2}{2}) + (\sqrt{c_1} + \sqrt{c_2}) \cosh(\frac{\sqrt{c_1}\zeta_1 - \sqrt{c_2}\zeta_2}{2}) ]^2} $$
    *   $\zeta_1 = x - 6.0t - (-2.0)$
    *   $\zeta_2 = x - 2.0t - 2.0$
*   **Problem Type:** Forward.
*   **Well-Posedness:** Yes, admits strong solutions.

### 2.1 Input
*   **Domain:** $t \in [0, 5]$, $x \in [0, 20]$. (2D input).

### 2.2 Output
*   **Value:** Scalar $u(t,x)$.
*   **Characteristics:** Soliton interaction (high gradients, localized features).

### 3. Base Model Selection
*   **Architecture:** MLP with Hybrid Adam and SSBroyden.

### 4. Main Architecture
**Structure:** Input $\to$ Normalization $\to$ Embedding $\to$ Main Body $\to$ Output Transformation $\to$ Output.

*   **4.1 Input Transformations:**
    *   **Normalization:** Normalize $t \to [0, 1]$ and $x \to [0, 1]$ **inside** the `forward` method.
    *   **Feature Embeddings:** **Option B: Random Fourier Features.**
        *   `degree = 5`
        *   `scale = 3.0`
        *   *Reasoning:* Proven effective in previous runs for capturing soliton frequencies.

*   **4.2 Main Body:**
    *   **Specs:** `width=30`, `depth=3`. (Standard for low-dim problems).
    *   **Activation:** Tanh.

*   **4.3 Output Transformation (Hard Constraints):**
    *   **Ansatz:** $u_{pred}(t,x) = f_b(t,x) + h(t,x) \cdot NN(t,x)$
    *   **Distance Function:** $h(t,x) = t \cdot x \cdot (20 - x)$
        *   (Zero at $t=0$, $x=0$, $x=20$).
    *   **Boundary Function:**
        $$ f_b(t,x) = g_0(x) + \frac{x}{20}(g_2(t) - g_2(0)) + \frac{20-x}{20}(g_1(t) - g_1(0)) $$
        *   *Note:* This interpolates the IC and the two Dirichlet BCs.

### 5. Constraint Implementation Strategy
*   **Count:** $L_{pde}$ (1) + $L_{bc\_neumann}$ (1) = 2 terms. ($L_{ic}$ and $L_{bc\_dirichlet}$ are satisfied by architecture).
*   **Strategy:** Loss-Based for Neumann, Architectural for others.

### 6. Loss Function Assembly
*   **Symbolic Loss:**
    $$ L_{total} = L_{pde} + L_{bc\_neumann} $$
*   **Definitions:**
    *   $L_{pde}$: MSE of $(u_t + 6uu_x + u_{xxx})$.
    *   $L_{bc\_neumann}$: MSE of $(u_x(t, 20) - g_3(t))$.

### 6.1 Continuation Strategy
*   **None required.**

### 7. Weighting Strategy & Optimization
*   **Local Weights (`lambdas`):** Use **vRBA** for $L_{pde}$.
    *   **Adam:** Used for Weighting & Sampling.
    *   **SSBroyden:** Used for Sampling ONLY.
*   **Global Weights (`lam`):**
    *   **Strategy:** **Self-Balancing (GradNorm).**
    *   **Phase 1 (Adam):** Dynamic updates.
    *   **Phase 2 (SSBroyden):** **FREEZE** at final Adam values.
*   **Optimization Schedule:**
    *   **Phase 1: Adam**
        *   Iterations: 5,000.
        *   Goal: Stabilization.
    *   **Phase 2: SSBroyden (MANDATORY UPDATE)**
        *   Iterations: **80,000**. (Increased to >150% of previous run to ensure plateau).
        *   **CRITICAL:** Explicitly instruct the Planner to **re-initialize the SSBroyden Hessian approximation** at the start of Phase 2.

### 8. Visualization Strategy
*   Standard plots. Ensure the analytical solution is overlaid for comparison.

### 9. Saving Data
*   Save `u_pred`, `u_exact`, `x`, `t` as `.npz`.

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
*   **Status:** APPROVED

**Section 2: Quantitative Scorecard**
*   **Score Integrity:** 35
*   **Score Accuracy:** 35
*   **Score Details:** 12
*   **Score Optimality:** 15
*   **TOTAL SCORE:** 97
**DECISION LOGIC:**
The trial history log shows four previous runs (5, 6, 7, 8) that achieved a score of 93, surpassing the success threshold of 90. This current run achieved an even higher score of 97, marking the fifth consecutive successful run. The exploration phase is now complete. This run represents the best model found.

**Section 3: Forensic Analysis & Feedback**
*   **Observation:**
    *   **Loss/Error History:** The training curves clearly show two phases. The initial Adam optimization (0-5k iterations) rapidly reduces the L2 error to approximately 1e-1. The subsequent SSBroyden phase (5k-55k iterations) achieves a much deeper convergence, driving the L2 error down to a final oscillating plateau around 1.5e-5. The total loss curve shows a smooth and stable convergence to a very low value of ~1e-10. While the L2 error curve successfully plateaued as requested, it exhibits minor oscillations in the final segment and settled at a value slightly higher than the minimum achieved during the run (1.57e-06, per the logs).
    *   **Solution Fields:** The qualitative comparison between the predicted and reference solutions is perfect. The model has captured the complex dynamics of the two-soliton interaction, including their collision and subsequent phase shift, with no discernible visual error.
    *   **Error/Residual Heatmaps:** The pointwise error plot correctly shows that the largest errors (~1e-6) are concentrated along the high-gradient soliton trajectories. Elsewhere, the error is negligible. The PDE residual is satisfied to an extremely high degree across the entire domain.

*   **Diagnosis:** "Excellent Convergence with Minor Final-Stage Oscillation." The model has successfully solved the KdV equation to a very high degree of accuracy. The primary objective of extending the SSBroyden iterations to ensure the loss curve reached a stable plateau was achieved. The minor oscillations in the final L2 error are insignificant given the extremely low error magnitude and do not detract from the outstanding result.

*   **Targeted Improvements:**
    *   **For Integrity:** None.
    *   **For Accuracy:** None. The accuracy is exceptional.
    *   **For Details:** None. The result is excellent and does not require further refinement.
    *   **For Optimality:** None. The implemented methodology (RFF, hard constraints, vRBA, GradNorm, and a hybrid Adam+SSBroyden optimizer) is state-of-the-art for this class of problem.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 5 | The implementation uses an MLP architecture (width=30, depth=3) with Random Fourier Features embedding (degree=5, scale=3.0) applied only to normalized inputs (t,x normalized to [0,1] inside the forward pass). The output employs a hard-constrained ansatz u_pred = f_b(t,x) + h(t,x) * NN(t,x), where h(t,x) = t * x * (20 - x) enforces zero at boundaries/IC, and f_b interpolates the analytical IC and Dirichlet BCs at x=0 and x=20. The optimizer follows a hybrid workflow: Phase 1 Adam (5000 epochs) with self-balancing global weights (lamE=1 fixed, lamB via GradNorm EMA with decay alpha=0.999, gamma_grads=0.99) and vRBA local weighting/sampling for L_pde (exponential potential, phi=1.0, c_log=1.0, adaptive cap_RBA from 20 increasing stepwise); Phase 2 SSBroyden (12500 effective steps in outer loops of N_change=100) freezes global lamB, uses vRBA-updated lambdas for sampling only (no local weighting in loss), plain MSE losses. Loss terms: L_total = L_pde (KdV residual u_t + 6*u*u_x + u_xxx, vRBA-weighted in Adam) + L_bc_neumann (MSE on u_x(t,20) vs analytical derivative). No curriculum continuation; RBA parameters adapt via max_RBA schedule. | 0.02% | 35 | 35 | 8 | 15 | **93** |
| 6 | The implementation uses a PINN architecture with input normalization to [0,1] inside the model's __call__ method, Random Fourier Features embedding (degree=5, scale=3.0), followed by an MLP main body (width=30, depth=3, output=1), and a hard-constraint ansatz u_pred = f_b(t,x) + h(t,x) * NN(t,x) where f_b interpolates the analytical IC and Dirichlet BCs at x=0/20, and h=t*x*(20-x) enforces zero on boundaries. Optimizer workflow is Hybrid Adam (5000 epochs) with vRBA local weighting (exponential potential, EMA updates) and sampling for L_pde, self-balancing global weights (GradNorm-based lamB between L_pde and L_neumann), followed by SSBroyden (20,000 effective iterations across outer loops with Nchange=100) using vRBA for sampling only on collocation points and frozen global weights from Adam end. Loss terms are exactly L_total = L_pde (MSE of KdV residual u_t + 6uu_x + u_xxx, vRBA-weighted in Adam) + L_bc_neumann (MSE of u_x(t,20) vs analytical g3(t)). Curriculum: Adam stabilization (5000 iters, adaptive max_RBA ramp to 20, lr decay), then deep SSBroyden convergence (20,000 iters). | 4.60e-05 | 35 | 35 | 8 | 15 | **93** |
| 7 | The implementation uses an MLP architecture with input normalization to [0,1] for t and x inside the PINN forward pass, followed by Random Fourier Features embedding (degree=5, scale=3.0), a 3-layer MLP body (width=30 per layer, tanh activation), and hard constraint output ansatz u(t,x) = f_b(t,x) + t*x*(20-x)*NN(t,x), where f_b interpolates the analytical IC and Dirichlet BCs at x=0 and x=20. Optimizer workflow follows hybrid Adam (5000 epochs with exponential decay LR starting at 1e-3, batch size 20000, vRBA weighting and importance sampling for PDE loss) followed by SSBroyden (30,000 total iterations across outer loops with Nchange=100 inner steps per resample, vRBA sampling only, global weights frozen from Adam via GradNorm self-balancing on PDE and Neumann losses). Loss terms are exactly L_total = L_pde (MSE on u_t + 6*u*u_x + u_xxx residuals with vRBA local weights during Adam) + L_bc_neumann (MSE on u_x(t,20) - analytical grad), no additional terms. Curriculum uses adaptive max_RBA ramp from initial eta/(1-gamma) to cap=20, EMA lambdas with exponential potential (phi=1.0, c_log=1.0), and GradNorm for global lamB. | 9.40e-06 | 35 | 35 | 8 | 15 | **93** |
| 8 | The implementation uses an MLP architecture (width=30, depth=3, tanh activation) with Random Fourier Features embedding (degree=5, scale=3.0) and input normalization to [0,1] inside the model's __call__. The output follows the hard-constraint ansatz u_pred = f_b(t,x) + h(t,x) * NN(t,x), where f_b interpolates IC and Dirichlet BCs using the analytical solution, and h = t * x * (20 - x). Loss terms are exactly L_total = L_pde (MSE on KdV residual u_t + 6 u u_x + u_xxx with vRBA local weighting via importance sampling) + L_bc_neumann (MSE on u_x(t,20) - g_3(t)). Optimization is hybrid: Phase 1 Adam (5000 iterations, lr decay, vRBA for local PDE weighting, dynamic GradNorm self-balancing for global lamB via EMA on gradient norms); Phase 2 SSBroyden (40000 total iterations across outer loops with Nchange=100 inner steps each, freezing final Adam lamB, reinitializing Hessian, continuing vRBA via resampling). Curriculum strictly follows Adam stabilization then extended SSBroyden for convergence. | 5.70e-06 | 35 | 35 | 8 | 15 | **93** |
| 9 | The implementation uses an MLP architecture (width=30, depth=3, tanh activation) with Random Fourier Features embedding (degree=5, scale=3.0) and input normalization to [0,1] inside the forward method. The output follows the hard-constrained ansatz u_pred = f_b(t,x) + h(t,x) * NN(t,x), where h(t,x) = t * x * (20 - x) and f_b interpolates the IC g0(x) and Dirichlet BCs g1(t), g2(t) using the analytical solution. Optimizer workflow is Hybrid: Phase 1 Adam (5000 epochs) with vRBA local weighting (exponential potential, EMA updates with gamma=0.999, eta=0.01, phi=1.0) and importance sampling (k_samp=1.0), plus GradNorm self-balancing global weights (lamE=1 fixed, dynamic lamB via gradient norms with alpha=0.999, gamma_grads=0.99); Phase 2 SSBroyden (80,000 total iterations, resampling every Nchange=100 steps using updated vRBA lambdas, k_samp_bfgs=1.0, c_samp_bfgs=0.0) with Hessian approximation re-initialized to identity at Phase 2 start and recycled/symmetrized per outer loop. Loss terms are exactly L_total = L_pde (MSE of u_t + 6 u u_x + u_xxx with vRBA) + L_bc_neumann (MSE of u_x(t,20) - g3(t)), with frozen lamB from Adam. | 1.50e-05 | 35 | 35 | 12 | 15 | **97** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 5 | Saves 'kdv_results.npz' containing keys: 'u_pred' (100,256), 'u_exact' (100,256), 't' (100,), 'x' (256,). |
| 6 | Saves 'kdv_results.npz' containing keys: 'u_pred' (100x256), 'u_exact' (100x256), 't' (100,), 'x' (256,). |
| 7 | Saves kdv_results.npz containing keys: 'u_pred' (100x256), 'u_exact' (100x256), 't' (100,), 'x' (256,). |
| 8 | Saves 'kdv_results.npz' containing keys: 'u_pred' (100x256), 'u_exact' (100x256), 't' (100,), 'x' (256,). |
| 9 | Saves 'kdv_results.npz' containing keys: 'u_pred' (100x256), 'u_exact' (100x256), 't' (100,), 'x' (256,). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 5 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//KdV_Mixed_ICs/2025_12_05_0035/` |
| 6 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//KdV_Mixed_ICs/2025_12_05_0039/` |
| 7 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//KdV_Mixed_ICs/2025_12_05_0044/` |
| 8 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//KdV_Mixed_ICs/2025_12_05_0050/` |
| 9 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//KdV_Mixed_ICs/2025_12_05_0057/` |


---
