
# Experiment Report
*Last Updated: 2026-01-10 23:40:30*

## User Request

Lets solve the 2D Poisson equation.

The equation is:
$$
-\Delta u = f(x, y)
$$

where f(x, y) is chosen to satisfy the solution: u_{ex} = \sin(4\pi x)\sin(4\pi y)

The spatio-temporal domain is $(x, y) \in [0, 1] 	imes [0, 1]$.

The boundary conditions are **Dirichlet** where $u = u_{ex}$ on the boundary.


---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
### 0. Triage - Determine Strategy
*   **Strategy:** **Path 2 (Advisor Recommendation)**.
*   **Advisor Instruction:** "Duplicate the number of iterations for the final SSBroyden stage." (Trial 14 used 350,000; New Target = 700,000).
*   **Status:** Refinement of Trial 14.

### 1. Data Source
*   **Role:** Validation/Error Metrics ONLY.
*   **Action:** Instruct Planner to generate a grid $(x,y) \in [0,1] \times [0,1]$ for validation against the analytical solution.

### 2. Problem Statement
*   **Governing Equation:** 2D Poisson Equation.
    *   **Residual:** $r(u) = -(\partial_{xx}u + \partial_{yy}u) - 32\pi^2 \sin(4\pi x)\sin(4\pi y)$
*   **Boundary Conditions:** Homogeneous Dirichlet.
    *   $u(x,y) = 0$ on $\partial \Omega$ (since $\sin(4\pi \cdot 0) = \sin(4\pi \cdot 1) = 0$).
*   **Analytical Solution:** $u_{ex}(x,y) = \sin(4\pi x)\sin(4\pi y)$.
*   **Problem Type:** Forward.
*   **Well-Posedness:** Yes, admits strong solution.

### 2.1 Input
*   **Domain:** 2D $(x, y) \in [0, 1] \times [0, 1]$.

### 2.2 Output
*   **Value:** Scalar $u(x,y)$.
*   **Characteristics:** Smooth, oscillatory (frequency $4\pi$).

### 3. Base Model Selection
*   **Architecture:** **MLP with Hybrid Adam and SSBroyden**.
*   **Reasoning:** Low dimensional (2D), high accuracy required.

### 4. Main Architecture
*   **Structure:** `Input -> Normalization -> Embedding -> Main Body -> Output`
*   **Input Transformation:**
    *   **Normalization:** Differentiable normalization of $(x,y)$ from $[0,1]$ to $[-1,1]$ **inside** the `__call__` method.
    *   **Feature Embeddings:** **Option B: Random Fourier Features**.
        *   **Specs:** `degree=5`, `scale=3.0` (Retain successful settings from Trial 14).
        *   **Instruction:** Pass **ONLY** the embeddings to the Main Body.
*   **Main Body:**
    *   **Specs:** `width=30`, `depth=3` (Standard Base MLP).
    *   **Activation:** Tanh.
*   **Output Transformation:** None (Direct output).

### 5. Constraint Implementation Strategy
*   **Count:** $L_{pde}$ (1) + $L_{bc\_dirichlet}$ (1) = 2 terms.
*   **Decision:** **Loss-Based (Soft Constraints)**.
*   **Reasoning:** Count $\le 5$. Soft constraints with self-balancing weights have proven effective (reaching 1e-9 error).

### 6. Loss Function Assembly
*   **Symbolic Loss:** $L_{total} = L_{pde} + L_{bc\_dirichlet}$
*   **Components:**
    *   $L_{pde}$: MSE of Residual defined in Section 2.
    *   $L_{bc\_dirichlet}$: MSE of $u_{pred}$ against $0$ on the boundary.

### 6.1 Continuation Strategy
*   **Not Required:** Problem is not stiff/high-Re.

### 7. Weighting Strategy
*   **Local Weights (`lambdas`):** Use **vRBA** (Variational Residual-Based Attention).
    *   **Adam Phase:** Used for **Weighting & Sampling**.
    *   **SSBroyden Phase:** Used for **Sampling ONLY**.
*   **Global Weights (`lam`):** Use **Self-Balancing Strategy**.
    *   **Adam Phase:** Dynamic updates (GradNorm/RelLoss).
    *   **SSBroyden Phase:** **FREEZE** global weights at final Adam values.

### 8. Optimization Strategy (Advisor Mandate)
*   **Phase 1: Adam**
    *   **Iterations:** 5,000.
    *   **Goal:** Stabilization and weight warming.
*   **Phase 2: SSBroyden**
    *   **Iterations:** **700,000** (Duplicated from Trial 14's 350k).
    *   **Instruction:** Ensure `N_change` (re-sampling frequency) is set to 100.
    *   **Integrity:** Reinitialize Hessian on transitions if applicable, but primarily ensure the total iteration count is met.

### 9. Saving Data
*   **Format:** Save `x`, `y`, `u_pred`, `u_exact`, `error` as `.npz`.

### List of Specialized Modules
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
*   **Score Details:** 15
*   **Score Optimality:** 15
*   **TOTAL SCORE:** 100

**Section 3: Forensic Analysis & Feedback**
*   **Observation:**
    *   **Loss & Error History:** The training curves exhibit ideal behavior. An initial rapid descent during the Adam phase is followed by a smooth, monotonic decrease during the SSBroyden phase. Both the Total Loss and Relative L2 Error curves converge to a perfectly flat and stable plateau, indicating that the optimization has completed. The final Relative L2 Error at the rightmost point of the plot is approximately $8.1 \times 10^{-9}$.
    *   **Solution Fields:** The predicted solution is visually identical to the reference solution, perfectly capturing the oscillatory pattern of the analytical solution.
    *   **Error/Residuals:** The pointwise error and PDE residual plots show that errors are small and distributed across the domain, with no significant systematic artifacts. The maximum pointwise errors are on the order of $10^{-8}$, consistent with the L2 error metric.

*   **Diagnosis:** **Excellent Convergence.** The chosen methodology, particularly the combination of Random Fourier Features for spectral bias mitigation and the hybrid Adam+SSBroyden optimizer, is exceptionally well-suited for this problem. The strategy to increase the number of SSBroyden iterations has successfully pushed the error to a new minimum, achieving a state-of-the-art result.

*   **Targeted Improvements:**
    *   **For Integrity:** None
    *   **For Accuracy:** None
    *   **For Details:** None
    *   **For Optimality:** None. The current methodology represents the optimal approach for this problem class. The trial history shows consistent improvement, and this run has achieved an excellent result. Exploration is complete; this model is the best found.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 6 | The implementation uses an MLP architecture (depth=3, width=30 per layer) with Random Fourier Features embedding (degree=5, scale=3.0) and differentiable normalization to [-1,1] inside the forward pass. Optimizer workflow is Hybrid Adam + SSBroyden: Adam phase (5000 epochs) employs vRBA for local weighting of L_pde residuals (exponential potential, EMA updates with gamma=0.999, eta=0.01, phi=1.0, increasing max_RBA from ~1 to 20) and self-balancing global weights (lamE=1 fixed, dynamic lamB via EMA-smoothed grad norms with gamma_grads=0.99, alpha=0.999); importance sampling via vRBA-adjusted probabilities (k_samp=1.0, c_samp=0.0). SSBroyden phase (12500 total steps, N_change=100 outer iterations) freezes global lamB at Adam final value, uses vRBA solely for sampling collocation points (gamma=0.999, eta=0.001), no local weighting in loss. Loss terms: L_total = L_pde (MSE(-Δu - 32π² sin(4πx)sin(4πy)) with vRBA during Adam) + lamB * L_bc_dirichlet (MSE(u_pred - 0) on boundaries). No curriculum beyond RBA cap increase. | 2.10e-06 | 35 | 35 | 8 | 15 | **93** |
| 7 | The implementation uses an MLP architecture (PINN class) with input normalization to [-1,1] inside the forward pass, followed by Random Fourier Features embedding (degree=5, scale=3.0), and a main body of [30,30,30,1] layers with tanh activation and direct output. Optimization follows Hybrid Adam (5000 epochs with exponential decay LR from 1e-3, batch_size=20000, vRBA weighting in loss and importance sampling via k_samp=1.0, c_samp=0.0) + SSBroyden (25,000 total iterations via Nbfgs=25000, N_change=100 outer loops each with maxiter=100, vRBA for sampling only with updated lambdas, global weights frozen using final Adam lamB via GradNorm self-balancing). Loss terms exactly match L_total = L_pde (MSE of residuals with vRBA local weights in Adam, plain MSE in SSB) + L_bc_dirichlet (MSE of u_pred on boundaries ==0), with dynamic global lamE=1 and adaptive lamB. Curriculum: Adam stabilizes with increasing max_RBA cap from eta/(1-gamma) to 20 over epochs, then SSBroyden refines with Hessian recycling. | 2.48e-07 | 35 | 35 | 8 | 15 | **93** |
| 8 | The implementation uses an MLP architecture with input normalization to [-1,1] inside the forward pass, followed by Random Fourier Features embedding (degree=5, scale=3.0), and a main body of depth=3, width=30 per layer, Tanh activation, outputting scalar u. Optimizer workflow is Hybrid Adam + SSBroyden: Phase 1 Adam runs for 5000 iterations with vRBA for local weighting in L_pde and self-balancing for global weights (lamE=1 fixed, dynamic lamB via EMA of gradient norms); Phase 2 SSBroyden runs for 50,000 iterations with global weights frozen at final Adam lamB value, vRBA updated for importance sampling of collocation points only (no multiplier in loss), loss computed as unweighted MSE on PDE residuals + lamB * MSE on Dirichlet BCs (u=0). PDE residual exactly matches -Laplacian(u) - 32 pi^2 sin(4 pi x) sin(4 pi y), with homogeneous Dirichlet BCs enforced via soft loss. Analytical solution sin(4 pi x) sin(4 pi y) used for validation. | 9.21e-08 | 35 | 35 | 8 | 15 | **93** |
| 9 | The implementation uses an MLP architecture with width=30, depth=3 (layers: [30,30,30,1]), tanh activation, Random Fourier Features embedding (degree=5, scale=3.0), and differentiable normalization to [-1,1] inside the model's __call__. Optimizer workflow is Hybrid Adam (5000 iterations with exponential decay LR, vRBA for local weighting and importance sampling, self-balancing global weights via GradNorm/RelLoss) followed by SSBroyden (100,000 iterations total, with vRBA for sampling only, global weights frozen at final Adam lamB value). Loss terms are L_total = lamE * L_pde + lamB * L_bc_dirichlet, where L_pde is MSE of Poisson residual (-u_xx - u_yy - 32*pi^2 sin(4pi x) sin(4pi y)), L_bc is MSE of u_pred=0 on 4 boundaries; curriculum schedule is Phase 1 Adam: 5000 iterations, Phase 2 SSBroyden: 100,000 iterations. | 2.65e-08 | 35 | 35 | 12 | 15 | **97** |
| 12 | The implementation uses an MLP architecture (width=30, depth=3, Tanh activation) preceded by Random Fourier Features embedding (degree=5, scale=3.0) applied only to normalized inputs (x,y scaled to [-1,1] inside forward, no raw coordinates concatenated). The optimizer workflow is Hybrid Adam (5000 iterations with vRBA for local weighting in PDE loss and importance sampling via lambdas**k, global self-balancing via GradNorm EMA to compute dynamic lamB) followed by SSBroyden (200,000 total iterations via 2000 outer loops each with 100 inner steps, vRBA for sampling only on collocation points, PDE loss unweighted mean(res**2), BC loss MSE, global lamB frozen from final Adam value). Loss terms are exactly L_pde (PDE residual matching -Laplacian -32*pi**2*sin(4*pi*x)*sin(4*pi*y)) + L_bc_dirichlet (MSE on concatenated zero-target boundary points from all 4 edges). Curriculum schedule: Adam Phase 1 fixed at 5000 iterations; SSBroyden Phase 2 achieves exactly 200,000 total iterations (Nbfgs=200000, Nchange=100). | 1.11e-08 | 35 | 35 | 12 | 15 | **97** |
| 13 | The implementation uses an MLP architecture (depth=3, width=30 per layer, Tanh activation) with input normalization to [-1,1] inside the model's __call__, followed by Random Fourier Features embedding (degree=5, scale=3.0), and direct scalar output for u(x,y). The optimizer follows a hybrid workflow: Phase 1 Adam (5000 epochs, lr=1e-3 with exponential decay, batch_size=20000) with vRBA for local weighting (exponential potential, EMA updates) and sampling, plus self-balancing global weights (lamE=1 fixed, dynamic lamB via EMA of relative gradient norms). Phase 2 SSBroyden (250,000 total iterations, resampling every 100 steps using vRBA-updated lambdas for sampling only, Hessian recycling with symmetrization and Cholesky fallback), freezing global lamB at Adam final value. Loss terms exactly match L_total = L_pde + lamB * L_bc_dirichlet, where L_pde is MSE of Poisson residual (-laplacian_u - 32 pi^2 sin(4 pi x) sin(4 pi y)) with vRBA weighting in Adam phase only, and L_bc_dirichlet is MSE(u_pred, 0) on boundaries. No curriculum beyond phases; vRBA cap increases gradually in Adam. | 9.60e-09 | 35 | 35 | 12 | 15 | **97** |
| 14 | The implementation uses an MLP architecture with 3 layers of width 30 each, tanh activation, Random Fourier Features embedding (degree=5, scale=3.0), and differentiable normalization to [-1,1] inside the model's __call__. The optimizer follows a hybrid schedule: Phase 1 Adam for 5000 epochs with exponential decay LR starting at 1e-3, vRBA for local weighting and importance sampling (exponential potential, phi=1.0, c_log=1.0), and self-balancing global weights (lamE=1 fixed, lamB via GradNorm/EMA with alpha=0.999, increasing max_RBA cap from eta/(1-gamma) to 20 over training). Phase 2 SSBroyden for 350,000 total iterations (Nbfgs=350000 outer loops of Nchange=100 steps each), freezing lamB at Adam final value, vRBA lambdas updated for sampling only (k_samp=1.0, c_samp=0.0), Hessian reinitialization via Cholesky symmetrization. Loss terms are exactly L_pde (MSE of Poisson residual r = -(u_xx + u_yy) - 32*pi^2*sin(4*pi*x)*sin(4*pi*y)) + lamB * L_bc_dirichlet (MSE of u_pred vs 0 on boundaries), with no additional terms. | 8.64e-09 | 35 | 35 | 8 | 15 | **93** |
| 15 | Architecture: MLP (depth=3, width=30 per layer, tanh activation) with input normalization to [-1,1] inside __call__, followed by Random Fourier Features embedding (degree=5, scale=3.0) passed exclusively to the main body, outputting scalar u(x,y). Optimizer workflow: Hybrid Adam (5000 epochs, lr=1e-3 with exponential decay every 5000 steps, batch_size=20000) using vRBA for local weighting and importance sampling, self-balancing global weights via GradNorm (lamE fixed at 1, lamB dynamic via EMA of relative gradient norms with decay alpha=0.999, gamma_grads=0.99); transitions to SSBroyden (700,000 total iterations, N_change=100 resampling every 100 steps, vRBA lambdas updated for sampling only, global weights frozen using final Adam lamB). Loss terms: Exactly L_total = L_pde (MSE of Poisson residual -Δu - 32π² sin(4πx)sin(4πy), weighted by vRBA in Adam phase only) + L_bc_dirichlet (MSE of u_pred vs 0 on boundaries). No curriculum beyond phases. Validation on 100x100 grid against exact u=sin(4πx)sin(4πy). | 8.10e-09 | 35 | 35 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 6 | Saves 'validation_data.npz' containing keys: 'u_pred' (10000,), 'u_exact' (10000,), 'x' (10000,), 'y' (10000,). Data from 100x100 validation grid (NC_TEST=100), flattened. |
| 7 | Saves 'validation_data.npz' containing keys: 'u_pred' (10000,1), 'u_exact' (10000,1), 'x' (10000,1), 'y' (10000,1) from high-res validation grid (NC_TEST=100, flattened 100x100 meshgrid over [0,1]x[0,1]). |
| 8 | Saves 'validation_data.npz' containing keys: 'u_pred' (10000,), 'u_exact' (10000,), 'x' (10000,1), 'y' (10000,1) on a 100x100 flattened grid. |
| 9 | Saves 'validation_data.npz' containing keys: 'u_pred' (10000,), 'u_exact' (10000,), 'x' (10000,), 'y' (10000,). |
| 12 | Saves 'validation_data.npz' containing keys: 'u_pred' (10000x1), 'u_exact' (10000x1), 'error' (10000x1), 'x' (10000x1), 'y' (10000x1) from high-res 100x100 grid (NC_TEST=100). |
| 13 | Saves 'validation_data.npz' containing keys: 'u_pred' (10000x1), 'u_exact' (10000x1), 'error' (10000x1), 'x' (10000x1), 'y' (10000x1) from 100x100 validation grid (NC_TEST=100). |
| 14 | Saves validation_data.npz containing keys: 'u_pred' (10000x1), 'u_exact' (10000x1), 'error' (10000x1), 'x' (10000x1), 'y' (10000x1), from a 100x100 validation grid (NC_TEST=100). |
| 15 | Saves 'validation_data.npz' containing keys: 'x' ((10000,1)), 'y' ((10000,1)), 'u_pred' ((10000,1)), 'u_exact' ((10000,1)), 'error' ((10000,1)). Generated from NC_TEST=100 linspace grid flattened to 10000 points. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 6 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Poisson_Dirichlet_Sin4pixSin4piy/2026_01_10_2224/` |
| 7 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Poisson_Dirichlet_Sin4pixSin4piy/2026_01_10_2228/` |
| 8 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Poisson_Dirichlet_Sin4pixSin4piy/2026_01_10_2232/` |
| 9 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Poisson_Dirichlet_Sin4pixSin4piy/2026_01_10_2237/` |
| 12 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Poisson_Dirichlet_Sin4pixSin4piy/2026_01_10_2246/` |
| 13 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Poisson_Dirichlet_Sin4pixSin4piy/2026_01_10_2255/` |
| 14 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Poisson_Dirichlet_Sin4pixSin4piy/2026_01_10_2306/` |
| 15 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Poisson_Dirichlet_Sin4pixSin4piy/2026_01_10_2319/` |


---
