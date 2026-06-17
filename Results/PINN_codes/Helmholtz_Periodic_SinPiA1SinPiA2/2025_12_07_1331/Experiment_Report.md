
# Experiment Report
*Last Updated: 2025-12-07 13:35:18*

## User Request

Lets solve the 2D Helmholtz equation.

The equation is:
$$
u_{xx} + u_{yy} + k^2 u = g(x, y)
$$

where g(x, y)  is chosen to satisfy the  solution: u_{an}=sin(pi a1)sin (pi a2)

The spatio-temporal domain is $(x, y) \in [-1, 1] 	imes [-1, 1]$.

The boundary conditions are **periodic** in both $x$ and $y$. And let a_1=1 and a_2=4 and k=1.


---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
### Analysis Steps

**0. Triage - Determine Strategy**
*   **User Request:** Solve 2D Helmholtz equation with periodic BCs.
*   **Advisor Report:** **REJECTED_FOR_REFINEMENT**.
    *   **Diagnosis:** Unfinished convergence. Error was still dropping at the end of the previous run.
    *   **Mandatory Action:** Increase SSBroyden phase to **50,000 iterations**.
*   **Critic Report:** Fix the symbolic definition of the PDE loss to explicitly use the residual $r$ with $k^2 u$.
*   **Strategy:** Re-run the successful architecture from Run 8 (Periodic Fourier Features + MLP) with the extended optimization schedule mandated by the Advisor.

**1. Data Source**
*   **Role:** Validation/Error Metrics ONLY.
*   **Action:** No external data loading. Generate grid for validation against analytical solution.

**2. Problem Statement**
*   **Governing Equation:** 2D Helmholtz Equation.
    *   **Residual:** $r(x,y) = u_{xx} + u_{yy} + k^2 u - g(x,y)$
    *   **Constants:** $k=1$.
    *   **Source Term:** $g(x,y) = (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$.
*   **Domain:** $(x, y) \in [-1, 1] \times [-1, 1]$.
*   **Boundary Conditions:** Periodic in $x$ and $y$.
*   **Analytical Solution:** $u_{an}(x,y) = \sin(\pi x)\sin(4\pi y)$.
*   **Problem Type:** Forward. Well-posed for PINNs.

**3. Base Model Selection**
*   **Type:** 2D Forward.
*   **Selection:** MLP with Hybrid Adam and SSBroyden.

**4. Main Architecture**
*   **Structure:** Input $\rightarrow$ Input Transformation $\rightarrow$ Main Body $\rightarrow$ Output.
*   **4.1 Input Transformation:**
    *   **Normalization:** Normalize $(x,y)$ to $[-1, 1]$ (already in range).
    *   **Feature Embeddings:** **Periodic Fourier Features**.
        *   **Reasoning:** Domain has Periodic BCs.
        *   **Config:** `degree=2` (proven effective in Run 8).
        *   **Constraint:** Pass **ONLY** the embeddings to the network. Do not pass raw $x, y$.
*   **4.2 Main Body:**
    *   **Specs:** `width=30`, `depth=3` (Case A: Low Dim).
    *   **Activation:** Tanh.
*   **4.3 Output Transformation:**
    *   None required (Periodicity handled by embedding).

**5. Constraint Implementation Strategy**
*   **Analysis:**
    *   `L_pde`: 1
    *   `L_bc_periodic`: Removed (Architectural).
    *   **Total Count:** 1.
*   **Decision:** **Loss-Based** (Trivial). No hard constraints for BCs needed as embedding handles periodicity.

**6. Loss Function Assembly**
*   **Symbolic Equation:** $L_{total} = L_{pde}$
*   **Terms:**
    *   Define residual: $r = u_{xx} + u_{yy} + k^2 u - g(x,y)$
    *   $L_{pde} = \text{MSE}(r)$
*   **Continuation:** Not required ($k=1$ is small).

**7. Weighting Strategy**
*   **Local Weights (`lambdas`):** Use **vRBA** (Variational Residual-Based Attention).
    *   Adam: Used for Weighting & Sampling.
    *   SSBroyden: Used for Sampling ONLY.
*   **Global Weights (`lam`):**
    *   **Total Terms:** 1.
    *   **Strategy:** No balancing needed.

**8. Optimization Strategy (Advisor Mandate)**
*   **Phase 1: Adam**
    *   Iterations: 5,000.
    *   Learning Rate: Standard decay.
*   **Phase 2: SSBroyden**
    *   **Iterations:** **50,000** (Strict Advisor Requirement).
    *   **Logic:** Re-initialize Hessian/Sampling periodically (e.g., every 500-1000 iters) to maintain stability over this long duration.

**9. Saving Data**
*   Save `u_pred`, `u_exact`, `x`, `y`, and `error` fields as `.npz`.

### Main Model
[MLP with Hybrid Adam and SSBrodyden]

### Enhancements
[Periodic Fourier Features]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
Based on the provided inputs, here is the forensic analysis and scoring report.

**Section 1: Status Decision**
*   **Status:** APPROVED

**Section 2: Quantitative Scorecard**
*   **Score Integrity:** 35
*   **Score Accuracy:** 35
*   **Score Details:** 15
*   **Score Optimality:** 15
*   **TOTAL SCORE:** 100

**Section 3: Forensic Analysis & Feedback**
*   **Observation:** The training run demonstrates textbook convergence for a PINN. The initial Adam phase (0-5k iterations) rapidly reduces the error to $\sim 10^{-3}$. The subsequent SSBroyden phase (5k-38k iterations) provides a smooth, monotonic decrease in error, eventually reaching a noise floor around $10^{-8}$ to $10^{-9}$. The Relative $L^2$ Error history shows that in the final segment (iterations 30k-38k), the error oscillates stably around the minimum, indicating that the optimizer has extracted the maximum precision possible given the network capacity and floating-point limits. The visual fields match the reference perfectly, and the residual heatmap shows uniform, low-magnitude noise ($< 10^{-6}$) with no localized artifacts.
*   **Diagnosis:** **Optimal Convergence.** The model successfully solved the 2D Helmholtz problem to high precision ($7.26 \times 10^{-9}$), significantly outperforming the standard threshold. The early stop (at ~38k instead of 55k) suggests the optimizer reached its convergence tolerance, which is a positive indicator.
*   **Targeted Improvements:**
    *   **For Integrity:** None.
    *   **For Accuracy:** None. The error is extremely low.
    *   **For Details:** None. Boundary conditions (periodic) are perfectly satisfied via the embedding.
    *   **For Optimality:** The architecture (Periodic Fourier Features + MLP) and optimizer schedule (Adam + SSBroyden) were perfectly chosen for this problem.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 6 | The implementation uses an MLP architecture (depth=3, width=30 per layer, output=1) preceded by Periodic Fourier Features embedding (degree=2, Lx=2.0, enforcing periodicity with fundamental frequency π, passing only embeddings to MLP body). Optimizer workflow is Hybrid Adam + SSBroyden: Adam phase (5000 epochs, batch_size=20000, exponential decay LR from 1e-3) with vRBA local weighting (exponential potential, EMA update with gamma=0.999, eta=0.01, phi=1.0) inside L_pde = mean((lambdas * residuals)^2) and importance sampling (k_samp=1.0); followed by SSBroyden (12500 total iterations in outer loops of N_change=100 steps each) using vRBA-updated lambdas for sampling only (k_samp_bfgs=1.0) with L_pde = mean(residuals^2). Adaptive curriculum via increasing max_RBA from ~1.0 to 20.0. Loss assembly is strictly L_total = L_pde (Helmholtz residual: u_xx + u_yy + k^2 u - g with k=1, g=(1-17π²)sin(πx)sin(4πy)). No boundary losses (periodic BCs architectural). | 6.70e-08 | 35 | 35 | 13 | 15 | **98** |
| 8 | The implementation uses a PINN architecture: inputs (x,y) fed to Periodic Fourier Features (embedding_degree=2, Lx=2.0 enforcing periodicity over [-1,1]) followed by an MLP (3 layers of width 30 neurons each, tanh activation, output 1 for u). Optimizer workflow is Hybrid Adam (5000 iterations with exponential decay LR starting at 1e-3, vRBA for local weighting via exponential potential and importance sampling with k_samp=1.0, c_samp=0.0) followed by SSBroyden (25000 total iterations, N_change=100 for resampling Hessian stability, vRBA sampling only with gamma_bfgs=0.1). Loss terms include only L_pde (Helmholtz residual r = u_xx + u_yy + k^2 u - g with k=1, g=(1-17*pi^2)*sin(pi x)*sin(4 pi y); weighted by vRBA lambdas in Adam phase, unweighted mean squared in SSBroyden with separate lambda updates). No curriculum training specified or used. Reference solution u_exact=sin(pi x) sin(4 pi y) matches exactly. | 2.00e-08 | 35 | 35 | 15 | 15 | **100** |
| 10 | Architecture: MLP (depth=3, width=30, tanh activation) with input transformation via Periodic Fourier Features (degree=2, Lx=2.0) applied exclusively to concatenated (x,y) coordinates; no raw coordinates passed to the MLP. Optimizer workflow: Hybrid Adam (Phase 1: 5000 iterations with exponential decay LR starting at 1e-3, decay_step=5000, decay_rate=0.9) followed by SSBroyden (Phase 2: 50,000 total iterations via 500 outer loops of Nchange=100 inner steps each, with periodic Hessian reinitialization via cholesky stabilization and vRBA importance sampling using k_samp=1.0, c_samp=0.0). Loss terms: L_total = L_pde = MSE(r) where r = u_xx + u_yy + k^2 u - g (k=1, g=(1-17π²)sin(πx)sin(4πy)); vRBA (exponential potential, phi=1.0, c_log=1.0, gamma=0.999, eta=0.01) provides local weighting in Adam phase and sampling in both phases; no additional terms or balancing needed (single term). Curriculum schedule: Adam pretraining with adaptive max_RBA ramp-up from eta/(1-gamma) to cap_RBA=20 over epochs, then pure SSBroyden sampling-only refinement. | 7.26e-09 | 35 | 35 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 6 | Saves 'validation_data.npz' containing keys: 'u' (256x256), 'u_exact' (256x256). |
| 8 | Saves 'validation_data.npz' containing keys: 'u_pred' (256x256), 'u_exact' (256x256), 'x' (65536x1), 'y' (65536x1). |
| 10 | Saves 'validation_data.npz' containing keys: 'u_pred' (256x256), 'u_exact' (256x256), 'x' (65536x1), 'y' (65536x1), 'error' (256x256). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 6 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Helmholtz_Periodic_SinPiA1SinPiA2/2025_12_07_1318/` |
| 8 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Helmholtz_Periodic_SinPiA1SinPiA2/2025_12_07_1324/` |
| 10 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Helmholtz_Periodic_SinPiA1SinPiA2/2025_12_07_1331/` |


---
