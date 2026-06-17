
# Experiment Report
*Last Updated: 2025-12-05 14:25:19*

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
Role: Validation/Error Metrics ONLY.

### 1. Data Source
*   **Type:** Analytical Solution Generator.
*   **Instruction:** Instruct Planner to generate a grid of $(x, y)$ points in $[-1, 1] \times [-1, 1]$ and compute the exact solution $u_{an}$ for validation.

### 2. Problem Statement
*   **Governing Equation (Helmholtz):**
    $$ R_{pde} = u_{xx} + u_{yy} + k^2 u - g(x, y) $$
    *   Where $k=1$.
    *   Where $g(x, y) = (1 - 17\pi^2) \sin(\pi x) \sin(4\pi y)$.
*   **Boundary Conditions:**
    *   Periodic in $x$: $u(-1, y) = u(1, y)$.
    *   Periodic in $y$: $u(x, -1) = u(x, 1)$.
*   **Analytical Solution:**
    $$ u_{an}(x, y) = \sin(\pi x) \sin(4\pi y) $$
*   **Problem Type:** Forward.
*   **Well-Posedness:** Well-posed for PINNs (Strong form).

### 2.1 Input
*   Domain: 2D $(x, y) \in [-1, 1]^2$.

### 2.2 Output
*   Value: Scalar $u(x, y)$.
*   Characteristics: Smooth, oscillatory, periodic.

### 3. Base Model Selection
*   **Model:** MLP with Hybrid Adam and SSBrodyden.
*   **Reasoning:** Low-dimensional (2D) forward problem.

### 4. Main Architecture
*   **Structure:** Input $\rightarrow$ Input Transformation $\rightarrow$ Main Body $\rightarrow$ Output.
*   **4.1 Input Transformation:**
    *   **Normalization:** None (Domain is already $[-1, 1]$).
    *   **Feature Embeddings:** **Periodic Fourier Features**.
        *   **Degree:** 2.
        *   **CRITICAL:** Pass **ONLY** the embeddings to the Main Body. Do **NOT** pass raw $x, y$ coordinates. This enforces the periodic BCs as a hard constraint.
*   **4.2 Main Body:**
    *   **Specs:** `width=30`, `depth=3`.
    *   **Activation:** Tanh.
*   **4.3 Output Transformation:**
    *   None required (Direct output).

### 5. Constraint Implementation Strategy
*   **Strategy:** Architectural (Hard Constraints).
*   **Logic:** Periodic Fourier Features satisfy the periodic BCs exactly. No boundary loss terms are needed.
*   **Active Terms:** Only $L_{pde}$.

### 6. Loss Function Assembly
*   **Symbolic Equation:**
    $$ L_{total} = L_{pde} $$
*   **Components:**
    *   $L_{pde}$: MSE of $R_{pde}$.
    *   $L_{bc}$: Excluded (Satisfied architecturally).

### 6.1 Continuation Strategy
*   None required (Linear PDE, low frequency).

### 7. Weighting Strategy
*   **Local Weights (`lambdas`):** Use **vRBA**.
    *   **Adam Phase:** Use for Weighting & Sampling.
    *   **SSBroyden Phase:** Use for Sampling ONLY.
*   **Global Weights (`lam`):** None (Single loss term).

### 8. Optimization Strategy (Advisor Mandate)
*   **Phase 1: Adam**
    *   Iterations: 5,000.
    *   Learning Rate: Standard decay.
*   **Phase 2: SSBroyden**
    *   **Iterations:** **75,000** (Increased per Advisor instruction to ensure convergence plateau).
    *   **Constraint:** Freeze global weights (if any) and reinitialize Hessian.

### 9. Saving Data
*   Instruct Planner to save `x`, `y`, `u_pred`, `u_exact`, and `error` as `.npz` for visualization.

Main Model:
[MLP with Hybrid Adam and SSBrodyden]

Enhancements:
[Periodic Fourier Features]

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
* **Score Details:** 15
* **Score Optimality:** 15
* **TOTAL SCORE:** 100

**Section 3: Forensic Analysis & Feedback**
* **Observation:**
    1.  **Loss & Error History:** The training history shows two distinct, successful phases. The initial Adam phase (0-5,000 iterations) rapidly reduces the L2 error from ~1.0 to ~1e-2. The subsequent switch to SSBroyden at iteration 5,000 causes a much faster convergence, driving the L2 error down to a noisy plateau around $10^{-8}$ and the total loss to a plateau around $10^{-12}$. The final 50% of the training run shows a stable, flat, albeit noisy, plateau, indicating that the model has fully converged. The final L2 error from the log is an excellent $7.25 \times 10^{-9}$.
    2.  **Residual/Pointwise Error:** The PDE residual heatmap is uniformly distributed with very low magnitude (mostly between $10^{-10}$ and $10^{-8}$), showing no problematic regions. The pointwise error is also extremely low, peaking at ~4e-8, confirming the high accuracy across the entire domain.
    3.  **Solution Fields:** The predicted solution field is visually identical to the reference solution, demonstrating a qualitatively perfect reconstruction.

* **Diagnosis:** Excellent Convergence. The chosen methodology, particularly the use of Periodic Fourier Features to architecturally enforce boundary conditions combined with a hybrid Adam+SSBroyden optimizer and vRBA, is exceptionally well-suited for this problem. The model achieved a high-precision solution, and the training dynamics indicate a stable and complete convergence.

* **Targeted Improvements:**
    * **For Integrity:** None
    * **For Accuracy:** None
    * **For Details:** None
    * **For Optimality:** None. The current configuration is state-of-the-art for this problem class and has achieved a near-optimal result. This run should be considered a final, successful model.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 4 | The implementation uses a PINN architecture consisting of Periodic Fourier Features (degree=2, kmax=2, Lx=2.0 generating separate cos/sin basis for x and y dimensions, concatenated) applied to raw coordinates (x,y in [-1,1]), followed by an MLP main body with structure [30,30,30,1] using tanh activation; raw coordinates are NOT passed to the MLP, only embeddings, enforcing periodic BCs architecturally. Optimizer workflow is Hybrid Adam + SSBroyden: first Adam phase (5000 epochs, batch_size=20000, LR=1e-3 with exponential decay every 5000 steps, vRBA exponential potential with phi=1.0, c_log=1.0, gamma=0.999, eta=0.01, adaptive max_RBA starting at ~1.0 increasing stepwise to cap=20 via step_RBA); then SSBroyden phase (12500 total iterations, Nchange=100 steps per outer iter with resampling using k_samp=1.0, c_samp=0.0, gamma_bfgs=0.999, updating vRBA lambdas each outer step). Loss terms: L_total = L_pde only (MSE on Helmholtz residual R_pde = u_xx + u_yy + k^2 u - g(x,y) with k=1, g=(1-17π²)sin(πx)sin(4πy), computed via double jvp forward-mode AD; weighted by vRBA via local lambdas updated in EMA with importance sampling in both phases). No BC losses or other terms. Curriculum: Adam pretraining followed by SSBroyden refinement with vRBA throughout. | 1.57e-07 | 35 | 35 | 12 | 15 | **97** |
| 5 | The implementation uses an MLP architecture (3 layers, width 30 per layer, tanh activation) preceded by Periodic Fourier Features embedding (degree 2, enforcing periodic BCs; only embeddings passed to MLP, no raw coordinates). Optimizer workflow is Hybrid Adam (5000 iterations with exponential decay LR schedule starting at 1e-3, vRBA for both weighting in loss and importance sampling) followed by SSBroyden (20,000 total iterations, resampling every 100 inner steps using vRBA lambdas for sampling only; global weight lamE=1.0 frozen post-Adam). Loss is L_total = L_pde (MSE on Helmholtz residual R_pde = u_xx + u_yy + k^2 u - g(x,y) with k=1 and exact g; weighted by vRBA lambdas during Adam phase only). Curriculum: Adam stabilization (5000 iters) then SSB refinement (20k iters) with frozen global weights. | 4.56e-08 | 35 | 35 | 8 | 15 | **93** |
| 6 | The implementation uses an MLP architecture (width=30, depth=3, tanh activation) preceded by Periodic Fourier Features embedding (degree=2, generating cos/sin terms for k=1,2 in x and y dimensions, passing only embeddings to MLP, no raw coordinates). Optimizer workflow is Hybrid: Phase 1 Adam for 5000 epochs with vRBA local weights (lambdas) for both weighting in MSE loss (mean((lambdas * residuals)^2)) and importance sampling (exponential potential, updated via EMA); global weight lamE frozen at 1.0. Phase 2 SSBroyden for 30,000 effective iterations (outer loop with Nchange=100 inner steps), using vRBA lambdas for sampling only (MSE on unweighted residuals), Hessian recycling, and frozen global weight. Loss is strictly L_total = L_pde (MSE of Helmholtz residual R_pde = u_xx + u_yy + k^2 u - g(x,y) with k=1, g=(1-17π²)sin(πx)sin(4πy)). Curriculum via adaptive max_RBA increase and gamma scheduling during Adam. | 1.64e-08 | 35 | 35 | 12 | 15 | **97** |
| 8 | The implementation uses a PINN architecture consisting of Periodic Fourier Features (degree=2, kmax=2, Lx=2.0) applied to raw 2D coordinates (x,y), passing only the embeddings (no raw coordinates) to a 3-layer MLP with width=30 per layer and tanh activations, outputting a single scalar field u. Training follows a hybrid optimizer workflow: Phase 1 Adam for 5000 iterations with exponential-decay LR (initial 1e-3), using vRBA for both weighting (EMA-updated lambdas inside PDE loss: mean((new_lambdas * residuals)^2)) and importance sampling (k_samp=1.0); Phase 2 SSBroyden for 50000 total iterations (outer loops of Nchange=100 steps each with maxiter=100), using vRBA-updated lambdas for sampling only on plain MSE PDE residuals (no weighting in loss). Loss is strictly L_total = L_pde (MSE of Helmholtz residual: u_xx + u_yy + k^2 u - g(x,y) with k=1, g=(1-17*pi^2)*sin(pi x) sin(4 pi y)), enforced via jvp-based second derivatives; periodic BCs handled architecturally. Curriculum via adaptive max_RBA ramp-up from eta/(1-gamma) to cap=20. | 7.54e-09 | 35 | 35 | 8 | 15 | **93** |
| 9 | The implementation uses an MLP architecture with Periodic Fourier Features (degree=2) as input embedding enforcing periodic BCs, passing only embeddings to the main body (width=30, depth=3, tanh activation). Optimization follows Hybrid Adam (5000 epochs with exponential decay LR starting at 1e-3) followed by SSBroyden (75,000 iterations with resampling every N_change=100 steps). Loss is strictly L_total = L_pde (MSE of Helmholtz residual R_pde = u_xx + u_yy + k^2 u - g(x,y) with k=1 and exact g matching (1-17π²)sin(πx)sin(4πy); no BC or other terms). vRBA provides local weighting (exponential potential) and importance sampling in Adam phase, sampling only in SSBroyden phase. No continuation/curriculum schedule beyond phases. Analytical solution u_an=sin(πx)sin(4πy) used for validation. | 7.25e-09 | 35 | 35 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 4 | Saves 'validation_data.npz' containing keys: 'u_pred' ((65536,1)), 'u_exact' ((65536,1)), 'x' ((65536,1)), 'y' ((65536,1)) from 256x256 validation grid (flattened). |
| 5 | Saves 'validation_data.npz' containing keys: 'u_pred' (65536,), 'u_exact' (65536,), 'x' (65536,1), 'y' (65536,1) from 256x256 validation grid. |
| 6 | Saves 'validation_data.npz' containing keys: 'u_pred' (65536,), 'u_exact' (65536,), 'x' (65536,1), 'y' (65536,1). |
| 8 | Saves 'validation_data.npz' containing keys: 'u_pred' (65536,), 'u_exact' (65536,), 'error' (65536,), 'x' (65536,1), 'y' (65536,1) from 256x256 validation grid. |
| 9 | Saves 'validation_data.npz' containing keys: 'u_pred' (65536,), 'u_exact' (65536,), 'error' (65536,), 'x' (65536,1), 'y' (65536,1) from 256x256 validation grid. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 4 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Helmholtz_Periodic_SinPiA1SinPiA2/2025_12_05_1404/` |
| 5 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Helmholtz_Periodic_SinPiA1SinPiA2/2025_12_05_1408/` |
| 6 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Helmholtz_Periodic_SinPiA1SinPiA2/2025_12_05_1412/` |
| 8 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Helmholtz_Periodic_SinPiA1SinPiA2/2025_12_05_1417/` |
| 9 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Helmholtz_Periodic_SinPiA1SinPiA2/2025_12_05_1421/` |


---
