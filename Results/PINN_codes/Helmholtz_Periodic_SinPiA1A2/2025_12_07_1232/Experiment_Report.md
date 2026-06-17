
# Experiment Report
*Last Updated: 2025-12-07 12:35:59*

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
*   **User Request:** Solve 2D Helmholtz with Periodic BCs.
*   **Advisor Report:** **REJECTED_FOR_REFINEMENT**. The previous run was excellent (Error $\approx 10^{-8}$) but the loss was still decreasing at the end.
*   **Mandatory Action:** Follow Advisor's recommendation to **increase SSBroyden iterations to 50,000** to find the convergence floor.

**1. Data Source**
*   **Role:** Validation/Error Metrics ONLY.
*   **Action:** No external data loading required. Generate grid for validation against analytical solution.

**2. Problem Statement**
*   **Governing Equation:** 2D Helmholtz Equation.
    $$ r = u_{xx} + u_{yy} + k^2 u - g(x,y) $$
    *   Where $k=1$.
    *   Source term: $g(x,y) = (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$.
*   **Domain:** $(x, y) \in [-1, 1] \times [-1, 1]$.
*   **Boundary Conditions:** Periodic in $x$ and $y$.
*   **Problem Type:** Forward.
*   **Well-Posedness:** $k^2=1$ is not an eigenvalue of the Laplacian on this domain (eigenvalues are $\approx \pi^2(n^2+m^2)$), so the problem admits a unique strong solution.

**2.1 Input**
*   Domain dimension: 2D ($x, y$).

**2.2 Output**
*   Value dimension: Scalar $u(x,y)$.
*   Characteristics: Smooth, oscillatory (wavenumbers 1 and 4).

**2.3 Analytical Solution**
*   $u_{an}(x,y) = \sin(\pi x)\sin(4\pi y)$.

**3. Base Model Selection**
*   **Model:** MLP with Hybrid Adam and SSBroyden.
*   **Reasoning:** Standard 2D problem, previous runs confirmed this architecture works well.

**4. Main Architecture**
*   **Structure:** Input $\rightarrow$ Normalization $\rightarrow$ Embedding $\rightarrow$ Main Body $\rightarrow$ Output.
*   **4.1 Input Transformations:**
    *   **Normalization:** None required (domain is already $[-1, 1]$).
    *   **Feature Embeddings:** **Periodic Fourier Features (Option A)**.
        *   **Degree:** 2 (Sufficient for this frequency).
        *   **Period:** $L=2$ (since domain is $[-1, 1]$).
        *   **Instruction:** Pass **ONLY** the embeddings to the network. Do not pass raw $x, y$. This enforces the periodic constraint architecturally.
*   **4.2 Main Body:**
    *   **Specs:** `width=30`, `depth=3`. (Retain successful configuration).
    *   **Activation:** Tanh.
*   **4.3 Output Transformation:**
    *   None required (Periodicity handled by input embedding).

**5. Constraint Implementation Strategy**
*   **Strategy:** **Architectural (Hard Constraints)** via Periodic Fourier Features.
*   **Loss Terms:**
    *   $L_{bc\_periodic}$: **EXCLUDE**. (Satisfied by architecture).
    *   $L_{bc\_dirichlet}$: None.
    *   $L_{ic}$: None.
    *   **Count:** Only $L_{pde}$ remains.

**6. Loss Function Assembly**
*   **Symbolic Equation:**
    $$ L_{total} = L_{pde} $$
*   **Components:**
    *   $L_{pde}$: MSE of residual $r$.

**6.1 Continuation Strategy**
*   Not required (Linear PDE, low frequency).

**7. Weighting Strategy**
*   **Local Weights (`lambdas`):** Use **vRBA** (Variational Residual-Based Attention).
    *   **Instruction:** "Use `lambdas` for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden."
*   **Global Weights (`lam`):**
    *   Total Terms = 1. No balancing needed.

**Optimization Schedule (Advisor Mandate):**
*   **Phase 1 (Adam):** 5,000 iterations.
*   **Phase 2 (SSBroyden):** **50,000 iterations**. (Increased from 25k as per Advisor).
    *   **Instruction:** Reinitialize Hessian recycling if necessary, but standard template handles this.

**8. Visualization Strategy**
*   Standard 2D plots.
*   **Instruction:** Do not modify visualization code.

**9. Saving Data**
*   Save `u_pred`, `u_exact`, `x`, `y` as `.npz`.

### Architecture & Modules
Main Model:
[MLP with Hybrid Adam and SSBrodyden]

Enhancements:
[Periodic Fourier Features]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
Based on the provided inputs, here is the forensic analysis and decision.

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
    *   **Loss & Error:** The training shows a textbook convergence profile. The Adam phase (0-5k) brings the error down to $\sim 10^{-2}$. The SSBroyden phase (5k+) induces a rapid descent, dropping the error by 5 orders of magnitude.
    *   **Convergence:** Crucially, the loss and error curves **plateau** significantly in the final segment (iterations 30,000 to ~40,000). The Relative L2 Error oscillates stably around a baseline of $10^{-7}$ to $10^{-8}$, indicating the optimization floor has been reached.
    *   **Fields:** The prediction is visually indistinguishable from the reference. The pointwise error map shows unstructured, low-frequency noise with a magnitude of $\sim 5 \times 10^{-8}$, which is effectively machine precision for this context.
    *   **Log Confirmation:** The log reports a minimum achieved error of $6.57 \times 10^{-9}$, confirming exceptional accuracy.

*   **Diagnosis:**
    *   **Perfect Convergence:** The model has successfully solved the 2D Helmholtz equation to a precision far exceeding standard requirements. The "Still Decreasing" issue from the previous run has been resolved; the extended iteration count allowed the model to find the true minimum and stabilize there.

*   **Targeted Improvements:**
    *   **For Integrity:** None.
    *   **For Accuracy:** None. The error is negligible ($10^{-9}$).
    *   **For Details:** None. The plateau is stable.
    *   **For Optimality:** The architecture (Periodic Embeddings + MLP + SSBroyden) is optimal for this problem.

*   **Conclusion:** This run is a definitive success. The error is extremely low, the physics are satisfied, and the training is stable. No further refinement is necessary for this problem configuration.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 9 | The implementation uses an MLP architecture (width=30, depth=3, tanh activation) with Periodic Fourier Features embeddings (degree=2, Lx=2.0 for periodicity over domain length 2), applied separately to x and y coordinates before concatenation and passing exclusively to the MLP body (no raw inputs). Optimizer workflow is Hybrid Adam + SSBroyden: Adam phase (5000 epochs, batch_size=20000, initial LR=1e-3 with exponential decay every 5000 steps at rate 0.9) incorporates vRBA for local weighting (exponential potential, phi=1.0, c_log=1.0, EMA with gamma=0.999, eta=0.01) inside the PDE loss and importance sampling (k_samp=1.0, c_samp=0.0); adaptive RBA cap increases from initial max_RBA0 (~1.0) to 20 over training via step_RBA scheduling. SSBroyden phase (12500 total iterations in outer loops of N_change=100 steps each) uses vRBA-updated lambdas for importance sampling only (k_samp_bfgs=1.0, c_samp_bfgs=0.0, gamma_bfgs=0.999). Loss is strictly L_total = L_pde (Helmholtz residual: u_xx + u_yy + k^2 u - g with k=1, exact g=(1-17π²)sin(πx)sin(4πy)), weighted by vRBA lambdas during Adam; no boundary losses (enforced architecturally). | 1.00e-07 | 35 | 35 | 15 | 15 | **100** |
| 10 | The implementation uses an MLP architecture (depth=3, width=30 per layer, tanh activation) preceded by Periodic Fourier Features embedding (degree=2, period L=2) applied separately to x and y coordinates, with only the concatenated embeddings passed to the MLP body; no raw coordinates are used. The optimizer follows a hybrid workflow: Phase 1 Adam for 5000 iterations with vRBA weighting and importance sampling (exponential potential, adaptive gamma via increasing max_RBA from initial eta/(1-gamma) up to cap=20, lr decay), followed by Phase 2 SSBroyden for 25000 iterations (outer loops with Nchange=100 inner steps per resample, vRBA for sampling only, Hessian recycling with cholesky stabilization). The loss is strictly L_total = L_pde (MSE of Helmholtz residual r = u_xx + u_yy + u - g with g=(1-17π²)sin(πx)sin(4πy)), augmented by vRBA lambdas during Adam and used for sampling in SSBroyden; no boundary losses or extra terms. Curriculum involves Adam with self-scaling RBA cap increase every 50000 effective steps and SSBroyden resampling every 100 steps. | 1.30e-08 | 35 | 35 | 12 | 15 | **97** |
| 11 | The code implements an MLP architecture (depth=3, width=30 per layer, tanh activation) with Periodic Fourier Features embedding (degree=2, period L=2) applied separately to x and y coordinates (concatenated, no raw inputs passed to MLP). Optimizer workflow is Hybrid: Adam phase for 5000 iterations with vRBA weighting in loss (lambdas * residuals^2) and importance sampling (batch_size=20000), followed by SSBroyden phase for 50000 iterations with vRBA for sampling only (resampling every Nchange=100 steps, unweighted MSE residual loss). Loss terms include only L_pde (MSE of Helmholtz residual: uxx + uyy + k^2 u - g with k=1, g=(1-17π²)sin(πx)sin(4πy)). Curriculum schedule: Adam (5000 iters, exponential LR decay), SSBroyden (50k total iters with Hessian recycling). Analytical solution u=sin(πx)sin(4πy) matches exactly. | 6.57e-09 | 35 | 35 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 9 | Saves 'helmholtz_results.npz' containing keys: 'model_name' (string), 'all_errors' (N_epochs x 1), 'all_its' (N_epochs x 1), 'all_loss' (N_epochs x 1), 'all_gamma' (N_adam x 1), 'all_max_RBA' (N_adam x 1), 'all_L_infty' (N_total x 1), 'all_L_2' (N_total x 1), 'u_pred' (100x100), 'u_exact' (100x100), 'X' (100x100), 'Y' (100x100). |
| 10 | Saves 'helmholtz_results.npz' in npz format containing keys: 'model_name' (string), 'all_errors' (Nx1), 'all_its' (Nx1), 'all_loss' (Nx1), 'all_gamma' (Nx1), 'all_max_RBA' (Nx1), 'all_L_infty' (Nx1), 'all_L_2' (Nx1), 'u_pred' (100x100), 'u_exact' (100x100), 'X' (100x100), 'Y' (100x100). |
| 11 | Saves 'helmholtz_results.npz' containing keys: 'model_name' (string), 'all_errors' (Nx1), 'all_its' (Nx1), 'all_loss' (Nx1), 'all_gamma' (Nx1), 'all_max_RBA' (Nx1), 'all_L_infty' (Nx1), 'all_L_2' (Nx1), 'u_pred' (100x100), 'u_exact' (100x100), 'X' (100x100), 'Y' (100x100). Validation grids derived from NC_TEST=100 linspace points. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 9 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Helmholtz_Periodic_SinPiA1A2/2025_12_07_1224/` |
| 10 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Helmholtz_Periodic_SinPiA1A2/2025_12_07_1228/` |
| 11 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Helmholtz_Periodic_SinPiA1A2/2025_12_07_1232/` |


---
