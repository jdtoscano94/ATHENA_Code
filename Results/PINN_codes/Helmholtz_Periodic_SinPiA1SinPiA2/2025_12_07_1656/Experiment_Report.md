
# Experiment Report
*Last Updated: 2025-12-07 16:59:59*

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
*   **Advisor Report:** The previous run was highly accurate ($2 \times 10^{-8}$) but rejected because the loss was still decreasing.
*   **Mandatory Action:** The Advisor explicitly requests to **increase iterations** for the final stage to ensure convergence plateau. I will increase SSBroyden iterations from 25,000 to 50,000.

**1. Data Source**
*   **Role:** Validation/Error Metrics ONLY.
*   **Action:** No external data loading. Generate collocation points internally.

**2. Problem Statement**
*   **Governing Equation:** 2D Helmholtz Equation.
    *   Residual: $r = u_{xx} + u_{yy} + k^2 u - g(x,y)$
    *   Constants: $k=1$.
    *   Source Term: $g(x,y) = (1 - 17\pi^2) \sin(\pi x) \sin(4\pi y)$.
*   **Domain:** $(x, y) \in [-1, 1] \times [-1, 1]$.
*   **Boundary Conditions:** Periodic in $x$ and $y$.
*   **Problem Type:** Forward. Well-posed for PINNs.
*   **Analytical Solution:** $u(x,y) = \sin(\pi x) \sin(4\pi y)$.

**3. Base Model Selection**
*   **Model:** MLP with Hybrid Adam and SSBroyden.
*   **Reasoning:** Standard 2D problem.

**4. Main Architecture**
*   **Structure:** Input ($x,y$) -> Normalization -> **Periodic Fourier Features** -> MLP -> Output ($u$).
*   **Input Transformation:**
    *   **Normalization:** Normalize inputs to $[-1, 1]$ (already in range, but good practice).
    *   **Embedding (Mandatory):** **Periodic Fourier Features**.
        *   **Configuration:** `degree=2`.
        *   **Periodicity:** The domain length is $L=2$. The fundamental frequency is $\pi$.
        *   **Constraint:** Pass **ONLY** the embeddings to the MLP. Do **NOT** pass raw $x, y$. This strictly enforces periodic BCs.
*   **Main Body:**
    *   **Specs:** `width=30`, `depth=3` (Case A: Low Dim).
    *   **Activation:** Tanh or Sin.
*   **Output Transformation:** None required (Periodicity handled by input embedding).

**5. Constraint Implementation Strategy**
*   **Strategy:** Architectural (Hard Constraints via Embeddings).
*   **BC Handling:** Since Periodic Fourier Features are used, the output is automatically periodic.
*   **Loss Terms:**
    *   $L_{bc\_periodic}$: **EXCLUDE** (Redundant).
    *   $L_{bc\_dirichlet}$: None.
    *   $L_{ic}$: None.
    *   **Active Terms:** Only $L_{pde}$.

**6. Loss Function Assembly**
*   **Symbolic Equation:** $L_{total} = L_{pde}$
*   **Continuation:** Not required ($k=1$ is low frequency).

**7. Weighting Strategy**
*   **Global Weights:** None (Only 1 term).
*   **Local Weights:** Use **vRBA**.
    *   **Instruction:** "`lambdas` are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden."

**8. Optimization Strategy (Advisor Correction)**
*   **Phase 1 (Adam):** 5,000 iterations.
*   **Phase 2 (SSBroyden):** **50,000 iterations**. (Increased from 25k per Advisor request to ensure convergence plateau).
    *   **Constraint:** Do not modify SSBroyden internal logic.

**9. Saving Data**
*   Save `x`, `y`, `u_pred`, `u_exact`, `error` as `.npz`.

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
    *   **Loss & Error History:** The Total Loss (blue curve) shows a classic two-stage convergence. The Adam phase (first 5k iterations) reduces loss to $\sim 10^{-2}$. The SSBroyden phase triggers a sharp, rapid descent, driving the loss down to $\sim 10^{-12}$. Crucially, unlike the previous run, the final segment (iterations 25k to 36k) shows a distinct **plateau**. The curve flattens out, indicating the optimizer has reached the machine precision/noise floor limit.
    *   **Error Convergence:** The Relative L2 Error (green curve) drops to a minimum of $\sim 10^{-8}$ around iteration 20,000 and remains stable (oscillating slightly within the same order of magnitude) until the end. It is no longer strictly decreasing, confirming convergence.
    *   **Solution Fields:** The predicted solution is visually indistinguishable from the reference. The periodic boundaries are perfectly respected.
    *   **Residuals:** The PDE residual heatmap shows unstructured noise at a magnitude of $10^{-6}$ to $10^{-10}$, which is excellent.
    *   **Pointwise Error:** The error distribution is smooth and low-magnitude ($2-3 \times 10^{-8}$), with no specific boundary artifacts.

*   **Diagnosis:**
    *   **Successful Convergence:** The strategy to increase SSBroyden iterations was successful. The model exhausted its capacity and reached a stable noise floor ($10^{-8}$ error), resolving the "still decreasing" issue from the previous run.
    *   **Optimal Configuration:** The combination of Periodic Fourier Features (enforcing BCs exactly) and vRBA + SSBroyden has solved the problem to near machine precision.

*   **Targeted Improvements:**
    *   **For Integrity:** None.
    *   **For Accuracy:** None. The error is $1 \times 10^{-8}$, far surpassing standard targets.
    *   **For Details:** None. The convergence tail is stable.
    *   **For Optimality:** None. The architecture and optimization schedule are optimal for this problem type.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 9 | The implementation uses an MLP architecture (width=30, depth=3 hidden layers, output=1) with Periodic Fourier Features embedding (kmax=2, Lx=2.0 ensuring periodicity over domain length 2 with fundamental frequency π; only embeddings passed to MLP, no raw coordinates). Optimizer workflow is Hybrid Adam (5000 epochs, exponential decay LR from 1e-3, batch_size=20000) followed by SSBroyden (12500 total iterations in blocks of N_change=100, BFGS method with Hessian recycling). Loss terms: L_total = L_pde only (Helmholtz residual u_xx + u_yy + k^2 u - g with k=1 and exact g=(1-17π²)sin(πx)sin(4πy)); during Adam, vRBA provides local weighting lambdas (exponential potential, EMA update with γ=0.999, η=0.01, φ=1.0) and importance sampling (k_samp=1.0); during SSBroyden, vRBA lambdas used for sampling only (no weighting in loss=mean(res^2)). No boundary losses (periodicity enforced architecturally). No curriculum/continuation (k=1 low). Analytical solution u=sin(πx)sin(4πy) correctly implemented. | 1.10e-07 | 35 | 35 | 12 | 15 | **97** |
| 12 | Architecture is PINN with MLP (depth=3, width=30 per layer, output=1) preceded by differentiable normalization and Periodic Fourier Features (kmax=2, Lx=2.0); only embeddings are passed to the MLP body, enforcing periodic BCs. Optimizer workflow: Phase 1 Adam (5000 iterations) with vRBA for local weighting (exponential potential, phi=1.0, c_log=1.0) inside the loss and importance sampling (k_samp=1.0, c_samp=0.0); Phase 2 SSBroyden (25,000 total iterations in blocks of N_change=100) with vRBA updated on full collocation points and used for sampling only (no weighting in loss). Loss terms: L_total = L_pde only (Helmholtz residual r = u_xx + u_yy + k^2 u - g with k=1, g=(1-17π²)sin(πx)sin(4πy)); no BC or data losses. No curriculum training parameters specified. | 2.00e-08 | 35 | 35 | 12 | 15 | **97** |
| 13 | The implementation uses an MLP architecture (width=30, depth=3, tanh activation) with input processing via normalization to [-1,1] followed by Periodic Fourier Features embedding (degree=2, Lx=2.0), passing only embeddings to the MLP to enforce periodic BCs, outputting u. Optimizer workflow is Hybrid: Phase 1 Adam for 5000 iterations with vRBA for both weighting in the loss (L_pde = mean((lambdas * residuals)^2)) and importance sampling (batch_size=20000); Phase 2 SSBroyden for 50,000 effective iterations (outer loops with Nchange=100 inner steps per minimize call, recycling Hessian), using vRBA for sampling only. Loss is strictly L_total = L_pde (Helmholtz residual: u_xx + u_yy + k^2 u - g with k=1, g=(1-17π²)sin(πx)sin(4πy)), no BC/IC terms. No explicit curriculum schedule beyond vRBA adaptation and self-scaling max_RBA. | 1.00e-08 | 35 | 35 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 9 | Saves 'validation_data.npz' containing keys: 'u' (256x256), 'u_exact' (256x256). |
| 12 | Saves 'validation_data.npz' containing keys: 'u_pred' (256x256), 'u_exact' (256x256), 'x' (256x256), 'y' (256x256), 'error' (256x256). |
| 13 | Saves 'validation_data.npz' containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256), 'error' (256x256). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 9 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Helmholtz_Periodic_SinPiA1SinPiA2/2025_12_07_1645/` |
| 12 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Helmholtz_Periodic_SinPiA1SinPiA2/2025_12_07_1652/` |
| 13 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Helmholtz_Periodic_SinPiA1SinPiA2/2025_12_07_1656/` |


---
