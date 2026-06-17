
# Experiment Report
*Last Updated: 2025-12-07 17:12:56*

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
*   **Strategy:** Standard Forward PINN with Periodic Architecture.
*   **Role:** Validation/Error Metrics ONLY (Method of Manufactured Solutions).

**1. Data Source**
*   **Type:** Synthetic/Analytical.
*   **Instruction:** Generate a grid of collocation points in $(x, y) \in [-1, 1] \times [-1, 1]$.
*   **Validation Data:** Generate exact solution $u_{an} = \sin(\pi x)\sin(4\pi y)$ on a dense grid for error computation.

**2. Problem Statement**
*   **Governing Equation:** 2D Helmholtz Equation.
    *   Residual: $r = u_{xx} + u_{yy} + k^2 u - g(x, y)$
    *   Constants: $k=1$.
    *   Source Term: $g(x, y) = (1 - 17\pi^2) \sin(\pi x) \sin(4\pi y)$.
        *   *Derivation:* $\Delta u_{an} = -(\pi^2 + 16\pi^2)u_{an} = -17\pi^2 u_{an}$. Thus LHS is $(-17\pi^2 + 1)u_{an}$.
*   **BCs:** Periodic in $x$ and $y$.
*   **Analytical Solution:** $u(x, y) = \sin(\pi x)\sin(4\pi y)$.
*   **Problem Type:** Forward.
*   **Well-Posedness:** $k^2=1$ is not an eigenvalue of the Laplacian on this domain (lowest non-zero eigenvalue $\approx 9.8$), so the problem is well-posed.

**2.1 Input**
*   Domain: 2D $(x, y)$.

**2.2 Output**
*   Value: Scalar $u$.
*   Characteristics: Smooth, oscillatory, periodic.

**3. Base Model Selection**
*   **Model:** MLP with Hybrid Adam and SSBroyden. (Case A: 2D).

**4. Main Architecture**
*   **Structure:** Input $\to$ Normalization $\to$ Embedding $\to$ Main Body $\to$ Output.
*   **4.1 Input Transformations:**
    *   **Normalization:** Normalize $(x, y)$ from $[-1, 1]$ to $[0, 1]$ **inside the model** before passing to embeddings. This ensures the domain matches the standard period of the embedding.
    *   **Feature Embeddings:** **Periodic Fourier Features** (Mandatory).
        *   **Degree:** Start with `degree=2`.
        *   **Instruction:** Pass *only* the embedded features to the Main Body.
*   **4.2 Main Body:**
    *   **Specs:** `width=30`, `depth=3`.
*   **4.3 Output Transformation:**
    *   Direct output (Periodicity is handled by input embedding).

**5. Constraint Implementation Strategy**
*   **Periodic BCs:** Enforced architecturally via **Periodic Fourier Features**.
*   **Dirichlet/Neumann:** None.
*   **Constraint Count:** 1 ($L_{pde}$).
*   **Decision:** Loss-based (Trivial).

**6. Loss Function Assembly**
*   **Symbolic Equation:** $$ L_{total} = L_{pde} $$
*   **Notes:**
    *   $L_{bc\_periodic}$ is **EXCLUDED** (Redundant due to architecture).
    *   $L_{data}$ is **EXCLUDED** (Forward problem).

**6.1 Continuation Strategy**
*   Not required ($k=1$ is small).

**7. Weighting Strategy**
*   **Global Weights (`lam`):** None (Single loss term).
*   **Local Weights (`lambdas`):** Use **vRBA** for $L_{pde}$.
    *   *Instruction:* "lambdas are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden."

**8. Visualization Strategy**
*   Standard 2D plots.
*   Include Error Map $|u_{pred} - u_{an}|$.

**9. Saving Data**
*   Save `u_pred` and `u_exact` as .npz.

### Main Model
[MLP with Hybrid Adam and SSBrodyden]

### Enhancements
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
    *   **Loss History:** The total loss drops smoothly from $10^2$ to $10^{-10}$. The final segment (SSBroyden phase, iterations 5000-14500) shows a consistent linear descent on the log scale, indicating the optimizer was still finding better weights to minimize the residual.
    *   **Error History:** The Relative L2 Error drops precipitously during the SSBroyden phase, reaching a floor around $10^{-7}$. While the Loss continues to drop, the Error curve exhibits noise/oscillations in the final 20% of the run (bouncing between $1e-7$ and $3e-7$), suggesting the model has reached the accuracy limit imposed by the architecture or floating-point precision, even if the residual can be pushed lower.
    *   **Visuals:** The Prediction is visually indistinguishable from the Reference. The Error Map shows unstructured high-frequency noise at a magnitude of $10^{-7}$, confirming no structural bias remains.
    *   **Log Confirmation:** The log reports a "Minimum achieved RL2 error: 0.00000008" ($8 \times 10^{-8}$), which is an exceptional result.

*   **Diagnosis:**
    *   **Perfect Convergence:** The combination of Periodic Fourier Features (correctly handling BCs) and the Hybrid Adam+SSBroyden optimizer worked perfectly.
    *   **Saturation:** The divergence between the continuing drop in Loss (PDE Residual) and the plateauing Error suggests the model has saturated its capacity or the precision limits. Further iterations would likely reduce the residual without significantly improving the L2 error.

*   **Targeted Improvements:**
    *   **For Integrity:** None.
    *   **For Accuracy:** None. The error ($8e-8$) is effectively zero for this problem class.
    *   **For Details:** None.
    *   **For Optimality:** The methodology (Periodic Embeddings + SSBroyden + vRBA) is optimal for this problem.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | The implementation uses an MLP architecture (PINN class) with input normalization from [-1,1] to [0,1] inside the forward method, followed by separate Periodic Fourier Features embeddings (degree=2, kmax=2, Lx=1.0 for normalized domain) for x and y, concatenated and passed to the main body MLP ([30,30,30,1], depth=3 hidden layers, tanh activation). Optimizer workflow is Hybrid Adam + SSBroyden: Adam phase (5000 epochs, batch_size=20000, LR=1e-3 with exponential decay every 5000 steps, vRBA with exponential potential, phi=1.0, c_log=1.0, gamma=0.999 initially adapting up to cap_RBA=20 via self-scaling) uses vRBA for both weighting (in loss: mean((lambdas * residuals)^2)) and importance sampling (k_samp=1.0); SSBroyden phase (12500 total iterations in chunks of N_change=100, gamma_bfgs=0.999, k_samp_bfgs=1.0, c_samp_bfgs=0.0) uses vRBA lambdas for sampling only with unweighted mean(residuals^2) loss. Loss is strictly L_total = L_pde (Helmholtz residual: uxx + uyy + k^2 u - g with k=1, g=(1-17π²)sin(πx)sin(4πy)), no BC or data terms. Curriculum via adaptive max_RBA increase during Adam. | 8.00e-08 | 35 | 35 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves 'results.npz' containing keys: 'u_pred' (256x256), 'u_exact' (256x256). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Helmholtz_Periodic_SinPiA1SinPiA2/2025_12_07_1709/` |


---
