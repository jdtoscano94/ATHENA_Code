
# Experiment Report
*Last Updated: 2025-12-05 12:12:10*

## User Request

Now lets solve the Allen Cahn equation in particular:
u_t-epsilon u_xx+k(u^3-u)=0

u(x,0)=x^2cos(pi x)

The domain in x goes from -1 to 1 and in t from 0 to 1 with periodic boundary conditions in x. Here epsilon=0.0001 and k=5

### Reference Data: 
You can find the data here:
/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Documents/codes/Data/allen_cahn.mat

The solutions are saved as a dictionary with keys 't', 'x' and 'usol'.


---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
### 0. Triage - Determine Strategy
*   **Role:** Validation/Error Metrics ONLY.
*   **Strategy:** Path 1 (User Request) & Path 2 (Critic Correction).
*   **Advisor Report:** None.
*   **Critic Report:** **ACCEPTED**. (Implementing strict discrete continuation).

### 1. Data Source
*   **Path:** `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Documents/codes/Data/allen_cahn.mat`
*   **Format:** `.mat` file. Keys: `t`, `x`, `usol`.
*   **Instruction:** Load this data for validation and error plotting only.

### 2. Problem Statement
*   **Governing Equation:** Allen-Cahn Equation.
    $$ \text{Residual} = u_t - \epsilon u_{xx} + k(u^3 - u) $$
    *   Where $\epsilon$ is the diffusion parameter (Target: $0.0001$) and $k=5$.
*   **BCs/ICs:**
    *   **IC:** $u(x,0) = x^2 \cos(\pi x)$.
    *   **BCs:** Periodic in $x$ ($u(-1,t)=u(1,t)$ and $u_x(-1,t)=u_x(1,t)$).
*   **Problem Type:** Forward.
*   **Well-Posedness:** Well-posed for PINNs (Strong form).
*   **Input:** Domain $x \in [-1, 1]$, $t \in [0, 1]$. ($d=2$).
*   **Output:** Scalar $u$. ($q=1$).
*   **Characteristics:** Stiff reaction-diffusion system. Small $\epsilon$ leads to sharp interfaces (shocks).

### 3. Base Model Selection
*   **Model:** MLP with Hybrid Adam and SSBroyden.
*   **Reasoning:** 1D+t problem.

### 4. Main Architecture
**Structure:** Input $\to$ Input Transformation $\to$ Main Body $\to$ Output Transformation $\to$ Output.

**4.1. Input Transformations**
*   **Normalization:** Normalize $t$ to $[0, 1]$. Normalize $x$ to $[-1, 1]$ (already in range).
*   **Feature Embeddings:** **Option A: Periodic Fourier Features**.
    *   **Variable:** Apply to $x$ **ONLY**.
    *   **Specs:** `degree=2` (Strict start).
    *   **Instruction:** Pass *only* the embedded $x$ and normalized $t$ to the Main Body.

**4.2. Main Body Architecture**
*   **Specs:** Case A (1D+t).
    *   `width=30`
    *   `depth=3`
*   **Optimizer:** Hybrid Adam + SSBroyden.

**4.3. Output Transformation**
*   **Ansatz:** None (Direct output).
*   **Reasoning:** Periodic BCs are handled by Input Embeddings. IC is handled by Soft Loss.

### 5. Constraint Implementation Strategy
*   **Count:** $L_{pde}$ (1) + $L_{ic}$ (1) = 2. (Periodic BC is architectural).
*   **Decision:** **Loss-Based** for IC. Architectural for BC.

### 6. Loss Function Assembly
$$ L_{total} = L_{pde} + L_{ic} $$

*   **L_pde:** Residual with variable $\epsilon$.
*   **L_ic:** MSE against $u(x,0) = x^2 \cos(\pi x)$.
*   **L_bc_periodic:** **EXCLUDED** (Satisfied by architecture).

**6.1. Continuation Strategy (Discrete Viscosity Ladder)**
*   **Context:** $\epsilon=0.0001$ is stiff. We must anneal $\epsilon$ using discrete, fixed steps.
*   **Phase 1 (Adam - Stabilization):**
    *   **Parameter:** Fixed $\epsilon = 0.1$.
    *   **Duration:** 10,000 iterations.
*   **Phase 2 (SSBroyden - Discrete Continuation):**
    *   **Strategy:** Execute a loop over the schedule below. **Reinitialize Hessian** at the start of each step.
    *   **Schedule (Epsilon, Iterations):**
        1.  $\epsilon=0.05$, Iters=5,000
        2.  $\epsilon=0.01$, Iters=5,000
        3.  $\epsilon=0.005$, Iters=10,000
        4.  $\epsilon=0.001$, Iters=15,000
        5.  $\epsilon=0.0005$, Iters=20,000
        6.  **Target:** $\epsilon=0.0001$, **Iters=25,000**
    *   **Constraint:** $\epsilon$ is **CONSTANT** within each training block. No dynamic ramping inside the optimizer loop.

### 7. Weighting Strategy
*   **Local Weights (`lambdas`):** Use **vRBA** for $L_{pde}$.
*   **Global Weights (`lam`):**
    *   **Total Terms:** 2 ($L_{pde}, L_{ic}$).
    *   **Strategy:** **Self-Balancing** (GradNorm/RelLoss).
    *   **Phase 1 (Adam):** Dynamic updates.
    *   **Phase 2 (SSBroyden):** **FREEZE** weights at final Adam values.

### 8. Visualization Strategy
*   Standard 1D+t plots.
*   **Instruction:** Ensure validation data (`usol`) is overlaid on the prediction plots for error assessment.

### 9. Saving Data
*   Save `u_pred`, `x`, `t` as `.npz`.

### Main Model:
[MLP with Hybrid Adam and SSBrodyden]

### Enhancements:
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
    *   **Loss & Error History:** The training history clearly shows the discrete continuation strategy. The initial Adam phase (0-10k iterations) smoothly reduces the error. Subsequent stages, handled by SSBroyden, begin with sharp increases in loss as the physics parameter `epsilon` is made more challenging, followed by rapid convergence. The final stage (iterations ~40,000 to ~65,000) shows the Relative L2 Error converging and achieving a stable, flat plateau at an exceptionally low value of approximately $2 \times 10^{-6}$.
    *   **Solution Fields:** The predicted solution field is visually identical to the reference solution, perfectly capturing the phase separation dynamics characteristic of the Allen-Cahn equation.
    *   **Residual & Error Heatmaps:** The pointwise error is concentrated at the initial condition ($t=0$), which is expected, but remains low (order $10^{-4}$). The PDE residual is uniformly low across the entire domain, mostly below $10^{-7}$.
    *   **Log Confirmation:** The final `Minimum achieved RL2 error` from the log is $2.067 \times 10^{-6}$, which perfectly matches the final plateau observed in the error history plot.

* **Diagnosis:** Excellent Convergence via Discrete Continuation. The chosen strategy of gradually decreasing the diffusion term `epsilon` in discrete stages was perfectly executed and highly effective for this stiff problem. The combination of Periodic Fourier Features, a hybrid Adam+SSBroyden optimizer, and vRBA weighting represents a state-of-the-art approach that led to a highly accurate and stable solution.

* **Targeted Improvements:**
    *   **For Integrity:** None.
    *   **For Accuracy:** None. The model has achieved an outstandingly low error.
    *   **For Details:** None. The final training stage shows a perfect convergence plateau.
    *   **For Optimality:** None. The methodology employed is optimal for this class of problem. This run serves as an excellent baseline for success.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 6 | The implementation uses an MLP architecture with 3 hidden layers of width 30, output features 1, incorporating Periodic Fourier Features (degree=2) applied strictly to x only, concatenated with normalized t (already in [0,1]) before the main body; no output ansatz. Optimizer workflow is Hybrid: Phase 1 Adam for 10,000 iterations at fixed ε=0.1 with vRBA local weighting on L_pde via importance sampling and self-balancing global weights (GradNorm via EMA on gradient norms, lamE for PDE, dynamic lamB for IC); Phase 2 SSBroyden follows discrete continuation schedule (ε=0.05/5k, 0.01/5k, 0.005/10k, 0.001/15k, 0.0005/20k, 0.0001/25k iters), reinitializing Hessian each stage, freezing global weights at Adam final values, using vRBA-frozen lambdas for importance sampling in batches. Loss terms: L_total = L_pde (MSE on Allen-Cahn residual u_t - ε u_xx + 5(u^3 - u)) + L_ic (MSE on IC u(x,0)=x^2 cos(πx)); periodic BCs architectural via embeddings; no extra terms. | 2.07e-06 | 35 | 35 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 6 | Saves 'results.npz' containing keys: 'u_pred' (grid shape, e.g., len(t0) x len(x0) from meshgrid), 'x' (N,), 't' (N,) where N = len(t0)*len(x0) from flattened meshgrid of validation data t0, x0. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 6 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//AllenCahn_Periodic_X2CosPiX/2025_12_05_1208/` |


---
