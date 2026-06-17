
# Experiment Report
*Last Updated: 2025-12-05 14:20:35*

## User Request

Lets solve the viscous burgers equation u_t+uu_x=nu u_{xx} with IC=-sin(pi x) the domain goes from 0 to 1 in time and from -1 to 1 in x; periodic boundary conditions.

Here $nu=1/(100*pi)$

### Reference Data: 
You can find the data here:
 /users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Documents/codes/Data/burgers.mat
The solutions are saved as a dictionary with keys 't', 'x' and 'usol'.


---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
### 0. Triage - Determine Strategy
*   **Role:** Validation/Error Metrics ONLY.
*   **Strategy:** **Path 2 (Advisor Recommendation)**.
    *   **Diagnosis:** Incomplete Convergence. The error curve is still decreasing at the end of the previous run (approx 60k total iterations).
    *   **Cure:** Increase SSBroyden iterations significantly to ensure the loss plateaus.
    *   **Action:** Increase SSBroyden iterations to **100,000**.

### 1. Data Source
*   **Path:** `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Documents/codes/Data/burgers.mat`
*   **Format:** Dictionary with keys `t`, `x`, `usol`.
*   **Instruction:** Load this data for validation and error plotting only.

### 2. Problem Statement
*   **Governing Equation:** Viscous Burgers' Equation.
    $$ r = u_t + u \cdot u_x - \nu \cdot u_{xx} $$
    where $\nu = \frac{1}{100\pi}$.
*   **Domain:** $t \in [0, 1]$, $x \in [-1, 1]$.
*   **Initial Condition:** $u(0, x) = -\sin(\pi x)$.
*   **Boundary Conditions:** Periodic on $x$ ($u(t, -1) = u(t, 1)$).
*   **Problem Type:** Forward.
*   **Well-Posedness:** Yes, admits strong solutions.
*   **Analytical Solution:** Reference solution provided in `.mat` file.

### 2.1 Input
*   **Domain:** 1D space ($x$) + Time ($t$). Total dim = 2.

### 2.2 Output
*   **Value:** Scalar $u(t,x)$.
*   **Characteristics:** Shock formation, smooths over time due to viscosity.

### 3. Base Model Selection
*   **Model:** MLP with Hybrid Adam and SSBroyden. (Case A: Low Dim).

### 4. Main Architecture
**Structure:** Input $\to$ Input Transformation $\to$ Main Body $\to$ Output.

**4.1 Input Transformations**
*   **Normalization:**
    *   Normalize $t$ to $[-1, 1]$ (using domain $[0, 1]$).
    *   **CRITICAL:** Apply normalization *inside* the model's `forward` method.
*   **Feature Embeddings (Option A: Periodic Fourier Features):**
    *   **Variable:** Apply **ONLY** to $x$.
    *   **Specs:** `degree=2`, `period=2.0` (since $x \in [-1, 1]$).
    *   **Composition Rule:** Pass `[t_normalized, x_embedded]` to the Main Body. **DO NOT** pass raw $x$.

**4.2 Main Body Architecture**
*   **Type:** Base MLP.
*   **Specs:** `width=30`, `depth=3`.
*   **Activation:** Tanh.

**4.3 Output Transformation**
*   **None.** Direct output from MLP.

### 5. Constraint Implementation Strategy
*   **Periodic BCs:** Satisfied exactly by **Periodic Fourier Features** (Step 4.1). No loss term needed.
*   **IC:** Soft Loss ($L_{ic}$).
*   **Dirichlet/Neumann:** None.
*   **Count:** $L_{pde} + L_{ic} = 2$ terms.
*   **Strategy:** Loss-Based.

### 6. Loss Function Assembly
*   **Symbolic Formula:**
    $$ L_{total} = L_{pde} + L_{ic} $$
*   **Terms:**
    *   $L_{pde}$: MSE of Residual $r$ (defined in Sec 2).
    *   $L_{ic}$: MSE of $(u_{pred}(0, x) - (-\sin(\pi x)))$.

### 6.1 Continuation Strategy
*   **Not required.** $\nu \approx 0.003$ is not stiff enough to require continuation. Train directly.

### 7. Weighting Strategy
*   **Local Weights (`lambdas`):**
    *   Use **vRBA** (Variational Residual-Based Attention) for $L_{pde}$.
    *   **Instruction:** Used for Weighting & Sampling during Adam; Sampling ONLY during SSBroyden.
*   **Global Weights (`lam`):**
    *   **Strategy:** **Self-Balancing Strategy** (Total Terms = 2).
    *   **Phase 1 (Adam):** Update `lam_pde` and `lam_ic` dynamically (GradNorm/RelLoss).
    *   **Phase 2 (SSBroyden):** **FREEZE** global weights at their final Adam values.

### 8. Optimization Schedule
*   **Phase 1: Adam**
    *   Iterations: 10,000.
    *   State: Dynamic Weights.
*   **Phase 2: SSBroyden (Targeted Improvement)**
    *   Iterations: **100,000**. (Increased to ensure plateau).
    *   State: Frozen Global Weights. Hessian initialized to Identity.

### 9. Saving Data
*   Save `t`, `x`, `u_pred` as `.npz` for visualization.

### Main Model
[MLP with Hybrid Adam and SSBrodyden]

### Enhancements
[Periodic Fourier Features]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
APPROVED

**Section 1: Status Decision**
*   **Status:** `APPROVED`

**Section 2: Quantitative Scorecard**
*   **Score Integrity:** 35
*   **Score Accuracy:** 35
*   **Score Details:** 15
*   **Score Optimality:** 15
*   **TOTAL SCORE:** 100

**DECISION LOGIC:**
The current score of 100 is excellent and the loss/error curves have clearly plateaued in the final stage. The trial history shows multiple prior successful runs (Score > 90), indicating that the exploration phase for this problem is mature. This run represents a state-of-the-art result, and further exploration is unnecessary.

**Section 3: Forensic Analysis & Feedback**
*   **Observation:**
    1.  **Loss & Error History:** The training exhibits a textbook two-phase optimization. The first 10,000 iterations (Adam) rapidly decrease the loss and error by several orders of magnitude. The second phase (SSBroyden, from 10k iterations onward) achieves a much lower error floor, with the Relative L2 Error dropping from ~1e-2 to a stable plateau around 1.5e-8. The final segment of the training run is perfectly flat, indicating complete convergence.
    2.  **Residual/Pointwise Error:** The PDE residual and pointwise error plots show extremely low values across the entire domain, with the largest (but still very small) errors concentrated around the shock front at x=0, as is physically expected.
    3.  **Solution Fields:** The predicted solution is visually identical to the reference solution. The shock formation, its magnitude, and its dissipation over time are all captured perfectly.

*   **Diagnosis:** **Excellent Convergence and High Accuracy.** The chosen methodology is perfectly suited for this problem. The combination of architectural enforcement of periodic boundary conditions (Fourier Features), adaptive weighting/sampling (vRBA), and a powerful second-order optimizer (SSBroyden) has resulted in a highly accurate and stable solution.

*   **Targeted Improvements:**
    *   **For Integrity:** None
    *   **For Accuracy:** None. The achieved L2 error of 1.54e-8 is exceptional.
    *   **For Details:** None. The convergence is ideal.
    *   **For Optimality:** None. The methodology represents a best-in-class approach for this type of problem (2D, periodic, smooth solution with a shock).

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 5 | The implementation uses an MLP architecture (PINN class) with input normalization for t mapped from [0, T_max] to [-1, 1], Periodic Fourier Features embedding (degree=2, kmax=2, Lx=2.0) applied only to raw x coordinates, concatenation of normalized t and embedded x, followed by main body of depth=3 layers each with width=30 neurons (tanh activation), and direct scalar output for u. Optimizer workflow is Hybrid Adam + SSBroyden: Phase 1 Adam (10,000 epochs, batch_size=20,000 with importance sampling via vRBA exponential potential for local lambdas on PDE residuals, self-balancing global weights lamB/lamP via EMA of normalized gradient magnitudes with decay alpha=0.999 and gamma_grads=0.99, dynamic max_RBA ramp from initial eta/(1-gamma) to cap=20, lr decay from 1e-3); Phase 2 SSBroyden (12,500 total iterations in outer loops of Nchange=100 steps each with BFGS method, frozen global weights from Adam end, continued vRBA for lambda updates and importance sampling with k_samp=1.0, c_samp=0.0). Loss terms exactly match L_total = L_pde (Burgers residual with nu=1/(100*pi) via JVP/HVP) + L_ic (MSE on exact IC u=-sin(pi x)) + L_bc_periodic (MSE on u(t,-1) - u(t,1)), with no additional terms. Curriculum uses static collocation points (NC=200,000), fixed IC/BC samples (NB=512), vRBA-driven local weighting and sampling throughout. | 1.00e-06 | 35 | 35 | 10 | 15 | **95** |
| 8 | The implementation uses an MLP architecture with depth=3, width=30, tanh activation. Input processing includes Periodic Fourier Features (degree=2, period=2.0) applied only to x, with t normalized to [-1,1] before concatenation and passage to the MLP body. The hybrid optimizer workflow consists of Phase 1: Adam for 10,000 iterations with exponential LR decay, vRBA local weighting (exponential potential) on L_pde via importance sampling, and self-balancing global weights (GradNorm-based dynamic lamB updates); Phase 2: SSBroyden for 20,000 iterations (200 outer loops of 100 inner steps each), with Hessian re-initialized to identity at start, frozen global weights from end of Adam phase, continued vRBA local weighting and importance sampling on full collocation points. Loss terms are exactly L_total = L_pde (MSE on Burgers residual with vRBA) + L_ic (MSE on IC -sin(pi x)), no BC loss as periodic BCs are architecturally enforced. PDE uses nu=1/(100 pi), IC exactly -sin(pi x). | 1.35e-07 | 35 | 35 | 12 | 15 | **97** |
| 10 | The implementation uses an MLP architecture (depth=3, width=30 per layer, tanh activation) wrapped in a PINN with input transformations: time t normalized to [-1,1] inside the forward pass, and Periodic Fourier Features (degree/kmax=2, period/Lx=2.0) applied strictly to x (no raw x passed). The optimizer workflow is Hybrid Adam (Phase 1: 10,000 iterations with dynamic self-balancing global weights via GradNorm and vRBA local weighting/importance sampling for L_pde) followed by SSBroyden (Phase 2: 30,000 iterations with frozen global weights from Adam end-state, vRBA used for sampling collocation points). Loss terms are exactly L_total = L_pde (Burgers residual MSE with nu=1/(100*pi)) + L_ic (MSE on IC u(0,x)=-sin(pi x)); no BC terms. Curriculum: dynamic weights and vRBA updates in Adam, frozen globals and sampling-based vRBA in SSBroyden. | 2.00e-07 | 35 | 35 | 12 | 15 | **97** |
| 11 | The implementation uses an MLP architecture (PINN class) with input normalization inside the forward method (t mapped to [-1,1], x identity), Periodic Fourier Features (degree=2, period=2.0) applied only to x (concatenated as [t_norm, cos/sin(k*x)]), followed by a 3-layer MLP body (width=30 neurons per layer, tanh activation) outputting scalar u(t,x). The hybrid optimizer workflow consists of Phase 1: Adam for 10,000 iterations with dynamic self-balancing global weights (lamE=1 fixed for PDE, adaptive lamB via GradNorm EMA on gradient norms) and vRBA local weighting (exponential potential for lambdas on PDE residuals, updated via EMA and used for importance sampling). Phase 2: SSBroyden for 45,000 iterations (450 outer loops of 100 inner steps each), freezing global weights (lamE, lamB) from Adam phase end, while continuing vRBA (lambda updates and importance sampling for PDE points). Loss terms are exactly L_total = lamE * L_pde (MSE of Burgers residual r = u_t + u u_x - nu u_xx) + lamB * L_ic (MSE on u(0,x) + sin(pi x)), with no BC terms (handled by embeddings). No data loss terms used. | 2.71e-08 | 35 | 35 | 12 | 15 | **97** |
| 12 | The implementation uses an MLP architecture (width=30, depth=3, tanh activation) with input transformations: time normalized to [-1,1] and space via Periodic Fourier Features (degree=2, period=2.0), concatenated as [t_norm, x_embedded] with no raw x included. The optimizer follows a hybrid workflow: Phase 1 Adam for 10,000 iterations with vRBA (exponential potential, phi=1.0, c_log=1.0) for local weighting/importance sampling on L_pde and self-balancing global weights (GradNorm with lamE=1, dynamic lamB, adaptive max_RBA up to 20). Phase 2 SSBroyden for exactly 50,000 iterations (500 outer loops × 100 inner steps per outer, Hessian initialized once and recycled), freezing global weights at Adam finals and using vRBA lambdas for sampling only (no weighting in loss). Loss terms are exactly L_total = lamE * MSE(residual) + lamB * MSE(u(0,x) + sin(πx)), with no boundary loss (periodic enforced architecturally). | 2.00e-08 | 35 | 35 | 12 | 15 | **97** |
| 13 | The implementation uses a PINN architecture consisting of Periodic Fourier Features (degree=2, period L=2.0) applied only to x coordinates, concatenated with normalized t (mapped to [-1,1]), feeding into an MLP main body with width=30, depth=3 layers, Tanh activation, outputting scalar u. Optimizer workflow is Hybrid: Phase 1 Adam for 10,000 epochs with exponential decay LR schedule (initial 1e-3), vRBA (exponential potential, EMA updates with gamma=0.999, eta=0.01, phi=1.0) for local weighting on L_pde via importance sampling (k_samp=1.0, c_samp=0.0), self-balancing global weights (GradNorm EMA with alpha=0.999, gamma_grads=0.99) between L_pde and L_ic; Phase 2 SSBroyden for 75,000 iterations (Nchange=100 outer loops), freezing global weights at Adam finals, continuing vRBA lambda updates and sampling internally without modifying SSBroyden logic. Loss terms are exactly L_total = L_pde (MSE of Burgers residual r = u_t + u u_x - nu u_xx with nu=1/(100 pi), weighted by vRBA) + L_ic (MSE on u(0,x) + sin(pi x)); periodic BCs enforced architecturally via embeddings, no explicit BC loss. Curriculum via increasing max_RBA cap from eta/(1-gamma) to 20 over Adam phase. | 1.75e-08 | 35 | 35 | 12 | 15 | **97** |
| 14 | The implementation uses an MLP architecture (depth=3, width=30 per layer, tanh activation) with input transformations consisting of t normalized to [-1,1] inside the forward method and Periodic Fourier Features (degree=2, period=2.0) applied only to x, concatenating [t_normalized, x_embedded] to the main body. The optimizer follows a hybrid schedule: Phase 1 Adam for 10,000 epochs with dynamic self-balancing global weights (lam_pde and lam_ic via EMA of gradient norms) and vRBA for both weighting (exponential potential) and importance sampling in L_pde; Phase 2 SSBroyden for 100,000 total iterations (outer loops of 1,000 with inner maxiter=100 per minimize call, accumulating effective steps), freezing global weights at Adam finals, using vRBA-updated lambdas for sampling only (no weighting inside loss), and Hessian recycling initialized to identity. Loss is L_total = lamE * L_pde + lamB * L_ic, where L_pde is MSE of Burgers residual (u_t + u u_x - nu u_xx, nu=1/(100 pi)) with vRBA in Adam, plain MSE in SSB; L_ic is MSE against exact IC -sin(pi x). No additional terms or data losses. | 1.54e-08 | 35 | 35 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 5 | Saves 'burgers_prediction.npz' containing keys: 'u_pred' (N x 1, N = len(t.flatten()) matching validation grid size from burgers.mat, e.g., ~25,600 x 1 if 100 t x 256 x), 'x' (N x 1), 't' (N x 1), where t/x are flattened meshgrid coordinates from data['t']/data['x'] used for final u_pred evaluation. |
| 8 | Saves 'burgers_prediction.npz' in npz format containing keys: 'u_pred' ((25600,1)), 'x' ((25600,1)), 't' ((25600,1)) where N=256(x)*100(t) from meshgrid flattening of data['x'] and data['t']. |
| 10 | Saves 'burgers_prediction.npz' containing keys: 'u_pred' (N x 1, flattened grid prediction), 'x' (N x 1, flattened x-coordinates from data grid), 't' (N x 1, flattened t-coordinates from data grid), 'u_exact' (N x 1, flattened ground truth from burgers.mat 'usol'). N matches data grid size (e.g., len(t0)*len(x0)). |
| 11 | Saves 'burgers_prediction.npz' in npz format containing keys: 'u_pred' (NT*NX x 1, flattened prediction grid), 'x' (NT*NX x 1, flattened x grid), 't' (NT*NX x 1, flattened t grid), 'u_exact' (NT*NX x 1, flattened ground truth from burgers.mat). NT and NX derived from data['t'].shape and data['x'].shape (typically ~100x100 grid). |
| 12 | Saves 'burgers_prediction.npz' in npz format containing keys: 'u_pred' (N x 1, N=number of test points from full t-x meshgrid), 'x' (N x 1), 't' (N x 1), 'u_exact' (N x 1). |
| 13 | Saves 'burgers_prediction.npz' in NumPy .npz format containing keys: 'u_pred' (N,1) final model predictions on full validation grid, 'u_exact' (N,1) ground truth from burgers.mat['usol'], 'x' (N,1) flattened spatial coordinates from meshgrid(x0,t0), 't' (N,1) flattened temporal coordinates from meshgrid(x0,t0), where N = len(t0)*len(x0) ~ typical 256x256 or similar dense grid. |
| 14 | Saves 'burgers_prediction.npz' in npz format containing keys: 'u_pred' (N x 1), 'x' (N x 1), 't' (N x 1), 'u_exact' (N x 1), where N is the flattened evaluation grid size (product of t0 and x0 lengths from burgers.mat, typically ~10,000). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 5 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2025_12_05_1332/` |
| 8 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2025_12_05_1342/` |
| 10 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2025_12_05_1349/` |
| 11 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2025_12_05_1354/` |
| 12 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2025_12_05_1402/` |
| 13 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2025_12_05_1408/` |
| 14 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2025_12_05_1414/` |


---
