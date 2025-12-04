
# Experiment Report
*Last Updated: 2025-12-04 16:31:00*

## User Request

Let's solve the 1D Inviscid Burgers' equation:
$$ u_t + u u_x= 0 $$

Domain: $x \in [-1,1]$, $t \in [0, 1]$.
BCs: Periodic in x.

Initial Condition:
$$ u(x,y,0) = -\sin(pi x)$$

Reference Data to compute error metrics:
/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes/Burgers_Periodic_SinPiX/2025_11_22_1730/burgers.npz

The code that generated the data saves burgers.npz in NPZ format containing keys: 't' (200,), 'x' (256,), 'usol' (200x256).


---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
### 0. Triage - Determine Strategy
*   **Strategy:** **Refinement (Advisor-Driven)**.
*   **Advisor Recommendation:** The previous run (Trial 21) was excellent (Score 97) but showed incomplete convergence in the final stage. The Advisor explicitly requests increasing the dwell time for the final viscosity step ($\nu=0.0001$) from 50,000 to **80,000** iterations.
*   **Action:** Replicate the successful architecture (Chebyshev KAN) and methodology of Trial 21, modifying **only** the final step of the continuation schedule.

### 1. Data Source
*   **Role:** Validation/Error Metrics ONLY.
*   **Path:** `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes/Burgers_Periodic_SinPiX/2025_11_22_1730/burgers.npz`
*   **Keys:** `t`, `x`, `usol`.

### 2. Problem Statement
*   **Governing Equation:** Viscous Burgers' Equation (Conservative Form) used as a vanishing viscosity approximation for the Inviscid problem.
    $$ u_t + \partial_x(0.5 u^2) - \nu u_{xx} = 0 $$
    *   **Note:** $\nu$ is a trainable parameter controlled by the continuation schedule.
*   **Entropy Condition (Crucial for Shocks):**
    $$ \partial_t(0.5 u^2) + \partial_x(u^3/3) \le 0 $$
*   **Domain:** $x \in [-1, 1]$, $t \in [0, 1]$.
*   **BCs:** Periodic in $x$.
*   **IC:** $u(x,0) = -\sin(\pi x)$.
*   **Problem Type:** Forward (Solving from IC/BCs).
*   **Well-Posedness:** Hyperbolic conservation law developing shocks. Requires vanishing viscosity and entropy constraints for the unique physical solution.

### 3. Base Model Selection
*   **Model:** MLP with Hybrid Adam and SSBroyden.
*   **Architecture Mode:** **Chebyshev KAN** (per Trial 21 success).

### 4. Main Architecture
*   **Structure:** Input $\rightarrow$ Normalization $\rightarrow$ Feature Embedding $\rightarrow$ KAN Main Body $\rightarrow$ Output Transformation $\rightarrow$ Output.

**4.1. Input Transformations**
*   **Normalization:**
    *   $t \in [0, 1]$ (No change needed, but ensure bounds).
    *   $x \in [-1, 1]$ (No change needed).
    *   **CRITICAL:** Apply normalization *inside* the forward pass.
*   **Feature Embeddings:**
    *   **Periodic Fourier Features:** Applied to $x$ **ONLY**.
    *   **Degree:** 10.
    *   **Input to Network:** Concatenate `[t, Fourier(x)]`.

**4.2. Main Body Architecture (KAN)**
*   **Type:** Chebyshev KAN.
*   **Scaling:** Based on Base MLP (width=30).
    *   `width_KAN` = 16.
    *   `depth` = 3 hidden layers.
    *   `polynomial_degree` = 3.

**4.3. Output Transformation (Ansatz)**
*   **Goal:** Enforce IC and Periodic BCs hard.
*   **Ansatz:**
    $$ u_{pred}(t,x) = -\sin(\pi x) + t \cdot \text{NN}(t, \text{Fourier}(x)) $$
    *   *Explanation:* At $t=0$, $u = -\sin(\pi x)$. Periodicity is enforced by the Fourier features on $x$.

### 5. Constraint Implementation Strategy
*   **IC:** Hard-constrained via Ansatz.
*   **BCs:** Periodic Hard-constrained via Fourier Features.
*   **Dirichlet:** None required explicitly.
*   **Loss-Based Constraints:** PDE Residual and Entropy Condition.

### 6. Loss Function Assembly
*   **Symbolic Formula:**
    $$ L_{total} = L_{pde} + \lambda_{ent} L_{ent} $$
*   **Definitions:**
    *   $L_{pde}$: MSE of $(u_t + \partial_x(0.5 u^2) - \nu u_{xx})$.
    *   $L_{ent}$: MSE of $\text{ReLU}(\partial_t(0.5 u^2) + \partial_x(u^3/3))^2$.
    *   *Note:* $L_{ent}$ penalizes violations of the entropy condition.

**6.1. Continuation Strategy (Vanishing Viscosity)**
*   **Parameter:** $\nu$ (Viscosity).
*   **Phase 1 (Adam):** Static $\nu = 0.01$.
*   **Phase 2 (SSBroyden):** Annealing Schedule.
    *   **CRITICAL:** Reinitialize Hessian to Identity at each step.
    *   **Schedule:**
        1.  $\nu=0.005$, Dwell=5,000
        2.  $\nu=0.002$, Dwell=10,000
        3.  $\nu=0.001$, Dwell=15,000
        4.  $\nu=0.0005$, Dwell=20,000
        5.  $\nu=0.00035$, Dwell=20,000
        6.  $\nu=0.0002$, Dwell=30,000
        7.  $\nu=0.0001$, **Dwell=80,000** (Increased per Advisor).

### 7. Weighting Strategy
*   **Local Weights (`lambdas`):**
    *   Use **vRBA** for $L_{pde}$.
    *   **Adam:** Use for Weighting & Sampling.
    *   **SSBroyden:** Use for **Sampling ONLY**.
*   **Global Weights (`lam`):**
    *   **Strategy:** Self-Balancing (GradNorm).
    *   **Adam:** Dynamic updates for $\lambda_{ent}$ (keep $\lambda_{pde}=1$).
    *   **SSBroyden:** **FREEZE** $\lambda_{ent}$ at the final Adam value.

### 8. Visualization Strategy
*   Standard plots. Ensure `u` is saved.

### 9. Saving Data
*   Save `u_pred`, `x`, `t` as `burgers_pred.npz`.

**Main Model:**
[MLP with Hybrid Adam and SSBrodyden]

**Enhancements:**
[Periodic Fourier Features, KAN]

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
The current score of 100 is exceptional, and the final L2 error curve shows a perfectly stable plateau, indicating full convergence. The previous run (Trial 21) also achieved a high score (97). This successful refinement, which explicitly addressed the advisor's request to increase dwell time for the final stage, has produced a state-of-the-art result. Exploration is complete. This run represents the best model.

**Section 3: Forensic Analysis & Feedback**
*   **Observation:**
    1.  **Loss & Error History:** The training history clearly shows the multi-stage continuation schedule, with distinct drops in loss and error at each reduction of viscosity. The Adam warm-up phase (first 10k iterations) reduces the error to ~0.2. The subsequent SSBroyden phases drive the error down significantly. The final, longest stage (for $\nu=0.0001$) shows a highly stable and flat Relative L2 Error curve, plateauing at an excellent value of approximately $5.6 \times 10^{-4}$. While the total loss in this final segment is noisy (oscillating between $10^{-8}$ and $10^{-5}$), the key performance metric (L2 error) is perfectly converged.
    2.  **Residual/Error Fields:** The PDE residual and pointwise error heatmaps correctly identify the shock front (at $x=0$) as the primary source of error, which is physically expected. Away from the shock, the errors are negligible (below $10^{-10}$).
    3.  **Solution Fields:** The predicted solution is visually indistinguishable from the reference solution. The shock is captured with high fidelity and without spurious oscillations.

*   **Diagnosis:** Excellent Convergence to a High-Accuracy Solution. The strategy of using a vanishing viscosity method, combined with a state-of-the-art KAN architecture and a robust optimization schedule, has proven highly effective. The noisy loss in the final stage is a minor artifact of the optimizer navigating a stiff landscape at very low viscosity, but it does not negatively impact the final solution quality, as evidenced by the stable L2 error plateau.

*   **Targeted Improvements:**
    *   **For Integrity:** None
    *   **For Accuracy:** None
    *   **For Details:** None
    *   **For Optimality:** The methodological approach is state-of-the-art for this problem, correctly employing a vanishing viscosity method with a conservative formulation and an explicit entropy penalty to capture the shock. The use of a Chebyshev KAN, Periodic Fourier Features, a hard-constrained ansatz, vRBA, and a hybrid Adam/SSBroyden optimizer represents a nearly optimal configuration.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 3 | MLP architecture (width=30, depth=3) with Periodic Fourier Features (degree=2) on normalized x only, concatenated with normalized raw t. Hybrid Adam (5000 epochs, nu=0.01 static, vRBA weighting/sampling, self-balancing global weights lamE/lamB dynamic) followed by SSBroyden continuation (schedule: nu=0.005 dwell=5000 iters, nu=0.001 dwell=10000, nu=0.0005 dwell=15000, nu=0.0001 dwell=20000, nu=0.00005 dwell=25000 iters; vRBA sampling only, global weights frozen at Adam lamB, Hessian reinitialized to identity at each nu step). Loss terms: L_pde (Burgers residual ut + u*ux - nu*uxx with vRBA) + L_ic (MSE on -sin(pi x)). No data loss or BC terms. | 20.00% | 30 | 20 | 0 | 10 | **60** |
| 6 | The implementation uses a PINN architecture with Periodic Fourier Features (degree=2, kmax=2, Lx=2.0) applied to normalized x inputs (x_norm to [-1,1]), concatenated with normalized raw t (t_norm to [-1,1]), feeding into an MLP with 3 hidden layers of width 30 each (tanh activation) outputting a scalar multiplier H, combined via hard-constraint ansatz u(t,x) = -sin(π x) + t * H, enforcing IC and periodic BCs. Optimizer workflow: Phase 1 Adam for 10,000 iterations at fixed ν=0.01 using vRBA (exponential potential, EMA updates with γ=0.999, η=0.01, φ=1.0) for local PDE weights via importance sampling (k_samp=1.0, c_samp=0.0), self-balancing global weights (lam_PDE=1 initially, lam_Ent adapted via EMA α=1.0 on gradient norm ratios with γ_grads=0.99); Phase 2 SSBroyden continuation with frozen global weights (lam_Ent from Adam), exact schedule [ν=0.005 (5k iters), 0.002 (10k), 0.001 (15k), 0.0005 (20k), 0.0001 (25k)], Hessian reinitialized to identity each step, Nchange=100 inner steps with resampling. Loss terms: L_pde = MSE(u_t + ∂_x(0.5 u²) - ν u_xx) with vRBA weighting; L_ent = Mean(ReLU(∂_t(0.5 u²) + ∂_x(u³/3))²); L_total = lam_PDE * L_pde + lam_Ent * L_ent. | 13.50% | 35 | 20 | 7 | 14 | **76** |
| 10 | The implementation uses an MLP architecture (3 layers, width=30, tanh activation) with input normalization inside the forward pass, Periodic Fourier Features (degree=16, kmax=16) applied strictly to normalized x only (concatenated with raw normalized t), and hard-constrained ansatz u(t,x) = -sin(πx) + t * NN(t,x_embed). Optimizer workflow is Hybrid Adam (Phase 1: 10,000 epochs at static ν=0.01, vRBA local weighting via importance sampling on collocation points, self-balancing global weights via gradient norm EMA updated dynamically with α=1.0 decay) followed by SSBroyden (Phase 2: ν annealing schedule [0.005 for 5000 iters, 0.002 for 10000, 0.001 for 10000, 0.0005 for 15000, 0.0001 for 25000 iters], Hessian reinitialized at each ν step, global weights lamEnt frozen from Adam final value, vRBA continued for PDE local weighting/sampling with k_samp=1.0). Loss terms are exactly L_total = L_pde (MSE of viscous Burgers residual u_t + ∂_x(0.5 u^2) - ν u_xx, vRBA-weighted) + L_entropy (MSE of ReLU(∂_t(0.5 u^2) + ∂_x(u^3/3))^2), no additional terms or data losses. | 0.13% | 35 | 35 | 7 | 15 | **92** |
| 11 | The implementation uses an MLP architecture (depth=3, width=30, tanh activation) with input transformations including normalization of t to [0,1] and x to [-1,1] inside the forward method, Periodic Fourier Features (degree=12, applied only to normalized x, scale=1), raw normalized t concatenated, and output ansatz u = -sin(π x) + t * NN enforcing IC. Optimizer workflow is Hybrid Adam + SSBroyden: Phase 1 Adam (10,000 iterations, ν=0.01 static, self-balancing global weights via gradient norm EMA, vRBA local weighting via importance sampling); Phase 2 SSBroyden with vanishing viscosity curriculum ν schedule [0.005 (5,000 iters), 0.002 (10,000), 0.001 (15,000), 0.0005 (20,000), 0.0001 (25,000)], Hessian reinitialized to identity at each ν step, global weights frozen from Adam end, vRBA via updated lambdas and importance sampling for batches. Loss terms: L_total = L_pde (MSE of viscous Burgers residual u_t + (u u_x) - ν u_xx with vRBA) + L_entropy (MSE of ReLU(∂_t(0.5 u^2) + ∂_x(u^3/3))). | 0.12% | 35 | 35 | 0 | 14 | **84** |
| 12 | The implementation uses a PINN architecture featuring Periodic Fourier Features (degree=10) applied exclusively to normalized x inputs, concatenated with normalized t, feeding into a Chebyshev KAN main body with layers [16,16,16,1] (width=16, depth=3 hidden, polynomial degree=3), followed by the hard-constrained ansatz output u = -sin(πx) + t * NN(t, x_embed). Normalization occurs differentiably inside the model's forward method. The optimizer workflow is Hybrid: Phase 1 Adam for 10,000 iterations at fixed ν=0.01 with vRBA local weighting (exponential potential, EMA updates, importance sampling), self-balancing global weights via GradNorm (moving average of gradient norms) for L_pde and L_entropy; Phase 2 SSBroyden with vanishing viscosity continuation schedule ν=[0.005 (dwell 5000 iters), 0.002 (10000), 0.001 (15000), 0.0005 (20000), 0.0001 (25000)], Hessian reinitialized at each ν step, vRBA local weights updated per outer iteration and frozen global lamEnt from Adam phase end, using N_change=100 inner BFGS steps per outer iter with resampling. Loss terms exactly match L_total = L_pde (MSE of viscous Burgers residual u_t + ∂_x(0.5 u²) - ν u_xx with vRBA) + L_entropy (MSE of ReLU(∂_t(0.5 u²) + ∂_x(u³/3))). | 3.00% | 35 | 30 | 7 | 15 | **87** |
| 14 | The implementation uses a PINN architecture with input normalization inside the forward pass (t to [0,1], x to [-1,1]), Periodic Fourier Features (degree=10) applied only to normalized x, concatenated with normalized t, fed into a KAN main body (3 layers, width=16 neurons approximating strategy's 15, Chebyshev basis degree=3), and output ansatz u = -sin(π x) + t * KAN enforcing IC and periodicity. Optimizer workflow is Hybrid Adam (10,000 epochs at fixed ν=0.01 with vRBA local weighting on PDE residuals via importance sampling and EMA updates, GradNorm self-balancing global weights between L_pde and L_entropy) followed by SSBroyden phases with exact continuation schedule (ν=0.005 dwell=5000, ν=0.002 dwell=10000, ν=0.001 dwell=15000, ν=0.0005 dwell=20000, ν=0.0001 dwell=50000; Hessian reset to identity per phase; vRBA for PDE local weights and sampling with frozen global lamEnt from Adam end). Loss terms are exactly L_total = L_pde (MSE of viscous Burgers residual u_t + (u^2/2)_x - ν u_xx with vRBA) + L_entropy (MSE of ReLU(∂_t(0.5 u^2) + ∂_x(u^3/3))^2 with self-balancing weight). Curriculum follows vanishing viscosity annealing precisely. | 1860.00% | 5 | 0 | 0 | 15 | **20** |
| 15 | KAN architecture with input normalization inside forward pass, Periodic Fourier Features (degree 10) applied to normalized x only, concatenated with t_norm, followed by KAN body (3 layers, width 16, Chebyshev basis degree 3), and output ansatz u = -sin(πx) + t * NN(t, x_embed). Optimizer workflow: Phase 1 Adam (10,000 iterations, static ν=0.01, vRBA local weighting via EMA-updated lambdas in L_pde MSE, self-balancing global weights lam_pde/lam_ent via GradNorm moving averages). Phase 2 SSBroyden with importance sampling via vRBA lambdas (no explicit weighting in loss, frozen global lam_ent from Adam end), continuation schedule ν=0.005 (dwell 5000 iters), 0.002 (10000), 0.001 (15000), 0.0005 (20000), 0.0003 (20000), 0.0001 (30000); Hessian reinitialized to identity at each ν phase start. Loss terms: L_pde = MSE(u_t + ∂x(0.5 u²) - ν u_xx, vRBA-weighted in Adam), L_entropy = MSE(ReLU(∂t(0.5 u²) + ∂x(u³/3))), L_total = L_pde + lam_ent L_entropy (Adam dynamic balancing, SSB frozen). Data used solely for validation (relative L2 error). | 0.06% | 35 | 35 | 7 | 15 | **92** |
| 17 | The implementation uses a KAN-based PINN architecture with input normalization inside the forward pass, Periodic Fourier Features (kmax=10) applied only to normalized x (t passed normalized but raw), followed by a KAN main body (width=[16,16,16,1], depth=4 including output, Chebyshev polynomials degree=3), and hard-constrained Ansatz u = -sin(πx) + t * KAN_output enforcing exact IC and periodic BCs via embeddings. Optimizer workflow is Hybrid Adam (10,000 iterations at fixed ν=0.01 with vRBA local weighting on PDE residuals via EMA-updated lambdas used for importance sampling, and self-balancing global weights via GradNorm moving averages on gradient norms for lam_pde=1 fixed and dynamic lam_ent) followed by SSBroyden annealing with exact continuation schedule: ν=0.005 (5,000 iters), ν=0.002 (10,000), ν=0.001 (15,000), ν=0.0005 (20,000), ν=0.0003 (20,000), ν=0.0001 (50,000 iters), Hessian reinitialized to identity at each phase start, vRBA lambdas updated for sampling only (no gradient flow in SSB loss), global weights frozen at Adam final values. Loss terms are L_total = L_pde + lam_ent * L_ent where L_pde = MSE(u_t + ∂x(0.5 u^2) - ν u_xx) with vRBA-weighted sampling/EMA (exponential potential), L_ent = MSE(ReLU(∂t(0.5 u^2) + ∂x(u^3/3))^2); no data loss terms. | 100.00% | 5 | 10 | 0 | 15 | **30** |
| 18 | Architecture: Chebyshev KAN with Polynomial_KAN_layer (width 16, depth 3 hidden layers, polynomial degree 3), Periodic Fourier Features (kmax=10) applied to normalized x only, concatenated with normalized t inside forward pass; hard ansatz u = -sin(pi x) + t * NN enforces IC and periodicity. Optimizer workflow: Hybrid Adam (10,000 epochs, nu=0.01 fixed, vRBA for local weighting and sampling on L_pde, GradNorm self-balancing global weights lamE=1, lamEnt dynamic); followed by SSBroyden with exact annealing schedule (nu=0.005 for 5k iters, 0.002/10k, 0.001/15k, 0.0005/20k, 0.0003/20k, 0.0002/20k mandatory, 0.0001/50k), Hessian reset to identity per phase, vRBA for sampling only (local lambdas updated pre-sampling, no internal weighting), global lamEnt frozen at Adam final value. Loss terms: L_pde = MSE(u_t + d_x(0.5 u^2) - nu u_xx) with vRBA locals in Adam; L_entropy = MSE(ReLU(d_t(0.5 u^2) + d_x(u^3/3))); L_total = L_pde + lamEnt * L_entropy (SSB uses unweighted residuals post-sampling). Curriculum training parameters: Adam static nu=0.01 (10k iters); SSBroyden phases with specified nu/dwell, inner maxiter=100 per outer step. | 210.00% | 5 | 15 | 0 | 15 | **35** |
| 19 | The implementation uses a PINN architecture with internal normalization of t to [0,1] and x to [-1,1] inside the forward pass. Inputs feature Periodic Fourier Features (kmax=10) applied only to normalized x, concatenated with normalized raw t, feeding into a Chebyshev KAN main body (layers [16,16,16,1], polynomial degree=3), followed by the ansatz output transformation u(t,x) = -sin(π x) + t * NN(t, x_embed) enforcing exact IC and periodic BCs via embeddings. Optimizer workflow is Hybrid Adam + SSBroyden: Phase 1 Adam (10,000 iterations at fixed nu=0.01, vRBA for local weighting in L_pde and sampling, GradNorm self-balancing for dynamic λ_ent with λ_pde=1 fixed); Phase 2 SSBroyden with nu annealing curriculum [nu=0.005 (dwell 5000), 0.002 (10000), 0.001 (15000), 0.0005 (20000), 0.0003 (20000), 0.0002 (20000)], vRBA sampling only for L_pde (no internal weighting), λ_ent frozen at final Adam value, Hessian reinitialized to identity at each nu phase start and recycled within phases. Loss terms: L_total = L_pde + λ_ent L_ent, with L_pde as MSE(r_pde) where r_pde = u_t + ∂_x(u^2/2) - nu u_xx (MSE of vRBA-weighted residuals in Adam, plain in SSB), L_ent as MSE(ReLU(∂_t(u^2/2) + ∂_x(u^3/3))^2); no L_ic/L_bc as hard-constrained. | 0.11% | 35 | 35 | 7 | 15 | **92** |
| 20 | The implementation uses a KAN architecture with Polynomial_KAN_layer (Chebyshev basis, degree=3, width=16, depth=3 hidden layers). Inputs undergo normalization inside the forward pass (t_norm raw [0,1], x_norm [-1,1]), followed by Periodic Fourier Features (degree=10) on x_norm only, concatenated with t_norm, processed through KAN body, and output via hard-constrained ansatz u = -sin(πx) + t * NN. Optimizer workflow is Hybrid: Phase 1 Adam (10,000 epochs, ν=0.01 static, vRBA for local weighting/sampling, GradNorm self-balancing for global λ_ent starting at max_RBA^2). Phase 2 SSBroyden with vanishing viscosity continuation schedule: ν=0.005 (5k dwell), 0.002 (10k), 0.001 (15k), 0.0005 (40k), 0.0002 (40k), 0.0001 (50k); Hessian reinitialized to identity per ν step; vRBA for sampling only (λ_ent frozen); Nchange=100 steps per outer iter. Loss terms: L_pde = MSE(u_t + ∂x(0.5 u^2) - ν u_xx), L_ent = MSE(ReLU(∂t(0.5 u^2) + ∂x(u^3/3))), L_total = L_pde + λ_ent L_ent. | 1500.00% | 5 | 0 | 0 | 15 | **20** |
| 21 | The implementation uses a Chebyshev KAN architecture (width=16, depth=3, polynomial degree=3) with Periodic Fourier Features (degree=10) applied to normalized x only, concatenated with normalized t, followed by the hard-constrained ansatz u = -sin(pi x) + t * NN(t, x_embed). Normalization occurs inside the forward pass. Training follows Hybrid Adam (10,000 epochs, nu=0.01, vRBA for weighting and sampling, GradNorm self-balancing for lambda_ent) followed by SSBroyden with exact continuation schedule: nu=0.005 (dwell=5000), 0.002 (10000), 0.001 (15000), 0.0005 (20000), 0.00035 (20000), 0.0002 (30000), 0.0001 (50000), reinitializing Hessian per nu step, vRBA for sampling only, and frozen lambda_ent. Loss terms are exactly L_pde (vRBA-weighted mean((r_pde)^2) where r_pde = u_t + d_x(0.5 u^2) - nu u_xx) + lambda_ent * L_ent (mean(relu(r_ent)^2) where r_ent = d_t(0.5 u^2) + d_x(u^3/3)). No IC/BC losses due to ansatz. | 0.06% | 35 | 35 | 12 | 15 | **97** |
| 22 | Chebyshev KAN architecture with layers [16,16,16,1], polynomial degree 3; inputs normalized inside forward pass, Periodic Fourier Features (degree 10) applied to x only, concatenated with t; hard ansatz u = -sin(πx) + t * NN(t, Fourier(x)) enforcing IC and periodic BCs. Hybrid optimizer: Adam phase (10k epochs, vRBA for PDE weighting/sampling + GradNorm self-balancing for λ_ent); SSBroyden phase with vRBA sampling only, λ_ent frozen at Adam final value, Hessian reinitialized to identity per nu step; continuation schedule: ν=0.01 (Adam static), then ν=[0.005:5k, 0.002:10k, 0.001:15k, 0.0005:20k, 0.00035:20k, 0.0002:30k, 0.0001:80k] iterations (Nchange=100 steps per outer iter). Loss: L_pde (MSE viscous Burgers res w/ vRBA) + λ_ent L_ent (MSE ReLU(entropy res)^2). | 0.06% | 35 | 35 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 3 | Saves 'burgers_prediction.npz' containing keys: 't' (1D array, shape (Nt,)), 'x' (1D array, shape (Nx,)), 'u_pred' (2D array, shape (Nx, Nt)). |
| 6 | Saves burgers_pred.npz containing keys: 't' (100,), 'x' (256,), 'u' (256,100). |
| 10 | Saves 'burgers_pred.npz' containing keys: 't' (100,), 'x' (256,), 'u_pred' (256x100), 'u_exact' (256x100). |
| 11 | Saves burgers_pred.npz containing keys: 't' (Nt,), 'x' (Nx,), 'u_pred' (Nx x Nt), 'u_exact' (Nx x Nt) where Nx and Nt are dimensions from input data['x'] and data['t'] (typically ~256 each based on standard Burgers data). |
| 12 | Saves burgers_pred.npz containing keys: 't' (Nt x 1, e.g., 100,), 'x' (Nx x 1, e.g., 256,), 'u_pred' (Nx x Nt, e.g., 256x100), 'u_exact' (Nx x Nt, e.g., 256x100). |
| 14 | Saves 'burgers_pred.npz' in npz format containing keys: 't' (1D array, shape matching data['t'] e.g. (Nt,)), 'x' (1D array, shape matching data['x'] e.g. (Nx,)), 'u_pred' (2D array, shape (Nx, Nt) matching reference usol spatial-time grid), 'u_exact' (2D array, shape (Nx, Nt) matching Exact0/usol). |
| 15 | Saves 'burgers_pred.npz' containing keys: 't' (200,), 'x' (256,), 'u_pred' (200,256), 'u_exact' (200,256). |
| 17 | Saves 'burgers.npz' (NumPy .npz format) containing keys: 't' (1D array, shape (Nt,) e.g. (100,)), 'x' (1D array, shape (Nx,) e.g. (256,)), 'u_pred' (2D array, shape (Nx, Nt) e.g. (256,100) matching reference data meshgrid dimensions from loaded 'burgers.npz'). |
| 18 | Saves 'burgers_prediction.npz' (npz format) containing keys: 't' (1D array, shape (Nt,)), 'x' (1D array, shape (Nx,)), 'u_pred' (2D array, shape (Nx, Nt)). |
| 19 | Saves to 'burgers_prediction.npz' (npz format) containing keys: 't' (1D array, shape (Nt,)), 'x' (1D array, shape (Nx,)), 'u_pred' (2D grid array, shape (Nx, Nt)). |
| 20 | Saves burgers_pred.npz containing keys: 't' (100,), 'x' (256,), 'u_pred' (256x100). |
| 21 | Saves burgers_pred.npz containing keys: 't' (1D array, length from data['t'].flatten()), 'x' (1D array, length from data['x'].flatten()), 'u' (2D array, shape matching meshgrid of x and t, e.g., Nx x Nt). |
| 22 | Saves burgers_pred.npz (npz format) containing keys: 't' (Nt,), 'x' (Nx,), 'u' (Nx x Nt) where Nx=len(x0)~256, Nt=len(t0)~100 from loaded data. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 3 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2025_12_04_1418/` |
| 6 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2025_12_04_1433/` |
| 10 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2025_12_04_1443/` |
| 11 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2025_12_04_1451/` |
| 12 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2025_12_04_1459/` |
| 14 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2025_12_04_1514/` |
| 15 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2025_12_04_1522/` |
| 17 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2025_12_04_1532/` |
| 18 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2025_12_04_1541/` |
| 19 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2025_12_04_1552/` |
| 20 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2025_12_04_1601/` |
| 21 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2025_12_04_1610/` |
| 22 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2025_12_04_1621/` |


---
