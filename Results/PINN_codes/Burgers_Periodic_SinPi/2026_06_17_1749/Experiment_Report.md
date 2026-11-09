
# Experiment Report
*Last Updated: 2026-06-17 17:56:01*

## User Request
Lets solve the viscous Burgers equation u_t + u u_x = nu u_{xx} with IC = -sin(pi x), the domain goes from 0 to 1 in time and from -1 to 1 in x; periodic boundary conditions.

Here $nu = 1/(100*pi)$.

### Reference Data:
You can find the data here:
/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Documents/codes/Data/burgers.mat
The solutions are saved as a dictionary with keys 't', 'x' and 'usol'.

---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
**Role: Validation/Error Metrics ONLY.**

### 1. Data Source
- **Path:** `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Documents/codes/Data/burgers.mat`
- **Format:** Dictionary with keys `t`, `x`, `usol`. Instruct the Planner to load this strictly for validation/error tracking.

### 2. Problem Statement
- **Governing Equation:** Viscous Burgers Equation.
  Residual: $L_{pde} = u_t + u u_x - \nu u_{xx} = 0$, where $\nu = 1/(100\pi)$.
- **Boundary/Initial Conditions:** 
  - IC: $u(0,x) = -\sin(\pi x)$
  - BC: Periodic boundaries at $x = -1$ and $x = 1$.
- **Problem Type:** Forward. Well-posed for PINNs.
- **Input:** $t \in [0, 1]$, $x \in [-1, 1]$ (1D+t).
- **Output:** Scalar $u(t,x)$.

### 3. Base Model Selection
- **Optimizer:** Hybrid Adam + SSBroyden.

### 4. Main Architecture
**Structure:** Input -> Input Transformation -> Main Body -> Output
- **Input Transformations:**
  - $x \in [-1, 1]$ and $t \in [0, 1]$ are already in nice ranges. **DO NOT** apply any normalization (MinMax/scalers). Pass them as-is.
  - **Periodic Fourier Features:** Apply to $x$ ONLY. 
    - **CRITICAL:** Since raw $x \in [-1, 1]$, derive $L_x = 1 - (-1) = 2.0$. Set `degree = 2`.
  - **Input Composition Rule:** Concatenate the embedded $x$ features with the raw $t$ coordinate. **DO NOT** pass the raw $x$ coordinate to the Main Body.
- **Main Body:** Base MLP with `width=30`, `depth=3`.
- **Output Transformation:** None required.

### 5. Constraint Implementation Strategy
- **Count:** $L_{pde}$ (1) + $L_{ic}$ (1) = 2 terms. (Periodic BC is satisfied by the architecture).
- **Strategy:** Loss-Based.

### 6. Loss Function Assembly
- **Formula:** $L_{total} = L_{pde} + L_{ic}$
- **Note:** $L_{bc\_periodic}$ is EXCLUDED because Periodic Fourier Features enforce it exactly. $L_{data}$ is EXCLUDED because data is for validation only.

### 6.1 Continuation / Training Schedule (ADVISOR MANDATED)
- **CRITICAL ADVISOR CURE:** The Advisor explicitly noted that the error is still decreasing at the end of the run and mandated a 2-5x increase in the final stage iterations.
- **Phase 1 (Adam):** 5,000 iterations.
- **Phase 2 (SSBroyden):** **150,000 iterations** (Increased from 50k to ensure the loss and error curves completely flatten).

### 7. Weighting Strategy
- **Local Weights (`lambdas`):** Use `vRBA` (Variational Residual-Based Attention). `lambdas` are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden.
- **Global Weights (`lam`):** Total terms = 2. Use the **Self-Balancing Strategy**.
  - **Phase 1 (Adam):** Update global weights dynamically.
  - **Phase 2 (SSBroyden):** **FREEZE** global weights at their final Adam values. Strictly forbid dynamic global weighting updates during SSBroyden.
- **SSBroyden Integrity:** Do NOT interfere with the internal weighting, sampling, or batching strategies of the SSBroyden optimizer.

### 8. Visualization & Saving Strategy
- Do NOT modify, delete, or "improve" the visualization code in the template.
- Save the predicted fields as `.npz` arrays with keys specifying the variable names for visualization.

### Main Model:
[MLP with Hybrid Adam and SSBrodyden]

### Enhancements:
[Periodic Fourier Features]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
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
  1. **Loss & Error History:** The x-axis extends to 100,000 iterations. The final training stage (SSBroyden) spans from iteration 5,000 to 100,000. The 50% mark of this final stage is around iteration 50,000.
     - **Loss curve:** The average band level at the 50% midpoint (50k) is `A` $\approx 10^{-13}$. The average band level at the end (100k) is `B` $\approx 10^{-13}$. The loss curve is completely flat and has reached its numerical floor.
     - **Error curve:** The average band level at the 50% midpoint (50k) is `A` $\approx 2 \times 10^{-8}$. The average band level at the end (100k) is `B` $\approx 2 \times 10^{-8}$. The error curve is genuinely horizontal and plateaued.
  2. **Residual/Pointwise Error:** The maximum PDE residual is $\sim 10^{-6}$ and the maximum pointwise error is $\sim 10^{-7}$. The errors are uniformly distributed across the domain and exceptionally low.
  3. **Solution Fields:** The predicted solution is visually indistinguishable from the reference solution, perfectly capturing the shock formation without any artifacts.

* **Diagnosis:** Optimal Convergence. The mandated increase in iterations allowed the SSBroyden optimizer to fully settle into the loss minimum, resulting in a perfectly flat plateau for both the loss and error curves. The model has achieved exceptional accuracy ($\sim 10^{-8}$) and fully resolved the physics.

* **Targeted Improvements:**
    * **For Integrity:** None
    * **For Accuracy:** None
    * **For Details:** None
    * **For Optimality:** None

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | MLP (width=30, depth=3) with Periodic Fourier Features (degree=2, Lx=2.0) on x only; raw t concatenated after embedding; no coordinate normalization. Hybrid optimizer: Adam phase (5000 epochs) with vRBA local weighting + GradNorm/RelLoss global self-balancing (lamB), followed by SSBroyden phase (12500 steps) that freezes global weights and uses vRBA only for sampling. Loss exactly L_pde + L_ic (no periodic BC loss, no data loss). PDE residual, IC (-sin(πx)), and periodic enforcement via embedding all match strategy. | 5.00e-07 | 35 | 35 | 7 | 15 | **92** |
| 2 | The implementation uses a PINN with Periodic Fourier Features (degree=2, Lx=2.0) applied only to x, concatenated with raw t, feeding an MLP (width=30, depth=3). Training follows Hybrid Adam (5000 iters) + SSBroyden (25000 iters). Loss is exactly L_pde + L_ic (IC enforced via -sin(πx) data; periodic BCs via embedding). vRBA used for weighting/sampling in Adam and sampling in SSBroyden; self-balancing global weights (lamB) updated in Adam and frozen in SSBroyden. PDE residual matches Burgers equation with ν=1/(100π). Data loaded from burgers.mat for validation only. | 1.00e-07 | 35 | 35 | 7 | 15 | **92** |
| 3 | Architecture: MLP (width=30, depth=3) with Periodic Fourier Features (degree=2, Lx=2.0) applied to x only; embedded x concatenated with raw t fed to main body. No input normalization. Optimizer workflow: Adam (5000 iterations) with vRBA local weighting + GradNorm self-balancing global weights, followed by SSBroyden (50000 iterations) with frozen global weights and vRBA sampling only. Loss terms: exactly L_pde + L_ic (periodic BC satisfied by embedding, no extra terms). | 3.00e-08 | 35 | 35 | 7 | 15 | **92** |
| 4 | Architecture: PINN with Periodic_Fourier_Features (degree=2, Lx=2.0) applied to x only, concatenated with raw t, fed into MLP (width=30, depth=3). Optimizer workflow: Adam (5000 iterations) with vRBA local weighting + self-balancing global weights (lamE, lamB), followed by SSBroyden (150000 iterations) with frozen global weights and vRBA sampling only. Loss terms: exactly L_pde (Burgers residual) + L_ic (-sin(πx)), no periodic BC loss or data loss. Curriculum schedule matches the mandated 2-5x increase (Adam 5k + SSBroyden 150k). | 2.00e-08 | 35 | 35 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves predictions.npz containing keys: 't' (flattened), 'x' (flattened), 'u_pred' (flattened). |
| 2 | Saves 'predictions.npz' containing keys: 't' (flattened mesh), 'x' (flattened mesh), 'u_pred' (meshgrid shape), 'u_ref' (Exact0 meshgrid shape). Also saves 'Log_files_SSBroyden.mat' with training logs. |
| 3 | Saves 'predictions.npz' containing keys: 't', 'x', 'u_pred', 'u_ref', 'error'. |
| 4 | Saves predictions.npz containing keys: 't' (256000x1), 'x' (256000x1), 'u_pred' (256000x1), 'u_ref' (256000x1), 'error' (256000x1). Also saves Log_files_SSBroyden.mat with training logs. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/PINN_codes//Burgers_Periodic_SinPi/2026_06_17_1736/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/PINN_codes//Burgers_Periodic_SinPi/2026_06_17_1740/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/PINN_codes//Burgers_Periodic_SinPi/2026_06_17_1744/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/PINN_codes//Burgers_Periodic_SinPi/2026_06_17_1749/` |


---
