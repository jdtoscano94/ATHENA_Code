
# Experiment Report
*Last Updated: 2025-11-09 18:31:17*

## User Request

Lets solve the viscous burgers equation u_t+uu_x=nu u_xx the domain is specified on the file; periodic boundary conditions.

Here $nu=1/(1000)$

# Problem
This will be an operator network that maps the initial condition to the full solution up to t=t_end (extracted from the data)

### Reference Data: 
You can find the data here:
/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Data/Operator_Network_Examples/Burger_1000_5000.mat

Here the domain in x \in[0,1] and t\in [0,1]. 

The data set contains the arrays for the initial conditions, and corresponding full trajectories, along with the spatial grid.


---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
## Scientific Strategy Report

### 0. Triage - Determine Strategy
The `[ADVISOR_REPORT]` indicates a scientific failure in the previous run (Trial 10), which was a `QR-DeepONet (KKAN Trunk, eMLP Branch)` with a Fourier basis, achieving a 5.72% error. The Advisor recommends escalating the decomposition method. Specifically, the Advisor recommends re-running the *best performing* QR-DeepONet strategy, which was **Run 9: QR-DeepONet (KKAN Trunk, eMLP Branch) with an RBF basis** (5.67% error), but changing the decomposition from QR to SVD. Therefore, the strategy for this run will be an **SVD-DeepONet with a KKAN Trunk and an eMLP Branch, using an RBF basis for the outer KAN layer.**

### 1. Data Source
The data for this problem is located at `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Data/Operator_Network_Examples/Burger_1000_5000.mat`. The dataset contains initial conditions, corresponding full trajectories, and the spatial grid. The Planner must update the data loading cell to use this path and verify the keys/variables.

### 2. Problem Statement
This problem involves solving the viscous Burgers equation: `∂u/∂t + u∂u/∂x = ν∂²u/∂x²`, where `ν = 1/1000`. The domain is `x ∈ [0,1]` and `t ∈ [0,1]`, with periodic boundary conditions. This is an operator network problem that maps the initial condition `u₀(x)` to the full solution `u(t,x)`. This is a function-to-trajectory problem where `F_space != S_space`.

### 3. Input (F_space)
The input `u₀(x)` is a 1D scalar function defined on a spatial grid.
- **Domain dimension (d):** 1D (x)
- **Value dimension (k):** Scalar
- **Structure:** Grid

### 4. Output (S_space)
The output `u(t,x)` is a 2D scalar function.
- **Domain dimension (d_y):** 2D (t,x)
- **Value dimension (q):** Scalar
- **Characteristics:** The viscous Burgers equation can exhibit shock-like features, implying the output may contain high-frequency components. Periodic boundary conditions are specified.

### 5. Normalization (CRITICAL: In-Graph)
All normalization must be differentiable and implemented *inside* the model's `apply` function.
- **Trunk (per-dimension):** The Planner must pass the `(t, x)` coordinates to the model *un-normalized*. The model itself must compute `t_min, t_max, x_min, x_max` from the training data and perform the normalization to `[-1, 1]` *inside* the `__call__` method before passing to feature layers.
- **Branch (global):** The Planner must pass the input function `u(x, t=0)` *un-normalized*. The model's Branch network must compute `u_min, u_max` and normalize the input to `[-1, 1]` *inside* the `__call__` method before the MLP.
- **Activations:** Based on the `[ADVISOR_REPORT]` for Run 9, both Branch and Trunk inputs were normalized to `[-1, 1]`. This will be maintained for consistency with the best performing previous run.

### 6. Architecture

**Methodology:** `SVD_DeepONet`

**7a. Branch Network:**
- **Type:** `eMLP`
- **Architecture:** The Branch will use an `eMLP` with `degree=7` and `layers=[32, 32, 32, 101]`.
- **Instruction for Planner:** Use `eMLP` for Branch. Set `degree=7`. Use `layers=[32, 32, 32, 101]`.

**7b. Trunk Network:**
- **Type:** `KKAN` (Kurkova Kolmogorov-Arnold Network)
- **Architecture:** The Trunk will use a `KKAN` with `output_dim=101` (to match the Branch embedding dimension) and an outer KAN `degree=5`. The inner `eMLP`s within the KKAN will have `features=[32, 32, 32, 101]` and `degree_T=5`.
- **Basis:** `RBF` basis for the outer KAN layer, as per the Advisor's recommendation to replicate Run 9.
- **Instruction for Planner:** Use `KKAN` for Trunk. Set `output_dim=101` and `degree=5` (for *outer* KAN). For the *inner* `eMLP`s, set `features=[32, 32, 32, 101]` and `degree_T=5`. Use `RBF` basis for the outer KAN.

**Number of iterations:**
Instruct the Planner to set `max_iterations=200000`.

### 7. Loss Function

**7a. Lambda (vRBA):**
- The loss function will incorporate `Lambda_spatial (N,P)` and `Lambda_latent (N,embed_dim)` for the SVD-DeepONet.

**7b. Error_fn:**
- The primary error function will be `MSE`.

**7c. Weights (W):**
- Since the output `u(t,x)` is scalar (`q=1`), the weight `W` will be `1.0 / max(abs(S))`.

### 8. Optimization (Learning Rate)
The `[ADVISOR_REPORT]` does not contain specific recommendations for learning rate adjustments. Therefore, default learning rates for QR/SVD DeepONet strategies will be applied.
- **Instruction for Planner:** Instruct the Planner to set the initial learning rate to `2e-3` for the Branch optimizer.
- **Instruction for Planner:** Instruct the Planner to set the Trunk optimizer learning rate to `(2e-3) / 3` (approximately `6.66e-4`).
- **Instruction for Planner:** Instruct the Planner to use the default decay schedule: set `decay_rate = 0.99` and `decay_steps = 2500` for all optimizers.

IMPLEMENTATION CODE REQUIRED:

Main Model:
SVD_DeepONet

Enhancements:
KKAN
eMLP
RBF basis


---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
The Final Test Error is 0.0627 (6.27%), which is above the 5% threshold. This confirms a scientific failure.

**Current Strategy State:** The [STRATEGIST_REPORT] and [TRIAL_HISTORY_LOG] confirm the strategy just run was **Step 12: SVD-DeepONet (Best)**. This was the final strategy in the 12-step escalation ladder, attempting to improve upon the best QR-DeepONet run by changing the decomposition method.

All 12 strategies in the escalation ladder have now been exhausted. The final 'SVD-DeepONet' strategy has failed to meet the success criterion. **We must now switch to refinement mode and revert to the best-performing model from history.**

Recommendation:
This is a **Strategist-level** and **Planner-level** recommendation.

1.  **(To Strategist):** Analyze the `[TRIAL_HISTORY_LOG]` and find the run/strategy with the **lowest historical test error**. This is **Run 9: QR-DeepONet (KKAN Trunk, eMLP Branch) with an RBF basis**, which achieved a 5.67% error.

2.  **(To Strategist):** Recommend the Strategist to **re-run this best-performing strategy** (the configuration from Run 9).

3.  **(To Planner):** Instruct the **Planner** to **refine this model further** by **doubling the iterations** from 200,000 to 400,000 to see if the slightly better architecture can converge below the 5% threshold with more training time.
---
## Full Trial History
Run | Model | Error
--- | --- | ---
1 | Str 1: MLP Branch / MLP Trunk / Periodic Fourier Features / LRs: 1e-3, 1e-3 / Act: tanh / Norm: [-1,1] / Loss: MSE | 16.37%
2 | Str 2: Standard DeepONet (MLP Branch, KAN Trunk) / Basis: sin series / LRs: 1e-3, 3.33e-4 / Act: tanh / Norm: [-1,1] / Loss: MSE | 17.60%
3 | Str 3: Standard DeepONet (MLP Branch, KAN Trunk) / Wavelet basis / LRs: 1e-3, 3.33e-4 / Act: tanh / Norm: [-1,1] / Loss: MSE | 8.98%
4 | Str 3: eMLP Branch / KKAN Trunk / LRs: 1e-3, 3.33e-4 / Act: tanh / Norm: Branch [-1,1], Trunk [0,1] / Loss: MSE | 10.40%
5 | Str 3: eMLP Branch / KKAN Trunk / Fourier basis / LRs: 1e-3, 3.33e-4 / Act: tanh / Norm: [-1,1] / Loss: MSE | 21.50%
6 | Standard DeepONet: MLP Branch (nn.tanh) / KAN Trunk (Wavelet basis, degree=5) / LRs: 1e-3, 3.33e-4 / Act: tanh / Norm: [-1,1] / Loss: R-L2 | 20.02%
7 | Str 4: QR-DeepONet (MLP) / LRs: 2e-3, 2e-3 / Act: tanh / Norm: [-1,1] / Loss: MSE | 5.95%
8 | Str 8: QR-DeepONet (MLP) + Periodic Fourier Features / LR: 2e-3 / Act: tanh / Norm: [-1,1] / Loss: MSE | 6.02%
9 | Str 5: QR-DeepONet (KKAN Trunk, eMLP Branch) / Basis: RBF / LRs: 2e-3, 6.66e-4 / Loss: MSE / Norm: [-1,1] | 5.67%
10 | Str 5: QR-DeepONet + KKAN Trunk / eMLP Branch / Fourier basis / LRs: 2e-3, 6.66e-4 / Act: tanh / Norm: [-1,1] / Loss: MSE | 5.72%
11 | Str 12: SVD-DeepONet (KKAN Trunk, eMLP Branch) / RBF basis / LRs: 2e-3, 6.66e-4 / Loss: MSE / Norm: [-1,1] | 6.27%

