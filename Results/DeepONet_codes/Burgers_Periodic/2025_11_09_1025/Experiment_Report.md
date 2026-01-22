
# Experiment Report
*Last Updated: 2025-11-09 10:32:50*

## User Request

Lets solve the viscous burgers equation u_t+uu_x=nu u_xx the domain is specified on the file; periodic boundary conditions.

Here $nu=1/(1000)$

# Problem
This will be an operator network that maps the initial condition to the full solution up to t=t_end (extracted from the data)

### Reference Data: 
You can find the data here:
/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Data/Operator_Network_Examples/Burger_1000_5000.mat

The data set contains the arrays for the initial conditions, and corresponding full trajectories, along with the spatial grid.


---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
## Scientific Strategy Report

### 0. Triage - Determine Strategy

The `[USER_QUESTION]` asks to solve the viscous Burgers equation and provides a data path. There are no specific architectural requests from the user.

The `[ADVISOR_REPORT]` indicates a **Quantitative Failure** for the previous run (Run 2), which utilized a `Standard DeepONet with KAN Trunk` and a `sin series basis`. The reported error was 17.60%, exceeding the 5% threshold. The Advisor's analysis highlighted that the `sin series basis` struggled to capture sharp, shock-like features present in the Burgers equation solutions.

The Advisor's **Recommendation** is to **refine** the `Standard DeepONet (KAN)` by **changing the basis function** to a **Wavelet basis**. This recommendation directly guides the current strategy.

Therefore, the strategy for this run will be to continue with `Strategy 2: Standard DeepONet + KAN Trunk`, but with the specified change in the KAN basis function.

### 1. Data Source

The data for this problem is located at:
`/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Data/Operator_Network_Examples/Burger_1000_5000.mat`

**Instruction for Planner:** Update the data loading cell to use this path. Verify that the `.mat` file contains arrays for initial conditions, corresponding full trajectories, and the spatial grid.

### 2. Problem Statement

-   **Governing Equation:** Viscous Burgers equation: `∂u/∂t + u∂u/∂x = ν∂²u/∂x²`
-   **Operator Mapping:** The network will map the initial condition `u₀(x)` to the full solution trajectory `u(t,x)`.
-   **Type:** This is a function-to-trajectory problem where the input function space (`F_space`) is different from the output solution space (`S_space`).

### 3. Input (F_space)

-   **Domain Dimension (d):** 1D (spatial, `x`).
-   **Value Dimension (k):** Scalar (`u`).
-   **Structure:** Grid of spatial points.

### 4. Output (S_space)

-   **Domain Dimension (d_y):** 2D (time `t` and space `x`).
-   **Value Dimension (q):** Scalar (`u`).
-   **Characteristics:** The problem specifies periodic boundary conditions. The Advisor's report notes the presence of "sharp, shock-like features" in the Burgers equation solutions, indicating high-frequency components.

### 5. Normalization (CRITICAL: In-Graph)

**CRITICAL INSTRUCTION FOR PLANNER:** All normalization MUST be differentiable and implemented *inside* the model's `apply` function (i.e., inside the Flax Module).

-   **Trunk (per-dimension):** The Trunk network processes the `(t, x)` coordinates.
    -   **Instruction for Planner:** The model must compute `t_min, t_max, x_min, x_max` from the training data and normalize the `(t, x)` coordinates to `[-1, 1]` *inside* the `__call__` method before passing them to the KAN layers.
-   **Branch (global):** The Branch network processes the input function `u(x, t=0)`.
    -   **Instruction for Planner:** The model's Branch network must compute `u_min, u_max` from the training data and normalize the input `F` to `[-1, 1]` *inside* the `__call__` method before passing it to the MLP layers.
-   **Activations:**
    -   **Trunk:** Using a KAN with Wavelet basis, normalization to `[-1, 1]` is appropriate.
    -   **Branch:** Using a standard MLP with `nn.tanh` activation, normalization to `[-1, 1]` is appropriate.

### 6. Architecture

**Methodology:** `Standard_DeepONet`

**7a. Branch Network:**
-   **Type:** Standard MLP.
-   **Architecture:** Based on Sizing Rule 1 and the previous successful run's configuration, the Branch will have 6 layers with 100 neurons per layer, and a final embedding dimension of 120.
-   **Instruction for Planner:** Use a standard MLP for the Branch with `layers=[100, 100, 100, 100, 100, 120]` and `nn.tanh` activation.

**7b. Trunk Network:**
-   **Type:** KAN (Kolmogorov-Arnold Network).
-   **Architecture:** Based on Sizing Rule 2, deriving from the Branch's MLP baseline `[100, 100, 100, 100, 100, 120]`. Assuming a `degree=5` for the Wavelet basis, `sqrt(degree+1) = sqrt(6) ≈ 2.45`. The layer before the final 120-dim layer should be `~120 / 2.45 ≈ 49`.
-   **Instruction for Planner:** Use a KAN for the Trunk with `layers=[50, 50, 50, 50, 120]`.
-   **Basis:** As per the Advisor's recommendation, use a `Wavelet basis`.
-   **Instruction for Planner:** Set the KAN `basis_function` to `Wavelet` with `degree=5`. Use `nn.tanh` activation for the KAN layers.

**Number of iterations:**
-   **Instruction for Planner:** Instruct the Planner to set `max_iterations=50000`.

### 7. Loss Function

**7a. Lambda (vRBA):**
-   **Instruction for Planner:** Implement a single learnable `Lambda` parameter of shape `(N, P)` for the standard DeepONet.

**7b. Error_fn:**
-   **Instruction for Planner:** Use Mean Squared Error (MSE) as the primary loss function.

**7c. Weights (W):**
-   Since the output `u` is a scalar (`q=1`), the weight `W` will be `1.0 / max(abs(S_true))`.
-   **Instruction for Planner:** Implement a loss function that calculates `MSE: mean(W*(Lambda*(S_pred-S_true))**2)` where `W = 1.0 / max(abs(S_true))` and `Lambda` is a learnable parameter.

### 8. Optimization (Learning Rate)

The `[ADVISOR_REPORT]` did not contain any specific recommendations regarding the learning rate. Therefore, default values for `Strategy 2` will be applied.

-   **Instruction for Planner:** Instruct the Planner to set the initial learning rate to `1e-3` for the Branch optimizer.
-   **Instruction for Planner:** For the KAN Trunk optimizer, set the initial learning rate to `(1e-3) / 3` (approximately `3.33e-4`).

---
IMPLEMENTATION CODE REQUIRED:

Main Model: Standard DeepONet

Enhancements:
- KAN with Wavelet basis


---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
The current run (Run 3) utilized a `Standard DeepONet with KAN Trunk` and a `Wavelet basis`, as recommended by the previous triage. 1. **Visual Readability:** The 'Pointwise Error' plots are clear and readable, showing the distribution of errors across the domain. This is not a cosmetic failure. 2. **Quantitative Failure:** The `[STDOUT LOG]` reports a "Final average Relative L2 error for the functions on the testing dataset" of **0.08981688 (8.98%)**. This value is above the 5% threshold, indicating a **Quantitative Failure**. 3. **Convergence:** The 'Test Relative Error (Mean)' plot shows a rapid initial decrease, followed by a very slow, gradual decrease in the latter half of the training. While not completely plateaued, the rate of improvement is minimal, suggesting the model is struggling to reduce the error further with the current architecture and basis. 4. **Qualitative Match:** The 'Prediction' and 'Reference' plots show a good visual agreement, capturing the overall wave-like patterns of the Burgers equation solutions. However, the 'Error' plots reveal structured errors, particularly along the diagonal features, indicating that the `Wavelet basis` was not fully effective in capturing the nuances of the solution, despite its ability to handle sharp features. The error is distributed globally rather than localized to sharp singularities. Given the quantitative failure (8.98% error) and the slow convergence, the current `Standard DeepONet (KAN) with Wavelet basis` is insufficient.

Recommendation:
This is a **Strategist-level** recommendation. The `DeepONet (KAN) + New Basis` (Wavelet) has just failed to meet the 5% error threshold. KAN basis refinements have proven insufficient for this problem. **Action:** `Escalate` **Cure:** Recommend **escalating** to **Standard DeepONet with KKAN Trunk and eMLP Branch**. **Analysis for KKAN Basis:**  **Recommendation for KKAN Basis:**  Recommend `KKAN with Radial basis functions basis` for the outer KAN layer.
---
## Full Trial History
Run | Model | Error
--- | --- | ---
1 | Str 1: MLP Branch / MLP Trunk / Periodic Fourier Features / LRs: 1e-3, 1e-3 / Act: tanh / Norm: [-1,1] / Loss: MSE | 16.37%
2 | Str 2: Standard DeepONet (MLP Branch, KAN Trunk) / Basis: sin series / LRs: 1e-3, 3.33e-4 / Act: tanh / Norm: [-1,1] / Loss: MSE | 17.60%
3 | Str 3: Standard DeepONet (MLP Branch, KAN Trunk) / Wavelet basis / LRs: 1e-3, 3.33e-4 / Act: tanh / Norm: [-1,1] / Loss: MSE | 8.98%

