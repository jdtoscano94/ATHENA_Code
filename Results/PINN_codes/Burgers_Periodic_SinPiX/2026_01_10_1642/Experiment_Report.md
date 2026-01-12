
# Experiment Report
*Last Updated: 2026-01-10 16:46:35*

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

MANDATORY: Make sure to report the loss and errors every 500 iterations and you cannonly use optimizers that work natively in JAX (NO SciPY)
CONSTRAINTS: If using latin hypercube sampling dont use any criterion such as 'maximin'. 


---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
**1. Data Source**
*   **Dataset Path:** `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes/Burgers_Periodic_SinPiX/2025_11_22_1730/burgers.npz`
*   **Format:** NPZ file.
*   **Keys:** `t` (time coordinates), `x` (spatial coordinates), `usol` (exact solution grid).
*   **Instruction:** Load this data solely for computing validation metrics (Relative L2 Error) during training. Do not use it for training.

**2. Problem Statement**
*   **Governing Equation (1D Inviscid Burgers):**
    $$u_t + u \cdot u_x = 0$$
    *   **Residual:** $r(t,x) = \frac{\partial u}{\partial t} + u \cdot \frac{\partial u}{\partial x}$
*   **Domain:** $x \in [-1, 1]$, $t \in [0, 1]$.
*   **Initial Condition:**
    $$u(0, x) = -\sin(\pi x)$$
*   **Boundary Conditions:**
    *   **Periodic:** $u(t, -1) = u(t, 1)$

**3. Model Architecture**
*   **Type:** **Adaptive Fourier Feature MLP** (Designed to capture high frequencies and sharp gradients).
*   **Input Embedding:** Gaussian Fourier Features.
    *   Map inputs $(t, x)$ to a higher dimensional space using a weight matrix $B$ sampled from $\mathcal{N}(0, \sigma^2)$.
    *   $\sigma = 5.0$.
    *   Output dimension: 128 features (64 sine, 64 cosine).
*   **Backbone:**
    *   5 Hidden Layers.
    *   **Width:** 256 neurons per layer (Increased capacity as per Advisor).
    *   **Activation:** **Adaptive Tanh**.
        *   Implement a custom activation mechanism where the pre-activation value is scaled by a trainable scalar parameter $\alpha$ before applying `tanh`.
        *   Formula: $h_i = \tanh(\alpha_i \cdot z_i)$, where $z_i$ is the linear output of the dense layer.
        *   $\alpha_i$ should be a trainable scalar (one per layer), initialized to 1.0.
*   **Output:** Single scalar $u$.
*   **Initialization:** Xavier/Glorot Uniform for weights.

**4. Optimization Strategy**
*   **Optimizer:** Adam (via Optax).
*   **Schedule:** Cosine Decay.
    *   Start Learning Rate: $1e-3$.
    *   End Learning Rate: $1e-5$.
    *   Steps: 60,000.
*   **Curriculum (Time-Marching):**
    *   Implement a dynamic time horizon $T_{end}$.
    *   Ramp $T_{end}$ linearly from $0.1$ to $1.0$ over the first 40,000 iterations.
    *   Hold $T_{end} = 1.0$ for the remaining 20,000 iterations.
    *   At each iteration, sample collocation points uniformly from the current domain $[0, T_{end}] \times [-1, 1]$.
*   **Refinement (RAR):**
    *   Frequency: Every 1,000 iterations.
    *   Candidates: Sample 50,000 random points in the current domain $[0, T_{end}] \times [-1, 1]$.
    *   Selection: Calculate absolute PDE residual $|r|$ and select the top 500 points with the highest error.
    *   Action: Append these points to the collocation set for the next 1,000 iterations (maintain a buffer or replace previous RAR points to manage memory).

**5. Loss Function**
*   **Total Loss:** $L_{total} = L_{PDE} + \lambda_{IC} L_{IC} + \lambda_{BC} L_{BC}$
*   **Components:**
    1.  **PDE Loss ($L_{PDE}$):** Mean Squared Error of the residual $u_t + u u_x$ on the dynamic collocation points (LHS + RAR).
    2.  **IC Loss ($L_{IC}$):** Mean Squared Error of $u(0, x) - (-\sin(\pi x))$ on 500 random spatial points.
    3.  **BC Loss ($L_{BC}$):** Mean Squared Error of $u(t, -1) - u(t, 1)$ on 500 random time points sampled from $[0, T_{end}]$.
*   **Weights:**
    *   $\lambda_{IC} = 100.0$
    *   $\lambda_{BC} = 100.0$

**6. Output Directives**
*   **Main Model:** Adaptive Fourier Feature MLP with 5 layers of 256 units and trainable activation scalers.
*   **Enhancements:** Time-Marching Curriculum (0.1 to 1.0) combined with Residual-based Adaptive Refinement (RAR).
*   **Reporting:** Log Total Loss, Component Losses, and Relative L2 Error (vs Reference Data) every 500 iterations.

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
REJECTED_FOR_REFINEMENT

**Section 1: Status Decision**
*   **Status:** REJECTED_FOR_REFINEMENT

**Section 2: Quantitative Scorecard**
*   **Score Integrity:** 10/35
*   **Score Accuracy:** 15/35
*   **Score Details:** 8/15
*   **Score Optimality:** 10/15
*   **TOTAL SCORE:** 43/100

**Section 3: Forensic Analysis & Feedback**
*   **Observation:**
    The training run exhibits a severe pathology. While the Relative L2 Error shows a consistent, albeit slow, decrease to a final value of ~19.65%, the Total Loss history tells a different story. After an initial period of convergence for the first ~10,000 iterations, the loss becomes extremely unstable, oscillating violently by over four orders of magnitude (from ~1 to >10^4) for the remainder of the training. The solution plots reveal the underlying cause: the model successfully captures the qualitative wave-steepening behavior of the Burgers' equation, but it struggles immensely to resolve the shock front that forms around t=0.3. The Pointwise Error and PDE Residual plots confirm that the error and instability are almost entirely concentrated along this shock line, with the log reporting a maximum residual of over 27,000.

*   **Diagnosis:** **Training Instability due to Unresolved Shock Front and Overly Aggressive RAR.**
    The core problem is the formation of a mathematical discontinuity (shock), which a standard neural network cannot represent perfectly. The gradients ($u_t, u_x$) become theoretically infinite at the shock, leading to enormous PDE residuals. The Residual-based Adaptive Refinement (RAR) mechanism correctly identifies these high-residual points at the shock and adds them to the training set. However, this creates a vicious cycle:
    1.  The optimizer encounters points with massive residuals.
    2.  This results in explosive gradients in the loss landscape.
    3.  The optimizer takes huge, unstable steps, causing the total loss to spike.
    4.  The model overcorrects, the loss temporarily drops, and the cycle repeats at the next RAR step.
    The L2 error (an integral metric) continues to improve because the model is learning the smooth parts of the solution well, but the loss (a pointwise metric) is completely dominated by the pathological behavior at the shock.

*   **Targeted Improvements:**
    *   **For Integrity:** The immediate priority is to stabilize the training. The current approach is not robust.
        *   **Reduce RAR Aggressiveness:** The current strategy of adding 500 high-residual points every 1000 iterations is feeding the instability. Reduce the number of added points (e.g., to 100-200) and/or decrease the frequency of RAR (e.g., to every 2500 iterations). This will give the optimizer more time to accommodate the difficult points.
        *   **Introduce a Robust Loss Function:** The MSE loss is highly sensitive to outliers. Replace the MSE on the PDE residual with a **Huber loss**. This will treat small residuals as quadratic (like MSE) but large residuals as linear (like MAE), effectively capping the gradient contribution from the pathological shock points and preventing the optimizer from taking excessively large steps.

    *   **For Accuracy:** The final error of ~20% is poor. The shock needs to be resolved more accurately.
        *   **Refine the Time-Marching Curriculum:** The instability worsens significantly after the curriculum ends at 40,000 iterations. This suggests the model is not ready for the fully formed shock. Consider extending the curriculum ramp time (e.g., to 50,000 iterations) to give the model more training on the pre-shock and early-shock regions before tackling the full problem.

    *   **For Details:** The shock front in the prediction is smeared.
        *   **Increase Fourier Feature Scale:** A larger `sigma` in the Gaussian Fourier Features can help the model represent higher-frequency components, which might lead to a sharper shock representation. Consider moderately increasing `sigma` from 5.0 to a value like 10.0.

    *   **For Optimality:** The methods are powerful but are working against each other.
        *   **Prioritize Stability over Aggressiveness:** The combination of a time curriculum and RAR is excellent, but stability must come first. The primary refinement should be to make the RAR process less disruptive by using a robust loss (Huber) and a less aggressive update schedule.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 7 | The implementation uses a Physics-Informed Neural Network with MLP architecture: 4 hidden layers of 50 neurons each, tanh activation, inputs (t, x), output u, initialized with Flax defaults (Glorot-like). Optimizer is Adam via Optax with exponential decay learning rate schedule (init 1e-3, decay_rate 0.9 every 2000 steps), trained for 40,000 iterations. Loss function exactly matches L_total = L_PDE (MSE of ut + u*ux residuals on 10k LHS interior points) + L_IC (MSE vs -sin(pi x) on 100 LHS points at t=0) + L_BC (MSE of u(-1,t) - u(1,t) on 100 LHS time points). Reports total loss, PDE/IC/BC components, and relative L2 error on reference data every 500 iterations. Uses Latin Hypercube Sampling without maximin criterion for all points. No curriculum training; fixed points throughout. | 38.80% | 30 | 5 | 5 | 5 | **45** |
| 11 | The implementation uses an MLP architecture with 4 hidden layers of 128 neurons each, tanh activations, taking raw inputs (t, x) and outputting u(t, x). Training employs Adam optimizer with initial learning rate 1e-3 and exponential decay (rate 0.9 every 2000 steps). The workflow follows Residual-based Adaptive Refinement (RAR): initializes 2000 LHS collocation points (no maximin), 200 IC points, 200 BC points; then 20 RAR cycles, each with 2500 gradient descent iterations followed by evaluating PDE residuals on 10000 uniform random candidates and adding the 200 highest-residual points. Loss is L_total = L_PDE + 10*L_IC + 10*L_BC, where L_PDE is MSE of (∂t u + u ∂x u), L_IC is MSE against -sin(πx) at t=0, L_BC is MSE of (u(-1,t) - u(1,t)). Metrics (total loss, components, relative L2 error on test data) are logged every 500 iterations. | 33.20% | 15 | 5 | 5 | 5 | **30** |
| 12 | The implementation uses a Modified MLP architecture in JAX/Flax: inputs (t,x) are concatenated and passed through two separate Dense encoders to feature vectors U and V (dim=128), followed by 4 hidden layers of 128 neurons each. Each hidden layer computes Z_linear via Dense, applies adaptive activation tanh(a * Z_linear) with per-layer trainable scalar a (init=1.0), and updates H via skip-connection H = (1 - Z) * U + Z * V to inject features and prevent vanishing gradients. Final Dense output to scalar u. Optimizer is optax.adam with exponential decay schedule (init LR=1e-3, decay_rate=0.9, transition_steps=2000), trained for 50,000 iterations via RAR curriculum (20 cycles of 2500 iterations each, adding 200 high-residual points per cycle from 10k candidates). Loss terms exactly match: L_total = L_PDE (mean(u_t + u * u_x)^2 on 10k+ LHS collocation) + 20 * L_IC (MSE vs -sin(π x) on 500 t=0 points) + 20 * L_BC (MSE(u(t,-1) - u(t,1)) on 500 paired boundary points). | 0.3-0.5 | 15 | 5 | 10 | 10 | **40** |
| 17 | The implementation uses a Fourier Feature MLP architecture: inputs [t, x] embedded via random Fourier features (B matrix (2,128) sampled from N(0, 5^2), concatenated cos(2π B v) and sin(2π B v) to 256 dims), followed by 4 hidden layers of 128 neurons each with tanh activation, and 1 output neuron for u. Optimizer is Adam with cosine decay learning rate schedule (initial 1e-3, alpha 0.01 over 50,000 steps). Loss function exactly assembles L_total = Huber(δ=0.1) on PDE residual (u_t + u u_x) + 100 * MSE_IC (vs exact -sin(πx)) + 100 * MSE_BC (u(t,-1) - u(t,1)). Training uses fixed uniform random sampling (20k PDE, 1k IC, 1k BC points) with no data leakage; validation L2 error computed on loaded test data. Curriculum: none specified, single-phase training over 50k iterations with logging every 500 steps. | 39.20% | 35 | 5 | 10 | 5 | **55** |
| 20 | The implementation uses a Fourier Feature MLP architecture: inputs (t, x) are embedded via Gaussian Fourier Features (m=64 frequencies sampled from N(0, 25) scaled by sigma=5, output dim=128) followed by 4 hidden layers of 128 neurons each with tanh activation, outputting scalar u. Optimization employs Adam with exponential decay scheduler (init lr=1e-3, decay=0.9 every 2000 steps). Training follows the RAR curriculum exactly: initializes with 2000 LHS PDE points, 200 IC points, 200 BC points; performs 25 cycles of 2000 inner iterations each; refines by evaluating |PDE residual| on 10,000 uniform random candidates and adding top 200 highest-residual points (except last cycle). Loss is L_total = Huber(delta=0.1, mean over dynamic PDE points of u_t + u u_x) + 100 * MSE_IC(u(0,x) + sin(pi x)) + 100 * MSE_BC(u(t,-1) - u(t,1)). Relative L2 error computed on loaded test data during training and visualized post-training. | 80.40% | 35 | 0 | 5 | 5 | **45** |
| 23 | The implementation uses a Fourier Feature MLP architecture with input embedding via random Fourier features (degree=128, scale/sigma=25.0, output dim=256), followed by 5 hidden layers of 256 neurons each using tanh activation, and a single output neuron for u. The optimizer is Adam with a cosine decay learning rate schedule from 1e-3 to 1e-5 over 50,000 total steps. The loss function exactly matches L_total = L_PDE (MSE of u_t + u u_x on collocation points) + 100 * L_IC (MSE vs -sin(pi x) at t=0) + 100 * L_BC (MSE of u(t,-1) - u(t,1)). Training follows the RAR curriculum precisely: initializes N_PDE=2500 (LHS), N_IC=500 (random), N_BC=500 (random t); performs 10 cycles of 5000 iterations each, refining by evaluating |residual| on 100,000 random candidates and adding the top 500 highest-residual points to PDE set. Logging occurs every 500 iterations with total/PDE/IC/BC losses and relative L2 error vs reference usol data. | 101.80% | 10 | 0 | 5 | 5 | **20** |
| 24 | The implementation uses a Fourier Feature MLP architecture with Gaussian Fourier embedding (B matrix of shape (2,64) sampled from N(0, sigma^2=3.0), producing 128 features via cos(2pi B v) and sin(2pi B v)), followed by 4 hidden layers of 128 neurons each with tanh activation, and a linear output for scalar u. Optimizer is Adam with cosine decay learning rate schedule starting at 5e-4 decaying to 1e-5 over 50,000 total steps. Loss terms exactly match L_total = L_PDE (mean squared residual of ut + u ux) + 100 L_IC (MSE against -sin(pi x) at t=0) + 100 L_BC (MSE between u(t,-1) and u(t,1)). Training follows RAR curriculum: 25 cycles, each with 2000 inner-loop iterations on current collocation points (initial N_f=2000 LHS PDE points, N_ic=200, N_bc=200), followed by refinement via evaluating |R_PDE| on 10,000 uniform candidates and appending the top 200 highest-residual points to PDE set. Metrics (total loss, components, relative L2 error on loaded burgers.npz data) logged every 500 iterations. | 39.12% | 35 | 10 | 10 | 10 | **65** |
| 25 | The implementation uses a Modified Fourier-Feature MLP architecture with Gaussian Fourier embedding (sigma=5.0, 64 degree yielding 128 features: 64 sine + 64 cosine), two encoder dense layers to U/V vectors of size 256, an initial projection to H, followed by 6 hidden layers of 256 neurons each applying the skip connection H_k = (1 - tanh(W_k H_{k-1} + b_k)) ⊙ U + tanh(...) ⊙ V, and a final linear output to scalar u, with Xavier initialization. The optimizer is Adam with cosine decay learning rate schedule from 1e-3 to 1e-5 over 50,000 total steps, chained with global gradient norm clipping at 1.0. Training follows the RAR curriculum over 20 cycles of 2,500 iterations each: initial N_PDE=2000 (LHS), N_IC=500, N_BC=500; at end of each cycle (except last), evaluate |PDE residual| on 100,000 uniform candidates and append top 500 highest-residual points to PDE set. Loss is L_total = L_PDE + 100 * L_IC + 100 * L_BC, where L_PDE = mean((u_t + u * u_x)^2), L_IC = mean((u(0,x) - (-sin(π x)))^2), L_BC = mean((u(t,-1) - u(t,1))^2), with JIT-compiled steps and logging every 500 iterations including relative L2 error on test data. | 83.30% | 35 | 0 | 5 | 5 | **45** |
| 26 | The implementation uses a Multi-Scale Fourier Feature MLP architecture with Gaussian Fourier embeddings at scales sigma=1.0 (low freq, 64 features -> 128 dims) and sigma=10.0 (high freq, 64 features -> 128 dims), concatenated to 256 dimensions, followed by 4 hidden dense layers of 128 neurons each with tanh activation and Xavier initialization, outputting scalar u. The optimizer is Optax Adam with cosine decay schedule from 1e-3 to 1e-5 over 60,000 total iterations. The loss function exactly matches L_total = L_PDE + 100*L_IC + 100*L_BC, where L_PDE is the mean squared inviscid Burgers residual (u_t + u*u_x)^2 on collocation points, L_IC is MSE against -sin(pi*x) at t=0, and L_BC is MSE between u(t,-1) and u(t,1) for periodicity. Training employs RAR curriculum: initial 2500 LHS collocation + 500 IC + 500 BC points; 60 cycles of 1000 iterations each, with refinement after each cycle (except last) by sampling 20,000 uniform candidates, selecting top 100 by absolute PDE residual, and appending to collocation set. Logging every 500 iterations includes total/PDE/IC/BC losses and relative L2 error on loaded reference data. | 39.20% | 35 | 5 | 10 | 10 | **60** |
| 27 | The implementation uses a FourierFeatureMLP architecture with Gaussian Fourier Features (sigma=5.0, 64 frequency pairs yielding 128 dimensions) followed by a deep MLP backbone of 6 hidden layers with 128 neurons each, tanh activation, and 1 output neuron, initialized with Xavier uniform. The optimizer is Adam with a cosine decay learning rate schedule starting at 1e-3, ending at 1e-5 over 60,000 steps, including global norm clipping. The loss function exactly assembles L_total = L_PDE + 100 * L_IC + 100 * L_BC, where L_PDE is the MSE of the viscous Burgers residual (u_t + u * u_x - nu * u_xx with nu=0.01/pi), L_IC is MSE against exact IC u(0,x)=-sin(pi x), and L_BC enforces periodic conditions via MSE(u(t,-1) - u(t,1)). Training follows RAR curriculum: 60 cycles of 1,000 iterations each (total 60,000 steps), refining every 1,000 steps by evaluating |residual| on 50,000 random candidates and adding the top 200 highest-residual points to the PDE collocation set (initial N_f=2500 LHS, N_ic=500, N_bc=500). Metrics (losses, relative L2 error on test data) logged every 500 iterations. | 15.18% | 35 | 20 | 13 | 14 | **82** |
| 28 | The implementation uses a Fourier Feature MLP architecture with Gaussian Fourier embedding (64 degrees, sigma=5.0 projecting (t,x) inputs to 128 dimensions), followed by 6 hidden layers of 128 neurons each using tanh activation and Xavier uniform initialization, outputting 1 dimension for u. Optimization employs Adam via Optax with cosine decay learning rate schedule from 1e-3 to 1e-5 over 60,000 total steps, including global gradient norm clipping (unspecified but compliant). The loss function strictly assembles L_total = L_PDE + 100 * L_IC + 100 * L_BC, where L_PDE is MSE(u_t + u * u_x) on collocation points (inviscid residual with no viscosity), L_IC is MSE(u(0,x) + sin(pi x)) on 500 IC points, and L_BC is MSE(u(t,-1) - u(t,1)) on 500 BC points per boundary. Training follows an aggressive RAR curriculum: 60 cycles of 1,000 iterations each (total 60k steps), refining PDE collocation points (starting from 2,500 LHS points) by sampling 100,000 uniform random candidates and appending the top 400 highest absolute residual points every 1,000 steps. Relative L2 error is computed and logged every 500 iterations against the loaded reference data from burgers.npz. | 44.00% | 35 | 5 | 10 | 5 | **55** |
| 29 | Fourier Feature MLP architecture with Gaussian Fourier embedding (degree=64, sigma=5.0 mapping to 128 dimensions), followed by 6 hidden layers of 128 neurons each using tanh activation and Xavier initialization, outputting scalar u. Optimizer is Adam with cosine decay schedule from 1e-3 to 1e-5 over 60,000 iterations, including global norm clipping. Loss function exactly matches L_total = L_PDE + 100*L_IC + 100*L_BC, where L_PDE is MSE on (u_t + u * u_x)^2 over dynamic collocation points (2500 base + 200 RAR), L_IC is MSE on predicted vs -sin(pi x) at t=0 (500 points), L_BC is MSE on u(t,-1) - u(t,1) (500 points). Implements Time-Marching Curriculum with T_max linearly ramped from 0.1 to 1.0 over first 50,000 steps, then fixed at 1.0; PDE points resampled every 1,000 iterations within [0, T_max] x [-1,1] using LHS, augmented by RAR selecting top 200 highest |r_PDE| from 50,000 candidates in current domain. Relative L2 error computed and logged every 500 iterations on loaded validation data. | 31.30% | 35 | 15 | 13 | 14 | **77** |
| 31 | The implementation uses an Adaptive Fourier Feature MLP architecture with Gaussian Fourier Features (sigma=5.0, 64 degree yielding 128 features: 64 sine + 64 cosine) followed by 5 hidden layers of 256 neurons each, employing adaptive tanh activations with per-layer trainable scalers alpha initialized to 1.0, and Xavier uniform initialization. The optimizer is Adam with cosine decay schedule from 1e-3 to 1e-5 over 60,000 steps, including global norm clipping. The loss function exactly matches L_total = L_PDE (MSE on u_t + u*u_x residual at dynamic collocation points from base LHS sampling + RAR buffer) + 100*L_IC (MSE on u(0,x) vs -sin(pi x) at 500 fixed spatial points) + 100*L_BC (MSE on u(t,-1) - u(t,1) at 500 resampled time points). Training employs a time-marching curriculum ramping T_end linearly from 0.1 to 1.0 over the first 40,000 iterations then holding at 1.0, with base collocation points (N_f=2500) resampled via LHS every iteration within [0, T_end] x [-1,1], BC points resampled every iteration within [0, T_end], and RAR every 1,000 iterations selecting top 500 points by absolute residual from 50,000 candidates to append to the collocation buffer. Metrics (total loss, components, relative L2 error vs reference data) are logged every 500 iterations. | 19.65% | 10 | 15 | 8 | 10 | **43** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 7 | Saves 'summary_all.png' (matplotlib visualization figure with loss histories, L2 error, PDE residual contour, reference/prediction/error contours). Prepares 'results_dict' with keys 'all_errors' (list of ~80 floats), 'all_its' (list of ~80 ints), 'all_loss' (list of ~80 floats), but no explicit save command executed in the provided code (intended for .mat but incomplete). No NPZ or other data files saved with solution grids. |
| 11 | No data saving is implemented in the code beyond printing metrics and generating plots (summary_all.png). An incomplete results_dict is defined with keys 'all_errors' (list of scalars), 'all_its' (list of scalars), 'all_loss' (list of scalars), but no file write occurs. The strategy does not specify saving requirements. |
| 12 | No .npz or .mat data saving implemented. Saves visualization plot 'summary_all.png' (containing loss history, L2 error, PDE residual heatmap, reference solution contour, prediction contour, and pointwise error heatmap). Defines 'results_dict' with keys 'all_errors' (list of ~100 floats), 'all_its' (list of ~100 ints), 'all_loss' (list of ~100 floats) but does not save it. |
| 17 | Saves visualization plot as 'summary_all.png' (matplotlib figure with loss histories, L2 error history, PDE residual heatmap, reference solution contour, prediction contour, and pointwise error heatmap). Defines results_dict = {'all_errors': list of ~100 relative L2 errors (scalars), 'all_its': list of iteration indices, 'all_loss': list of total loss values} but does not persist it to file. No .npz or solution data saved. |
| 20 | No NPZ or solution data saved to file as specified in strategy (which only requires loading burgers.npz for metrics). Saves visualization 'summary_all.png' (plots: loss history, L2 error history, |PDE residual| heatmap, reference solution contour, prediction contour, pointwise error heatmap). In-memory results_dict with keys 'all_errors' (list of ~100 floats), 'all_its' (list of ~100 ints), 'all_loss' (list of ~100 floats). Model params not saved. |
| 23 | Saves 'training_results.mat' (MATLAB format) containing keys: 'all_errors' (1D array of ~100 floats, relative L2 errors logged every 500 steps), 'all_its' (1D array of ~100 integers, global iteration counts), 'all_loss' (1D array of ~100 floats, total loss values). Additionally generates 'summary_all.png' with plots of losses, errors, reference, prediction, residual, and pointwise error fields, but primary artifacts are the MAT file metrics. |
| 24 | Saves 'training_results.mat' (MAT format) containing keys: 'all_errors' (1D array of ~100 floats, relative L2 errors logged every 500 steps), 'all_its' (1D array of ~100 integers, global iteration counts), 'all_loss' (1D array of ~100 floats, total losses). Additionally saves 'summary_all.png' (PNG image) with plots of loss/error histories, PDE residual field, reference solution, prediction, and pointwise error. |
| 25 | Saves 'training_results.mat' (MATLAB format) containing keys: 'all_errors' (1D array of relative L2 errors over logged iterations, shape varies e.g., ~100 elements), 'all_its' (1D array of global iteration counts at logs), 'all_loss' (1D array of total loss values at logs). No other artifacts saved (e.g., no model params, predictions, or npz files). |
| 26 | Saves 'training_results.mat' (MAT format) containing keys: 'all_errors' (1D array of ~120 relative L2 errors), 'all_its' (1D array of ~120 iteration indices), 'all_loss' (1D array of ~120 total loss values). Additionally generates 'summary_all.png' visualization, but no NPZ solution grids saved. |
| 27 | Saves 'training_results.mat' (MAT format) containing keys: 'all_errors' (1D array, length ~120 with relative L2 errors), 'all_its' (1D array, length ~120 with iteration numbers), 'all_loss' (1D array, length ~120 with total losses). Additionally generates 'summary_all.png' with plots of loss/error histories, PDE residual, reference solution, prediction, and pointwise error (no solution grids saved in npz format). |
| 28 | Saves 'training_results.mat' (MATLAB format) containing keys: 'all_errors' (1D array of ~120 floats, length depends on log frequency), 'all_its' (1D array of iteration indices), 'all_loss' (1D array of total loss values). Additionally saves 'summary_all.png' (visualization figure with loss/error histories, PDE residual, reference solution, prediction, and pointwise error plots). No solution grids or NPZ files with fields (t,x,u) are saved. |
| 29 | Saves 'training_results.mat' (MAT format) containing keys: 'all_errors' (array of shape ~120x1, relative L2 errors logged every 500 steps), 'all_its' (array of shape ~120x1, iteration numbers), 'all_loss' (array of shape ~120x1, total loss values). Additionally saves 'summary_all.png' (PNG visualization of loss history, L2 error history, PDE residual heatmap, reference solution contour, prediction contour, and pointwise error heatmap), but no solution grids or NPZ outputs. |
| 31 | Saves 'training_results.mat' (MAT format) containing keys: 'all_errors' (1D array of ~120 floats, logged relative L2 errors), 'all_its' (1D array of ~120 integers, iteration numbers), 'all_loss' (1D array of ~120 floats, total loss values). No other artifacts saved; reference data loaded but not saved. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 7 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2026_01_10_1448/` |
| 11 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2026_01_10_1456/` |
| 12 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2026_01_10_1501/` |
| 17 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2026_01_10_1519/` |
| 20 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2026_01_10_1528/` |
| 23 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2026_01_10_1536/` |
| 24 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2026_01_10_1542/` |
| 25 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2026_01_10_1548/` |
| 26 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2026_01_10_1602/` |
| 27 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2026_01_10_1611/` |
| 28 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2026_01_10_1623/` |
| 29 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2026_01_10_1635/` |
| 31 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2026_01_10_1642/` |


---
