
# Experiment Report
*Last Updated: 2026-01-10 18:07:35*

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
*   **Path:** `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes/Burgers_Periodic_SinPiX/2025_11_22_1730/burgers.npz`
*   **Format:** NPZ file.
*   **Variables:** Load `t` (time coordinates), `x` (spatial coordinates), and `usol` (exact solution grid).
*   **Usage:** Use this data solely for computing the Relative L2 Error metric during logging. Do not use it for training.

**2. Problem Statement**
*   **Governing Equation:** 1D Inviscid Burgers' Equation.
    $$ u_t + u u_x = 0 $$
    *   **PDE Residual:** $r(t,x) = \partial_t u + u \cdot \partial_x u$
*   **Domain:** $x \in [-1, 1]$, $t \in [0, 1]$.
*   **Initial Condition (Dirichlet):**
    $$ u(0, x) = -\sin(\pi x) $$
*   **Boundary Conditions (Periodic):**
    $$ u(t, -1) = u(t, 1) $$

**3. Model Architecture**
*   **Type:** Fourier Feature MLP (High Capacity).
*   **Input Embedding:** Gaussian Fourier Features.
    *   Map inputs $(t, x)$ to Fourier space.
    *   **Scale ($\sigma$):** 10.0 (Increased to capture sharp shock gradients).
    *   **Dimensions:** 64 sine/cosine pairs (Output dimension = 128).
*   **Backbone:**
    *   **Layers:** 8 hidden layers (Increased depth).
    *   **Neurons:** 256 neurons per layer (Increased width).
    *   **Activation:** `tanh`.
    *   **Initialization:** Xavier/Glorot Uniform.
*   **Output:** Single scalar $u$.

**4. Optimization Strategy**
*   **Optimizer:** Adam (via Optax).
*   **Learning Rate:** Cosine Decay Schedule.
    *   Start: 1e-3
    *   End: 1e-5
    *   Steps: 80,000
*   **Gradient Clipping:** Global norm clip at 1.0.
*   **Iterations:** 80,000 total.

**5. Training Curriculum & Sampling (Critical Updates)**
*   **Time-Marching Curriculum:**
    *   Define a variable upper bound for time sampling, $T_{end}$.
    *   Ramp $T_{end}$ linearly from 0.1 to 1.0 over the first **60,000** iterations.
    *   Hold $T_{end} = 1.0$ for the remaining 20,000 iterations.
*   **Base Collocation Sampling (Low Frequency):**
    *   **Resample Frequency:** Every **5,000** iterations (Stabilization fix).
    *   **PDE Points:** 2,500 points sampled via Latin Hypercube Sampling (LHS) within domain $[0, T_{end}] \times [-1, 1]$.
    *   **BC Points:** 500 random time points $t \in [0, T_{end}]$.
    *   **IC Points:** 500 fixed points at $t=0$.
*   **Residual-based Adaptive Refinement (RAR):**
    *   **Frequency:** Every **5,000** iterations (Synchronized with base resampling).
    *   **Candidates:** Sample 100,000 dense uniform random points in $[0, T_{end}] \times [-1, 1]$.
    *   **Selection:** Calculate absolute PDE residual $|r|$ for all candidates. Select the top **500** points with the highest residuals.
    *   **Buffer:** Append these points to a FIFO buffer (Max size: 4,000).
    *   **Training Set:** In every iteration, use (Current Base PDE Points + All Points in RAR Buffer).

**6. Loss Function**
*   **Formulation:**
    $$ \mathcal{L}_{total} = \mathcal{L}_{PDE} + \lambda_{IC} \mathcal{L}_{IC} + \lambda_{BC} \mathcal{L}_{BC} $$
*   **Terms:**
    *   $\mathcal{L}_{PDE}$: **Huber Loss** ($\delta=0.1$) of the residual $u_t + u u_x$. (Huber is crucial for shock stability).
    *   $\mathcal{L}_{IC}$: MSE of $(u(0, x) - (-\sin(\pi x)))$.
    *   $\mathcal{L}_{BC}$: MSE of $(u(t, -1) - u(t, 1))$.
*   **Weights:**
    *   $\lambda_{IC} = 100.0$
    *   $\lambda_{BC} = 100.0$

**7. Output & Logging**
*   **Frequency:** Every 500 iterations.
*   **Metrics:** Total Loss, PDE Loss, IC Loss, BC Loss, Relative L2 Error (computed against the loaded `burgers.npz` data).

**Main Model:** Fourier Feature MLP (sigma=10.0, 8 layers, 256 units) with Adam optimizer (Cosine Decay).
**Enhancements:** Stabilized Time-Marching Curriculum (60k ramp), Low-Frequency Resampling (5k steps), Aggressive RAR (Top 500/100k), Huber Loss for shocks.

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
REJECTED_FOR_REFINEMENT

**Section 1: Status Decision**
*   **Status:** REJECTED_FOR_REFINEMENT

**Section 2: Quantitative Scorecard**
*   **Score Integrity:** 10/35
*   **Score Accuracy:** 0/35
*   **Score Details:** 0/15
*   **Score Optimality:** 5/15
*   **TOTAL SCORE:** 15/100

**DECISION LOGIC (Standard):**
The model failed to converge to a physically meaningful solution, achieving a final relative L2 error greater than 100%. The training process was plagued by severe, periodic instabilities directly linked to the chosen sampling methodology. This constitutes a fundamental failure requiring a significant strategic revision.

**Section 3: Forensic Analysis & Feedback**
*   **Observation:**
    *   **Loss & Error History:** The total loss plot exhibits extreme periodic instability, with sharp spikes occurring precisely every 5,000 iterations. The relative L2 error starts high (~85%), quickly jumps to over 100%, and remains there for the majority of the training, indicating a complete failure to learn the correct solution.
    *   **Solution Fields:** The predicted solution is qualitatively incorrect. It is a noisy, blocky field that bears no resemblance to the reference solution's characteristic wave steepening and shock formation. The pointwise error plot confirms that the error is O(1) across almost the entire domain. The PDE residual plot is noisy and unstructured, suggesting the model is only locally minimizing the loss on specific point clouds without learning the global structure.

*   **Diagnosis:**
    The root cause of this failure is **Catastrophic Training Instability induced by the infrequent resampling strategy**.
    *   The decision to resample the entire set of base collocation points only every 5,000 iterations is the primary flaw. The optimizer spends 5,000 steps adapting to a static loss landscape defined by one set of points. When this set is abruptly replaced, the loss function changes drastically, causing the loss to spike and effectively resetting the learning process. This prevents the model from ever converging to a stable, generalizable solution that satisfies the PDE across the entire domain.
    *   The benefits of advanced techniques like the time-marching curriculum and RAR are completely nullified by this core instability. The model cannot learn the physics progressively when its training objective is radically altered at fixed intervals.

*   **Targeted Improvements:**
    *   **For Integrity:**
        *   **CRITICAL:** Modify the sampling strategy to resample the base collocation points **every iteration**. This is the standard and most effective approach. It provides a smooth and consistent training signal, allowing the optimizer to make steady progress. The current strategy of resampling every 5,000 iterations is the direct cause of the failure and must be abandoned.
    *   **For Accuracy:**
        *   Accuracy is impossible without a stable training process. After fixing the resampling frequency, the existing high-capacity model and curriculum have a much better chance of succeeding. If the error is still high, consider a more gradual time-marching ramp (e.g., over 70,000 of the 80,000 iterations) to give the model more time to adapt to the evolving shock.
    *   **For Details:**
        *   This is not a primary concern until the model can produce a qualitatively correct solution.
    *   **For Optimality:**
        *   The entire concept of "low-frequency resampling" for the base collocation points has been proven detrimental here. The optimal strategy is high-frequency (per-iteration) resampling to ensure the model is constantly challenged to satisfy the PDE over the entire continuous domain, rather than on discrete, static point sets.

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
| 35 | The implementation uses a Fourier Feature MLP architecture with Gaussian Fourier Features (degree=64, sigma=10.0) mapping (t,x) to 128 dimensions, followed by 6 hidden layers of 128 neurons each with tanh activation and Xavier Uniform initialization, outputting scalar u. Optimization employs Adam with cosine decay schedule (1e-3 to 1e-5 over 80,000 iterations) and global gradient norm clipping at 1.0. Training follows a time-marching curriculum where T_end ramps linearly from 0.1 to 1.0 over the first 50,000 iterations. Base sampling resamples 2500 LHS collocation points and 500 BC time points every 1000 iterations within [0, T_end] x [-1,1]. RAR every 2500 iterations samples 50,000 uniform random candidates, selects top 100 by absolute PDE residual, and appends to a persistent buffer (training on base + buffer). Loss is L_total = Huber(delta=0.1) on PDE residuals + 100 * MSE on fixed 500 IC points (vs -sin(pi x)) + 100 * MSE on periodic BC (u(t,-1) - u(t,1)). JIT-compiled updates, logging every 500 iterations includes total/component losses and relative L2 error vs reference data. | 60.00% | 15 | 5 | 5 | 10 | **35** |
| 37 | The implementation uses a Fourier Feature MLP architecture with Gaussian Fourier Features (sigma=5.0, degree=64 yielding 128 features: 64 sine + 64 cosine), followed by a backbone of 6 hidden layers with 128 neurons each, tanh activations, and Xavier/Glorot uniform initialization, outputting a single scalar u. The optimizer is Adam via Optax with a cosine decay schedule from initial LR 1e-3 to final 1e-5 over 80,000 total iterations, including global gradient norm clipping at max_norm=1.0. The loss function exactly matches L_total = L_PDE (Huber loss delta=0.1 on |u_t + u u_x|) + 100 * L_IC (MSE on exact IC u(0,x)=-sin(pi x)) + 100 * L_BC (MSE on u(t,-1) - u(t,1)). Training employs a time-marching curriculum linearly ramping T_end from 0.1 to 1.0 over the first 70,000 iterations then holding at 1.0; base collocation uses 5,000 LHS points resampled every 1,000 iterations within [0, T_end] x [-1,1] combined with persistent RAR buffer; RAR runs every 5,000 iterations sampling 50,000 uniform candidates and appending the top 50 highest |PDE residual| points to the buffer; IC uses 500 fixed points at t=0; BC uses 500 points resampled every 1,000 iterations at x=-1/1 within [0, T_end]. Validation uses loaded npz data for relative L2 error. | 28.71% | 30 | 15 | 5 | 14 | **64** |
| 39 | The implementation uses a Fourier Feature MLP architecture with Gaussian Fourier Features (degree=64 pairs, sigma/s=10.0 for high-frequency shock capture) embedding concatenated (t,x) inputs, followed by an 8-layer backbone of 256 tanh-activated neurons each (Xavier init) outputting scalar u. Optimizer is optax.adam with cosine decay schedule (1e-3 start to 1e-5 end over 80,000 total steps) and global gradient norm clipping at 1.0. Loss is L_total = Huber(delta=0.1) on PDE residual (u_t + u u_x via JVP) + 100 * MSE IC (vs -sin(pi x)) + 100 * MSE periodic BC (u(t,-1) - u(t,1)). Training follows aggressive curriculum: linear time-marching ramp of T_max from 0.1 to 1.0 over first 40,000 iterations then hold at 1.0; every 1,000 iterations resample 5,000 LHS PDE base points and 500 uniform BC points in [0,T_max]x[-1,1] (IC fixed 500 points), perform RAR on 100,000 uniform candidates selecting top 200 by |r_PDE| appended to persistent buffer (training on base + buffer); logs every 500 steps with relative L2 on loaded validation data. | 60.00% | 10 | 0 | 0 | 5 | **15** |
| 44 | The implementation uses a Fourier Feature MLP architecture with Gaussian Fourier Features (64 degree, sigma=5.0, output dim 128) followed by a 6-layer backbone of 128 neurons each, tanh activations, and Xavier uniform initialization, outputting a scalar u. Optimization employs Adam via Optax with cosine decay learning rate schedule (5e-4 to 1e-5 over 80,000 steps) and global gradient norm clipping at 1.0. The loss function exactly matches L_total = Huber(delta=0.1, mean) on PDE residual (u_t + u u_x) + 100 * MSE on IC (u(0,x) + sin(pi x)) + 100 * MSE on periodic BC (u(t,-1) - u(t,1)). Training follows a time-marching curriculum ramping T_max linearly from 0.1 to 1.0 over the first 60,000 iterations then holding at 1.0; base points (N_PDE=2500 LHS no maximin + N_BC=500) resampled every 1,000 steps within [0,T_max]; fixed N_IC=500 at t=0; RAR every 2,500 steps selects top 50/50k candidates by |residual| into FIFO buffer (max 800, truncate oldest); trains on base + buffer points each step. Relative L2 error computed on loaded test data during logging. | 38.00% | 25 | 5 | 5 | 10 | **45** |
| 45 | Fourier Feature MLP with Gaussian Fourier embedding (sigma=5.0, 128 features from 64 sin/cos pairs), 6 hidden layers of 128 units each with tanh activation and Xavier uniform initialization. Optimizer is Adam with cosine decay learning rate schedule from 1e-3 to 1e-5 over 80,000 steps and global gradient norm clipping at 1.0. Loss function is L_total = Huber(delta=0.1) on inviscid Burgers residual (u_t + u u_x) + 100 * MSE on IC (-sin(pi x)) + 100 * MSE on periodic BC (u(t,-1) - u(t,1)). Training uses stabilized time-marching curriculum ramping T_end linearly from 0.1 to 1.0 over first 70,000 iterations then holding at 1.0; base 2,500 LHS collocation points resampled every 5,000 iterations within [0, T_end] x [-1,1]; RAR every 5,000 iterations selecting top 200 highest |residual| points from 50,000 candidates into FIFO buffer of max 2,000 points; training set is base + RAR buffer. | 38.04% | 35 | 5 | 5 | 10 | **55** |
| 46 | The code implements a Fourier Feature MLP architecture with Gaussian Fourier embedding (sigma=5.0, degree=64 yielding 128 features: cos(2pi B v) and sin(2pi B v) where B ~ N(0,25)), followed by a backbone of 6 hidden layers with 128 neurons each using tanh activation and Xavier uniform initialization, outputting a single scalar u. The optimizer workflow uses Optax Adam with cosine decay schedule from 1e-3 to 1e-5 over 80,000 steps, preceded by global gradient norm clipping at 1.0. The training employs a time-marching curriculum ramping T_max linearly from 0.1 to 1.0 over the first 50,000 iterations then holding at 1.0; base sampling resamples N_PDE=2500 LHS points and N_BC=500 uniform time points every 1,000 iterations within [0, T_max] x [-1,1]; RAR every 1,000 iterations generates 50,000 uniform candidates, selects top 200 by absolute PDE residual, adds to a FIFO buffer (max 4000 points, truncating oldest); training set is union of base and buffer points. Loss is L_total = Huber(delta=0.1, mean) on PDE residual (u_t + u u_x) + 100 * MSE_IC (vs -sin(pi x)) + 100 * MSE_BC (u(t,-1) - u(t,1)). Metrics (losses, relative L2 error vs reference data) logged every 500 iterations. | 38.00% | 25 | 5 | 5 | 10 | **45** |
| 48 | Fourier Feature MLP architecture with Gaussian Fourier embedding (sigma=10.0, 64 degree pairs outputting 128 features), followed by 8 hidden layers of 256 neurons each using tanh activation and Xavier uniform initialization, outputting a single scalar u. Optimizer workflow uses Adam with cosine decay learning rate schedule from 1e-3 to 1e-5 over 80,000 steps and global gradient norm clipping at 1.0. Loss terms exactly match L_total = Huber(delta=0.1) on PDE residual (u_t + u*u_x) + 100*MSE_IC (against -sin(pi x) at t=0) + 100*MSE_BC (u(t,-1) - u(t,1)). Training employs time-marching curriculum ramping T_end linearly from 0.1 to 1.0 over first 60,000 iterations then holding at 1.0; base collocation resampling every 5,000 steps (2,500 LHS PDE points + 500 IC fixed + 500 BC times in [0,T_end]); RAR every 5,000 steps (100,000 uniform candidates, top 500 by |residual| appended to FIFO buffer max 4,000, combined with base PDE points). Logging every 500 iterations includes total/PDE/IC/BC losses and relative L2 error. | >1.0 | 10 | 0 | 0 | 5 | **15** |


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
| 35 | Saves 'training_results.mat' (MAT format) containing keys: 'all_errors' (1D array of length ~160, relative L2 errors logged every 500 iterations), 'all_its' (1D array of length ~160, iteration numbers), 'all_loss' (1D array of length ~160, total losses). No model parameters or solutions saved; visualization saved as 'summary_all.png'. |
| 37 | Saves 'training_results.mat' (MAT format) containing keys: 'all_errors' (1D array of ~160 floats, relative L2 errors logged every 500 steps), 'all_its' (1D array of ~160 integers, iteration numbers), 'all_loss' (1D array of ~160 floats, total losses). Additionally saves 'summary_all.png' (PNG image) with plots of loss histories, L2 error history, PDE residual field, reference solution, prediction, and pointwise error. |
| 39 | Saves 'training_results.mat' (MAT format) containing keys: 'all_errors' (1D array of ~160 floats, relative L2 errors logged every 500 steps), 'all_its' (1D array of ~160 integers, iteration numbers), 'all_loss' (1D array of ~160 floats, total losses). Additionally generates 'summary_all.png' with training plots (loss/error histories, PDE residual, reference/prediction/error fields). Loads but does not overwrite input 'burgers.npz'. |
| 44 | Saves 'training_results.mat' (MAT format) containing keys: 'all_errors' (1D array, length ~160 floats), 'all_its' (1D array, length ~160 ints), 'all_loss' (1D array, length ~160 floats). Additionally generates 'summary_all.png' visualization but no NPZ or model params saved. |
| 45 | Saves 'training_results.mat' (MAT format) containing keys: 'all_errors' (1D array, length ~160), 'all_its' (1D array, length ~160), 'all_loss' (1D array, length ~160). Also saves visualization plot 'summary_all.png' with loss histories, L2 error history, PDE residual heatmap, reference solution contour, prediction contour, and pointwise error heatmap. No solution grids or NPZ files saved. |
| 46 | Saves 'training_results.mat' (MAT format) containing keys: 'all_errors' (1D array of ~160 floats, relative L2 errors logged every 500 steps), 'all_its' (1D array of ~160 integers, iteration numbers), 'all_loss' (1D array of ~160 floats, total losses). Additionally generates and saves visualization plot 'summary_all.png' with loss histories, L2 error history, PDE residual heatmap, reference solution contour, prediction contour, and pointwise error heatmap. No NPZ solution save. |
| 48 | Saves 'training_results.mat' (MAT format) containing keys: 'all_errors' (1D array of ~160 floats), 'all_its' (1D array of ~160 integers), 'all_loss' (1D array of ~160 floats). Additionally saves visualization 'summary_all.png' with loss histories, L2 error history, PDE residual heatmap, reference solution contour, prediction contour, and pointwise error heatmap. |


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
| 35 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2026_01_10_1654/` |
| 37 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2026_01_10_1704/` |
| 39 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2026_01_10_1712/` |
| 44 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2026_01_10_1737/` |
| 45 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2026_01_10_1743/` |
| 46 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2026_01_10_1749/` |
| 48 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Burgers_Periodic_SinPiX/2026_01_10_1758/` |


---
