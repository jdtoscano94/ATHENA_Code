
# Experiment Report
*Last Updated: 2026-01-09 18:12:07*

## User Request

We are solving a forward problem for the Korteweg-De Vries (KdV) equation:

$$
u_t + 6uu_x + u_{xxx} = 0
$$

**Domain:**
- Spatial: $x \in [0, 20]$
- Temporal: $t \in [0, 5]$

**Boundary Conditions:**

$$
u(0,x) = g_0(x) \quad 	ext{(Initial Condition)}
$$

$$
u(t,0) = g_1(t) \quad 	ext{(BC at } x=0	ext{)}
$$

$$
u(t,20) = g_2(t) \quad 	ext{(BC at } x=20	ext{)}
$$

$$
u_x(t,20) = g_3(t) \quad 	ext{(Neumann BC at } x=20	ext{)}
$$

**Analytical Solution:**

The solution is given by a two-soliton interaction formula:

$$
u(x,t) = rac{2(c_1 - c_2) \left[ c_1 \cosh^2\left(rac{\sqrt{c_2}\zeta_2}{2}ight) + c_2 \sinh^2\left(rac{\sqrt{c_1}\zeta_1}{2}ight) ight]}{\left[ \left(\sqrt{c_1} - \sqrt{c_2}ight) \cosh\left(rac{\sqrt{c_1}\zeta_1 + \sqrt{c_2}\zeta_2}{2}ight) + \left(\sqrt{c_1} + \sqrt{c_2}ight) \cosh\left(rac{\sqrt{c_1}\zeta_1 - \sqrt{c_2}\zeta_2}{2}ight) ight]^2}
$$


where:
- $\zeta_1 = x - c_1 t - x_1$
- $\zeta_2 = x - c_2 t - x_2$

**Parameters:**
- $c_1 = 6.0$ (velocity of first soliton)
- $c_2 = 2.0$ (velocity of second soliton)
- $x_1 = -2.0$ (initial position of first soliton)
- $x_2 = 2.0$ (initial position of second soliton)

The boundary conditions $g_0, g_1, g_2, g_3$ are derived by evaluating the analytical solution at the appropriate boundaries.

MANDATORY: Make sure to report the loss and errors every 500 iterations and you cannonly use optimizers that work natively in JAX (NO SciPY)
CONSTRAINTS: If using latin hypercube sampling dont use any criterion such as 'maximin'. 


---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
To the Planner:

The previous attempt (Run 16) achieved stable training but suffered from **Mode Collapse**, capturing only the dominant soliton and missing the second one, resulting in a 52% error. The Advisor recommends increasing model capacity, adjusting loss weights, and implementing **Residual-based Adaptive Refinement (RAR)** to force the model to resolve the missing physics.

Here is the scientific plan to solve the KdV equation.

### 1. Data Source & Problem Statement
**Governing Equation:**
The Korteweg-De Vries equation:
$$u_t + 6uu_x + u_{xxx} = 0$$
*   **Residual:** $f = \frac{\partial u}{\partial t} + 6u\frac{\partial u}{\partial x} + \frac{\partial^3 u}{\partial x^3}$

**Domain:**
$t \in [0, 5]$, $x \in [0, 20]$.
*   **Normalization:** You **MUST** normalize inputs $(t, x)$ to the range $[-1, 1]$ before passing them into the network layers.

**Boundary/Initial Conditions (Exact Data Generation):**
You must implement the analytical solution function `u_exact(t, x)` exactly as provided in the prompt to generate the training data for boundaries and initial conditions ($g_0, g_1, g_2, g_3$).
*   **IC:** $u(0, x)$ for $x \in [0, 20]$
*   **BC Left:** $u(t, 0)$ for $t \in [0, 5]$
*   **BC Right:** $u(t, 20)$ for $t \in [0, 5]$
*   **Neumann BC Right:** $u_x(t, 20)$ for $t \in [0, 5]$

### 2. Model Architecture
**Main Model:**
*   **Type:** Deep MLP (Multi-Layer Perceptron).
*   **Depth/Width:** **8 hidden layers** with **128 neurons** each (Increased capacity as per Advisor).
*   **Activation:** `tanh` (Hyperbolic Tangent).
*   **Output:** 1 neuron ($u$).
*   **Initialization:** Xavier/Glorot normal.

### 3. Optimization Strategy
*   **Optimizer:** `optax.adam` (JAX native).
*   **Learning Rate:** Cosine Decay Schedule.
    *   Start: `1e-3`
    *   End: `1e-5`
    *   Steps: 50,000
*   **Gradient Clipping:** Apply global gradient clipping with max norm `1.0` to maintain stability.

### 4. Training & Loss Formulation
**Loss Function:**
$$ \mathcal{L}_{total} = \lambda_{PDE}(i) \mathcal{L}_{PDE} + \lambda_{BC} (\mathcal{L}_{IC} + \mathcal{L}_{BC\_left} + \mathcal{L}_{BC\_right} + \mathcal{L}_{Neumann}) $$

**Weights:**
*   $\lambda_{BC} = 100.0$ (Reduced from 1000 to allow PDE physics to influence the interior more).
*   $\lambda_{PDE}(i)$: **Annealing Schedule**
    *   Iterations 0 - 5,000: $\lambda_{PDE} = 0.0$ (Pre-training on BCs/ICs).
    *   Iterations 5,000 - 10,000: Linearly ramp from $0.0 \to 1.0$.
    *   Iterations 10,000+: $\lambda_{PDE} = 1.0$.

**Enhancements (CRITICAL):**
1.  **Residual-based Adaptive Refinement (RAR):**
    *   **Initial Collocation:** 2,000 points (Latin Hypercube Sampling, `criterion=None`).
    *   **Refinement Loop:** Every **2,500 iterations** (starting after iter 10,000):
        1.  Generate a dense candidate set of random points (e.g., 10,000) in the domain.
        2.  Compute the PDE residual $|f|$ for all candidate points.
        3.  Select the **top 200 points** with the highest residual.
        4.  Add these 200 points to the training collocation set for subsequent iterations.
    *   This strategy explicitly targets the "missing soliton" region where errors are high.

### 5. Execution Plan
1.  **Define `u_exact`**: Implement the two-soliton formula carefully.
2.  **Data Prep**: Generate fixed BC/IC points (200 each). Generate initial Collocation points (2000).
3.  **Training Loop (50,000 steps)**:
    *   Update Learning Rate.
    *   Update $\lambda_{PDE}$.
    *   **If** `step % 2500 == 0` and `step >= 10000`: Perform RAR (evaluate residuals, add points).
    *   Compute Loss and Gradients.
    *   Update Parameters.
    *   **Every 500 steps**: Report Loss terms and Relative L2 Error against the analytical solution on a validation grid ($256_t \times 100_x$).

**Output Section:**
**Main Model:** Deep MLP (8 layers, 128 units, tanh), Optax Adam with Cosine Decay.
**Enhancements:** Residual-based Adaptive Refinement (RAR), Loss Annealing (PDE weight ramp-up), Gradient Clipping.

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
APPROVED

**Section 1: Status Decision**
*   **Status:** APPROVED

**Section 2: Quantitative Scorecard**
*   **Score Integrity:** 35/35
*   **Score Accuracy:** 35/35
*   **Score Details:** 15/15
*   **Score Optimality:** 15/15
*   **TOTAL SCORE:** 100/100

**DECISION LOGIC (Standard):**
This run is the first to achieve a score above the 90-point passing threshold, marking a significant breakthrough after previous failures. The solution is highly accurate and stable. Per protocol, this successful configuration now serves as the baseline for subsequent exploration runs.

**Section 3: Forensic Analysis & Feedback**
*   **Observation:**
    *   **Quantitative:** The training achieved an excellent final Relative L2 Error of **1.76%**, as confirmed by both the final point on the error history plot and the STDOUT log (`Minimum achieved RL2 error: 0.01757...`).
    *   **Qualitative:** The solution plots demonstrate a near-perfect match between the PINN prediction and the reference solution. The previous critical failure of "Mode Collapse" (Run 16) has been completely resolved. The model successfully captures both the fast and slow solitons, their individual propagation, and their complex, non-linear interaction phase.
    *   **Training Dynamics:** The loss history plot shows characteristic spikes that perfectly align with the Residual-based Adaptive Refinement (RAR) schedule (every 2500 iterations). These spikes indicate the successful injection of high-error points into the training set, forcing the model to correct its deficiencies. This is a clear sign that the RAR strategy was not only implemented correctly but was instrumental to the success of the run.

*   **Diagnosis:**
    The run is a textbook case of **Successful Complex Dynamics Capture**, directly attributable to a well-designed, multi-faceted strategy that correctly diagnosed and targeted the previous failure mode. The key factors were:
    1.  **Residual-based Adaptive Refinement (RAR):** This was the decisive element. By adaptively sampling points with high PDE residual, the training process was forced to resolve the physics of the second, weaker soliton and the interaction region, which it had previously ignored.
    2.  **Increased Model Capacity:** The deeper and wider network (8x128) provided the necessary representational power to model the complex two-soliton solution.
    3.  **Curriculum Learning:** The PDE loss annealing schedule allowed the model to first learn the simpler boundary conditions before tackling the more difficult PDE dynamics, ensuring a stable optimization path.

*   **Targeted Improvements:**
    The current result is outstanding. Future work should focus on efficiency and exploration rather than fixing flaws.
    *   **For Integrity:** N/A. The method is robust.
    *   **For Accuracy:** N/A. The accuracy is already excellent.
    *   **For Details:** N/A. The fine-grained features of the interaction are well-captured.
    *   **For Optimality:**
        *   **Efficiency Study:** Investigate if the same accuracy can be achieved with fewer RAR steps or a smaller network to reduce computational cost.
        *   **Alternative Architectures:** Now that a strong baseline is established, one could explore if architectures like a modified Fourier Network (M-FN) or a SIREN could achieve similar or better results with fewer parameters.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | The implementation uses a Fully Connected Neural Network (MLP) with 5 hidden layers of 64 neurons each and tanh activation. Inputs (t, x) are normalized to [-1, 1] inside the model's forward method for stability and differentiability. The optimizer is Optax Adam with an exponential decay learning rate schedule starting at 1e-3, decaying by 0.9 every 2000 steps to approximately 1e-5, over 40,000 iterations. Loss function assembles exactly as specified: L_total = 1.0 * MSE_PDE (u_t + 6*u*u_x + u_xxx on collocation points) + 10.0 * MSE_IC (Dirichlet at t=0) + 10.0 * MSE_BC (Dirichlet at x=0 and x=20) + 1.0 * MSE_Neumann (u_x at x=20), with ground truth from JAX-differentiated analytical two-soliton solution. Reporting occurs every 500 iterations, including relative L2 error on 256x100 validation grid. | 90.00% | 25 | 0 | 0 | 5 | **30** |
| 12 | The implementation uses a Fourier Feature MLP architecture: inputs (t,x) are normalized to [-1,1], mapped via Periodic Fourier Features (m=20 frequencies, sigma=1.0, sin/cos(2pi x B)) to 40D, followed by 4 hidden layers of 64 neurons each with tanh activation, outputting 1D u. Optimizer is Optax Adam with Curriculum Learning: Phase 1 (iterations 0-5000) uses constant LR=1e-3 on L_total = L_IC (MSE u at t=0) + L_BC_left (MSE u at x=0) + L_BC_right (MSE u at x=20) + L_Neumann (MSE u_x at x=20); Phase 2 (5001-50000) switches to L_total = 1.0 * L_PDE (MSE (u_t + 6 u u_x + u_xxx)) + 100.0 * (L_IC + L_BC_left + L_BC_right + L_Neumann) with exponential decay LR schedule (init=1e-3, decay every 1000 steps by 0.9, approaching 1e-5). Reports relative L2 error and losses every 500 iterations on 256x100 validation grid. | 90.00% | 35 | 0 | 5 | 5 | **45** |
| 16 | The implementation uses a standard MLP architecture with 5 hidden layers of 128 neurons each, tanh activation, single output for u, and critical input normalization to [-1,1] for t and x performed inside the model's __call__ method. The optimizer is optax.adam with cosine decay learning rate schedule from 1e-3 to 1e-5 over 50,000 iterations, chained with global gradient clipping at norm 1.0. The loss function exactly matches L_total = 1000.0 * (L_IC + L_BC_left_Dirichlet + L_BC_right_Dirichlet + L_Neumann_right) + lambda_PDE(i) * L_PDE, where L_PDE is MSE on collocation residuals (u_t + 6*u*u_x + u_xxx), and lambda_PDE follows the annealing schedule: 0.0 for iterations 0-4999, linearly ramps from 0.0 to 1.0 during 5000-14999, and fixed at 1.0 for 15000-49999. All derivatives use JAX automatic differentiation. Relative L2 error is computed and reported every 500 iterations on a 256(t) x 100(x) validation grid against the analytical solution. | 52.40% | 35 | 5 | 5 | 10 | **55** |
| 19 | The implementation uses a Deep MLP architecture with 8 hidden layers of 128 neurons each, tanh activation, and Glorot normal initialization. Inputs (t, x) are normalized to [-1, 1] inside the model's forward method. The optimizer is Optax Adam with a cosine decay learning rate schedule from 1e-3 to 1e-5 over 50,000 steps, chained with global gradient clipping at norm 1.0. The loss function is L_total = λ_PDE(i) * L_PDE + λ_BC * (L_IC + L_BC_left + L_BC_right + L_Neumann), where λ_BC=100.0 is fixed, and λ_PDE follows the curriculum: 0.0 for iterations 0-5000, linearly ramps from 0.0 to 1.0 during 5000-10000, then fixed at 1.0 thereafter. Residual-based Adaptive Refinement (RAR) is implemented every 2500 iterations starting from iteration 10000, evaluating PDE residuals on 10,000 random candidate points and adding the top 200 highest-residual points to the collocation set (initially 2000 LHS points). Training reports relative L2 error every 500 steps on a 256x100 validation grid against the analytical two-soliton solution. | 1.76% | 35 | 35 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves 'KdV_PINN_results.mat' (MATLAB format) containing keys: 'model_name' (string: 'KdV_PINN'), 'all_errors' (array ~80x1 of relative L2 errors), 'all_its' (array ~80x1 of iteration numbers), 'all_loss' (array ~80x1 of total losses), 'all_L_infty' (array ~80x1 of L-infty errors), 'all_L_2' (array ~80x1 of L2 norms). No solution fields or grids saved in npz; visualization plots generated but not persisted beyond 'summary_all.png'. |
| 12 | Saves KdV_PINN_Fourier_Curriculum_results.mat (MAT format) containing keys: 'model_name' (string), 'all_errors' (1D array ~100 elements), 'all_its' (1D array ~100 elements), 'all_loss' (1D array ~100 elements), 'all_L_infty' (1D array ~100 elements), 'all_L_2' (1D array ~100 elements). No solution fields or grids saved. |
| 16 | Saves 'KdV_PINN_MLP_Annealing_results.mat' in MATLAB format (.mat) containing keys: 'model_name' (string), 'all_errors' (list of ~100 relative L2 errors), 'all_its' (list of ~100 iteration numbers), 'all_loss' (list of ~100 total loss values), 'all_L_infty' (list of ~100 L-infty errors), 'all_L_2' (list of ~100 L2 norms). Additionally generates 'summary_all.png' with training plots (loss, error histories, contours of reference/prediction/error, PDE residual). No solution grids or NPZ files saved. |
| 19 | Saves KdV_PINN_MLP_Annealing_RAR_results.mat (MAT format) containing keys: 'model_name' (string), 'all_errors' (list of ~100 floats for relative L2 errors), 'all_its' (list of ~100 iteration indices), 'all_loss' (list of ~100 total loss values), 'all_L_2' (list of ~100 L2 norms), 'all_L_infty' (list of ~100 L-infinity errors). Additionally generates summary_all.png with loss histories, error plots, PDE residual heatmap, reference/prediction contours, and pointwise error map. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//KdV_Mixed_AnalyticalIC/2026_01_09_1717/` |
| 12 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//KdV_Mixed_AnalyticalIC/2026_01_09_1740/` |
| 16 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//KdV_Mixed_AnalyticalIC/2026_01_09_1752/` |
| 19 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//KdV_Mixed_AnalyticalIC/2026_01_09_1804/` |


---
