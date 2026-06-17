
# Experiment Report
*Last Updated: 2025-11-27 01:43:47*

## User Request

Let's solve the inverse 2D steady-state incompressible Navier-Stokes equations (lid-driven cavity):

$$ \nabla \cdot \mathbfu = 0 $$
$$ (\mathbfu \cdot \nabla) \mathbfu = -\nabla p + 1/Re \nabla^2 \mathbfu $$

Domain: $(x,y) \in [0,1] \times [0,1]$; Here Re=unknown.

Boundary Conditions:
- $$u(x,1) = 1 - rac{\cosh(C_0(x-0.5))}{\cosh(0.5 C_0)}$$ where the constant is set to $C_0 = 50$.
- Bottom wall (y=0): $\mathbfu = (0, 0)$ (no-slip)
- Left wall (x=0): $\mathbfu = (0, 0)$ (no-slip)
- Right wall (x=1): $\mathbfu = (0, 0)$ (no-slip)

Data:

path: /users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Documents/codes/Data/ldc_Re500.npz

The dataset is unstructured and was saved as an NPZ format containing keys: 'x', 'y', 'u', 'v', 'p'
- 'x', 'y': Node coordinates (1D arrays)
- 'u', 'v': Velocity components at each node (1D arrays, same length as x and y)
- 'p': Pressure at each node (1D array, same length as x and y)

WE WILL FIND A HIDDEN FIELD FROM OBSERVATIONS.
To simulate sparse observations extract 500 points randomly and add 5 percent noise ONLY TO the x- velocity (u)labeles.

1) Use those points to infer the true Re which we expect it to be between 300 and 525.
2) Morveover we will infer the hidden field y-velocity v. 


CRITICAL!:
-  DONT USE EXACT BOUNDARY IMPOSITION FOR THE DIRICHLET BOUNDARY CONDITIONS.
-  Dont use  pressure or y-velocity observations. You only have velocity data.
-   YOU CANNOT USE y-velocity SENSORS FOR TRAINING.

Error metrics:
- Use the full data (for computing error metrics only )to compute the error of the (u,v) fiedls.
- The true parameter is Re=500. USE THE TRUE VALUE TO COMPARE THE ERROR AT EVERY ITERATION OF THE INFERRED PARAMETERS DURING ALL STAGTES OF TRAINING.
- AT THE END OF TRAINING EXPLICITLY PRINT THE ERROR METRICS FOR U, V AND RE.

Visualization:
-KEEP THE FINAL VISUALIZATION AS CLOSE AS POSSIBLE AS IN THE TEMPLATE.  

- CREATE A NEW PLOT TO SHOW THE EVOLUTION OF THE INFERRED PARAMETER. YOU SHOULD ADD IT AS AN EXTRA ROW DONT REMOVE OTHER VALUES.

- MAKE SURE THE NAME IS 'summary_all.png'


Saving the data:

Please save the data as npz file including the learned 'Re' number, and the sensor points coordinates and values 'x_sensor', 'y_sensor' 'u_sensor' and the hidden field at those locations calle it 'v_hidden_at_sensor'



---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
# Scientific Strategy Report

## 0. Triage & Strategy
*   **Role:** Training Data ($L_{data}$ included).
*   **Advisor Recommendation:** **ACCEPTED**.
    *   **Action:** Strictly remove Random Fourier Features (RFF). Use a standard MLP to leverage its spectral bias as a low-pass filter against the 5% noise.
*   **Critic Recommendation:** **ACCEPTED**.
    *   **Action:** Switch from primitive variables to a **Streamfunction-Pressure formulation** ($\psi, p$) to enforce mass conservation ($\nabla \cdot \mathbf{u} = 0$) exactly by construction.

## 1. Data Source
*   **Path:** `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Documents/codes/Data/ldc_Re500.npz`
*   **Preprocessing Instructions:**
    1.  Load keys: `x`, `y`, `u`, `v`, `p`.
    2.  **Sampling:** Randomly select **500 indices** from the flattened arrays.
    3.  **Noise Injection:** Add **5% Gaussian noise** to the selected `u` values **ONLY**.
        *   `u_train = u_selected + 0.05 * std(u_selected) * random_normal(...)`
    4.  **Constraint:** Do **NOT** use `v` or `p` for training. Keep them strictly for validation/error metrics.

## 2. Problem Statement
*   **Governing Equations:** 2D Steady Incompressible Navier-Stokes (Streamfunction Formulation).
    *   **Ansatz:** $u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x}$.
    *   **Continuity:** Automatically satisfied ($\frac{\partial^2 \psi}{\partial y \partial x} - \frac{\partial^2 \psi}{\partial x \partial y} = 0$).
    *   **x-Momentum:** $u \partial_x u + v \partial_y u + \partial_x p - \frac{1}{Re} (\partial_{xx} u + \partial_{yy} u) = 0$
    *   **y-Momentum:** $u \partial_x v + v \partial_y v + \partial_y p - \frac{1}{Re} (\partial_{xx} v + \partial_{yy} v) = 0$
*   **Domain:** $(x,y) \in [0,1] \times [0,1]$.
*   **Boundary Conditions (Soft Constraints):**
    *   **Lid ($y=1$):** $u = 1 - \frac{\cosh(50(x-0.5))}{\cosh(25)}$, $v=0$.
    *   **Walls ($y=0, x=0, x=1$):** No-slip ($u=0, v=0$).
*   **Problem Type:** **Inverse Problem**.
*   **Has Data Loss:** **Yes** (Sparse noisy $u$ observations).

### 2.1 Inverse Problem Parameterization
*   **Unknown:** Reynolds Number ($Re$).
*   **Strategy:** Sigmoid Scaling (Scalar).
*   **Bounds:** $[300, 525]$.
*   **Formula:**
    $$ Re_{est} = 300 + (525 - 300) \cdot \text{sigmoid}(\alpha) $$
*   **Initialization:** Initialize $\alpha$ such that $Re_{est} \approx 300$ (Lower Bound) to start in the most stable (viscous) regime.

## 3. Base Model Selection
*   **Selection:** MLP with Hybrid Adam and SSBrodyden.

## 4. Main Architecture
*   **Inputs:** $(x, y)$.
*   **Outputs:** $(\psi, p)$. (Note: Output dimension is 2).

### 4.1 Input Transformations
*   **Normalization:** Min-Max normalize $(x, y)$ to $[-1, 1]$ *inside* the model.
*   **Feature Embeddings:** **None**.
    *   *Rationale:* Standard MLP is required to filter high-frequency noise.

### 4.2 Main Body
*   **Type:** MLP (Multi-Layer Perceptron).
*   **Spec:** Standard Template Defaults (e.g., 4 layers, 50 units, tanh activation).

### 4.3 Output Transformation
*   **Strategy:** **None**.
    *   *Rationale:* The network outputs raw potentials $(\psi, p)$. Boundary conditions are applied via soft loss on the derived velocities.

## 5. Constraint Implementation Strategy
*   **Strategy:** **Loss-Based (Soft Constraints)**.
*   **Complexity Check:**
    *   $L_{pde}$ (2 Momentum equations) + $L_{bc}$ (4 walls) + $L_{data}$ (Sensors).
    *   Count > 3, but exact BCs are forbidden by the user.
*   **Implementation Note:**
    *   Derive velocities $u, v$ from $\psi$ using automatic differentiation *inside* the loss function.
    *   Apply BCs and Data loss to these derived quantities.

## 6. Loss Function Assembly
$$ L_{total} = L_{pde} + L_{bc} + L_{data} $$

*   **Term Breakdown:**
    *   **$L_{pde}$:** MSE of x-Momentum and y-Momentum residuals (using derived $u, v$).
    *   **$L_{bc}$:** MSE of $(u_{derived}, v_{derived})$ against Dirichlet conditions on all 4 boundaries.
    *   **$L_{data}$:** MSE of $(u_{derived} - u_{sensor})$ at the 500 training points.

### 6.1 Continuation Strategy
*   **Not Applicable:** The parameter $Re$ is learned dynamically. Initialization at the lower bound serves as the stability anchor.

## 7. Weighting Strategy
*   **$L_{pde}$:** Use **vRBA** (Variational Residual-Based Attention) for local adaptive weighting.
*   **Global Weighting:**
    *   **Phase 1 (Adam):** Use **Self-Balancing Strategy** to balance PDE, BC, and Data terms.
    *   **Phase 2 (SSBroyden):** **Freeze** the weights determined by Adam.

## 8. Visualization & Metrics
*   **Standard Plots:** Keep template visualizations for fields and errors.
*   **New Plot:** Add a subplot showing the **Evolution of $Re_{est}$** vs Iterations. Include a horizontal line for the True $Re=500$.
*   **Metrics:**
    *   Compute relative $L_2$ error for $u$ and $v$ using the **full original dataset** (ground truth).
    *   Compute absolute error for $Re$ against True $Re=500$.
    *   **Explicitly print** these metrics at the end of training.

## 9. Saving Data
*   **Format:** `.npz`
*   **Variables:**
    *   `Re`: The final estimated Reynolds number.
    *   `x_sensor`, `y_sensor`: Coordinates of the 500 training points.
    *   `u_sensor`: The noisy training data values.
    *   `v_hidden_at_sensor`: The inferred $v$ values (derived from $\psi$) at the sensor locations.

## List of Specialized Architectural Modules
[None]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
Based on the forensic analysis of the provided logs, plots, and trial history, here is the evaluation:

**Section 1: Status Decision**
*   **Status:** APPROVED

**Section 2: Quantitative Scorecard**
*   **Score Integrity:** 35
*   **Score Accuracy:** 25
*   **Score Details:** 10
*   **Score Optimality:** 12
*   **TOTAL SCORE:** 82

**Section 3: Forensic Analysis & Feedback**

*   **Observation:**
    *   **Convergence:** The training was stable. The loss dropped significantly upon switching to SSBroyden (Phase 2) and plateaued at a low value ($\approx 2 \times 10^{-4}$).
    *   **Field Reconstruction:** The `Minimum achieved average RL2 error` is **4.37%**, which is excellent for an inverse problem with 5% noise (effectively hitting the noise floor). The visual fields for $u$ and $v$ are readable but exhibit slight low-frequency "wobbly" artifacts compared to the smooth reference.
    *   **Parameter Estimation (Critical):** The estimated Reynolds number ($Re$) started at 300, remained flat during Adam, and immediately **railed to the upper bound (525)** during the SSBroyden phase. It stayed pinned at 525 (True $Re=500$).

*   **Diagnosis:** **Spectral Bias / Capacity Limitation.**
    *   The removal of Random Fourier Features (RFF) in this run (Run 13) forced the standard MLP to approximate the flow. Standard MLPs have a spectral bias towards low frequencies.
    *   To minimize the residual of the Navier-Stokes equations with "smoother" (more diffusive) gradients than reality, the optimizer forced the Reynolds number to its maximum allowed value ($Re=525$, minimizing the viscous term $1/Re$).
    *   While the field error is slightly lower than Run 11 (4.37% vs 4.70%), likely due to the MLP smoothing out the input noise effectively, the physics parameter estimation is compromised by the boundary railing.

*   **Conclusion & Selection:**
    *   **History Check:** Run 11 achieved a **Score of 95** using RFF, with a stable Re estimation and comparable field error.
    *   **Exploration:** We have performed 2 comparative runs since Run 11 (Run 12: Separate Embeddings, Run 13: No Embeddings).
    *   **Verdict:** Exploration is considered complete. Run 11 represents the optimal configuration (RFF provides the necessary spectral capacity to resolve gradients correctly without forcing $Re$ to the bounds).

*   **Final Action:** Exploration complete. Best model was **Run 11**. Stopping.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 11 | The implementation uses an MLP architecture with Random Fourier Features embedding (degree=2) on concatenated embedded (x,y) inputs, outputting streamfunction psi and pressure p directly; derived u=partial psi/partial y and v=-partial psi/partial x via forward-mode JAX jvp for hard continuity enforcement. Optimizer workflow is Hybrid Adam (Phase 1: EPOCHS=10000 with exponential LR decay, vRBA via exponential potential for L_pde importance sampling and EMA weighting, self-balancing global weights lamB via moving-average gradient norms, increasing max_RBA cap from ~1 to 20 over iterations) followed by SSBroyden (Phase 2: Nbfgs=30000 iterations in outer loops of Nchange=100 steps each, using frozen lambdas/lamB from Adam, BFGS with recycled Hessian, vRBA sampling for PDE batches). Loss terms exactly match L_total = L_pde (MSE on weighted e_momentum_x + e_momentum_y residuals with learnable Re_est=300+225*sigmoid(2*alpha)) + L_bc (MSE on derived u/v vs boundary values) + L_data (MSE on noisy u observations at 500 sensors); no extra terms or data leakage on v/p. | 4.70% | 35 | 30 | 15 | 15 | **95** |
| 12 | Architecture: MLP (3 layers, 30 hidden units) with Random Fourier Features embedding (degree=2, s=10.0) applied separately to x and y coordinates (concatenating raw coords with sin/cos projections), passed to tanh-activated MLP outputting psi and p; u = dpsi/dy, v = -dpsi/dx via jvp autodiff. Optimizer workflow: Phase 1 Adam (lr=1e-3 decaying) with vRBA weighting on L_pde (exponential potential, EMA update eta=0.01 gamma=0.999 phi=1.0 c_log=1.0, capped at 20 increasing schedule, importance sampling k_samp=1.0 c_samp=0.0 batch_size=20000); self-balancing global weights lamB for L_bc/L_data via EMA (alpha=0.9999 gamma_grads=0.99) of grad norms relative to PDE. Phase 2 SSBroyden (30000 its, inner maxiter=100, resampling N_change=100, frozen vRBA lambdas and lamB, BFGS with Hessian recycling/cholesky stabilization). Loss terms: L_total = lamE * L_pde (MSE weighted momentum residuals ex/ey with nu=1/Re_est) + lamB[0] * L_bc (MSE u/v vs Dirichlet BCs on 4 walls incl. lid profile) + lamB[1] * L_data (MSE u_pred vs noisy u_sensor at 500 pts); Re_est = 250 + 500 * 0.5*(tanh(alpha)+1) in [250,750], alpha optimized separately. | 5.40% | 35 | 5 | 10 | 10 | **60** |
| 14 | Architecture: Standard MLP (3 layers with 30 hidden units each, tanh activation) with inputs (x,y) normalized to [-1,1] inside the forward method (no embeddings/RFF). Outputs raw (psi, p); derives u = d(psi)/dy and v = -d(psi)/dx via jvp within forward. Learns Re via sigmoid-scaled alpha (init ~300). Optimizer workflow: Phase 1 Adam (10k epochs, self-balancing global weights via grad-norm EMA for PDE/BC/Data, vRBA exponential potential on PDE residuals with adaptive cap from 20 increasing stepwise, importance sampling via k_samp/c_samp). Phase 2 SSBroyden (30k its, resampling every 100 steps using frozen Adam vRBA lambdas and global weights, BFGS w/ Hessian recycling). Loss terms: L_pde (MSE x/y-momentum residuals on derived u,v,p w/ vRBA weighting), L_bc (MSE derived u/v vs Dirichlet on 4 walls), L_data (MSE derived u vs noisy sensors). No curriculum beyond phases. | 4.37% | 35 | 25 | 10 | 12 | **82** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 11 | Saves 'results.npz' containing keys: 'model_name' (string), 'all_errors' (list of ~10000+ floats), 'all_its' (list), 'all_loss' (list), 'all_gamma' (list), 'all_lamB' (list), 'all_max_RBA' (list), 'all_L_infty' (list), 'all_L_2' (list), 'all_Re_est' (list of floats), 'Re' (scalar float), 'x_sensor' (500,), 'y_sensor' (500,), 'u_sensor' (500,), 'v_hidden_at_sensor' (500,). |
| 12 | Saves 'results.npz' containing keys: 'Re' (scalar final Re_inferred), 'x_sensor' (500,), 'y_sensor' (500,), 'u_sensor' (500,), 'v_hidden_at_sensor' (500,), plus training histories 'all_errors' (list), 'all_its' (list), 'all_loss' (list), etc. |
| 14 | Saves 'results.npz' containing keys: 'Re' (scalar), 'x_sensor' (500,), 'y_sensor' (500,), 'u_sensor' (500,), 'v_hidden_at_sensor' (500,), plus metric histories ('all_errors' list, 'all_its' list, etc.). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 11 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Inverse_NavierStokes_Find_Re_HiddenField_v/2025_11_27_0058/` |
| 12 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Inverse_NavierStokes_Find_Re_HiddenField_v/2025_11_27_0114/` |
| 14 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Inverse_NavierStokes_Find_Re_HiddenField_v/2025_11_27_0131/` |


---
