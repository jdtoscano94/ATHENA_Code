
# Experiment Report
*Last Updated: 2026-11-06 14:07:59*

## User Request
Lets solve the 2D Helmholtz equation.

The equation is:
$$
u_{xx} + u_{yy} + k^2 u = g(x, y)
$$

where g(x, y)  is chosen to satisfy the  solution: u_{an}=sin(pi a1)sin (pi a2)

The spatio-temporal domain is $(x, y) \in [-1, 1] \times [-1, 1]$.
 
The boundary conditions are **periodic** in both $x$ and $y$. And let a_1=1 and a_2=4 and k=1.

---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
### Strategy Plan

**1. Data Source**
* Role: Validation/Error Metrics ONLY.

**2. Problem Statement**
* **Governing Equation (2D Helmholtz):** 
  $u_{xx} + u_{yy} + k^2 u - g(x, y) = 0$
  where $k = 1$.
* **Analytical Solution:** $u_{an}(x,y) = \sin(\pi x) \sin(4\pi y)$
* **Forcing Term:** $g(x,y) = (1 - 17\pi^2) \sin(\pi x) \sin(4\pi y)$
* **Domain:** $x \in [-1, 1]$, $y \in [-1, 1]$.
* **Boundary Conditions:** Periodic in both $x$ and $y$.
* **Problem Type:** Forward. Well-posed for PINNs.

**3. Base Model Selection**
* Hybrid Adam + SSBroyden.

**4. Main Architecture**
* **Structure:** Input -> Input Transformation (Embedding) -> Main Body -> Output.
* **Input Transformations:**
  * $x$ and $y$ are already in $[-1, 1]$. **DO NOT** normalize them.
  * Apply **Periodic Fourier Features** to both $x$ and $y$.
  * **CRITICAL:** Since raw inputs are in $[-1, 1]$, set `Lx = 2.0` and `Ly = 2.0`.
  * Set `degree = 2`.
  * **CRITICAL:** Pass **ONLY** the embedded features to the main body. Do not pass the raw $x$ or $y$ coordinates.
* **Main Body:** Base MLP with `width=30`, `depth=3`.
* **Output Transformation:** None. The periodic embeddings inherently enforce the periodic boundary conditions.

**5. Constraint Implementation Strategy**
* **Count:** `L_pde` (1).
* **Strategy:** Periodic BCs are strictly enforced by the Periodic Fourier Features. No boundary loss terms are required.

**6. Loss Function Assembly**
* **Total Loss:** $L_{total} = L_{pde}$
* **Iteration Budget (ADVISOR MANDATE):**
  * **Phase 1 (Adam):** 10,000 iterations.
  * **Phase 2 (SSBroyden):** **80,000 iterations** (Increased per Advisor to ensure full convergence).

**7. Weighting Strategy**
* **Local Weights (`lambdas`):** Use `vRBA` for local adaptive weighting inside the PDE loss. `lambdas` are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden.
* **Global Weights:** Only 1 loss term exists, so no global balancing is needed.

**8. Visualization Strategy**
* Do not modify the default visualization callbacks.

**9. Saving Data**
* Save the predicted and reference fields as `.npz` arrays with keys specifying the variable names for visualization.

### Required Modules
Main Model:
[MLP with Hybrid Adam and SSBrodyden]
Enhancements:
[Periodic Fourier Features]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
* APPROVED

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
  1. **Loss & Error History:** The training exhibits a clear two-phase structure. The Adam phase (0-10k) reduces the loss to roughly $10^{-3}$. The SSBroyden phase (10k-60k) initiates with a sharp drop. The final stage spans from 10k to 60k. At the 50% midpoint of this final stage (iteration 35k), the average loss band is `A` ≈ $10^{-12}$. At the end of the training (iteration 60k), the average loss band is `B` ≈ $10^{-12}$. Similarly, for the Relative L2 Error, the average band at the midpoint (35k) is `A` ≈ $5 \times 10^{-8}$, and at the end (60k) it is `B` ≈ $5 \times 10^{-8}$. Both curves have established a genuine horizontal plateau across the last 50% of the final stage.
  2. **Residual/Pointwise Error:** The PDE residuals are exceptionally low (maximum $\sim 2 \times 10^{-6}$), and the pointwise error is uniformly distributed with a maximum of only $\sim 4 \times 10^{-8}$.
  3. **Solution Fields:** The predicted solution is visually indistinguishable from the reference analytical solution, perfectly capturing the high-frequency periodic grid.
* **Diagnosis:** Optimal Convergence & Success. The model has perfectly solved the 2D Helmholtz equation. The combination of Periodic Fourier Features (which strictly enforce the periodic boundary conditions and provide the necessary frequency bandwidth) and the Hybrid Adam + SSBroyden optimizer with vRBA has allowed the network to reach the limits of numerical precision. The early termination of SSBroyden (stopping at 60k total iterations instead of the scheduled 90k) is completely acceptable here because the dual-curve plateau was genuinely achieved before termination.
* **Targeted Improvements:**

    **FEEDBACK SECTIONS:**
    * **For Integrity:** None
    * **For Accuracy:** None
    * **For Details:** None
    * **For Optimality:** None

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | The implementation uses a PINN with MLP (width=30, depth=3) preceded by Periodic Fourier Features (degree=2, Lx=Ly=2.0) on separate x/y inputs; embeddings are concatenated and fed exclusively to the MLP body with no raw coordinates or normalization. Training follows Hybrid Adam (with vRBA weighting/sampling) followed by SSBroyden (BFGS-based with vRBA sampling only). Loss assembly contains exactly one term: L_pde (Helmholtz residual matching the analytic g). No BC loss terms are present. No curriculum schedule is defined or used. | 1.00e-07 | 35 | 35 | 7 | 15 | **92** |
| 2 | Architecture: MLP (width=30, depth=3) with Periodic Fourier Features (degree=2, Lx=Ly=2.0) on raw x,y inputs in [-1,1]; only embedded features fed to MLP body. Optimizer workflow: Hybrid Adam (10,000 iterations with vRBA weighting/sampling) followed by SSBroyden (20,000 iterations with vRBA sampling only). Loss terms: exactly L_pde (Helmholtz residual) with vRBA; no BC, data, or ghost terms. Curriculum schedule matches the mandated 10k+20k split. Forward problem, periodic BCs satisfied by embedding, no normalization applied. | 1.00e-07 | 35 | 35 | 7 | 15 | **92** |
| 3 | The implementation uses a PINN with Periodic Fourier Features (degree=2, Lx=Ly=2.0) on raw coordinates in [-1,1] (no normalization) feeding only embeddings into a 3-layer MLP of width 30. Training follows Hybrid Adam (10,000 iterations) + SSBroyden (40,000 iterations) with vRBA for PDE residual weighting and importance sampling. Loss is strictly L_total = L_pde (Helmholtz residual with correct forcing term g(x,y)); no boundary or data losses are present. The analytical solution sin(πx)sin(4πy) is used only for error metrics. | 1.00e-07 | 35 | 35 | 7 | 15 | **92** |
| 4 | The implementation uses a PINN with Periodic Fourier Features (degree=2, Lx=Ly=2.0) applied independently to each coordinate, followed by an MLP (width=30, depth=3) that receives ONLY the concatenated embedded features. Training follows the Hybrid Adam (10,000 iterations) + SSBroyden (80,000 iterations) workflow. The loss is strictly L_pde (Helmholtz residual) with vRBA used for local adaptive weighting and importance sampling. No boundary losses are present. The analytical solution and forcing term are correctly implemented. | 5.00e-08 | 35 | 35 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 2 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 3 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 4 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run2/results//Helmholtz2D_Periodic_Sin/2026_11_06_1354/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run2/results//Helmholtz2D_Periodic_Sin/2026_11_06_1357/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run2/results//Helmholtz2D_Periodic_Sin/2026_11_06_1400/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run2/results//Helmholtz2D_Periodic_Sin/2026_11_06_1405/` |


---
