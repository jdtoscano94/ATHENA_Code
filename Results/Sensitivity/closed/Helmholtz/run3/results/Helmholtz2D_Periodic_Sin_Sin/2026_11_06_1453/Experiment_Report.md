
# Experiment Report
*Last Updated: 2026-11-06 15:14:47*

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
**1. Data Source**
* Role: Validation/Error Metrics ONLY.

**2. Problem Statement**
* **Governing Equation (2D Helmholtz):** $u_{xx} + u_{yy} + u = (1 - 17\pi^2) \sin(\pi x) \sin(4\pi y)$
* **Boundary Conditions:** Periodic in both $x$ and $y$ on the domain $[-1, 1] \times [-1, 1]$.
* **Analytical Solution:** $u(x,y) = \sin(\pi x) \sin(4\pi y)$
* **Problem Type:** Forward. Well-posed for PINNs.

**3. Base Model Selection**
* Hybrid Adam + SSBroyden.

**4. Main Architecture**
* **Structure:** Input -> Periodic Embedding -> Main Body -> Output.
* **Input Transformations:** 
  * $x$ and $y$ are already in $[-1, 1]$. Do **NOT** normalize them. Pass them unchanged to the embedding.
  * **Periodic Fourier Features:** Apply to both $x$ and $y$. Since the raw inputs are in $[-1, 1]$, set `Lx = 2.0` and `Ly = 2.0`. Use `degree = 2`.
  * **CRITICAL:** Do not pass raw $x$ or $y$ to the main body. Only pass their embeddings.
* **Main Body:** Base MLP with `width=30`, `depth=3`.
* **Output Transformation:** None required.

**5. Constraint Implementation Strategy**
* **Periodic BCs:** Enforced exactly by the Periodic Fourier Features. Do not include a boundary loss term.
* **Count:** $L_{pde}$ (1). Count < 5. Use separate loss terms (no hard constraints needed).

**6. Loss Function Assembly**
* $L_{total} = L_{pde}$
* **Iteration Budget (ADVISOR MANDATE):** 
  * **Phase 1 (Adam):** 10,000 iterations.
  * **Phase 2 (SSBroyden):** 960,000 iterations (increased from 480,000 to ensure full convergence). Reinitialize the Hessian at the start of this phase.

**7. Weighting Strategy**
* **Global Weights:** Only 1 loss term ($L_{pde}$). No global balancing needed.
* **Local Weights:** `lambdas` (vRBA) are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden. Do not interfere with SSBroyden's internal mechanisms.

**8. Visualization Strategy**
* Do not modify the default visualization callbacks.

**9. Saving Data**
* Save the fields as `.npz` arrays with keys specifying the variable names for visualization.

Main Model:
[MLP with Hybrid Adam and SSBrodyden]

Enhancements:
[Periodic Fourier Features]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
APPROVED

**Section 1: Status Decision**
* **Status:** APPROVED

**Section 2: Quantitative Scorecard**
* **Score Integrity:** 35
* **Score Accuracy:** 35
* **Score Details:** 15
* **Score Optimality:** 15
* **TOTAL SCORE:** 100

**Section 3: Forensic Analysis & Feedback**
* **Observation:** The loss and error curves show a clear plateau in the final stage. For the loss curve, the average band level at the 50% midpoint of the final stage (around 325,000 iterations) is approximately 4e-5, and at the end of the final stage (around 650,000 iterations) it is approximately 3e-5. These values are within a factor of 2, indicating a stable horizontal floor. Similarly, the error curve plateaus with an average band level around 8e-4 at the midpoint and 6e-4 at the end. The pointwise error and residual heatmaps show well-distributed, low-magnitude errors across the domain.
* **Diagnosis:** Exploration complete. Best model was Run 2. Stopping.
* **Targeted Improvements:** None.

**FEEDBACK SECTIONS:**
* **For Integrity:** None
* **For Accuracy:** None
* **For Details:** None
* **For Optimality:** None

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The implementation uses a PINN with Periodic Fourier Features (degree=2, Lx=Ly=2.0) on raw coordinates in [-1,1] followed by an MLP (width=30, depth=3). Training follows the Hybrid Adam + SSBroyden workflow with vRBA weighting/sampling applied exclusively inside the single PDE loss term. The PDE residual exactly matches the required Helmholtz form with the analytic source term; no BC loss terms are present. Output transformation is absent as specified. | 1.20e-07 | 35 | 35 | 5 | 15 | **90** |
| 1 | The implementation uses a 3-layer MLP (width 30) with Periodic Fourier Features (degree=2, Lx=Ly=2.0) as input embeddings and no output transformation or coordinate normalization. Training follows the Hybrid Adam (10,000 iterations) + SSBroyden (30,000 iterations) workflow. The loss is strictly L_pde (Helmholtz residual) with vRBA weighting/sampling; no BC or auxiliary loss terms are present. The analytical solution, source term, and periodic BC enforcement via embeddings all match the strategy exactly. | 1.00e-07 | 35 | 35 | 5 | 15 | **90** |
| 2 | The implementation uses a PINN with Periodic Fourier Features (degree=2, Lx=Ly=2.0) feeding a 3-layer MLP (width=30) that receives only embeddings. Training follows the mandated Hybrid schedule: Adam phase for 10,000 iterations followed by SSBroyden phase for exactly 60,000 iterations. Loss assembly contains only L_pde (Helmholtz residual with correct source term); periodic BCs are enforced exactly by the embeddings so no L_bc term appears. vRBA weighting/sampling is applied inside L_pde for both phases. No normalization is performed on inputs. Analytical solution and source term match the strategy exactly. | 5.00e-08 | 35 | 35 | 5 | 15 | **90** |
| 3 | Architecture: MLP (width=30, depth=3) with Periodic Fourier Features (degree=2, Lx=Ly=2.0) on inputs x,y in [-1,1] (no normalization, only embeddings passed to MLP body). Optimizer workflow: Phase 1 Adam (10k iterations) with vRBA pointwise weighting + sampling inside L_pde; Phase 2 SSBroyden (120k iterations) with Hessian reinitialized each outer loop and vRBA used for sampling only. Loss terms: exactly L_total = L_pde (Helmholtz residual with correct source term (1-17*pi^2)sin(pi x)sin(4 pi y)); no BC, data, or gauge terms. Curriculum schedule matches the mandated 10k/120k split. | 0.10% | 35 | 20 | 5 | 15 | **75** |
| 4 | The implementation uses a PINN with Periodic Fourier Features (degree=2, Lx=Ly=2.0) feeding only embeddings into an MLP (width=30, depth=3). Training follows the exact Hybrid schedule: Adam phase for 10,000 iterations using vRBA-weighted PDE loss, followed by SSBroyden phase for 240,000 iterations with Hessian reinitialization at the start of each outer loop and vRBA used for sampling only. Loss assembly is strictly L_total = L_pde with no additional terms or data losses. Inputs remain unnormalized in [-1,1]. | 0.20% | 35 | 20 | 5 | 15 | **75** |
| 5 | The implementation uses a PINN with Periodic Fourier Features (degree=2, Lx=Ly=2.0) feeding only embeddings into a 3-layer MLP (width=30). Training follows the exact Hybrid schedule: 10,000 Adam iterations with vRBA weighting/sampling, followed by 480,000 SSBroyden iterations (Hessian reinitialized each outer step) using vRBA for sampling only. Loss assembly is strictly L_total = L_pde with the correct Helmholtz residual (uxx + uyy + u - (1-17π²)sin(πx)sin(4πy)). Periodic BCs are enforced architecturally with no additional loss terms. Analytical solution and source term match the report exactly. | 0.08% | 35 | 35 | 5 | 15 | **90** |
| 6 | The implementation uses a PINN with Periodic Fourier Features (degree=2, Lx=Ly=2.0) on raw x/y inputs in [-1,1] feeding a base MLP (width=30, depth=3). Optimizer workflow is exactly Hybrid Adam (10,000 iterations with vRBA weighting/sampling) followed by SSBroyden (960,000 iterations with Hessian reinitialization each outer loop and vRBA sampling only). Loss assembly contains solely L_pde; no boundary, data, or ghost terms are present. Periodic BCs are enforced exactly via the embedding with no additional loss terms. | 0.06% | 35 | 35 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 1 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 2 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 3 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 4 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 5 | Saves helmholtz_2d_results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_an' (256x256), 'abs_error' (256x256). |
| 6 | Saves 'helmholtz_2d_results.npz' containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_an' (256x256), 'abs_error' (256x256). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1414/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1417/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1421/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1426/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1431/` |
| 5 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1440/` |
| 6 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1453/` |


---
