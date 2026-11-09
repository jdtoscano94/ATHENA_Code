
# Experiment Report
*Last Updated: 2026-11-06 14:24:18*

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

**0. Triage & Advisor Directives**
* **Advisor Directive (MANDATORY):** The previous run was budget-limited. You **MUST** increase the number of iterations of the final SSBroyden stage to **60,000 iterations** to ensure the loss curve plateaus.

**1. Data Source**
* **Role:** Validation/Error Metrics ONLY. (Forward problem).

**2. Problem Statement**
* **Governing Equation:** 2D Helmholtz Equation.
  * Residual: $L_{pde} = u_{xx} + u_{yy} + k^2 u - g(x, y) = 0$
  * Constants: $k = 1$, $a_1 = 1$, $a_2 = 4$.
  * Source Term $g(x,y)$: Derived from the analytical solution. $g(x,y) = (1 - 17\pi^2) \sin(\pi x) \sin(4\pi y)$.
* **Boundary Conditions:** Periodic in $x$ and $y$ on $[-1, 1] \times [-1, 1]$.
* **Analytical Solution:** $u_{an}(x,y) = \sin(\pi x) \sin(4\pi y)$.
* **Problem Type:** Forward.
* **Input:** 2D ($x, y$).
* **Output:** Scalar $u(x,y)$.

**3. Base Model Selection**
* **Optimizer:** Hybrid Adam + SSBroyden.
* **Base MLP Specs:** `width=30`, `depth=3`.

**4. Main Architecture**
* **Structure:** Input -> Input Transformation -> Main Body -> Output Transformation -> Output.
* **Input Transformations:**
  * **Normalization:** $x$ and $y$ are already in $[-1, 1]$. **DO NOT** normalize them. Pass them directly to the embedding.
  * **Embedding:** Use **Periodic Fourier Features** for both $x$ and $y$.
  * **Lx / Ly Derivation:** Since $x, y \in [-1, 1]$ enter the embedding directly, `Lx = 2.0` and `Ly = 2.0`.
  * **Degree:** `degree = 2`.
  * **Rule:** The main body must receive ONLY the periodic embeddings, NOT the raw $x$ or $y$ coordinates.
* **Main Body:** MLP (`width=30`, `depth=3`).
* **Output Transformation:** None.

**5. Constraint Implementation Strategy**
* **Count:** $L_{pde}$ (1).
* **Periodic BCs:** Handled exactly by the Periodic Fourier Features. **REMOVE** $L_{bc\_periodic}$ from the loss.
* **Strategy:** Loss-Based (Only PDE loss is required).

**6. Loss Function Assembly**
* **Symbolic Equation:** $L_{total} = L_{pde}$
* **Iteration Schedule (CRITICAL):**
  * **Phase 1 (Adam):** 10,000 iterations.
  * **Phase 2 (SSBroyden):** **60,000 iterations** (Mandated by Advisor to ensure convergence).

**7. Weighting Strategy**
* **Local Weights (`lambdas`):** Use **vRBA** inside the $L_{pde}$ term. `lambdas` are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden.
* **Global Weights (`lam`):** Total terms = 1. No global balancing is needed.
* **SSBroyden Integrity:** Do NOT interfere with the internal weighting, sampling, or batching strategies of the SSBroyden optimizer.

**8. Visualization Strategy**
* Do NOT modify, delete, or "improve" the visualization code in the base template.

**9. Saving Data**
* Save the predicted and exact fields as `.npz` arrays with keys specifying the variable names for visualization.

### Required Modules
Main Model:
[MLP with Hybrid Adam and SSBrodyden]
Enhancements:
[Periodic Fourier Features]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
REJECTED_FOR_REFINEMENT

**Section 1: Status Decision**
* **Status:** REJECTED_FOR_REFINEMENT

**Section 2: Quantitative Scorecard**
* **Score Integrity:** 35
* **Score Accuracy:** 35
* **Score Details:** 5
* **Score Optimality:** 15
* **TOTAL SCORE:** 90

**Section 3: Forensic Analysis & Feedback**
* **Observation:** The model successfully completed the run with excellent accuracy and stable physics. The final training stage (SSBroyden) begins after the Adam transition at iteration 10,000 and ends prematurely at approximately iteration 48,000. The 50% midpoint of this final stage is iteration 29,000. 
Applying the mandatory dual-curve test:
For the Loss curve:
`A` = average band level at the 50% midpoint (iter 29,000) = 3e-12
`B` = average band level at the END (iter 48,000) = 3e-13
For the Error curve:
`A` = average band level at the 50% midpoint (iter 29,000) = 8e-8
`B` = average band level at the END (iter 48,000) = 5e-8
Because `B` is lower than `A` for the loss curve (the floor of the noise band drops visibly from $10^{-11.5}$ to $10^{-12.5}$), the loss is still decreasing on average. The optimizer early-terminated before reaching a genuine horizontal plateau.
* **Diagnosis:** Budget-Limited / Premature Stop (Still Improving).
* **Targeted Improvements:**
    * **For Integrity:** None
    * **For Accuracy:** Score is excellent but the average loss is still decreasing across the last 50% of the final stage (from ~3e-12 to ~3e-13, judged by the band center). Duplicate the number of iterations for the final stage (e.g., raise the final SSBroyden stage to 120,000 iterations) and reinitialize the Hessian to prevent premature line-search termination.
    * **For Details:** Ensure both the loss and error curves genuinely plateau in the final segment before the run ends.
    * **For Optimality:** None

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The implementation uses a PINN with Periodic Fourier Features (degree=2, Lx=Ly=2.0) on raw coordinates in [-1,1] followed by an MLP (width=30, depth=3). Training follows the Hybrid Adam + SSBroyden workflow with vRBA weighting/sampling applied exclusively inside the single PDE loss term. The PDE residual exactly matches the required Helmholtz form with the analytic source term; no BC loss terms are present. Output transformation is absent as specified. | 1.20e-07 | 35 | 35 | 5 | 15 | **90** |
| 1 | The implementation uses a 3-layer MLP (width 30) with Periodic Fourier Features (degree=2, Lx=Ly=2.0) as input embeddings and no output transformation or coordinate normalization. Training follows the Hybrid Adam (10,000 iterations) + SSBroyden (30,000 iterations) workflow. The loss is strictly L_pde (Helmholtz residual) with vRBA weighting/sampling; no BC or auxiliary loss terms are present. The analytical solution, source term, and periodic BC enforcement via embeddings all match the strategy exactly. | 1.00e-07 | 35 | 35 | 5 | 15 | **90** |
| 2 | The implementation uses a PINN with Periodic Fourier Features (degree=2, Lx=Ly=2.0) feeding a 3-layer MLP (width=30) that receives only embeddings. Training follows the mandated Hybrid schedule: Adam phase for 10,000 iterations followed by SSBroyden phase for exactly 60,000 iterations. Loss assembly contains only L_pde (Helmholtz residual with correct source term); periodic BCs are enforced exactly by the embeddings so no L_bc term appears. vRBA weighting/sampling is applied inside L_pde for both phases. No normalization is performed on inputs. Analytical solution and source term match the strategy exactly. | 5.00e-08 | 35 | 35 | 5 | 15 | **90** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 1 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 2 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1414/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1417/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1421/` |


---
