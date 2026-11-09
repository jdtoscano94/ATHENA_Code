
# Experiment Report
*Last Updated: 2026-11-06 14:19:43*

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
- **Advisor Directive:** The previous run achieved excellent accuracy but was budget-limited. You **MUST** duplicate the number of iterations for the final SSBroyden stage to 30,000 iterations to allow the loss and error curves to fully plateau.

**1. Data Source**
- **Role:** Validation/Error Metrics ONLY.

**2. Problem Statement**
- **Governing Equation:** 2D Helmholtz Equation.
  - Residual: $L_{pde} = u_{xx} + u_{yy} + k^2 u - g(x, y) = 0$
  - Constants: $k = 1$, $a_1 = 1$, $a_2 = 4$.
  - Source Term $g(x,y)$: Derived from the analytical solution. $g(x,y) = (1 - 17\pi^2) \sin(\pi x) \sin(4\pi y)$.
- **Boundary Conditions:** Periodic in both $x$ and $y$ on $[-1, 1] \times [-1, 1]$.
- **Analytical Solution:** $u_{an}(x,y) = \sin(\pi x) \sin(4\pi y)$.
- **Problem Type:** Forward. Admits strong form solutions.
- **Input:** 2D ($x, y$).
- **Output:** Scalar $u(x,y)$.

**3. Base Model Selection**
- **Optimizer:** Hybrid Adam + SSBroyden.
- **Base MLP Specs:** `width=30`, `depth=3`.

**4. Main Architecture**
- **Structure:** Input -> Input Transformation -> Main Body -> Output Transformation -> Output.
- **Input Transformations:**
  - **Normalization:** The raw coordinates $x, y \in [-1, 1]$ are already in a nice, tanh-friendly range. **DO NOT** apply any normalization (pass them unchanged).
  - **Embedding:** Use **Periodic Fourier Features** for both $x$ and $y$ with `degree=2`.
  - **Lx/Ly Derivation:** Since $x$ and $y$ enter the embedding directly from $[-1, 1]$, their ranges are $1 - (-1) = 2.0$. Therefore, you **MUST** set `Lx = 2.0` and `Ly = 2.0`.
  - **Input Composition Rule:** The main body must receive **ONLY** the periodic embeddings of $x$ and $y$. **DO NOT** pass the bare $x$ or $y$ coordinates to the main body.
- **Output Transformation:** None required. Periodicity is strictly enforced by the input embeddings.

**5. Constraint Implementation Strategy**
- **Count:** $L_{pde}$ (1). Periodic BCs are enforced architecturally by the embeddings, so $L_{bc\_periodic}$ is REMOVED from the count.
- **Strategy:** Loss-Based (only PDE loss is required).

**6. Loss Function Assembly**
- **Symbolic Equation:** $L_{total} = L_{pde}$
- **Iteration Budget (CRITICAL):**
  - **Phase 1 (Adam):** 10,000 iterations.
  - **Phase 2 (SSBroyden):** **30,000 iterations** (Increased per Advisor report to ensure convergence plateau).

**7. Weighting Strategy**
- **Local Weights (`lambdas`):** Use **vRBA** (Variational Residual-Based Attention) inside the PDE loss. `lambdas` are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden.
- **Global Weights (`lam`):** Since there is only 1 loss term ($L_{pde}$), no global balancing is needed.
- **SSBroyden Integrity:** Do NOT interfere with the internal weighting, sampling, or batching strategies of the SSBroyden optimizer.

**8. Visualization Strategy**
- Do NOT modify, delete, or "improve" the visualization code in the base template.

**9. Saving Data**
- Save the predicted and reference fields as `.npz` arrays with keys specifying the variable names for visualization.

### Librarian Instructions

Main Model:
[MLP with Hybrid Adam and SSBrodyden]

Enhancements:
[Periodic Fourier Features]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
APPROVED
Wait, I must output EXACTLY one of the options.
The logic dictates REJECTED_FOR_REFINEMENT.

**Section 1: Status Decision**
* **Status:** REJECTED_FOR_REFINEMENT

**Section 2: Quantitative Scorecard**
* **Score Integrity:** 35
* **Score Accuracy:** 35
* **Score Details:** 5
* **Score Optimality:** 15
* **TOTAL SCORE:** 90

**Section 3: Forensic Analysis & Feedback**
* **Observation:** The loss and error history plots display an initial Adam phase (0-10,000 iterations) followed by a final SSBroyden phase (10,000-30,000 iterations). To evaluate convergence, we examine the final stage. The 50% midpoint of this final stage is at 20,000 iterations, where the average band level of the loss curve (A) is approximately 1e-11. At the end of the final stage (30,000 iterations), the average band level of the loss curve (B) is approximately 1e-12. Because B is lower than A, the loss curve is still decreasing on average across the last 50% of the window and has not plateaued. While the error curve appears relatively flat around 1e-7, the continued descent of the loss indicates the model was still learning when the iteration budget was exhausted. The residual and pointwise error maps show excellent accuracy, and the predicted solution visually matches the reference perfectly.
* **Diagnosis:** Budget-Limited / Premature Stop (Still Improving).
* **Targeted Improvements:**
    * **For Integrity:** None
    * **For Accuracy:** Increase the number of iterations of the LAST training stage (e.g. 2-5x) so both the loss and error curves flatten before the run ends. Raise the final SSBroyden stage from 20,000 to 60,000 iterations.
    * **For Details:** Ensure loss plateaus in final segment.
    * **For Optimality:** None

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The implementation uses a PINN with Periodic Fourier Features (degree=2, Lx=Ly=2.0) on raw coordinates in [-1,1] followed by an MLP (width=30, depth=3). Training follows the Hybrid Adam + SSBroyden workflow with vRBA weighting/sampling applied exclusively inside the single PDE loss term. The PDE residual exactly matches the required Helmholtz form with the analytic source term; no BC loss terms are present. Output transformation is absent as specified. | 1.20e-07 | 35 | 35 | 5 | 15 | **90** |
| 1 | The implementation uses a 3-layer MLP (width 30) with Periodic Fourier Features (degree=2, Lx=Ly=2.0) as input embeddings and no output transformation or coordinate normalization. Training follows the Hybrid Adam (10,000 iterations) + SSBroyden (30,000 iterations) workflow. The loss is strictly L_pde (Helmholtz residual) with vRBA weighting/sampling; no BC or auxiliary loss terms are present. The analytical solution, source term, and periodic BC enforcement via embeddings all match the strategy exactly. | 1.00e-07 | 35 | 35 | 5 | 15 | **90** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 1 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1414/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1417/` |


---
