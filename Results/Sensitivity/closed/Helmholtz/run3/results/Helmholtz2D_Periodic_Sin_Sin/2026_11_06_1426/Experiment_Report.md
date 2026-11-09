
# Experiment Report
*Last Updated: 2026-11-06 14:30:25*

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
Role: Validation/Error Metrics ONLY.

**2. Problem Statement**
* **Governing Equation (2D Helmholtz):** 
  $u_{xx} + u_{yy} + k^2 u = g(x, y)$
  **Residual:** $L_{pde} = u_{xx} + u_{yy} + k^2 u - g(x, y) = 0$
  where $k=1$, $a_1=1$, $a_2=4$.
  The source term $g(x,y)$ is derived from the analytical solution:
  $g(x,y) = (k^2 - \pi^2 a_1^2 - \pi^2 a_2^2) \sin(\pi a_1 x) \sin(\pi a_2 y)$
* **Boundary Conditions:** Periodic in both $x$ and $y$.
* **Analytical Solution:** $u_{an}(x,y) = \sin(\pi a_1 x) \sin(\pi a_2 y)$
* **Problem Type:** Forward. Well-posed for PINNs.
* **Domain:** 2D, $x \in [-1, 1]$, $y \in [-1, 1]$.

**3. Base Model Selection**
* **Optimizer:** Hybrid Adam + SSBroyden.
* **Base MLP Specs:** `width=30`, `depth=3`.

**4. Main Architecture**
* **Structure:** Input -> Periodic Fourier Features -> Main Body -> Output.
* **Input Transformations:** 
  * **Normalization:** Do NOT normalize $x$ and $y$. They are already in the nice range $[-1, 1]$. Pass them unchanged to the embedding.
  * **Embedding:** Use **Periodic Fourier Features** for both $x$ and $y$. 
  * **CRITICAL:** Since $x, y \in [-1, 1]$ enter the embedding directly, you MUST set `Lx = 2.0` and `Ly = 2.0`. Use `degree = 2`.
  * **Input Composition Rule:** The main body must receive ONLY the periodic embeddings. Do NOT pass the raw $x$ and $y$ coordinates to the MLP.

**5. Constraint Implementation Strategy**
* **Heuristic Count:** `L_pde` (1). `L_bc_periodic` is REMOVED because Periodic Fourier Features inherently enforce periodicity.
* **Strategy:** Loss-Based (only PDE loss is required). No hard constraints or output transformations are needed.

**6. Loss Function Assembly**
* **Symbolic Equation:** $L_{total} = L_{pde}$
* **Iteration Budget (PER ADVISOR MANDATE):** 
  * **Phase 1 (Adam):** 10,000 iterations.
  * **Phase 2 (SSBroyden):** **120,000 iterations**. (Increased to ensure the loss and error curves genuinely plateau). Ensure the Hessian is reinitialized at the start of this phase.

**7. Weighting Strategy**
* **Local Weights (`lambdas`):** Use **vRBA** (Variational Residual-Based Attention) applied pointwise inside the $L_{pde}$ term. `lambdas` are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden.
* **Global Weights (`lam`):** Since there is only 1 loss term ($L_{pde}$), no global balancing is needed.

**8. Visualization Strategy**
* Keep the default visualization callbacks. Do not modify or delete them.

**9. Saving Data**
* Save the predicted and analytical fields as `.npz` arrays with keys specifying the variable names for visualization.

**Main Model:**
[MLP with Hybrid Adam and SSBrodyden]

**Enhancements:**
[Periodic Fourier Features]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
**Section 1: Status Decision**
* **Status:** REJECTED_FOR_REFINEMENT

**Section 2: Quantitative Scorecard**
* **Score Integrity:** 35
* **Score Accuracy:** 20
* **Score Details:** 5
* **Score Optimality:** 15
* **TOTAL SCORE:** 75

**Section 3: Forensic Analysis & Feedback**
* **Observation:** The Total Loss curve shows an initial drop during the Adam phase (0-10k iterations), followed by a noisy but continuous descent during the SSBroyden phase. Applying the Dual-Curve Test to the final stage (10k to ~90k iterations): the average band level of the loss at the 50% midpoint (~50k) is approximately `3e-4`, and at the end (~90k) it has dropped to approximately `2e-4`. The Relative L2 Error curve is highly oscillatory, maintaining an average band level around `1e-3` throughout the final stage. 
* **Diagnosis:** Budget-Limited / Premature Stop. The loss curve is still descending on average across the last 50% of the final training stage. The model was still actively learning and had not yet reached its optimal floor before the training run terminated.
* **Targeted Improvements:**
    * **For Integrity:** None
    * **For Accuracy:** Increase the number of iterations of the LAST training stage (SSBroyden) from 120,000 to 240,000 iterations so both the loss and error curves genuinely flatten before the run ends.
    * **For Details:** Ensure the loss plateaus in the final segment.
    * **For Optimality:** None

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The implementation uses a PINN with Periodic Fourier Features (degree=2, Lx=Ly=2.0) on raw coordinates in [-1,1] followed by an MLP (width=30, depth=3). Training follows the Hybrid Adam + SSBroyden workflow with vRBA weighting/sampling applied exclusively inside the single PDE loss term. The PDE residual exactly matches the required Helmholtz form with the analytic source term; no BC loss terms are present. Output transformation is absent as specified. | 1.20e-07 | 35 | 35 | 5 | 15 | **90** |
| 1 | The implementation uses a 3-layer MLP (width 30) with Periodic Fourier Features (degree=2, Lx=Ly=2.0) as input embeddings and no output transformation or coordinate normalization. Training follows the Hybrid Adam (10,000 iterations) + SSBroyden (30,000 iterations) workflow. The loss is strictly L_pde (Helmholtz residual) with vRBA weighting/sampling; no BC or auxiliary loss terms are present. The analytical solution, source term, and periodic BC enforcement via embeddings all match the strategy exactly. | 1.00e-07 | 35 | 35 | 5 | 15 | **90** |
| 2 | The implementation uses a PINN with Periodic Fourier Features (degree=2, Lx=Ly=2.0) feeding a 3-layer MLP (width=30) that receives only embeddings. Training follows the mandated Hybrid schedule: Adam phase for 10,000 iterations followed by SSBroyden phase for exactly 60,000 iterations. Loss assembly contains only L_pde (Helmholtz residual with correct source term); periodic BCs are enforced exactly by the embeddings so no L_bc term appears. vRBA weighting/sampling is applied inside L_pde for both phases. No normalization is performed on inputs. Analytical solution and source term match the strategy exactly. | 5.00e-08 | 35 | 35 | 5 | 15 | **90** |
| 3 | Architecture: MLP (width=30, depth=3) with Periodic Fourier Features (degree=2, Lx=Ly=2.0) on inputs x,y in [-1,1] (no normalization, only embeddings passed to MLP body). Optimizer workflow: Phase 1 Adam (10k iterations) with vRBA pointwise weighting + sampling inside L_pde; Phase 2 SSBroyden (120k iterations) with Hessian reinitialized each outer loop and vRBA used for sampling only. Loss terms: exactly L_total = L_pde (Helmholtz residual with correct source term (1-17*pi^2)sin(pi x)sin(4 pi y)); no BC, data, or gauge terms. Curriculum schedule matches the mandated 10k/120k split. | 0.10% | 35 | 20 | 5 | 15 | **75** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 1 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 2 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 3 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1414/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1417/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1421/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1426/` |


---
