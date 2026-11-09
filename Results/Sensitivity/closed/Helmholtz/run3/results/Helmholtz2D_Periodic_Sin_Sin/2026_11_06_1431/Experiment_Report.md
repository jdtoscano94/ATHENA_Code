
# Experiment Report
*Last Updated: 2026-11-06 14:38:11*

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
*   **Governing Equation (Helmholtz):** $u_{xx} + u_{yy} + k^2 u - g(x, y) = 0$
*   **Constants:** $k = 1$, $a_1 = 1$, $a_2 = 4$.
*   **Analytical Solution:** $u_{an}(x,y) = \sin(\pi x) \sin(4\pi y)$
*   **Source Term:** $g(x, y) = (1 - 17\pi^2) \sin(\pi x) \sin(4\pi y)$
*   **Domain:** $x \in [-1, 1]$, $y \in [-1, 1]$ (2D).
*   **Boundary Conditions:** Periodic in both $x$ and $y$.
*   **Problem Type:** Forward. Well-posed for PINNs.

**3. Base Model Selection**
*   **Optimizer:** Hybrid Adam + SSBroyden.
*   **Base MLP Specs:** `width=30`, `depth=3`.

**4. Main Architecture**
Structure: Input -> Input Transformation (Embedding) -> Main Body -> Output.
*   **Input Transformations:**
    *   $x$ and $y$ are already in $[-1, 1]$. **DO NOT** apply any normalization (leave as identity).
    *   Apply **Periodic Fourier Features** to both $x$ and $y$.
    *   **CRITICAL:** Since inputs are unnormalized in $[-1, 1]$, set `Lx = 2.0` and `Ly = 2.0`.
    *   Set `degree = 2`.
    *   **Composition Rule:** Pass **ONLY** the periodic embeddings to the Main Body. Do NOT pass the raw $x$ or $y$ coordinates.
*   **Main Body:** MLP (`width=30`, `depth=3`).
*   **Output Transformation:** None required.

**5. Constraint Implementation Strategy**
*   **Count:** $L_{pde}$ (1).
*   **Periodic BCs:** Enforced exactly by the Periodic Fourier Features. **DO NOT** include a periodic boundary loss term.
*   **Strategy:** Loss-Based (only PDE loss remains).

**6. Loss Function Assembly**
*   $L_{total} = L_{pde}$
*   **Training Schedule (ADVISOR MANDATE):**
    *   **Phase 1 (Adam):** 10,000 iterations.
    *   **Phase 2 (SSBroyden):** **240,000 iterations** (Increased per Advisor to ensure convergence). Reinitialize the Hessian at the start of this phase.

**7. Weighting Strategy**
*   **Local Weights (`lambdas`):** Use **vRBA** (Variational Residual-Based Attention) inside the PDE loss. `lambdas` are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden.
*   **Global Weights (`lam`):** Total terms = 1. No global balancing is needed.
*   **SSBroyden Integrity:** Do not interfere with the internal weighting, sampling, or batching strategies of the SSBroyden optimizer.

**8. Visualization Strategy**
*   Do not modify the standard visualization callbacks.

**9. Saving Data**
*   Save the predicted and analytical fields as `.npz` arrays with keys specifying the variable names for visualization.

**Main Model:**
[MLP with Hybrid Adam and SSBrodyden]

**Enhancements:**
[Periodic Fourier Features]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
APPROVED
*Wait, let me re-evaluate the history check.*
First success: Run 0.
Runs since first success: Run 1, Run 2, Run 3, and this current run is Run 4.
Total runs after first success = 4.
The rule says: "IF >= 5 runs since first success: APPROVED... IF < 5 runs since first success: Continue to Step 2 to evaluate current run."
Since 4 < 5, I must evaluate and REJECT_FOR_REFINEMENT.

Let me output the correct format.

**Section 1: Status Decision**
* **Status:** REJECTED_FOR_REFINEMENT

**Section 2: Quantitative Scorecard**
* **Score Integrity:** 35
* **Score Accuracy:** 20
* **Score Details:** 5
* **Score Optimality:** 15
* **TOTAL SCORE:** 75

**Section 3: Forensic Analysis & Feedback**
* **Observation:** The training successfully completes the initial Adam phase and transitions to the SSBroyden phase. However, analyzing the final SSBroyden stage (from 10,000 to ~175,000 iterations), the loss curve has not reached a genuine plateau. The average band level of the loss at the 50% midpoint of this final stage (~92,500 iterations) is approximately `A = 2e-4`. At the end of the training (~175,000 iterations), the average band level has dropped to approximately `B = 1e-4`. Because `B` is lower than `A`, the loss curve is still descending on average. The relative L2 error oscillates heavily around 2e-3, but the continued descent of the loss indicates the model was still learning and the run terminated prematurely before reaching its true floor.
* **Diagnosis:** Budget-Limited / Premature Stop (Still Improving).
* **Targeted Improvements:** 
    * **For Integrity:** None
    * **For Accuracy:** Increase the number of iterations of the LAST training stage (e.g. 2-5x) so both the loss and error curves flatten before the run ends. Raise the final SSBroyden stage from 240,000 to 480,000 iterations.
    * **For Details:** Ensure loss plateaus in final segment.
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


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 1 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 2 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 3 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 4 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1414/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1417/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1421/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1426/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1431/` |


---
