
# Experiment Report
*Last Updated: 2026-11-06 15:31:05*

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
**1. Data Role**
Role: Validation/Error Metrics ONLY.

**2. Problem Statement**
*   **Governing Equation (2D Helmholtz):** $u_{xx} + u_{yy} + k^2 u = g(x, y)$
    *   Residual: $L_{pde} = u_{xx} + u_{yy} + u - g(x, y) = 0$
    *   Constants: $k = 1$, $a_1 = 1$, $a_2 = 4$.
    *   Source term: $g(x, y) = (1 - 17\pi^2) \sin(\pi x) \sin(4\pi y)$
*   **Domain:** $x \in [-1, 1]$, $y \in [-1, 1]$ (2D).
*   **Boundary Conditions:** Periodic in both $x$ and $y$.
*   **Analytical Solution:** $u_{an}(x, y) = \sin(\pi x) \sin(4\pi y)$
*   **Problem Type:** Forward. Well-posed for PINNs.

**3. Base Model Selection**
*   MLP with Hybrid Adam and SSBroyden.

**4. Main Architecture**
*   **Structure:** Input -> Periodic Fourier Features -> Main Body -> Output.
*   **Input Transformations:**
    *   Do NOT normalize $x$ and $y$ (they are already in $[-1, 1]$).
    *   **Periodic Fourier Features:** Apply to both $x$ and $y$.
    *   **Lx Derivation:** Since $x, y \in [-1, 1]$, the range is $1 - (-1) = 2.0$. Set `Lx = 2.0` for both coordinates.
    *   **Degree:** Set `degree = 2`.
    *   **CRITICAL:** The main body must receive ONLY the periodic embeddings, NOT the raw $x$ and $y$ coordinates.
*   **Main Body:** Base MLP Specs: `width=30`, `depth=3`.
*   **Output Transformation:** None required.

**5. Constraint Implementation Strategy**
*   **Periodic BCs:** Enforced architecturally via Periodic Fourier Features.
*   **Count:** $L_{pde}$ (1). Total soft loss terms = 1.
*   **Strategy:** Loss-Based (only PDE loss is needed).

**6. Loss Function Assembly**
*   $L_{total} = L_{pde}$
*   **Optimization Schedule (PER ADVISOR INSTRUCTION):**
    *   **Phase 1 (Adam):** 10,000 iterations.
    *   **Phase 2 (SSBroyden):** **40,000 iterations** (Increased to ensure full convergence plateau).

**7. Weighting Strategy**
*   **Local Weights (`lambdas`):** Use `vRBA` for local adaptive weighting inside the PDE loss. `lambdas` are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden.
*   **Global Weights (`lam`):** Total Terms == 1. No global balancing needed.

**8. Visualization Strategy**
*   Do NOT modify the default visualization callbacks.

**9. Saving Data**
*   Save the predicted and exact fields as `.npz` arrays with keys specifying the variable names for visualization.

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
* **Score Accuracy:** 35
* **Score Details:** 7
* **Score Optimality:** 15
* **TOTAL SCORE:** 92

**Section 3: Forensic Analysis & Feedback**
* **Observation:** The training consists of an Adam phase followed by an SSBroyden phase starting at 10,000 iterations. The final stage spans from 10,000 to approximately 35,000 iterations (where it early-terminated). 
For the Loss curve:
- `A` = average band level at the 50% midpoint of the final stage (~22,500 iterations) = 1e-11
- `B` = average band level at the END of the final stage (~35,000 iterations) = 1e-12
Comparing A and B explicitly: B (1e-12) is lower than A (1e-11). The loss curve is still decreasing on average.
For the Error curve:
- `A` = average band level at the 50% midpoint = 1e-7
- `B` = average band level at the END = 1e-7
Although the error curve appears relatively flat, the loss curve is still actively descending at the point of termination.

* **Diagnosis:** Score is excellent but the average loss is still decreasing across the last 50% of the final stage (from ~1e-11 to ~1e-12, judged by the band center). Duplicate the number of iterations for the final stage.

* **Targeted Improvements:**
    * **For Integrity:** None
    * **For Accuracy:** Increase the number of iterations of the LAST training stage (e.g., raise the final SSBroyden stage from 40,000 to 80,000 iterations) and reinitialize the Hessian to continue, so both the loss and error curves flatten before the run ends.
    * **For Details:** Ensure the loss genuinely plateaus in the final segment. Early termination is not evidence of a plateau.
    * **For Optimality:** None

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | MLP (width=30, depth=3) with Periodic Fourier Features (degree=2, Lx=2.0) as input embedding; main body receives only embeddings. Hybrid Adam + SSBroyden optimizer workflow with vRBA for local adaptive weighting/sampling. Loss assembly is exactly L_total = L_pde (no BC or other terms). No input normalization performed. Curriculum training: none specified or used. | 1.00e-07 | 35 | 35 | 7 | 15 | **92** |
| 1 | MLP (width=30, depth=3) with Periodic Fourier Features (degree=2, Lx=2.0) as sole input transformation; raw coordinates are excluded. Hybrid optimizer: Adam (10k iterations) followed by SSBroyden (20k iterations). Loss assembly contains exactly one term: L_pde (Helmholtz residual) with vRBA local adaptive weighting/sampling. Periodic BCs are satisfied by construction via the embedding; no BC loss terms present. Forward problem, validation metrics only. | 1.00e-07 | 35 | 35 | 7 | 15 | **92** |
| 2 | MLP architecture with Periodic Fourier Features (degree=2, Lx=2.0) feeding only embeddings into a 3-layer width-30 tanh MLP. Hybrid optimizer: Adam (10k iterations) followed by SSBroyden (40k iterations). Loss assembly contains exactly one term: L_pde (Helmholtz residual with vRBA local adaptive weighting and importance sampling). Forward problem with periodic BCs enforced architecturally. No input normalization, no extra loss terms, no data leakage on pressure. | 1.00e-07 | 35 | 35 | 7 | 15 | **92** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves predictions.npz containing keys: 'u_pred' (256x256), 'u_exact' (256x256), 'x' (256x256), 'y' (256x256). |
| 1 | Saves predictions.npz containing keys: 'u_pred' (256x256), 'u_exact' (256x256), 'x' (256x256), 'y' (256x256). Also saves Log_files_SSBroyden.mat with training logs. |
| 2 | Saves predictions.npz containing keys: 'u_pred' (256x256), 'u_exact' (256x256), 'x' (256x256), 'y' (256x256). Also saves Log_files_SSBroyden.mat with training logs. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run5/results//Helmholtz2D_Periodic_Sin_Solution/2026_11_06_1522/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run5/results//Helmholtz2D_Periodic_Sin_Solution/2026_11_06_1525/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run5/results//Helmholtz2D_Periodic_Sin_Solution/2026_11_06_1528/` |


---
