
# Experiment Report
*Last Updated: 2026-11-06 19:24:51*

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
* **Advisor Directive:** The previous run was stopped prematurely while the error was still decreasing. You **MUST** increase the final SSBroyden stage from 15,000 to at least 30,000 iterations to ensure convergence.

**1. Data Source**
* **Role:** Validation/Error Metrics ONLY.

**2. Problem Statement**
* **Governing Equation (2D Helmholtz):** $u_{xx} + u_{yy} + k^2 u - g(x, y) = 0$
* **Constants:** $k=1$, $a_1=1$, $a_2=4$.
* **Analytical Solution:** $u(x,y) = \sin(\pi a_1 x) \sin(\pi a_2 y) = \sin(\pi x) \sin(4\pi y)$
* **Source Term:** $g(x,y) = (1 - 17\pi^2) \sin(\pi x) \sin(4\pi y)$
* **Domain:** $x \in [-1, 1]$, $y \in [-1, 1]$
* **Boundary Conditions:** Periodic in $x$ and $y$.
* **Problem Type:** Forward. Well-posed for PINNs.

**3. Base Model Selection**
* **Optimizer:** Hybrid Adam + SSBroyden.
* **Base MLP Specs:** `width=30`, `depth=3`.

**4. Main Architecture**
* **Structure:** Input -> Input Transformation -> Main Body -> Output Transformation -> Output.
* **Input Transformations:**
    * **Normalization:** $x$ and $y$ are already in $[-1, 1]$. **DO NOT** apply any normalization. Pass them directly to the embedding.
    * **Embedding:** Use **Periodic Fourier Features** for both $x$ and $y$.
    * **Lx Derivation:** Since $x \in [-1, 1]$ and $y \in [-1, 1]$ enter the embedding without normalization, the ranges are $1 - (-1) = 2.0$.
    * **Instruction:** Set `Lx=2.0` for $x$ and `Ly=2.0` for $y$. Use `degree=2`.
    * **Composition Rule:** Feed ONLY the embedded features to the main body. Do NOT pass the raw $x$ or $y$ coordinates.
* **Output Transformation:** None required. The periodic embeddings inherently satisfy the periodic boundary conditions.

**5. Constraint Implementation Strategy**
* **Count:** $L_{pde}$ (1). (Periodic BCs are enforced architecturally by the embeddings, so $L_{bc\_periodic}$ is REMOVED).
* **Strategy:** Loss-Based.

**6. Loss Function Assembly**
* **Equation:** $L_{total} = L_{pde}$
* **Iteration Budget (CRITICAL):**
    * **Phase 1 (Adam):** 10,000 iterations.
    * **Phase 2 (SSBroyden):** **30,000 iterations** (Mandatory increase per Advisor Report).

**7. Weighting Strategy**
* **Global Weights:** Only 1 loss term exists. No global balancing is needed.
* **Local Weights:** Use **vRBA** (`lambdas`). `lambdas` are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden.

**8. Visualization Strategy**
* Do NOT modify, delete, or "improve" the visualization code in the template.

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
APPROVED
*Wait, let me follow the exact format requested.*

**Section 1: Status Decision**
* **Status:** REJECTED_FOR_REFINEMENT

**Section 2: Quantitative Scorecard**
* **Score Integrity:** 35
* **Score Accuracy:** 35
* **Score Details:** 7
* **Score Optimality:** 15
* **TOTAL SCORE:** 92

**Section 3: Forensic Analysis & Feedback**
* **Observation:** The model successfully runs and achieves a very low final error of approximately 1.99e-8. The predicted solution visually matches the reference perfectly, and the pointwise error is minimal. However, analyzing the Total Loss History plot, the final training stage (from iteration 10,000 to 30,000) does not show a plateau. The average band level of the loss at the 50% midpoint of this final stage (iteration 20,000) is `A` = 1e-10, while at the end of the stage (iteration 30,000) it is `B` = 1e-12. Since `B` is significantly lower than `A`, the loss curve is still descending on average. This holds true even though the relative L2 error curve appears to have mostly flattened around 1e-7.
* **Diagnosis:** Budget-Limited / Premature Stop (Still Improving).
* **Targeted Improvements:**
    * **For Integrity:** None
    * **For Accuracy:** Score is excellent but the average loss is still decreasing across the last 50% of the final stage (from ~1e-10 to ~1e-12, judged by the band center). Duplicate the number of iterations for the final stage (e.g., increase the final SSBroyden stage to 60,000 iterations).
    * **For Details:** Ensure both loss and error curves genuinely plateau in the final segment before terminating.
    * **For Optimality:** None

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | MLP architecture with Periodic Fourier Features (Lx=2.0, Ly=2.0) embeddings. Hybrid optimizer: Adam (10k iterations) followed by SSBroyden (exactly 15k iterations). Loss is strictly L_pde only. Global weights (lamE) frozen after Adam; lambdas active solely for importance sampling during SSBroyden. Analytical solution and PDE residual match exactly. | 1.00e-08 | 35 | 35 | 7 | 15 | **92** |
| 1 | Architecture: MLP (width=30, depth=3) with Periodic Fourier Features (degree=2, Lx=Ly=2.0) on raw coordinates in [-1,1] (no normalization), feeding only embeddings to the MLP body. Optimizer workflow: Phase 1 Adam (10,000 iterations) with vRBA for weighting+sampling, followed by Phase 2 SSBroyden (30,000 iterations) with vRBA for sampling only. Loss terms: exactly L_total = L_pde (no BC or other terms). Curriculum schedule matches the required 10k+30k split. | 1.99e-08 | 35 | 35 | 7 | 15 | **92** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves results.npz containing keys: 'x' (65536x1), 'y' (65536x1), 'u_pred' (65536x1), 'u_an' (65536x1), 'abs_error' (65536x1). |
| 1 | Saves results.npz containing keys: 'x' (65536x1), 'y' (65536x1), 'u_pred' (65536x1), 'u_an' (65536x1), 'abs_error' (65536x1). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run6/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1919/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run6/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1922/` |


---
