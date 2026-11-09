
# Experiment Report
*Last Updated: 2026-11-06 15:27:24*

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
- **Advisor Directive:** The previous run was budget-limited. The final stage (SSBroyden) must be extended to 20,000 iterations to allow the loss and error curves to fully plateau.
- **Data Role:** Validation/Error Metrics ONLY.

**1. Problem Statement**
- **Governing Equation:** 2D Helmholtz Equation.
  Residual: $L_{pde} = u_{xx} + u_{yy} + k^2 u - g(x, y) = 0$
  Constants: $k = 1$, $a_1 = 1$, $a_2 = 4$.
  Forcing term $g(x,y)$: Derived from the analytical solution, $g(x,y) = (1 - 17\pi^2) \sin(\pi x) \sin(4\pi y)$.
- **Boundary Conditions:** Periodic in both $x$ and $y$ on the domain $[-1, 1] \times [-1, 1]$.
- **Analytical Solution:** $u_{an}(x,y) = \sin(\pi x) \sin(4\pi y)$.
- **Problem Type:** Forward.
- **Input:** 2D ($x, y$).
- **Output:** 1D scalar ($u$).

**2. Base Model Selection**
- **Optimizer:** Hybrid Adam + SSBroyden.
- **Base MLP Specs:** `width=30`, `depth=3`.

**3. Main Architecture**
Instruct the Planner to strictly follow: Input -> Input Transformation -> Main Body -> Output Transformation -> Output.

*   **Input Transformations:**
    *   **Normalization:** The raw coordinates $x \in [-1, 1]$ and $y \in [-1, 1]$ are already in a tanh-friendly range. **DO NOT** apply any normalization (it is the identity).
    *   **Embedding:** Use **Periodic Fourier Features** for both $x$ and $y$.
    *   **Lx Derivation:** Since the inputs enter the embedding unnormalized with bounds $[-1, 1]$, `Lx = 1 - (-1) = 2.0` for both $x$ and $y$.
    *   **Degree:** `degree = 2`.
    *   **Input Composition Rule:** The main body must receive **ONLY** the periodic embeddings. Do NOT pass the raw $x$ or $y$ coordinates to the main body.
*   **Output Transformation:** None required. The periodic embeddings inherently satisfy the periodic boundary conditions.

**4. Constraint Implementation Strategy**
- **Count:** $L_{pde}$ (1). $L_{bc\_periodic}$ is **REMOVED** because Periodic Fourier Features enforce periodicity exactly by construction.
- **Total soft loss terms:** 1.
- **Strategy:** Loss-Based.

**5. Loss Function Assembly**
- **Symbolic Equation:** $L_{total} = L_{pde}$
- **Iteration Budget (CRITICAL):**
  - **Phase 1 (Adam):** 10,000 iterations.
  - **Phase 2 (SSBroyden):** **20,000 iterations** (Increased per Advisor's strict instruction to ensure convergence plateau).

**6. Weighting Strategy**
- **Local Weights (`lambdas`):** Use **vRBA** (Variational Residual-Based Attention) for local adaptive weighting inside the PDE loss. `lambdas` are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden.
- **Global Weights (`lam`):** Since there is only 1 loss term ($L_{pde}$), no global balancing is needed.
- **SSBroyden Integrity Rule:** Explicitly FORBID the Planner from interfering with the internal weighting, sampling, or batching strategies of the SSBroyden optimizer found in the template.

**7. Visualization & Saving Strategy**
- Do NOT modify, delete, or "improve" the visualization code in the base template.
- Save the predicted and exact fields as `.npz` arrays with keys specifying the variable names for visualization.

### Required Modules
Main Model:
[MLP with Hybrid Adam and SSBrodyden]
Enhancements:
[Periodic Fourier Features]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
APPROVED / REJECTED_FOR_REFINEMENT / REJECTED_FATAL
(Wait, the prompt says "Write EXACTLY one of these options:". I will output just the text as requested.)

**Section 1: Status Decision**
* **Status:** REJECTED_FOR_REFINEMENT

**Section 2: Quantitative Scorecard**
* **Score Integrity:** 35
* **Score Accuracy:** 35
* **Score Details:** 7
* **Score Optimality:** 15
* **TOTAL SCORE:** 92

**Section 3: Forensic Analysis & Feedback**
* **Observation:** 
  1. **Loss & Error History:** The training consists of an Adam phase (0-10k iterations) followed by an SSBroyden phase (10k-24k iterations). In the final SSBroyden stage, the 50% midpoint is at ~17,000 iterations. The average band level of the loss at this midpoint is `A` $\approx 10^{-10}$, while at the end of the run (~24k) it is `B` $\approx 10^{-11}$. For the relative $L^2$ error curve, the midpoint average is `A` $\approx 10^{-6}$ and the final average is `B` $\approx 10^{-7}$. Because `B` is lower than `A` for both curves, they are still descending on average and have not reached a genuine horizontal plateau.
  2. **Residual/Pointwise Error:** The maximum PDE residual is $\sim 10^{-5}$, and the pointwise error is $\sim 10^{-7}$, indicating very high precision across the domain.
  3. **Solution Fields:** The predicted solution is visually indistinguishable from the exact reference solution.
* **Diagnosis:** Budget-Limited / Premature Stop (Still Improving).
* **Targeted Improvements:** 
    * **For Integrity:** None
    * **For Accuracy:** Score is excellent but the average error is still decreasing across the last 50% of the final stage (from $\sim 10^{-6}$ to $\sim 10^{-7}$, judged by the band center). Duplicate the number of iterations for the final stage (increase the SSBroyden phase to 40,000 iterations).
    * **For Details:** Ensure both loss and error curves fully plateau in the final segment.
    * **For Optimality:** None

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | MLP (width=30, depth=3) with Periodic Fourier Features (degree=2, Lx=2.0) as input embedding; main body receives only embeddings. Hybrid Adam + SSBroyden optimizer workflow with vRBA for local adaptive weighting/sampling. Loss assembly is exactly L_total = L_pde (no BC or other terms). No input normalization performed. Curriculum training: none specified or used. | 1.00e-07 | 35 | 35 | 7 | 15 | **92** |
| 1 | MLP (width=30, depth=3) with Periodic Fourier Features (degree=2, Lx=2.0) as sole input transformation; raw coordinates are excluded. Hybrid optimizer: Adam (10k iterations) followed by SSBroyden (20k iterations). Loss assembly contains exactly one term: L_pde (Helmholtz residual) with vRBA local adaptive weighting/sampling. Periodic BCs are satisfied by construction via the embedding; no BC loss terms present. Forward problem, validation metrics only. | 1.00e-07 | 35 | 35 | 7 | 15 | **92** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves predictions.npz containing keys: 'u_pred' (256x256), 'u_exact' (256x256), 'x' (256x256), 'y' (256x256). |
| 1 | Saves predictions.npz containing keys: 'u_pred' (256x256), 'u_exact' (256x256), 'x' (256x256), 'y' (256x256). Also saves Log_files_SSBroyden.mat with training logs. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run5/results//Helmholtz2D_Periodic_Sin_Solution/2026_11_06_1522/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run5/results//Helmholtz2D_Periodic_Sin_Solution/2026_11_06_1525/` |


---
