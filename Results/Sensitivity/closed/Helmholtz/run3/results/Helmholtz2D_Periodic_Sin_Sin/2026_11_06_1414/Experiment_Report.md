
# Experiment Report
*Last Updated: 2026-11-06 14:16:01*

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

**0. Data Role**
Role: Validation/Error Metrics ONLY.

**1. Problem Statement**
*   **Governing Equation (2D Helmholtz):** $u_{xx} + u_{yy} + k^2 u = g(x, y)$
*   **PDE Residual:** $r = u_{xx} + u_{yy} + u - (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$
    *(Note: $g(x,y)$ is derived analytically from $u_{an}$ by substituting it into the LHS with $k=1, a_1=1, a_2=4$)*
*   **Boundary Conditions:** Periodic in both $x$ and $y$ on $[-1, 1] \times [-1, 1]$.
*   **Analytical Solution:** $u(x,y) = \sin(\pi x) \sin(4\pi y)$
*   **Problem Type:** Forward Problem. Well-posed for PINNs (admits strong form solution).

**2. Input & Output**
*   **Input:** 2D ($x, y$).
*   **Output:** 1D scalar ($u$).
*   **Characteristics:** Periodic, smooth, low-frequency.

**3. Base Model Selection**
*   **Optimizer:** Hybrid Adam + SSBroyden (Low Dim: 2D).

**4. Main Architecture**
Instruct the Planner to strictly follow: `Input -> Input Transformation -> Main Body -> Output Transformation -> Output`. All transformations must be differentiable and inside the model's `call` function.

*   **4.1 Input Transformations:**
    *   **Normalization:** $x$ and $y$ are already in $[-1, 1]$. **DO NOT** apply any normalization/MinMax scaling. Pass them unchanged to the embedding.
    *   **Feature Embedding:** Use **Periodic Fourier Features** for both $x$ and $y$.
    *   **$L_x$ and $L_y$ Derivation (CRITICAL):** Since raw coordinates enter the embedding directly, the ranges are $1 - (-1) = 2.0$. Set `Lx = 2.0` and `Ly = 2.0`.
    *   **Degree:** `degree = 2`.
    *   **Input Composition Rule:** The main body must receive **ONLY** the periodic embeddings of $x$ and $y$. **DO NOT** pass the raw $x$ or $y$ coordinates to the main body.
*   **4.2 Main Body Architecture:**
    *   Base MLP Specs: `width=30`, `depth=3`.
*   **4.3 Output Transformation:**
    *   None required. $u_{pred}(x,y) = \text{NN}(\text{Embed}(x), \text{Embed}(y))$.

**5. Constraint Implementation Strategy**
*   **Count:** $L_{pde}$ (1).
*   **Deduction:** $L_{bc\_periodic}$ is **REMOVED** because Periodic Fourier Features inherently enforce exact periodicity.
*   **Strategy:** Loss-based (only PDE loss remains).

**6. Loss Function Assembly**
*   $L_{total} = L_{pde}$
*   **Continuation Strategy:** Not required (no high Reynolds/Peclet number).

**7. Weighting Strategy**
*   **Local Weights (`lambdas`):** Use **vRBA** (Variational Residual-Based Attention) inside the PDE loss. `lambdas` are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden.
*   **Global Weights (`lam`):** Total Terms == 1. No global balancing is needed.
*   **SSBroyden Integrity Rule:** Explicitly FORBID the Planner from interfering with the internal weighting, sampling, or batching strategies of the SSBroyden optimizer.

**8. Visualization Strategy**
*   Do NOT modify, delete, or "improve" the visualization code in the base template.

**9. Saving Data**
*   Save the predicted and exact fields as `.npz` arrays with keys specifying the variable names (`x`, `y`, `u_pred`, `u_exact`) for visualization.

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
*Wait, let me re-evaluate based on the strict instructions.*
The instructions state:
"IF STILL DECREASING ON AVERAGE (noisy but trending down):
REJECTED_FOR_REFINEMENT
"Score is excellent but the average error is still decreasing across the last 50% of the final stage (from ~[value] to ~[value], judged by the band center). Duplicate the number of iterations for the final stage."
STOP. Do NOT check history. Do NOT suggest exploration."

I must output exactly `REJECTED_FOR_REFINEMENT`.

**Section 1: Status Decision**
* **Status:** REJECTED_FOR_REFINEMENT

**Section 2: Quantitative Scorecard**
* **Score Integrity:** 35
* **Score Accuracy:** 35
* **Score Details:** 5
* **Score Optimality:** 15
* **TOTAL SCORE:** 90

**Section 3: Forensic Analysis & Feedback**
* **Observation:** The model successfully captures the physics, achieving an excellent final relative L2 error of ~1.2e-7. The reference and predicted solution fields are visually identical, and the pointwise error is extremely low across the domain. However, analyzing the final training stage (the SSBroyden phase from iteration 5000 to ~13500), neither the loss nor the error curves have plateaued. Taking the last 50% of this final stage as a window (iterations ~9250 to 13500), the average loss band drops from `A` ≈ 1e-9 in the first half to `B` ≈ 1e-10 at the end. Similarly, the relative L2 error band drops from `A` ≈ 5e-7 to `B` ≈ 1e-7. Both curves are still visibly descending on average at the right edge of the plot.
* **Diagnosis:** Budget-Limited / Premature Stop. Score is excellent but the average error is still decreasing across the last 50% of the final stage (from ~5e-7 to ~1e-7, judged by the band center). 
* **Targeted Improvements:** 

    * **For Integrity:** None
    * **For Accuracy:** Duplicate the number of iterations for the final stage (e.g., increase the SSBroyden stage to 20,000 - 30,000 iterations) so both the loss and error curves genuinely flatten before the run ends.
    * **For Details:** Ensure loss plateaus in final segment.
    * **For Optimality:** None

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The implementation uses a PINN with Periodic Fourier Features (degree=2, Lx=Ly=2.0) on raw coordinates in [-1,1] followed by an MLP (width=30, depth=3). Training follows the Hybrid Adam + SSBroyden workflow with vRBA weighting/sampling applied exclusively inside the single PDE loss term. The PDE residual exactly matches the required Helmholtz form with the analytic source term; no BC loss terms are present. Output transformation is absent as specified. | 1.20e-07 | 35 | 35 | 5 | 15 | **90** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1414/` |


---
