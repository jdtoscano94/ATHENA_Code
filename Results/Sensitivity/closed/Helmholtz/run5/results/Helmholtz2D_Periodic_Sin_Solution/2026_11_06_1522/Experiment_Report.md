
# Experiment Report
*Last Updated: 2026-11-06 15:24:02*

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

**0. Triage & Data Role**
* **Role:** Validation/Error Metrics ONLY. (Forward Problem).

**1. Problem Statement**
* **Governing Equation (2D Helmholtz):** $u_{xx} + u_{yy} + k^2 u = g(x, y)$
* **Parameters:** $a_1 = 1$, $a_2 = 4$, $k = 1$.
* **Source Term $g(x,y)$:** Derived from the analytical solution: $g(x,y) = (1 - 17\pi^2) \sin(\pi x) \sin(4\pi y)$.
* **PDE Residual:** $r = u_{xx} + u_{yy} + u - (1 - 17\pi^2) \sin(\pi x) \sin(4\pi y)$
* **Boundary Conditions:** Periodic in both $x$ and $y$ on $[-1, 1]$.
* **Analytical Solution:** $u(x,y) = \sin(\pi x) \sin(4\pi y)$.
* **Problem Type:** Forward. Well-posed for PINNs.

**2. Input & Output**
* **Input:** 2D ($x, y$).
* **Output:** 1D scalar ($u$). Smooth, oscillatory features.

**3. Base Model Selection**
* **Optimizer:** Hybrid Adam + SSBroyden (2D Forward Problem).
* **Base MLP Specs:** `width=30`, `depth=3`.

**4. Main Architecture**
* **Structure:** Input -> Input Transformation -> Main Body -> Output Transformation -> Output.
* **Input Transformations:**
  * **Normalization:** $x$ and $y$ are already in $[-1, 1]$. **DO NOT** normalize them. Pass them directly to the embedding.
  * **Embedding:** Use **Periodic Fourier Features** for both $x$ and $y$.
  * **CRITICAL $L_x$ Derivation:** Since $x, y \in [-1, 1]$ enter the embedding directly, $L_x = x_{max} - x_{min} = 1 - (-1) = 2.0$. Set `Lx=2.0` for both coordinates.
  * **Degree:** Set `degree=2`.
  * **Composition Rule:** The main body must receive ONLY the periodic embeddings of $x$ and $y$. Do NOT pass the raw $x$ or $y$ coordinates to the main body.
* **Output Transformation:** None required. $u_{pred}(x,y) = \text{NN}(x,y)$.

**5. Constraint Implementation Strategy**
* **Count of Soft Loss Terms:** 1 ($L_{pde}$).
* **Deduction:** $L_{bc\_periodic}$ is **REMOVED** because Periodic Fourier Features inherently enforce exact periodicity.
* **Strategy:** No hard constraints or boundary loss terms are needed.

**6. Loss Function Assembly**
* **Symbolic Equation:** $L_{total} = L_{pde}$
* **Continuation Strategy:** Not required (no high Reynolds/Peclet number).

**7. Weighting Strategy**
* **Local Weights (`lambdas`):** Use **vRBA** (Variational Residual-Based Attention) for local adaptive weighting inside the PDE loss. `lambdas` are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden.
* **Global Weights (`lam`):** Total terms = 1. No global balancing is needed.
* **SSBroyden Integrity Rule:** Explicitly FORBID interfering with the internal weighting, sampling, or batching strategies of the SSBroyden optimizer.

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
*Wait, let me re-evaluate based on the strict instructions.*
The instructions state:
"IF STILL DECREASING ON AVERAGE (noisy but trending down):
REJECTED_FOR_REFINEMENT
"Score is excellent but the average error is still decreasing across the last 50% of the final stage (from ~[value] to ~[value], judged by the band center). Duplicate the number of iterations for the final stage."
STOP. Do NOT check history. Do NOT suggest exploration."

Let's write the correct output.

REJECTED_FOR_REFINEMENT

**Section 1: Status Decision**
* **Status:** REJECTED_FOR_REFINEMENT

**Section 2: Quantitative Scorecard**
* **Score Integrity:** 35
* **Score Accuracy:** 35
* **Score Details:** 7
* **Score Optimality:** 15
* **TOTAL SCORE:** 92

**Section 3: Forensic Analysis & Feedback**
* **Observation:** The training successfully transitions from Adam to SSBroyden around iteration 5000, marked by a sharp, smooth descent in both loss and error. However, analyzing the final stage (iterations 5000 to 14000), the curves have not plateaued. Looking at the last 50% of this final stage (iterations 9500 to 14000), the average band level of the loss at the midpoint (`A`) is $\sim 10^{-8}$, while at the end (`B`) it is $\sim 10^{-10}$. Similarly, the relative L2 error at the midpoint (`A`) is $\sim 10^{-6}$ and at the end (`B`) is $\sim 10^{-7}$. Because `B` is significantly lower than `A` for both metrics, the model is still actively learning and improving at the time of termination.
* **Diagnosis:** Budget-Limited / Premature Stop. Score is excellent but the average error is still decreasing across the last 50% of the final stage (from ~1e-6 to ~1e-7, judged by the band center). Duplicate the number of iterations for the final stage.
* **Targeted Improvements:**
    * **For Accuracy:** Increase the number of iterations of the LAST training stage (SSBroyden) by 2-3x (e.g., from ~9000 to 20000+ iterations) so both the loss and error curves genuinely flatten into a horizontal plateau before the run ends.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | MLP (width=30, depth=3) with Periodic Fourier Features (degree=2, Lx=2.0) as input embedding; main body receives only embeddings. Hybrid Adam + SSBroyden optimizer workflow with vRBA for local adaptive weighting/sampling. Loss assembly is exactly L_total = L_pde (no BC or other terms). No input normalization performed. Curriculum training: none specified or used. | 1.00e-07 | 35 | 35 | 7 | 15 | **92** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves predictions.npz containing keys: 'u_pred' (256x256), 'u_exact' (256x256), 'x' (256x256), 'y' (256x256). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run5/results//Helmholtz2D_Periodic_Sin_Solution/2026_11_06_1522/` |


---
