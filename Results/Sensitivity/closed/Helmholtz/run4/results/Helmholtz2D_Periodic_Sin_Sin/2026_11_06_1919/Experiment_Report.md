
# Experiment Report
*Last Updated: 2026-11-06 19:21:39*

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
Phase 2 (SSBroyden) is set to exactly 15,000 iterations. Freeze global weights (`lam`) at their final Adam values before SSBroyden; `lambdas` remain active for sampling only during SSBroyden. Save an `.npz` containing `x`, `y`, `u_pred`, `u_an`, and `abs_error`.

### Strategy Plan

**0 - 5. Problem Formulation & Architecture**
Initial logic and parameters are validated. 
Analytical Solution: $u(x,y) = \sin(\pi x) \sin(4\pi y)$
PDE Residual: $R = u_{xx} + u_{yy} + u - (1 - 17\pi^2) \sin(\pi x) \sin(4\pi y)$
Embeddings: Periodic Fourier Features ($Lx=2.0, Ly=2.0$).
Standard processing applied for remaining architectural constraints.

**6. Loss Function Assembly & Continuation**
$L_{total} = L_{pde}$
Phase 1 (Adam): 10,000 iterations.
Phase 2 (SSBroyden): 15,000 iterations.

**7. Weighting Strategy**
Freeze global weights (`lam`) at their final Adam values before SSBroyden; `lambdas` remain active for sampling only during SSBroyden.

**8 - 9. Visualization & Saving**
Save an `.npz` containing `x`, `y`, `u_pred`, `u_an`, and `abs_error`.

### Required Modules
Main Model:
[MLP with Hybrid Adam and SSBroyden]
Enhancements:
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
* **Observation:** 
  1. **Loss & Error History:** The y-axis for the total loss ranges from $10^2$ down to $10^{-12}$, and the relative L2 error ranges from $10^0$ down to $10^{-8}$. The curves show a clear transition at 10,000 iterations, marking the shift from Adam to SSBroyden. Analyzing the final SSBroyden stage (from 10k to ~22k iterations) for a plateau: at the 50% midpoint (~16,000 iterations), the average loss band level is `A` = $10^{-10}$ and the average error band level is `A` = $10^{-7}$. At the end of the run (~22,000 iterations), the average loss band level is `B` = $10^{-12}$ and the average error band level is `B` = $10^{-8}$. Because `B` is lower than `A` for both metrics, the curves are still decreasing on average and have not genuinely plateaued.
  2. **Residual/Pointwise Error:** The PDE residual heatmap shows a uniform distribution of error with a maximum around $10^{-6}$. The pointwise error is exceptionally low, peaking around $10^{-8}$.
  3. **Solution Fields:** The predicted solution field is visually identical to the reference analytical solution, capturing the high-frequency periodic features perfectly.

* **Diagnosis:** Budget-Limited / Premature Stop. Score is excellent but the average error is still decreasing across the last 50% of the final stage (from ~1e-7 to ~1e-8, judged by the band center). Duplicate the number of iterations for the final stage.

* **Targeted Improvements:**
    * **For Integrity:** None
    * **For Accuracy:** Increase the number of iterations of the LAST training stage (e.g. 2-5x) so both the loss and error curves flatten before the run ends. Raise the final SSBroyden stage from 15,000 to at least 30,000 iterations.
    * **For Details:** Ensure both the loss and error curves reach a genuine horizontal plateau in the final segment before terminating.
    * **For Optimality:** None

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | MLP architecture with Periodic Fourier Features (Lx=2.0, Ly=2.0) embeddings. Hybrid optimizer: Adam (10k iterations) followed by SSBroyden (exactly 15k iterations). Loss is strictly L_pde only. Global weights (lamE) frozen after Adam; lambdas active solely for importance sampling during SSBroyden. Analytical solution and PDE residual match exactly. | 1.00e-08 | 35 | 35 | 7 | 15 | **92** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves results.npz containing keys: 'x' (65536x1), 'y' (65536x1), 'u_pred' (65536x1), 'u_an' (65536x1), 'abs_error' (65536x1). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run6/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1919/` |


---
