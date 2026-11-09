
# Experiment Report
*Last Updated: 2026-11-06 13:07:03*

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
Role: Validation/Error Metrics ONLY.

**1. Problem Statement**
*   **Parameters:** $a_1 = 1$, $a_2 = 4$, $k = 1$.
*   **Analytical Solution:** $u(x,y) = \sin(\pi x)\sin(4\pi y)$
*   **PDE Residual:** $u_{xx} + u_{yy} + u - (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y) = 0$

**2-5. Setup & Architecture**
Initial logic and parameters are validated. Base MLP (30x3) with Periodic Fourier Features ($Lx=2.0$, degree=2). Standard processing applied.

**6. Loss Assembly & Finite Budget**
$L_{total} = L_{pde}$. 
Adam: 10,000 iterations; SSBroyden: 10,000 iterations. No open-ended stopping conditions permitted.

**7. Weighting Strategy**
*   **Global Weights (`lam`)**: Set `lam_pde = 1`. Freeze Global Weights at their final Adam values before entering SSBroyden.
*   **Local Weights (`lambdas`)**: vRBA applied pointwise inside the PDE loss. **`lambdas` are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden.**

**8-9. Visualization & Saving**
Standard processing applied.

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
* **Score Accuracy:** 15
* **Score Details:** 0
* **Score Optimality:** 10
* **TOTAL SCORE:** 60

**Section 3: Forensic Analysis & Feedback**
* **Observation:** The total loss history shows a two-stage training process. In the final SSBroyden stage (iterations 10,000 to ~17,500), the loss drops continuously and early-terminates before the budget is exhausted. Applying the mandatory dual-curve test to the final stage: the 50% midpoint is at iteration ~13,750, where the average loss band is at `A` = 1e-6. At the end of the stage (~17,500), the average loss band is at `B` = 1e-9. Since `B` is significantly lower than `A`, the loss curve is still descending on average and has not plateaued. The relative L2 error curve is also extremely high and slightly descending (`A` = 3.5e0, `B` = 3.0e0). Visually, the prediction plot shows a transposed pattern compared to the reference (4 waves vertically instead of horizontally). Given the PDE loss reached an impressive 1e-9, the model has successfully learned the physics, but a coordinate swap bug during evaluation is causing the massive 300% error.
* **Diagnosis:** Budget-Limited / Premature Stop (Still Improving) combined with an Evaluation Coordinate Swap. The optimizer early-terminated while the loss was still strictly descending. Furthermore, the massive error despite a near-zero PDE loss strongly indicates that the $x$ and $y$ coordinates are swapped during the evaluation/plotting phase, comparing the prediction to a transposed reference.
* **Targeted Improvements:**
    * **For Integrity:** Fix the evaluation and plotting pipeline. Ensure the grid coordinates `(x, y)` are not swapped when passed to the model for inference, and verify that the reshaping/transpose operations for `plt.imshow` align perfectly with the reference solution axes.
    * **For Accuracy:** Increase the number of iterations of the LAST training stage (e.g., raise the final SSBroyden stage from 10,000 to 30,000 iterations) and reinitialize the Hessian to continue, so both the loss and error curves flatten before the run ends.
    * **For Details:** Ensure the loss genuinely plateaus in the final segment rather than early-terminating while still descending.
    * **For Optimality:** Increase the embedding degree `kmax` from 2 to at least 4. The analytical solution contains a $4\pi y$ frequency, and a `kmax=2` only explicitly provides up to $2\pi$, forcing the MLP to work harder to generate the higher harmonics.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Architecture: PINN with Periodic_Fourier_Features (kmax=2, Lx=2.0) feeding a 30x3 MLP (tanh). Optimizer workflow: 10k Adam iterations (vRBA weighting + sampling) followed by 10k SSBroyden iterations (vRBA sampling only, plain MSE loss). Loss assembly: L_total = L_pde only (no data, BC, or gauge terms). Global lam_pde=1. Curriculum: fixed 10k+10k budget with vRBA EMA updates. | 300.00% | 35 | 15 | 0 | 10 | **60** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves Log_files_SSBroyden.mat containing keys: 'model_name' (str), 'all_errors' (list), 'all_its' (list), 'all_loss' (list), 'all_gamma' (list), 'all_max_RBA' (list), 'all_L_infty' (list), 'all_L_2' (list). No field arrays saved. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run1/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1303/` |


---
