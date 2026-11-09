
# Experiment Report
*Last Updated: 2026-11-06 13:16:22*

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
**Role:** Validation/Error Metrics ONLY.

**1. Problem Statement & Exact Save Schema**
*   **Governing Equation:** $u_{xx} + u_{yy} + u = g(x, y)$
*   **PDE Residual:** $L_{pde} = u_{xx} + u_{yy} + u - (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y) = 0$
*   **Analytical Solution:** $u_{an}(x,y) = \sin(\pi x)\sin(4\pi y)$
*   **Domain & BCs:** $x, y \in [-1, 1]$. Periodic BCs. Forward problem.

**2. Base Model Selection**
*   Hybrid Adam + SSBroyden. Base MLP: `width=30`, `depth=3`.

**3. Main Architecture & Transformations**
*   **Flow:** Input -> Periodic Fourier Features -> Main Body -> Output.
*   **Input Transformations:** $x$ and $y$ remain in $[-1, 1]$ (unnormalized). Do not pass raw coordinates to the main body.
*   **Feature Embeddings (Advisor Override):** Apply Periodic Fourier Features. Set `Lx = 2.0`. Set `degree = 4` to capture the $4\pi y$ harmonic.

**4. Constraint Implementation Strategy**
*   Periodicity is enforced architecturally via the embedding. No output transformation required ($u_{pred} = NN(x,y)$).

**5. Loss Function Assembly & Curriculum (Advisor Override)**
*   $L_{total} = L_{pde}$
*   **Phase 1 (Adam):** 10,000 iterations.
*   **Phase 2 (SSBroyden):** 30,000 iterations (increased to ensure convergence).

**6. Weighting Strategy (Critic Fix)**
*   **Local Weights (`lambdas`):** Use `vRBA` for local adaptive weighting inside the PDE loss (Weighting & Sampling in Adam; Sampling ONLY in SSBroyden).
*   **Global Weights (`lam`):** **CRITICAL:** Freeze global weights (e.g., `lam_pde=1`) at their final Adam values before entering SSBroyden. Strictly forbid dynamic global weighting updates during SSBroyden to preserve Hessian integrity.

**7. Visualization Strategy**
*   Ensure grid coordinates `(x, y)` are not transposed during inference or `plt.imshow` reshaping.

**8. Saving Data**
*   Save the evaluation grid and fields as `.npz` arrays with explicit keys: `x`, `y`, `u_pred`, `u_an`, and `g`.

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
* **Score Accuracy:** 0
* **Score Details:** 0
* **Score Optimality:** 15
* **TOTAL SCORE:** 50

**Section 3: Forensic Analysis & Feedback**
* **Observation:** 
  1. **Loss & Error History:** The final training stage (SSBroyden) spans from iteration 10,000 to 30,000. For the Loss curve, the average band level at the 50% midpoint (iteration 20,000) is `A = 1e-7`, and at the end (iteration 30,000) it is `B = 1e-8`. The loss is still decreasing. However, the Error curve exhibits a catastrophic failure: it jumps at the 10k transition and steadily INCREASES throughout the final stage (from `A ≈ 7` at 20k to `B ≈ 10` at 30k). The final reported error is 4.65 (465%).
  2. **Residual/Pointwise Error:** Despite the massive error, the PDE residual is extremely low (max $3.7 \times 10^{-4}$), indicating the network perfectly solved the PDE it was given.
  3. **Solution Fields:** The prediction is visibly transposed compared to the reference. The reference plot displays 2 half-waves along the horizontal axis and 8 half-waves along the vertical axis. The prediction plot displays the exact opposite: 8 half-waves horizontally and 2 vertically. 

* **Diagnosis:** Formulation Mismatch / Coordinate Transposition. The network has perfectly solved a PDE (loss reached $10^{-8}$), but it is the WRONG PDE. The transposed frequencies in the prediction plot definitively prove that the source term $g(x,y)$ was implemented with swapped coordinates (e.g., `sin(4*pi*x)*sin(pi*y)` instead of `sin(pi*x)*sin(4*pi*y)`), or the `a1` and `a2` constants were swapped in the code. Because the network is forced to satisfy this incorrect source term, it learns the transposed solution. As SSBroyden rapidly converges to this wrong solution, the distance to the true analytical solution increases, explaining why the loss drops while the error diverges.

* **Targeted Improvements:**
    * **For Integrity:** None.
    * **For Accuracy:** Fix the source term implementation in the PDE loss. Ensure that $g(x,y) = (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$ exactly, without swapping `x` and `y` or the frequency multipliers. 
    * **For Details:** Verify that `X, Y = meshgrid(x, y)` is not being flattened or reshaped in a way that transposes the coordinates during the evaluation and plotting phases.
    * **For Optimality:** None. The architecture (Periodic Fourier Features + MLP) and optimizer strategy are perfectly suited for this problem once the formulation bug is fixed.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Architecture: PINN with Periodic_Fourier_Features (kmax=2, Lx=2.0) feeding a 30x3 MLP (tanh). Optimizer workflow: 10k Adam iterations (vRBA weighting + sampling) followed by 10k SSBroyden iterations (vRBA sampling only, plain MSE loss). Loss assembly: L_total = L_pde only (no data, BC, or gauge terms). Global lam_pde=1. Curriculum: fixed 10k+10k budget with vRBA EMA updates. | 300.00% | 35 | 15 | 0 | 10 | **60** |
| 1 | The implementation uses a PINN with Periodic Fourier Features (degree=4, Lx=2.0) followed by an MLP (width=30, depth=3). Training follows Hybrid Adam (10k iterations) + SSBroyden (30k iterations). The PDE residual exactly implements L_pde = u_xx + u_yy + u - (1-17π²)sin(πx)sin(4πy) with only L_total = L_pde (no extra terms). vRBA provides local adaptive weighting (sampling in both phases; weighting in Adam). Global weights are frozen (lamE=1) entering SSBroyden. Analytical solution and source term match the report exactly. | 465.00% | 35 | 0 | 0 | 15 | **50** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves Log_files_SSBroyden.mat containing keys: 'model_name' (str), 'all_errors' (list), 'all_its' (list), 'all_loss' (list), 'all_gamma' (list), 'all_max_RBA' (list), 'all_L_infty' (list), 'all_L_2' (list). No field arrays saved. |
| 1 | Saves evaluation_fields.npz containing keys: 'x' (10000x1), 'y' (10000x1), 'u_pred' (10000x1), 'u_an' (10000x1), 'g' (10000x1). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run1/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1303/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run1/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1313/` |


---
