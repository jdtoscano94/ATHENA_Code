
# Experiment Report
*Last Updated: 2026-11-06 13:28:10*

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
* Role: Validation/Error Metrics ONLY.

**2. Problem Statement**
* **Governing Equation (2D Helmholtz):** $u_{xx} + u_{yy} + k^2 u = g(x,y)$
* **Residual:** $L_{pde} = u_{xx} + u_{yy} + k^2 u - g(x,y)$ where $g(x,y) = (k^2 - (\pi a_1)^2 - (\pi a_2)^2) \sin(\pi a_1 x) \sin(\pi a_2 y)$ with constants $a_1=1$, $a_2=4$, $k=1$.
* **Domain:** $x \in [-1, 1]$, $y \in [-1, 1]$ (2D input).
* **Boundary Conditions:** Periodic in both $x$ and $y$.
* **Analytical Solution:** $u_{an} = \sin(\pi a_1 x) \sin(\pi a_2 y)$ where $a_1=1$, $a_2=4$.
* **Problem Type:** Forward.

**3. Base Model Selection**
* MLP with Hybrid Adam and SSBroyden (2D input, low dimension).

**4. Main Architecture**
* **Structure:** Input -> Periodic Fourier Features -> Main Body -> Output.
* **Input Transformations:**
  * Do NOT normalize $x$ and $y$ as they are already in the nice range $[-1, 1]$.
  * **Periodic Fourier Features:** Apply to both $x$ and $y$.
  * **Derivation of Lx and Ly:** Since raw $x, y \in [-1, 1]$, $L_x = 2.0$ and $L_y = 2.0$.
  * **Degree:** 4.
  * **CRITICAL:** Pass ONLY the embedded features to the main body. Do NOT pass raw $x$ or $y$.
* **Main Body:** Base MLP Specs: `width=30`, `depth=3`.
* **Output Transformation:** None.

**5. Constraint Implementation Strategy**
* **Count:** $L_{pde}$ (1).
* Periodic BCs are exactly enforced by the Periodic Fourier Features. Do NOT add a loss term for them.

**6. Loss Function Assembly**
* $L_{total} = L_{pde}$
* **Iteration Budget (ADVISOR MANDATORY):** 
  * **Phase 1 (Adam):** 10,000 iterations.
  * **Phase 2 (SSBroyden):** 60,000 iterations (Duplicated from previous trial to ensure full convergence).

**7. Weighting Strategy**
* **Local Weights (`lambdas`):** Use vRBA inside the PDE loss. `lambdas` are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden.
* **Global Weights (`lam`):** Total terms = 1. No global balancing needed. Set `lam_pde = 1.0`.

**8. Visualization Strategy**
* Do not modify the default visualization callbacks.

**9. Saving Data**
* Save the fields as `.npz` arrays with exact keys `x`, `y`, `u_pred`, and `u_exact` for visualization.

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
* **Observation:** The training successfully transitions from Adam to SSBroyden at iteration 10,000, marked by a sharp drop in both loss and error. The final stage spans from 10,000 to approximately 46,000 iterations. Evaluating the last 50% of this final stage (from ~28,000 to 46,000):
  - For the Loss curve: The average band level at the 50% midpoint (`A`) is $\approx 10^{-11}$, while the average band level at the end (`B`) is $\approx 10^{-12}$.
  - For the Error curve: The average band level at the 50% midpoint (`A`) is $\approx 10^{-7}$, while the average band level at the end (`B`) is $\approx 10^{-8}$.
  Because `B` is explicitly lower than `A` for both metrics, the curves are STILL DECREASING ON AVERAGE. The model achieves an excellent final error of $8.63 \times 10^{-9}$, and the predicted solution visually matches the reference perfectly, but a genuine horizontal plateau has not been reached.
* **Diagnosis:** Budget-Limited / Premature Stop (Still Improving). The run early-terminated or hit its iteration cap while the model was still actively learning and improving its error floor. 
* **Targeted Improvements:** 

    * **For Integrity:** None
    * **For Accuracy:** Score is excellent but the average error is still decreasing across the last 50% of the final stage (from ~1e-7 to ~1e-8, judged by the band center). Duplicate the number of iterations for the final stage (e.g., raise the final SSBroyden stage from 60,000 to 120,000 iterations) and reinitialize the Hessian if necessary to continue training until the curves genuinely flatten.
    * **For Details:** Ensure both the loss and error curves plateau horizontally in the final segment before the run ends.
    * **For Optimality:** None

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Architecture: PINN with Periodic_Fourier_Features (kmax=2, Lx=2.0) feeding a 30x3 MLP (tanh). Optimizer workflow: 10k Adam iterations (vRBA weighting + sampling) followed by 10k SSBroyden iterations (vRBA sampling only, plain MSE loss). Loss assembly: L_total = L_pde only (no data, BC, or gauge terms). Global lam_pde=1. Curriculum: fixed 10k+10k budget with vRBA EMA updates. | 300.00% | 35 | 15 | 0 | 10 | **60** |
| 1 | The implementation uses a PINN with Periodic Fourier Features (degree=4, Lx=2.0) followed by an MLP (width=30, depth=3). Training follows Hybrid Adam (10k iterations) + SSBroyden (30k iterations). The PDE residual exactly implements L_pde = u_xx + u_yy + u - (1-17π²)sin(πx)sin(4πy) with only L_total = L_pde (no extra terms). vRBA provides local adaptive weighting (sampling in both phases; weighting in Adam). Global weights are frozen (lamE=1) entering SSBroyden. Analytical solution and source term match the report exactly. | 465.00% | 35 | 0 | 0 | 15 | **50** |
| 3 | Architecture: MLP (width=30, depth=3) with Periodic Fourier Features (degree=4, Lx=2.0) applied separately to each coordinate before the main body; raw coordinates passed with no normalization. Optimizer workflow: 10,000 Adam steps followed by 30,000 SSBroyden steps. Loss: exactly L_pde (utt + uxx + u - (1-17π²)sin(πt)sin(4πx)) with vRBA local adaptive weighting inside the PDE term and global lam_pde=1.0. Curriculum schedule matches the requested 10k Adam → 30k SSBroyden split. No data is saved. | 1.00e-07 | 35 | 35 | 7 | 15 | **92** |
| 4 | MLP (width=30, depth=3) with Periodic Fourier Features (degree=4, L=2.0) on both inputs; only embedded features passed to MLP. Hybrid optimizer: Adam (10k iterations) followed by SSBroyden (60k iterations). Loss: exactly L_pde (Helmholtz residual with correct source term (1-17π²)sin(πt)sin(4πx)); vRBA used for weighting+sampling in Adam and sampling-only in SSBroyden. No BC loss terms. Forward problem, no data leakage. | 8.63e-09 | 35 | 35 | 7 | 15 | **92** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves Log_files_SSBroyden.mat containing keys: 'model_name' (str), 'all_errors' (list), 'all_its' (list), 'all_loss' (list), 'all_gamma' (list), 'all_max_RBA' (list), 'all_L_infty' (list), 'all_L_2' (list). No field arrays saved. |
| 1 | Saves evaluation_fields.npz containing keys: 'x' (10000x1), 'y' (10000x1), 'u_pred' (10000x1), 'u_an' (10000x1), 'g' (10000x1). |
| 3 | No data saved; saving explicitly omitted per strategy. |
| 4 | Saves results.npz containing keys: 'x' (10000x1), 'y' (10000x1), 'u_pred' (10000x1), 'u_exact' (10000x1). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run1/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1303/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run1/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1313/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run1/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1320/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run1/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1325/` |


---
