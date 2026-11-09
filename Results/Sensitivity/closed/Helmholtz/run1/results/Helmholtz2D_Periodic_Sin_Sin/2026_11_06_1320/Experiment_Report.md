
# Experiment Report
*Last Updated: 2026-11-06 13:23:00*

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
**Atomic Conclusion:** The critical fix requires implementing the exact source term $g(x,y) = (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$ without coordinate transposition, and strictly omitting any hallucinated data-saving instructions not requested by the user.

### Strategy Plan

**1. Data Source**
* Role: Validation/Error Metrics ONLY.

**2. Problem Statement**
* **Governing Equation:** $u_{xx} + u_{yy} + u = g(x,y)$
* **PDE Residual:** $L_{pde} = u_{xx} + u_{yy} + u - (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$
* **Domain:** 2D, $x \in [-1, 1]$, $y \in [-1, 1]$.
* **Boundary Conditions:** Periodic in both $x$ and $y$.
* **Analytical Solution:** $u_{an} = \sin(\pi x)\sin(4\pi y)$.
* **Problem Type:** Forward. Well-posed for PINNs.

**3. Base Model Selection**
* **Optimizer:** Hybrid Adam + SSBroyden.
* **Base MLP Specs:** `width=30`, `depth=3`.

**4. Main Architecture**
* **Structure:** Input -> Periodic Fourier Features -> Main Body -> Output.
* **Input Transformations:**
  * **Normalization:** SKIP normalization. Pass raw coordinates directly.
  * **Periodic Embedding:** Apply to $x$ and $y$. `Lx = 2.0` (derived from $1 - (-1)$). `degree = 4`.
  * **Composition Rule:** The main body must ONLY receive the periodic embeddings.
* **Output Transformation:** None.

**5. Constraint Implementation Strategy**
* **Strategy:** Loss-Based. Periodic BCs are enforced architecturally.

**6. Loss Function Assembly**
* **Total Loss:** $L_{total} = L_{pde}$
* **Iteration Budget:** 10,000 Adam -> 30,000 SSBroyden.

**7. Weighting Strategy**
* **Local Weights:** `vRBA` for local adaptive weighting inside the PDE loss.
* **Global Weights:** `lam_pde = 1.0`.

**8. Visualization Strategy**
* Verify `X, Y = meshgrid(x, y)` indexing to prevent coordinate transposition. Standard callbacks apply.

**9. Saving Data**
* No specific saving instructions provided by the user. Do not save data.

### Main Model:
[MLP with Hybrid Adam and SSBrodyden]

### Enhancements:
[Periodic Fourier Features]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
APPROVED
*Wait, let me re-evaluate based on the strict instructions.*
The instructions state:
"If Current Score ≥ 90:
MANDATORY PLATEAU CHECK:
...
IF STILL DECREASING ON AVERAGE (noisy but trending down):
REJECTED_FOR_REFINEMENT
"Score is excellent but the average error is still decreasing across the last 50% of the final stage (from ~[value] to ~[value], judged by the band center). Duplicate the number of iterations for the final stage."
STOP. Do NOT check history. Do NOT suggest exploration."

I must output REJECTED_FOR_REFINEMENT.

**Section 1: Status Decision**
* **Status:** REJECTED_FOR_REFINEMENT

**Section 2: Quantitative Scorecard**
* **Score Integrity:** 35
* **Score Accuracy:** 35
* **Score Details:** 7
* **Score Optimality:** 15
* **TOTAL SCORE:** 92

**Section 3: Forensic Analysis & Feedback**
* **Observation:** The training successfully completes the Adam phase and transitions to the SSBroyden phase at iteration 10,000. In the final SSBroyden stage (10,000 to ~27,000), the loss and error curves are stable and reach excellent minimums, but they have not plateaued. For the Loss curve, the average band level at the 50% midpoint of the final stage (~18,500) is A $\approx 10^{-9}$, while at the end (~27,000) it is B $\approx 10^{-11}$. For the Error curve, the midpoint level is A $\approx 10^{-6}$ and the end level is B $\approx 10^{-7}$. Since B < A for both curves, they are still descending on average. The prediction visually matches the reference perfectly, and the pointwise error is extremely low.
* **Diagnosis:** Budget-Limited / Premature Stop. The model is highly accurate and stable, but the optimization was terminated while the loss and error were still actively decreasing. It has not reached its true convergence floor.
* **Targeted Improvements:**
    * **For Accuracy:** Score is excellent but the average error is still decreasing across the last 50% of the final stage (from ~1e-6 to ~1e-7, judged by the band center). Duplicate the number of iterations for the final stage (e.g., increase SSBroyden to 60,000 - 100,000 iterations).

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Architecture: PINN with Periodic_Fourier_Features (kmax=2, Lx=2.0) feeding a 30x3 MLP (tanh). Optimizer workflow: 10k Adam iterations (vRBA weighting + sampling) followed by 10k SSBroyden iterations (vRBA sampling only, plain MSE loss). Loss assembly: L_total = L_pde only (no data, BC, or gauge terms). Global lam_pde=1. Curriculum: fixed 10k+10k budget with vRBA EMA updates. | 300.00% | 35 | 15 | 0 | 10 | **60** |
| 1 | The implementation uses a PINN with Periodic Fourier Features (degree=4, Lx=2.0) followed by an MLP (width=30, depth=3). Training follows Hybrid Adam (10k iterations) + SSBroyden (30k iterations). The PDE residual exactly implements L_pde = u_xx + u_yy + u - (1-17π²)sin(πx)sin(4πy) with only L_total = L_pde (no extra terms). vRBA provides local adaptive weighting (sampling in both phases; weighting in Adam). Global weights are frozen (lamE=1) entering SSBroyden. Analytical solution and source term match the report exactly. | 465.00% | 35 | 0 | 0 | 15 | **50** |
| 3 | Architecture: MLP (width=30, depth=3) with Periodic Fourier Features (degree=4, Lx=2.0) applied separately to each coordinate before the main body; raw coordinates passed with no normalization. Optimizer workflow: 10,000 Adam steps followed by 30,000 SSBroyden steps. Loss: exactly L_pde (utt + uxx + u - (1-17π²)sin(πt)sin(4πx)) with vRBA local adaptive weighting inside the PDE term and global lam_pde=1.0. Curriculum schedule matches the requested 10k Adam → 30k SSBroyden split. No data is saved. | 1.00e-07 | 35 | 35 | 7 | 15 | **92** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves Log_files_SSBroyden.mat containing keys: 'model_name' (str), 'all_errors' (list), 'all_its' (list), 'all_loss' (list), 'all_gamma' (list), 'all_max_RBA' (list), 'all_L_infty' (list), 'all_L_2' (list). No field arrays saved. |
| 1 | Saves evaluation_fields.npz containing keys: 'x' (10000x1), 'y' (10000x1), 'u_pred' (10000x1), 'u_an' (10000x1), 'g' (10000x1). |
| 3 | No data saved; saving explicitly omitted per strategy. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run1/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1303/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run1/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1313/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run1/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1320/` |


---
