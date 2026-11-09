
# Experiment Report
*Last Updated: 2026-11-06 13:33:40*

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
**Role: Validation/Error Metrics ONLY.**

**1. Data Source**
No external training data. Use the analytical solution strictly for validation and error metrics.

**2. Problem Statement**
*   **Governing Equation (2D Helmholtz):** 
    $u_{xx} + u_{yy} + u - (1 - 17\pi^2) \sin(\pi x) \sin(4\pi y) = 0$
*   **Boundary Conditions:** Periodic in both $x$ and $y$.
*   **Analytical Solution:** $u(x,y) = \sin(\pi x) \sin(4\pi y)$
*   **Domain:** $x \in [-1, 1]$, $y \in [-1, 1]$
*   **Problem Type:** Forward

**3. Base Model Selection**
*   **Optimizer:** Hybrid Adam + SSBroyden
*   **Base MLP Specs:** `width=30`, `depth=3`

**4. Main Architecture**
*   **Structure:** Input -> Periodic Embedding -> Main Body -> Output
*   **Input Transformations:**
    *   $x$ and $y$ are already in $[-1, 1]$. **DO NOT** apply any normalization.
    *   **Periodic Fourier Features:** Apply to both $x$ and $y$.
    *   **CRITICAL `Lx` Derivation:** Since $x, y \in [-1, 1]$ enter the embedding directly, `Lx = 1 - (-1) = 2.0` for both coordinates.
    *   **Degree:** Set `degree = 4`.
    *   **Input Composition Rule:** Pass **ONLY** the embedded features to the MLP main body. Do not concatenate the raw $x$ or $y$ coordinates.

**5. Constraint Implementation Strategy**
*   **Periodic BCs:** Enforced exactly by the Periodic Fourier Features.
*   **Decision:** No additional hard constraints or boundary loss terms are required.

**6. Loss Function Assembly**
*   $L_{total} = L_{pde}$
*   **Iteration Budget (ADVISOR MANDATE):** 
    *   **Adam Phase:** 10,000 iterations.
    *   **SSBroyden Phase:** **120,000 iterations**. (Increased from 60k to ensure the loss and error curves reach a genuine horizontal plateau).

**7. Weighting Strategy**
*   **Local Weighting:** Use **vRBA** (Variational Residual-Based Attention) for local adaptive weighting inside the PDE loss. `lambdas` are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden.
*   **Global Weighting:** Since there is only one loss term ($L_{pde}$), set `lam_pde = 1.0`. No global balancing is needed.

**8. Visualization Strategy**
*   Keep standard visualization callbacks. Ensure the predicted $u$, analytical $u$, and absolute error fields are plotted over the 2D domain.

**9. Saving Data**
*   Save the final predicted fields and analytical solutions as `.npz` arrays with clear keys for visualization.

**Main Model:**
[MLP with Hybrid Adam and SSBrodyden]

**Enhancements:**
[Periodic Fourier Features]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
APPROVED
Wait, let me re-evaluate. The instructions state:
"IF STILL DECREASING ON AVERAGE (noisy but trending down):
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
* **Observation:** The final training stage (SSBroyden) spans from iteration 10,000 to 80,000. The 50% midpoint of this final stage is at iteration 45,000. For the Total Loss curve, the average band level at the midpoint (`A`) is approximately 1e-12, while at the end of the run (`B`) it is approximately 1e-13. For the Relative L2 Error curve, the average band level at the midpoint (`A`) is approximately 3e-8, and at the end (`B`) it is approximately 1e-8. Because `B` is lower than `A` for both curves, the training is still descending on average across the last 50% of the window and has not reached a genuine horizontal plateau. The pointwise error map shows a maximum error of ~2e-8, which is excellent, but the lack of a plateau indicates the model was still learning when the iteration budget ran out.
* **Diagnosis:** Budget-Limited / Premature Stop (Still Improving).
* **Targeted Improvements:** 

    * **For Integrity:** None
    * **For Accuracy:** Score is excellent but the average error is still decreasing across the last 50% of the final stage (from ~3e-8 to ~1e-8, judged by the band center). Duplicate the number of iterations for the final stage.
    * **For Details:** Ensure both the loss and error curves flatten completely into a horizontal plateau before the run ends.
    * **For Optimality:** None

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Architecture: PINN with Periodic_Fourier_Features (kmax=2, Lx=2.0) feeding a 30x3 MLP (tanh). Optimizer workflow: 10k Adam iterations (vRBA weighting + sampling) followed by 10k SSBroyden iterations (vRBA sampling only, plain MSE loss). Loss assembly: L_total = L_pde only (no data, BC, or gauge terms). Global lam_pde=1. Curriculum: fixed 10k+10k budget with vRBA EMA updates. | 300.00% | 35 | 15 | 0 | 10 | **60** |
| 1 | The implementation uses a PINN with Periodic Fourier Features (degree=4, Lx=2.0) followed by an MLP (width=30, depth=3). Training follows Hybrid Adam (10k iterations) + SSBroyden (30k iterations). The PDE residual exactly implements L_pde = u_xx + u_yy + u - (1-17π²)sin(πx)sin(4πy) with only L_total = L_pde (no extra terms). vRBA provides local adaptive weighting (sampling in both phases; weighting in Adam). Global weights are frozen (lamE=1) entering SSBroyden. Analytical solution and source term match the report exactly. | 465.00% | 35 | 0 | 0 | 15 | **50** |
| 3 | Architecture: MLP (width=30, depth=3) with Periodic Fourier Features (degree=4, Lx=2.0) applied separately to each coordinate before the main body; raw coordinates passed with no normalization. Optimizer workflow: 10,000 Adam steps followed by 30,000 SSBroyden steps. Loss: exactly L_pde (utt + uxx + u - (1-17π²)sin(πt)sin(4πx)) with vRBA local adaptive weighting inside the PDE term and global lam_pde=1.0. Curriculum schedule matches the requested 10k Adam → 30k SSBroyden split. No data is saved. | 1.00e-07 | 35 | 35 | 7 | 15 | **92** |
| 4 | MLP (width=30, depth=3) with Periodic Fourier Features (degree=4, L=2.0) on both inputs; only embedded features passed to MLP. Hybrid optimizer: Adam (10k iterations) followed by SSBroyden (60k iterations). Loss: exactly L_pde (Helmholtz residual with correct source term (1-17π²)sin(πt)sin(4πx)); vRBA used for weighting+sampling in Adam and sampling-only in SSBroyden. No BC loss terms. Forward problem, no data leakage. | 8.63e-09 | 35 | 35 | 7 | 15 | **92** |
| 5 | The implementation uses a 3-layer MLP (width 30) with Periodic Fourier Features (degree=4, Lx=2.0) applied independently to each coordinate; only the embedded features are passed to the MLP body. Training follows the exact Hybrid schedule: 10,000 Adam iterations followed by 120,000 SSBroyden iterations. The loss is strictly L_pde (with vRBA local adaptive weighting and importance sampling); no boundary, data, or auxiliary terms are present. The PDE residual, source term, domain, and periodic enforcement via embeddings all match the Helmholtz problem definition. Normalization is absent as required. | 1.00e-08 | 35 | 35 | 7 | 15 | **92** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves Log_files_SSBroyden.mat containing keys: 'model_name' (str), 'all_errors' (list), 'all_its' (list), 'all_loss' (list), 'all_gamma' (list), 'all_max_RBA' (list), 'all_L_infty' (list), 'all_L_2' (list). No field arrays saved. |
| 1 | Saves evaluation_fields.npz containing keys: 'x' (10000x1), 'y' (10000x1), 'u_pred' (10000x1), 'u_an' (10000x1), 'g' (10000x1). |
| 3 | No data saved; saving explicitly omitted per strategy. |
| 4 | Saves results.npz containing keys: 'x' (10000x1), 'y' (10000x1), 'u_pred' (10000x1), 'u_exact' (10000x1). |
| 5 | Saves results.npz containing keys: 'x' (10000x1), 'y' (10000x1), 'u_pred' (10000x1), 'u_exact' (10000x1). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run1/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1303/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run1/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1313/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run1/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1320/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run1/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1325/` |
| 5 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run1/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1329/` |


---
