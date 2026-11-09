
# Experiment Report
*Last Updated: 2026-11-06 13:44:48*

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
Role: Validation/Error Metrics ONLY. 

**2. Problem Statement**
- **Governing Equation:** $u_{xx} + u_{yy} + u = (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$
- **Domain:** $(x,y) \in [-1, 1] \times [-1, 1]$
- **Boundary Conditions:** Periodic in $x$ and $y$.
- **Analytical Solution:** $u(x,y) = \sin(\pi x)\sin(4\pi y)$
- **Problem Type:** Forward

**3. Base Model Selection**
- MLP with Hybrid Adam and SSBroyden.

**4. Main Architecture**
- **Input Transformations:** $x$ and $y$ are already in $[-1, 1]$. **DO NOT** normalize them. Apply **Periodic Fourier Features** to both $x$ and $y$. Set `Lx = 2.0` for both and `degree = 4`. **CRITICAL:** Pass ONLY the embedded features to the main body. Do not pass the raw $x$ and $y$ coordinates.
- **Main Body:** Base MLP Specs: `width=30`, `depth=3`.
- **Output Transformation:** None.

**5. Constraint Implementation Strategy**
- Periodic BCs are strictly enforced by the Periodic Fourier Features. Do NOT include any boundary loss terms.

**6. Loss Function Assembly**
- $L_{total} = L_{pde}$

**7. Optimization & Weighting Strategy**
- **Phase 1 (Adam):** 10,000 iterations.
- **Phase 2 (SSBroyden):** **240,000 iterations**.
- **Weighting (CRITICAL):** **Use vRBA (`lambdas`) for weighting and sampling during Adam, and for sampling ONLY during SSBroyden.** 
- **Global Weights:** Since there is only one loss term ($L_{pde}$), global balancing (`lam`) is not required. If any global `lam` scalars are present, they MUST be frozen before entering SSBroyden.

**8. Visualization Strategy**
- Standard processing applied. Do not modify the visualization code.

**9. Saving Data**
- Save the fields as `.npz` arrays with keys specifying the variable names for visualization.

**Main Model:**
[MLP with Hybrid Adam and SSBrodyden]

**Enhancements:**
[Periodic Fourier Features]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
APPROVED

**Section 1: Status Decision**
* **Status:** APPROVED

**Section 2: Quantitative Scorecard**
* **Score Integrity:** 35
* **Score Accuracy:** 35
* **Score Details:** 15
* **Score Optimality:** 15
* **TOTAL SCORE:** 100

**Section 3: Forensic Analysis & Feedback**
* **Observation:** 
  1. **Loss & Error History:** The final training stage (SSBroyden) spans from iteration 10,000 to 150,000. Evaluating the last 50% of this stage (iterations 80,000 to 150,000):
     - For the Loss curve, the average band level at the midpoint (iteration 80,000) is `A` $\approx 10^{-13}$, and at the end (iteration 150,000) is `B` $\approx 10^{-13}$.
     - For the Error curve, the average band level at the midpoint is `A` $\approx 10^{-8}$, and at the end is `B` $\approx 10^{-8}$.
     Both curves are genuinely flat, indicating full convergence.
  2. **Residual/Pointwise Error:** The PDE residual is uniformly distributed with a maximum around $10^{-6}$. The pointwise error is extremely low, peaking around $10^{-9}$.
  3. **Solution Fields:** The predicted solution visually matches the reference analytical solution perfectly.
* **Diagnosis:** Optimal Convergence. The model has successfully learned the underlying physics with high precision. The use of Periodic Fourier Features perfectly enforces the boundary conditions, and the hybrid optimizer with vRBA has driven the loss to its noise floor.

**Targeted Improvements:**
* **For Integrity:** None
* **For Accuracy:** None
* **For Details:** None
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
| 6 | MLP architecture with Periodic Fourier Features (degree=4, Lx=2.0) applied independently to x and y; only embedded features are concatenated and passed to a 3-layer width-30 MLP. Hybrid optimizer: Adam (10k iterations with vRBA weighting + importance sampling) followed by SSBroyden (240k iterations with vRBA sampling only). Loss is strictly L_pde (no boundary or other terms). vRBA lambdas used exactly as specified for weighting/sampling phases. | 1.00e-09 | 35 | 35 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves Log_files_SSBroyden.mat containing keys: 'model_name' (str), 'all_errors' (list), 'all_its' (list), 'all_loss' (list), 'all_gamma' (list), 'all_max_RBA' (list), 'all_L_infty' (list), 'all_L_2' (list). No field arrays saved. |
| 1 | Saves evaluation_fields.npz containing keys: 'x' (10000x1), 'y' (10000x1), 'u_pred' (10000x1), 'u_an' (10000x1), 'g' (10000x1). |
| 3 | No data saved; saving explicitly omitted per strategy. |
| 4 | Saves results.npz containing keys: 'x' (10000x1), 'y' (10000x1), 'u_pred' (10000x1), 'u_exact' (10000x1). |
| 5 | Saves results.npz containing keys: 'x' (10000x1), 'y' (10000x1), 'u_pred' (10000x1), 'u_exact' (10000x1). |
| 6 | Saves results.npz containing keys: 'x' (10000x1), 'y' (10000x1), 'u_pred' (10000x1), 'u_exact' (10000x1). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run1/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1303/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run1/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1313/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run1/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1320/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run1/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1325/` |
| 5 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run1/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1329/` |
| 6 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run1/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1339/` |


---
