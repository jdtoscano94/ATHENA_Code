
# Experiment Report
*Last Updated: 2026-11-06 14:52:13*

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
*   **Governing Equation (2D Helmholtz):** $u_{xx} + u_{yy} + k^2 u = g(x, y)$
*   **Parameters:** $a_1 = 1$, $a_2 = 4$, $k = 1$.
*   **Analytical Solution:** $u(x,y) = \sin(\pi x) \sin(4 \pi y)$
*   **Source Term:** $g(x,y) = (1 - 17\pi^2) \sin(\pi x) \sin(4 \pi y)$
*   **PDE Residual:** $u_{xx} + u_{yy} + u - (1 - 17\pi^2) \sin(\pi x) \sin(4 \pi y) = 0$
*   **Boundary Conditions:** Periodic in both $x$ and $y$ on $[-1, 1] \times [-1, 1]$.
*   **Problem Type:** Forward. Well-posed for PINNs.

**2. Inputs & Outputs**
*   **Input:** 2D ($x, y$).
*   **Output:** Scalar $u(x,y)$.

**3. Base Model Selection**
*   **Optimizer:** Hybrid Adam + SSBroyden.
*   **Base MLP Specs:** `width=30`, `depth=3`.

**4. Main Architecture**
*   **Structure:** Input -> Periodic Fourier Features -> Main Body -> Output.
*   **Input Transformations:**
    *   $x$ and $y$ are already in $[-1, 1]$. **DO NOT** normalize them. Pass them unchanged to the embedding.
    *   **Periodic Fourier Features:** Use `degree=2`.
    *   **CRITICAL `Lx` Derivation:** Since $x, y \in [-1, 1]$ enter the embedding directly, the range is $1 - (-1) = 2.0$. Set `Lx = 2.0` and `Ly = 2.0`.
    *   **Input Composition Rule:** Pass **ONLY** the embeddings to the Main Body. Do not pass the raw $x$ or $y$ coordinates.
*   **Output Transformation:** None. Periodicity is exactly satisfied by the embeddings.

**5. Constraint Implementation Strategy**
*   **Periodic BCs:** Enforced architecturally via Periodic Fourier Features. Do not add a loss term for them.
*   **Count of soft loss terms:** 1 ($L_{pde}$).
*   **Strategy:** Loss-Based (only PDE).

**6. Loss Function Assembly**
*   **Symbolic Equation:** $L_{total} = L_{pde}$
*   **Iteration Budget (MANDATORY ADVISOR FIX):**
    *   **Phase 1 (Adam):** 10,000 iterations.
    *   **Phase 2 (SSBroyden):** **480,000 iterations** (Increased per Advisor to ensure the loss plateaus). Reinitialize the Hessian at the start of this phase.

**7. Weighting Strategy**
*   **Local Weights (`lambdas`):** Use **vRBA** (Variational Residual-Based Attention) inside the PDE loss. `lambdas` are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden.
*   **Global Weights:** Only 1 term ($L_{pde}$), so no global balancing is needed. If any global `lam` weights are used, they must be frozen before SSBroyden.
*   **SSBroyden Integrity Rule:** Do not interfere with the internal weighting, sampling, or batching strategies of the SSBroyden optimizer.

**8. Visualization Strategy**
*   Do not modify, delete, or "improve" the visualization code from the template.

**9. Saving Data**
*   Save the fields as a `.npz` file named `helmholtz_2d_results.npz`. The dictionary must explicitly contain the keys: `x`, `y`, `u_pred`, `u_an`, and `abs_error` mapped to their respective evaluated arrays.

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
* **Score Details:** 5
* **Score Optimality:** 15
* **TOTAL SCORE:** 90

**Section 3: Forensic Analysis & Feedback**
* **Observation:** 
  1. **Loss & Error History:** The loss y-axis ranges from $10^2$ down to $10^{-4}$. The curve shows an initial sharp drop (Adam phase) followed by a long, noisy descent (SSBroyden phase). Identifying the final stage (SSBroyden, from ~10,000 to ~340,000 iterations), the 50% midpoint is around iteration 175,000. 
  For the Loss curve: `A` = average band level at the 50% midpoint = $1 \times 10^{-4}$; `B` = average band level at the END = $6 \times 10^{-5}$. Because `B` is lower than `A`, the loss curve is STILL DECREASING ON AVERAGE. 
  For the Error curve: `A` = average band level at the 50% midpoint = $1 \times 10^{-3}$; `B` = average band level at the END = $8 \times 10^{-4}$. The error curve is also still decreasing.
  2. **Residual/Pointwise Error:** The PDE residual heatmap shows a noisy, distributed error topology with a maximum around $10^{-2}$. The pointwise error is highest near the boundaries and corners, reaching $\sim 10^{-3}$.
  3. **Solution Fields:** The predicted solution visually matches the reference analytical solution perfectly, capturing the high-frequency periodic features.

* **Diagnosis:** Budget-Limited / Premature Stop (Still Improving). Score is excellent but the average error is still decreasing across the last 50% of the final stage (from ~1e-4 to ~6e-5, judged by the band center). The model ran out of iterations (or hit a loose early-stopping tolerance) before reaching its true mathematical floor.

* **Targeted Improvements:**
    * **For Accuracy:** Increase the number of iterations of the LAST training stage (e.g., duplicate the SSBroyden iterations from 480,000 to 960,000) so both the loss and error curves flatten before the run ends.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The implementation uses a PINN with Periodic Fourier Features (degree=2, Lx=Ly=2.0) on raw coordinates in [-1,1] followed by an MLP (width=30, depth=3). Training follows the Hybrid Adam + SSBroyden workflow with vRBA weighting/sampling applied exclusively inside the single PDE loss term. The PDE residual exactly matches the required Helmholtz form with the analytic source term; no BC loss terms are present. Output transformation is absent as specified. | 1.20e-07 | 35 | 35 | 5 | 15 | **90** |
| 1 | The implementation uses a 3-layer MLP (width 30) with Periodic Fourier Features (degree=2, Lx=Ly=2.0) as input embeddings and no output transformation or coordinate normalization. Training follows the Hybrid Adam (10,000 iterations) + SSBroyden (30,000 iterations) workflow. The loss is strictly L_pde (Helmholtz residual) with vRBA weighting/sampling; no BC or auxiliary loss terms are present. The analytical solution, source term, and periodic BC enforcement via embeddings all match the strategy exactly. | 1.00e-07 | 35 | 35 | 5 | 15 | **90** |
| 2 | The implementation uses a PINN with Periodic Fourier Features (degree=2, Lx=Ly=2.0) feeding a 3-layer MLP (width=30) that receives only embeddings. Training follows the mandated Hybrid schedule: Adam phase for 10,000 iterations followed by SSBroyden phase for exactly 60,000 iterations. Loss assembly contains only L_pde (Helmholtz residual with correct source term); periodic BCs are enforced exactly by the embeddings so no L_bc term appears. vRBA weighting/sampling is applied inside L_pde for both phases. No normalization is performed on inputs. Analytical solution and source term match the strategy exactly. | 5.00e-08 | 35 | 35 | 5 | 15 | **90** |
| 3 | Architecture: MLP (width=30, depth=3) with Periodic Fourier Features (degree=2, Lx=Ly=2.0) on inputs x,y in [-1,1] (no normalization, only embeddings passed to MLP body). Optimizer workflow: Phase 1 Adam (10k iterations) with vRBA pointwise weighting + sampling inside L_pde; Phase 2 SSBroyden (120k iterations) with Hessian reinitialized each outer loop and vRBA used for sampling only. Loss terms: exactly L_total = L_pde (Helmholtz residual with correct source term (1-17*pi^2)sin(pi x)sin(4 pi y)); no BC, data, or gauge terms. Curriculum schedule matches the mandated 10k/120k split. | 0.10% | 35 | 20 | 5 | 15 | **75** |
| 4 | The implementation uses a PINN with Periodic Fourier Features (degree=2, Lx=Ly=2.0) feeding only embeddings into an MLP (width=30, depth=3). Training follows the exact Hybrid schedule: Adam phase for 10,000 iterations using vRBA-weighted PDE loss, followed by SSBroyden phase for 240,000 iterations with Hessian reinitialization at the start of each outer loop and vRBA used for sampling only. Loss assembly is strictly L_total = L_pde with no additional terms or data losses. Inputs remain unnormalized in [-1,1]. | 0.20% | 35 | 20 | 5 | 15 | **75** |
| 5 | The implementation uses a PINN with Periodic Fourier Features (degree=2, Lx=Ly=2.0) feeding only embeddings into a 3-layer MLP (width=30). Training follows the exact Hybrid schedule: 10,000 Adam iterations with vRBA weighting/sampling, followed by 480,000 SSBroyden iterations (Hessian reinitialized each outer step) using vRBA for sampling only. Loss assembly is strictly L_total = L_pde with the correct Helmholtz residual (uxx + uyy + u - (1-17π²)sin(πx)sin(4πy)). Periodic BCs are enforced architecturally with no additional loss terms. Analytical solution and source term match the report exactly. | 0.08% | 35 | 35 | 5 | 15 | **90** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 1 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 2 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 3 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 4 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 5 | Saves helmholtz_2d_results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_an' (256x256), 'abs_error' (256x256). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1414/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1417/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1421/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1426/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1431/` |
| 5 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run3/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1440/` |


---
