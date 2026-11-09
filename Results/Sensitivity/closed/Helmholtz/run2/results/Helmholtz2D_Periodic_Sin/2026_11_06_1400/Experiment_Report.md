
# Experiment Report
*Last Updated: 2026-11-06 14:03:47*

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

**1. Problem Statement**
*   **Governing Equation:** 2D Helmholtz equation. The residual is: $L_{pde} = u_{xx} + u_{yy} + u - g(x, y) = 0$ (since $k=1$).
*   **Forcing Term:** $g(x, y) = (1 - 17\pi^2) \sin(\pi x) \sin(4\pi y)$.
*   **Boundary Conditions:** Periodic in both $x$ and $y$ on $[-1, 1] \times [-1, 1]$.
*   **Analytical Solution:** $u_{an}(x,y) = \sin(\pi x) \sin(4\pi y)$.
*   **Problem Type:** Forward.

**2. Input & Output**
*   **Input:** 2D ($x, y$).
*   **Output:** Scalar $u(x,y)$.

**3. Base Model Selection**
*   Hybrid Adam + SSBroyden.

**4. Main Architecture**
Instruct the Planner to strictly follow: Input -> Input Transformation -> Main Body -> Output Transformation -> Output.

*   **Input Transformations:**
    *   **Normalization:** $x$ and $y$ are already in $[-1, 1]$. **DO NOT** apply any normalization. Pass them unchanged to the embedding.
    *   **Embedding:** Use **Periodic Fourier Features** for both $x$ and $y$.
    *   **Lx / Ly Derivation:** Since $x, y \in [-1, 1]$ enter the embedding directly, `Lx = 2.0` and `Ly = 2.0`.
    *   **Degree:** `degree = 2`.
    *   **CRITICAL:** Feed **ONLY** the periodic embeddings to the Main Body. Do not concatenate or pass the raw $x, y$ coordinates.
*   **Main Body:**
    *   Base MLP: `width=30`, `depth=3`.
*   **Output Transformation:**
    *   None. $u_{pred}(x,y) = NN(x,y)$. Periodicity is guaranteed by the embeddings.

**5. Constraint Implementation Strategy**
*   **Count:** $L_{pde}$ (1). $L_{bc\_periodic}$ is **REMOVED** because Periodic Fourier Features inherently enforce exact periodicity.
*   **Strategy:** Loss-Based (only PDE loss is needed).

**6. Loss Function Assembly**
*   $L_{total} = L_{pde}$
*   **Iteration Budget (CRITICAL - ADVISOR MANDATED):**
    *   **Phase 1 (Adam):** 10,000 iterations.
    *   **Phase 2 (SSBroyden):** **40,000 iterations**. (The Advisor explicitly diagnosed a premature stop and mandated duplicating the final stage iterations to ensure the loss and error curves genuinely plateau).

**7. Weighting Strategy**
*   **Local Weights (`lambdas`):** Use **vRBA** for the PDE residual. `lambdas` are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden.
*   **Global Weights (`lam`):** Since there is only 1 loss term ($L_{pde}$), no global balancing is needed.
*   **SSBroyden Integrity:** Do not interfere with the internal weighting, sampling, or batching strategies of the SSBroyden optimizer.

**8. Visualization & Saving**
*   Do not modify, delete, or "improve" the visualization code in the template.
*   Save the predicted and analytical fields as `.npz` arrays with keys specifying the variable names.

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
* **Observation:** 
  1. **Loss & Error History (Top plots):** The training history displays two distinct stages: an initial Adam phase (0-10k) followed by an SSBroyden phase (10k-~38k). 
     * **FINAL REGION (10,000 to ~38,000):** The midpoint of this final stage is at ~24,000, and the end is at ~38,000.
     * **Loss curve:** `A` = average band level at the 50% midpoint = 1e-11. `B` = average band level at the END = 1e-12. Because `B` is lower than `A` (a 10x drop), the loss curve is **STILL DECREASING ON AVERAGE**.
     * **Error curve:** `A` = average band level at the 50% midpoint = 1e-7. `B` = average band level at the END = 1e-7. The error curve is **FLAT**.
  2. **Residual/Pointwise Error (Heatmap):** Both the PDE residual and the pointwise error are uniformly distributed across the periodic domain, lacking any problematic localized artifacts.
  3. **Solution Fields (Reference vs Prediction):** The predicted solution is visually identical to the reference analytical solution, successfully capturing the high-frequency periodic features.

* **Diagnosis:** Budget-Limited / Premature Stop. Score is excellent but the average loss is still decreasing across the last 50% of the final stage (from ~1e-11 to ~1e-12, judged by the band center). Duplicate the number of iterations for the final stage.

* **Targeted Improvements:**
    * **For Integrity:** None
    * **For Accuracy:** Increase the number of iterations of the LAST training stage (e.g. 2-5x) so both the loss and error curves flatten before the run ends. Raise the final SSBroyden stage from 40,000 to 80,000 iterations.
    * **For Details:** Ensure the loss curve genuinely plateaus in the final segment before terminating.
    * **For Optimality:** None

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | The implementation uses a PINN with MLP (width=30, depth=3) preceded by Periodic Fourier Features (degree=2, Lx=Ly=2.0) on separate x/y inputs; embeddings are concatenated and fed exclusively to the MLP body with no raw coordinates or normalization. Training follows Hybrid Adam (with vRBA weighting/sampling) followed by SSBroyden (BFGS-based with vRBA sampling only). Loss assembly contains exactly one term: L_pde (Helmholtz residual matching the analytic g). No BC loss terms are present. No curriculum schedule is defined or used. | 1.00e-07 | 35 | 35 | 7 | 15 | **92** |
| 2 | Architecture: MLP (width=30, depth=3) with Periodic Fourier Features (degree=2, Lx=Ly=2.0) on raw x,y inputs in [-1,1]; only embedded features fed to MLP body. Optimizer workflow: Hybrid Adam (10,000 iterations with vRBA weighting/sampling) followed by SSBroyden (20,000 iterations with vRBA sampling only). Loss terms: exactly L_pde (Helmholtz residual) with vRBA; no BC, data, or ghost terms. Curriculum schedule matches the mandated 10k+20k split. Forward problem, periodic BCs satisfied by embedding, no normalization applied. | 1.00e-07 | 35 | 35 | 7 | 15 | **92** |
| 3 | The implementation uses a PINN with Periodic Fourier Features (degree=2, Lx=Ly=2.0) on raw coordinates in [-1,1] (no normalization) feeding only embeddings into a 3-layer MLP of width 30. Training follows Hybrid Adam (10,000 iterations) + SSBroyden (40,000 iterations) with vRBA for PDE residual weighting and importance sampling. Loss is strictly L_total = L_pde (Helmholtz residual with correct forcing term g(x,y)); no boundary or data losses are present. The analytical solution sin(πx)sin(4πy) is used only for error metrics. | 1.00e-07 | 35 | 35 | 7 | 15 | **92** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 2 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 3 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run2/results//Helmholtz2D_Periodic_Sin/2026_11_06_1354/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run2/results//Helmholtz2D_Periodic_Sin/2026_11_06_1357/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run2/results//Helmholtz2D_Periodic_Sin/2026_11_06_1400/` |


---
