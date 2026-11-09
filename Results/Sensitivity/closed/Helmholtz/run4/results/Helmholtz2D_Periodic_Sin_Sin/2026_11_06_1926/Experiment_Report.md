
# Experiment Report
*Last Updated: 2026-11-06 19:29:52*

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

**Atomic Conclusion:** Execute 60,000 SSBroyden iterations and save outputs to `helmholtz_2d_results.npz` with keys `x`, `y`, `u_pred`, `u_exact`.

**1-5. Context & Architecture Summary**
Initial logic, 2D Helmholtz PDE residual ($u_{xx} + u_{yy} + u - g(x, y) = 0$), and periodic Fourier feature embeddings ($L_x=2.0, L_y=2.0$) are validated.

**6. Loss Function Assembly & Schedule**
$L_{total} = L_{pde}$
Phase 1 (Adam): 10,000 iterations.
Phase 2 (SSBroyden): 60,000 iterations.

**7-8. Weighting & Visualization**
Standard processing applied.

**9. Saving Data**
Instruct the Planner to save the predicted and exact fields to `helmholtz_2d_results.npz` with exact keys: `x`, `y`, `u_pred`, `u_exact`.

### Modules

Main Model:
[MLP with Hybrid Adam and SSBroyden]

Enhancements:
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
* **Observation:** The training successfully completes the Adam phase and transitions to SSBroyden at iteration 10,000. The SSBroyden phase terminates early around iteration 48,000. Evaluating the final stage (10k to 48k) by looking at the last 50% window (29k to 48k): The average band level of the loss at the midpoint (A) is ~1e-11, and at the end (B) it is ~3e-12. The relative L2 error at the midpoint (A) is ~1e-7, and at the end (B) it is ~5e-8. Both curves have essentially plateaued within a ~2x margin, oscillating around a fixed numerical floor in the final 10,000 iterations. The pointwise error is extremely low (max ~5e-8), and the predicted solution visually perfectly matches the reference.
* **Diagnosis:** Optimal Convergence. The model has fully resolved the 2D Helmholtz equation using Periodic Fourier Features, reaching the numerical precision limit of the optimizer.
* **Targeted Improvements:** 
    * **For Integrity:** None
    * **For Accuracy:** None
    * **For Details:** None
    * **For Optimality:** None

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | MLP architecture with Periodic Fourier Features (Lx=2.0, Ly=2.0) embeddings. Hybrid optimizer: Adam (10k iterations) followed by SSBroyden (exactly 15k iterations). Loss is strictly L_pde only. Global weights (lamE) frozen after Adam; lambdas active solely for importance sampling during SSBroyden. Analytical solution and PDE residual match exactly. | 1.00e-08 | 35 | 35 | 7 | 15 | **92** |
| 1 | Architecture: MLP (width=30, depth=3) with Periodic Fourier Features (degree=2, Lx=Ly=2.0) on raw coordinates in [-1,1] (no normalization), feeding only embeddings to the MLP body. Optimizer workflow: Phase 1 Adam (10,000 iterations) with vRBA for weighting+sampling, followed by Phase 2 SSBroyden (30,000 iterations) with vRBA for sampling only. Loss terms: exactly L_total = L_pde (no BC or other terms). Curriculum schedule matches the required 10k+30k split. | 1.99e-08 | 35 | 35 | 7 | 15 | **92** |
| 2 | Architecture: MLP with Periodic_Fourier_Features (Lx=Ly=2.0, embedding_degree=2) feeding a 3-layer tanh MLP (30 hidden units). Optimizer workflow: Phase 1 Adam (EPOCHS=10000) with vRBA importance sampling, followed by Phase 2 SSBroyden (Nbfgs=60000 iterations, Nchange=100). Loss function: strictly L_total = L_pde (no additional terms). Curriculum schedule exactly matches the strategy (Adam 10k -> SSBroyden 60k). PDE residual implements u_xx + u_yy + u - f with the correct Helmholtz source term. | 5.00e-08 | 35 | 35 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves results.npz containing keys: 'x' (65536x1), 'y' (65536x1), 'u_pred' (65536x1), 'u_an' (65536x1), 'abs_error' (65536x1). |
| 1 | Saves results.npz containing keys: 'x' (65536x1), 'y' (65536x1), 'u_pred' (65536x1), 'u_an' (65536x1), 'abs_error' (65536x1). |
| 2 | Saves helmholtz_2d_results.npz containing keys: 'x' (65536x1), 'y' (65536x1), 'u_pred' (65536x1), 'u_exact' (65536x1). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run6/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1919/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run6/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1922/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run6/results//Helmholtz2D_Periodic_Sin_Sin/2026_11_06_1926/` |


---
