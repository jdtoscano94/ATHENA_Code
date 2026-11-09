
# Experiment Report
*Last Updated: 2026-11-06 13:59:33*

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

**0. Triage & Advisor Instructions**
* **Advisor Directive:** The previous run stopped prematurely while the error was still decreasing. You **MUST** increase the SSBroyden iterations to at least 20,000 to ensure the loss and error curves genuinely plateau.
* **Data Role:** Validation/Error Metrics ONLY.

**1. Problem Statement**
* **Governing Equation (Helmholtz):** $u_{xx} + u_{yy} + k^2 u - g(x, y) = 0$
  * Constants: $k = 1$, $a_1 = 1$, $a_2 = 4$.
  * Forcing term $g(x,y)$ derived from the analytical solution: $g(x,y) = (1 - 17\pi^2) \sin(\pi x) \sin(4\pi y)$
* **Boundary Conditions:** Periodic in both $x$ and $y$ on $[-1, 1] \times [-1, 1]$.
* **Analytical Solution:** $u(x,y) = \sin(\pi x) \sin(4\pi y)$
* **Problem Type:** Forward.

**2. Inputs & Outputs**
* **Inputs:** $x, y$ (2D).
* **Outputs:** $u$ (Scalar).

**3. Base Model Selection**
* **Optimizer:** Hybrid Adam + SSBroyden.
* **Base MLP Specs:** `width=30`, `depth=3`.

**4. Main Architecture**
* **Structure:** Input -> Input Transformation (Embedding) -> Main Body -> Output.
* **Input Transformations:**
  * $x$ and $y$ are already in $[-1, 1]$. **DO NOT** apply any normalization.
  * Apply **Periodic Fourier Features** to both $x$ and $y$.
  * **CRITICAL:** Since raw inputs are in $[-1, 1]$, set `Lx = 2.0` and `Ly = 2.0`.
  * Set `degree = 2`.
  * **Rule:** Feed ONLY the embedded features to the Main Body. Do NOT pass raw $x$ or $y$.
* **Output Transformation:** None. The periodic embeddings guarantee exact periodicity.

**5. Constraint Implementation Strategy**
* **Count:** `L_pde` (1). `L_bc_periodic` is exactly satisfied by the architecture and MUST be excluded.
* **Total Soft Constraints:** 1.

**6. Loss Function Assembly**
* **Formula:** $L_{total} = L_{pde}$
* **Iteration Budget (MANDATORY ADVISOR FIX):**
  * **Phase 1 (Adam):** 10,000 iterations.
  * **Phase 2 (SSBroyden):** **20,000 iterations** (Increased to ensure plateau).

**7. Weighting Strategy**
* **Local Weights (`lambdas`):** Use **vRBA** for $L_{pde}$. `lambdas` are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden.
* **Global Weights (`lam`):** Only 1 loss term exists. No global balancing is needed.
* **SSBroyden Integrity:** Do NOT interfere with the internal weighting, sampling, or batching strategies of the SSBroyden optimizer.

**8. Visualization & Saving**
* Do NOT modify the legacy visualization callbacks.
* Save the predicted and analytical fields as `.npz` arrays with keys specifying the variable names.

### Required Modules
Main Model:
[MLP with Hybrid Adam and SSBrodyden]
Enhancements:
[Periodic Fourier Features]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
APPROVED / REJECTED_FOR_REFINEMENT / REJECTED_FATAL
(Wait, the prompt says "[Write EXACTLY one of these options:]" at the very top, but then "Section 1: Status Decision". I will just output the sections as requested.)

Actually, the prompt says:
[Write EXACTLY one of these options:]
* APPROVED
* REJECTED_FOR_REFINEMENT
* REJECTED_FATAL

Section 1: Status Decision
...

I will provide exactly that.

REJECTED_FOR_REFINEMENT

**Section 1: Status Decision**
* **Status:** REJECTED_FOR_REFINEMENT

**Section 2: Quantitative Scorecard**
* **Score Integrity:** 35
* **Score Accuracy:** 35
* **Score Details:** 7
* **Score Optimality:** 15
* **TOTAL SCORE:** 92

**Section 3: Forensic Analysis & Feedback**
* **Observation:** The training run successfully completes the Adam phase and transitions to SSBroyden at iteration 10,000. However, in the final stage (iterations 10,000 to ~23,000), the curves do not plateau. Evaluating the last 50% of this final stage (from ~16,500 to 23,000), the average band level of the loss curve drops from `A` $\approx 10^{-9}$ at the midpoint to `B` $\approx 10^{-11}$ at the end. The relative L2 error curve similarly drops from `A` $\approx 10^{-6}$ to `B` $\approx 10^{-7}$. Because `B` is significantly lower than `A` for both curves, the model is still actively improving and has not converged to a flat horizontal floor.
* **Diagnosis:** Budget-Limited / Premature Stop (Still Improving).
* **Targeted Improvements:**
    * **For Integrity:** None
    * **For Accuracy:** Score is excellent but the average error is still decreasing across the last 50% of the final stage (from ~$10^{-6}$ to ~$10^{-7}$, judged by the band center). Duplicate the number of iterations for the final stage (e.g., increase SSBroyden to 40,000 iterations) and reinitialize the Hessian to continue.
    * **For Details:** Ensure both loss and error curves genuinely plateau in the final segment before terminating.
    * **For Optimality:** None

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | The implementation uses a PINN with MLP (width=30, depth=3) preceded by Periodic Fourier Features (degree=2, Lx=Ly=2.0) on separate x/y inputs; embeddings are concatenated and fed exclusively to the MLP body with no raw coordinates or normalization. Training follows Hybrid Adam (with vRBA weighting/sampling) followed by SSBroyden (BFGS-based with vRBA sampling only). Loss assembly contains exactly one term: L_pde (Helmholtz residual matching the analytic g). No BC loss terms are present. No curriculum schedule is defined or used. | 1.00e-07 | 35 | 35 | 7 | 15 | **92** |
| 2 | Architecture: MLP (width=30, depth=3) with Periodic Fourier Features (degree=2, Lx=Ly=2.0) on raw x,y inputs in [-1,1]; only embedded features fed to MLP body. Optimizer workflow: Hybrid Adam (10,000 iterations with vRBA weighting/sampling) followed by SSBroyden (20,000 iterations with vRBA sampling only). Loss terms: exactly L_pde (Helmholtz residual) with vRBA; no BC, data, or ghost terms. Curriculum schedule matches the mandated 10k+20k split. Forward problem, periodic BCs satisfied by embedding, no normalization applied. | 1.00e-07 | 35 | 35 | 7 | 15 | **92** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |
| 2 | Saves results.npz containing keys: 'x' (256x256), 'y' (256x256), 'u_pred' (256x256), 'u_exact' (256x256). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run2/results//Helmholtz2D_Periodic_Sin/2026_11_06_1354/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/Helmholtz/run2/results//Helmholtz2D_Periodic_Sin/2026_11_06_1357/` |


---
