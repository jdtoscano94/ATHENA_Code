
# Experiment Report
*Last Updated: 2026-11-06 02:57:10*

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
# SciML Strategy: 2D Helmholtz (Periodic, Manufactured Solution)

## 0. Triage
No specific user requests, no Advisor report → **Initial Proposal**.

## 1. Data Source
No external dataset. Analytical solution used for **validation only**.

## 2. Problem Statement
- **Governing Equation (Helmholtz, k=1):**
$$u_{xx} + u_{yy} + k^2 u = g(x,y), \quad k=1$$
- **PDE Residual:** $r(x,y) = u_{xx} + u_{yy} + u - g(x,y)$
- **Source term (manufactured, derived from solution):**
$$g(x,y) = (k^2 - \pi^2 a_1^2 - \pi^2 a_2^2)\sin(\pi a_1 x)\sin(\pi a_2 y) = (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$$
- **BCs:** Periodic in x and y on $[-1,1]^2$ (consistent: solution has period 2 in both).
- **Analytical Solution (EXACT, for validation):**
$$u_{an}(x,y) = \sin(\pi x)\sin(4\pi y), \quad a_1=1,\ a_2=4,\ k=1$$
- **Data Role:** Condition B — **Validation/Error Metrics ONLY.** No $L_{data}$.
- **Problem Type:** Forward.
- **Well-posedness:** Yes. Smooth strong solution; $k^2=1$ is non-resonant (Laplacian eigenvalues $\pi^2(m^2+n^2) \ge \pi^2 > 1$). Ideal for strong-form PINNs.

## 2.1 / 2.2 Input & Output
- **Input:** d=2, $(x,y)\in[-1,1]^2$. No time.
- **Output:** scalar $u(x,y)$. Smooth, oscillatory (frequency $4\pi$ in y), periodic.

## 3. Base Model Selection
2D problem → **Case A: MLP with Hybrid Adam and SSBroyden**. Base MLP: `width=30, depth=3`.

## 4. Main Architecture
**Pipeline: Input → Input Transformation → Main Body → Output Transformation → Output.** All transformations differentiable, inside the model's call function.

**Input Transformation:**
- **Normalization: NONE.** x,y are already in $[-1,1]$ (nice range) — pass raw coordinates, identity. FORBID extra scalers.
- **Periodic Fourier Features (both x and y):** degree=5, $L_x = L_y = 2.0$ (range of coordinates as they enter the embedding, i.e., raw $[-1,1]$).
- **Justification (physics-driven):** solution contains harmonics $k=1$ (x) and $k=4$ (y); degree must be ≥ 4. Degree=5 gives margin. (< 20 constraint satisfied.)
- **Input composition:** MLP body receives **ONLY** the 20 embedding features. Do NOT pass raw x,y to the body.

**Main Body:** MLP, width=30, depth=3 (template default activation).

**Output Transformation:** Identity. No Dirichlet BCs exist; periodicity is exact by construction (embedding). No ansatz needed.

## 5. Constraint Implementation
- Count: $L_{pde}$ only (periodic BC satisfied architecturally — **exclude** $L_{bc}$; no IC; no data).
- Count = 1 < 5 → **Loss-based, no hard constraints.**

## 6. Loss Function Assembly
$$L_{total} = L_{pde}$$

## 6.1 Continuation
None. k=1 fixed, no Reynolds/Peclet. **No continuation schedule.**

**Phases:** Adam (≥10,000 iters, static) → SSBroyden (convergence). No parameter changes.

## 7. Weighting Strategy
- **Total terms = 1** → No global balancing needed.
- **Local weights:** vRBA on PDE residual. State: "**`lambdas` are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden.**"
- **SSBroyden Integrity:** Do NOT modify internal weighting/sampling/batching of the SSBroyden template.

## 8. Visualization
Do NOT modify template visualization callbacks. Ensure analytical field $u_{an}$ is supplied for error/reference plots.

## 9. Saving Data
Save prediction grid and reference as `.npz` with keys: `u_pred`, `u_analytical`, `X`, `Y`.

---

## Librarian Retrieval List
**Main Model:**
- MLP with Hybrid Adam and SSBroyden

**Enhancements:**
- Periodic Fourier Features (degree=5, applied to both x and y)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
REJECTED_FOR_REFINEMENT

## Section 1: Status Decision
* **Status:** **REJECTED_FOR_REFINEMENT**
* **Problem type:** **Type A — forward problem**, with no observational noise. The analytical solution is used only for validation.
* **Decision rationale:** The run is stable, visually readable, and highly accurate, and it solves the requested \(k=1\), \(a_1=1\), \(a_2=4\) problem. However, the final-stage loss and error bands are still decreasing. The run therefore fails the mandatory plateau gate despite reaching a total score of 90.

## Section 2: Quantitative Scorecard
* **Score Integrity:** **35/35**
* **Score Accuracy:** **35/35**
  * Numerical Precision: **20/20**
  * Consistency and Strategy: **15/15**
* **Score Details:** **6/15**
* **Score Optimality:** **14/15**
* **TOTAL SCORE:** **90/100**

No error tolerance was specified by the user. Even using a stringent scoring benchmark of \(\epsilon_{\text{target}}=10^{-3}\), the final plotted relative error of order \(10^{-6}\) easily qualifies for full precision credit.

## Section 3: Forensic Analysis & Feedback

### Observation

**Loss and error histories**

- The plots are readable and all displayed values are finite. There is no explosion, NaN, or divergence.
- Two training stages are visible, with the optimizer transition at approximately iteration \(10{,}000\).
- **Total Loss:** decreases from an initial value around \(10^2\) to roughly \(10^{-2}\) during the first stage. Following a brief transition jump, the second stage drops rapidly and then continues descending smoothly, ending near \(10^{-8}\)–\(10^{-9}\).
- **Relative \(L^2\) Error:** begins near \(10^0\), spends much of the first stage around a few \(10^{-3}\), then drops sharply after the transition. The final stage is noisy, with spikes around \(10^{-5}\) and dips approaching \(5\times10^{-7}\).

The visible final stage runs approximately from iteration \(10{,}000\) to \(17{,}000\)–\(17{,}500\), placing its midpoint near \(13{,}500\)–\(13{,}750\). Reading the oscillation-band centers rather than isolated dips:

| Curve | \(A\): final-stage midpoint | \(B\): final-stage end | Conclusion |
|---|---:|---:|---|
| Loss | \(\approx 10^{-6}\) | \(\approx 10^{-8}\) | Clearly still decreasing |
| Relative \(L^2\) error | \(\approx 3\times10^{-6}\) | \(\approx 10^{-6}\) | Noisy, but still drifting downward |

Thus, neither curve provides a genuine horizontal final plateau. In particular, the roughly two-order-of-magnitude loss reduction after the midpoint independently triggers the **loss gate**: the model was still learning when the run ended.

**Final-error extraction**

- The right edge of the error plot is of order \(10^{-6}\). For scoring, I conservatively use:
  \[
  \mathrm{RL2}_{\mathrm{final}}\approx 1.0\times10^{-6}.
  \]
- The exact best-ever value at the bottom of the log is:
  \[
  \boxed{\text{Minimum achieved RL2 error}=5.491462051369757\times10^{-7}}.
  \]
- The earlier `RL2 error: 0.00435141` entries belong to the Adam warm-up reporting and are not the final second-order-stage result. The logged minimum is consistent with the lowest plotted dips, but it is not evidence that the final oscillation band has flattened.

**Residual and solution fields**

- The **PDE residual heatmap** is mottled and broadly distributed across the domain, without a localized shock front, corner singularity, or coherent boundary failure. The log reports finite residuals ranging from \(2.3857\times10^{-9}\) to \(2.2808\times10^{-4}\).
- **Reference and prediction** are visually nearly indistinguishable: they share the same oscillatory stripe pattern, phase, and amplitude.
- The **pointwise error** is spatially structured, with broad red/orange higher-error regions and lower-error filamentary bands, but no single dominant localized defect.
- Periodic boundaries are enforced architecturally by the Fourier embedding. The implementation uses the correct manufactured source term, fixed \(k=1\), and no data-loss term, so there is no validation-data leakage.

### Diagnosis
**Budget-Limited / Premature Stop.** This is not a physics, stability, or capacity failure. The model has already reached excellent accuracy, but the final-stage loss and error average bands have not flattened. Early termination or a best-ever error dip cannot substitute for visible convergence.

### Targeted Improvements

* **For Integrity:** None. The execution, plots, and numerical values are valid.

* **For Accuracy:** **Increase the final SSBroyden/second-order stage by at least \(2\times\), preferably \(2\)–\(5\times\), so both curves can reach a genuine floor.** If continuing through a restart, reinitialize the Hessian and resume from the current weights. Do not change the PDE, source term, Fourier degree, or network capacity; the existing setup is working.

* **For Details:** Accept the next run only when both the loss and error band centers are approximately horizontal over the last 50% of the final stage. Do not use the single lowest RL2 dip or optimizer termination as plateau evidence.

* **For Optimality:** No architectural change is presently warranted. Retain the pipeline  
  `periodic Fourier features → MLP → identity output`, together with vRBA and SSBroyden fine-tuning. Entropy regularization is not applicable to this smooth Helmholtz problem, and KAN/RAR modifications would distract from the actual bottleneck: insufficient final-stage training.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The code implements a forward 2D Helmholtz PINN. Architecture: an MLP body with 3 hidden layers of width 30 and tanh activation; both input coordinates pass through Periodic_Fourier_Features with degree=5 and Lx=2.0, producing 20 embedding features that are concatenated before the MLP. No raw coordinates are passed to the body, no normalization is applied, and the output transformation is identity. The PDE residual is u_tt + u_xx + u - g with g=(1-17π²) sin(π t) sin(4π x), matching the manufactured solution sin(π x) sin(4π y) under the coordinate mapping used in the code. The loss contains only L_pde; no L_bc, L_ic, or L_data is used. Training uses Adam for 10,000 epochs with vRBA weighting and importance sampling, followed by SSBroyden/BFGS-style second-order optimization for 12,500 iterations in 100-step chunks with resampling. vRBA is used for weighting+sampling during Adam and for sampling only during SSBroyden. There is no PDE parameter continuation; k=1 is fixed. The analytical solution is used only for validation/error metrics. | 1.00e-06 | 35 | 35 | 6 | 14 | **90** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves 'Helmholtz_predictions.npz' in NumPy .npz format. The npz contains keys: 'u_pred' with shape (100, 100) float64, 'u_analytical' with shape (100, 100) float64, 'X' with shape (100, 100) float64 representing the first coordinate grid, and 'Y' with shape (100, 100) float64 representing the second coordinate grid. It also saves 'Log_files_SSBroyden.mat' with training log arrays. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run1//Helmholtz2D_Periodic_SinPiX_Sin4PiY/2026_11_06_0252/` |


---
