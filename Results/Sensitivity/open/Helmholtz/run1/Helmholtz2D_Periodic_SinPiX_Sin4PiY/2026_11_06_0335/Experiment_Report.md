
# Experiment Report
*Last Updated: 2026-11-06 03:41:56*

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
## Strategy: Exploration Run 4/5 — Controlled Single-Knob Variation

**Triage:** Advisor rejection is **protocol-only** (run 3/5 complete; runs 4/5 and 5/5 required before final selection). Run 3 (score 96, final RL2 = 1.42e-08) is the leading candidate. **Mandatory: preserve the verified pipeline exactly; vary ONE knob only.**

**THE KNOB (Run 4):** Periodic Fourier embedding **degree 6 → 7**. Rationale: degree 5→6 was the only knob that demonstrably improved final RL2 (3.51e-08 → 1.42e-08). Everything else identical to Run 3. (Run 5/5 will use a different single knob, e.g., width; not this plan.)

### 1. Data Source
- No new dataset. No changes to data loading.
- **Data Role: Validation/Error Metrics ONLY** (forward problem; analytical solution is reference only — no L_data, no leakage).

### 2. Problem Statement
- **Governing Equation (2D Helmholtz, k=1):**
$$u_{xx} + u_{yy} + k^2 u = g(x,y), \quad k=1$$
- **Analytical (manufactured) solution:** $u_{an} = \sin(\pi a_1 x)\sin(\pi a_2 y)$, $a_1=1$, $a_2=4$:
$$u_{an}(x,y) = \sin(\pi x)\sin(4\pi y)$$
- **Source term (derived):** $g = (k^2 - \pi^2(a_1^2+a_2^2))\sin(\pi x)\sin(4\pi y) = (1-17\pi^2)\sin(\pi x)\sin(4\pi y)$
- **PDE Residual:**
$$r(x,y) = u_{xx} + u_{yy} + u - (1-17\pi^2)\sin(\pi x)\sin(4\pi y)$$
(If template names inputs $(t,x)$: both are **spatial** — residual is $u_{tt}+u_{xx}+u-g$; no time derivative.)
- **BCs:** **Periodic** in both $x$ and $y$ on $[-1,1]^2$ (solution is exactly 2-periodic in both). No IC (steady problem).
- **Problem Type:** Forward. **Well-posed for PINNs:** smooth strong solution; $1-\pi^2(m^2+n^2)\neq 0$ for all integer modes → no resonance/null-space.

### 2.1/2.2 Input & Output
- Input: d=2, $(x,y)\in[-1,1]^2$. Output: scalar $u(x,y)$ — smooth, oscillatory ($4\pi$ in $y$), periodic.

### 3. Base Model
- 2D forward → **MLP with Hybrid Adam and SSBroyden** (Case A: width=30, depth=3).

### 4. Main Architecture
Pipeline (all differentiable, inside the model call):
**Input (x,y) → Periodic Fourier Embedding (both coords) → MLP body → identity → u**
- **Normalization:** NONE — both coordinates already in $[-1,1]$ (nice range); pass raw coordinates unchanged. Any normalization must live inside the call function (here: identity).
- **Input Transformation:** Periodic Fourier Features on **BOTH** coordinates: **degree=7 (THE KNOB)**, **Lx=2.0** each (raw range $[-1,1]$ → $L_x = 1-(-1) = 2.0$). Produces 14 features/coordinate → 28 concatenated.
- **CRITICAL:** NO raw coordinates to the MLP body — body receives ONLY the 28 embedding features.
- **Main Body:** MLP, width=30, depth=3, tanh.
- **Output Transformation:** identity (no Dirichlet constraints; periodicity is architectural).

### 5. Constraint Strategy
- Periodic BCs: enforced exactly by the embedding → no BC loss term.
- Soft loss count: $L_{pde}$ only = 1 < 5 → loss-based; no hard constraints needed.

### 6. Loss Assembly
$$L_{total} = L_{pde}$$
(No $L_{bc}$ — redundant with periodic embedding; no $L_{ic}$; no $L_{data}$.)

### 6.1 Continuation
- None. $k=1$ fixed; no Re/Péclet.

### 7. Weighting
- **Local:** vRBA. "**lambdas** are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden."
- **Global:** 1 term → no balancing needed.
- **SSBroyden Integrity:** Do NOT modify internal weighting/sampling/batching of the SSBroyden template.

### Training Workflow (preserve Run 3 exactly)
1. **Adam:** 10,000 iterations, vRBA weighting + sampling.
2. **Transition:** reinitialize Hessian to identity.
3. **SSBroyden:** 100,000 iterations, Nchange=100, vRBA sampling only, unweighted PDE loss.
4. **Plateau test (retain):** dual-curve (loss + RL2) band centers over trailing 50% of the SSBroyden stage, early/late halves, change factors ≤ 2; extend beyond budget if not flat.
5. Report **final-iteration RL2** and **best-ever RL2 separately** (best-ever is not convergence evidence).

### 8. Visualization
- Do NOT modify/delete template visualization callbacks.

### 9. Saving
- Save fields as .npz arrays with keys naming the variables (u_pred, u_analytical, pointwise error, residual) for visualization.

---
Main Model: [MLP with Hybrid Adam and SSBrodyden]
Enhancements: [Periodic Fourier Features (degree=7, Lx=2.0)]

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
REJECTED_FOR_REFINEMENT

**Section 1: Status Decision**
* **Status:** `REJECTED_FOR_REFINEMENT`
* **Reason:** **Protocol-only rejection.** The run is numerically successful, stable, and passes the dual-curve plateau check. However, Run 0 was the first scoring ≥90, and Runs 1–4 represent only **four** subsequent exploration attempts. The required fifth exploration run remains.

**Section 2: Quantitative Scorecard**
* **Score Integrity:** 35
* **Score Accuracy:** 35
* **Score Details:** 12
* **Score Optimality:** 14
* **TOTAL SCORE:** **96**

**Section 3: Forensic Analysis & Feedback**
* **Observation:**
  * This is a **Type A forward, noise-free problem**. The manufactured solution is used only for validation/RL2 metrics; there is no data-loss leakage. The implemented source term, \(g=(1-17\pi^2)\sin(\pi t)\sin(4\pi x)\), is consistent with the requested Helmholtz problem under the legacy coordinate naming.
  * **Loss history:** The y-axis spans approximately \(10^0\) to below \(10^{-12}\). Loss decreases rapidly during Adam, shows the expected optimizer transition near iteration 10,000, then decreases further during SSBroyden before entering a noisy floor around \(10^{-13}\).
  * **RL2 history:** The y-axis spans approximately \(10^0\) to below \(10^{-8}\). Error falls sharply through roughly iterations 20,000–40,000 and subsequently oscillates around an approximately \(10^{-8}\) floor, with isolated downward dips and upward spikes.
  * **Final-stage dual-curve test:** The final SSBroyden stage runs approximately from iteration 10,000 to 110,000, with midpoint near 60,000:
    * **Loss:** \(A\approx1.05\times10^{-13}\) at the midpoint region; \(B\approx7.28\times10^{-14}\) at the end. Ratio \(A/B\approx1.44\).
    * **RL2:** \(A\approx1.42\times10^{-8}\) at the midpoint region; \(B\approx1.13\times10^{-8}\) at the end. Ratio \(A/B\approx1.25\).
    * Both changes are within the prescribed factor-of-two band criterion. These are **noisy but genuinely horizontal floors**, not a still-descending final stage. The log’s early/late trailing-window diagnostics confirm these readings.
  * **Final versus best error:** The rightmost plot value and final log value give **final-iteration RL2 = \(8.2017640219490544\times10^{-9}\)**, which is used for scoring. The exact best-ever isolated minimum is **\(1.9060446981126568\times10^{-9}\)**; it is not convergence evidence. The earlier SSBroyden warm-up report of `RL2 error: 0.00666222` is not the final result.
  * **Residual heatmap:** Residuals are spatially mottled rather than concentrated at a shock, boundary, or coherent failure front. The log reports a range from \(6.5228\times10^{-11}\) to \(1.1079\times10^{-6}\).
  * **Solution fields:** Reference and prediction are visually indistinguishable in phase, amplitude, and periodic stripe structure. Pointwise errors are generally only a few \(10^{-9}\), with the largest localized patch in the upper-right portion of the domain. Periodic boundaries are enforced architecturally by the Fourier embedding.
  * Using the history-consistent precision scale \(\epsilon_{\text{target}}\approx10^{-6}\) (no explicit tolerance was supplied), the final error easily earns full numerical-precision credit.

* **Diagnosis:** Stable, highly accurate training with a **noisy-but-flat sustained plateau**. There is no explosion, premature-stop signature, physics mismatch, or data leakage. The sole reason for rejection is incomplete mandatory exploration.

* **Targeted Improvements:**
  * **For Integrity:** None.
  * **For Accuracy:** No physics or optimization correction is needed. Complete **Run 5/5** using the preplanned **single width-only variation**—for example, width \(30\rightarrow40\)—while keeping degree 7, depth, loss assembly, vRBA settings, optimizer workflow, and all other controls unchanged. Compare runs using sustained final-iteration RL2, not isolated best-ever dips.
  * **For Details:** None. Retain the dual-curve plateau test and separate reporting of final and best-ever RL2.
  * **For Optimality:** Preserve `Input → Periodic Fourier Features → MLP → identity output`, with no raw-coordinate bypass. Retain Adam→SSBroyden and vRBA. No entropy penalty is applicable because this smooth Helmholtz problem contains no shocks. After Run 5, select the best sustained result; Run 4 is currently the leading candidate by final RL2.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The code implements a forward 2D Helmholtz PINN. Architecture: an MLP body with 3 hidden layers of width 30 and tanh activation; both input coordinates pass through Periodic_Fourier_Features with degree=5 and Lx=2.0, producing 20 embedding features that are concatenated before the MLP. No raw coordinates are passed to the body, no normalization is applied, and the output transformation is identity. The PDE residual is u_tt + u_xx + u - g with g=(1-17π²) sin(π t) sin(4π x), matching the manufactured solution sin(π x) sin(4π y) under the coordinate mapping used in the code. The loss contains only L_pde; no L_bc, L_ic, or L_data is used. Training uses Adam for 10,000 epochs with vRBA weighting and importance sampling, followed by SSBroyden/BFGS-style second-order optimization for 12,500 iterations in 100-step chunks with resampling. vRBA is used for weighting+sampling during Adam and for sampling only during SSBroyden. There is no PDE parameter continuation; k=1 is fixed. The analytical solution is used only for validation/error metrics. | 1.00e-06 | 35 | 35 | 6 | 14 | **90** |
| 1 | The implementation matches the strategy. It uses a PINN with Periodic Fourier Features (degree=5, Lx=2.0) applied to both coordinates, producing 20 embedding features that are concatenated and passed into an MLP body with width=30 and depth=3 using tanh activations and identity output. The PDE residual is the Helmholtz residual u_tt + u_xx + u - g with k=1 and g=(1-17*pi^2) sin(pi t) sin(4*pi x); the analytical reference is sin(pi t) sin(4*pi x). The loss is only L_pde: during Adam it is vRBA-weighted and vRBA-sampled, and during SSBroyden the PDE loss is unweighted while vRBA is used for sampling only. Adam runs for 10,000 iterations, then SSBroyden runs for 50,000 iterations with the Hessian reinitialized to identity at the transition. No L_data and no L_bc are included. | 1.00e-07 | 35 | 35 | 6 | 14 | **90** |
| 2 | Implementation matches the strategy. Architecture: MLP PINN with width 30 and three hidden layers (features [30,30,30,1]) using tanh activation and identity output. Input transformation uses Periodic Fourier Features on BOTH coordinates only, degree=5 with Lx=2.0, concatenated before the MLP body; no raw coordinates and no external normalization. PDE is Helmholtz with k=1: residual u_tt + u_xx + u - g, with g=(1-17*pi^2)sin(pi*t)sin(4*pi*x), matching the strategy under the t->x, x->y mapping. Analytical solution sin(pi*T)sin(4*pi*X) is used for validation/RL2 only. Loss assembly contains exactly one PDE loss term: during Adam it is vRBA-weighted mean squared residual with vRBA weighting and sampling; during SSBroyden it is unweighted mean squared residual with vRBA used for sampling only. No data, BC, IC, Neumann, gauge, or other ghost loss terms are included. Optimizer workflow: Adam for 10,000 iterations with vRBA, followed by SSBroyden for Nbfgs=100,000 using Nchange=100 inner iterations and Hessian reinitialized to identity at the Adam-to-SSBroyden transition. The plateau stopping criterion is implemented: after the 100,000 budget, it checks loss and RL2 band centers over the trailing 50% of the SSBroyden stage split into early/late halves, requires both change factors <= 2, and continues beyond budget if not flat. Final-iteration RL2 and best-ever minimum RL2 are reported separately, with the text noting that best-ever is not convergence evidence. | 3.51e-08 | 35 | 35 | 12 | 14 | **96** |
| 3 | Implements the specified 2D Helmholtz forward PINN. Architecture: MLP width 30, depth 3, tanh activation. Inputs (t,x) are mapped to (x,y); both coordinates pass through Periodic Fourier Features with degree=6 and Lx=2.0, producing 12 features per coordinate and 24 concatenated features. No raw-coordinate bypass is used. The sole loss is L_pde; residual is u_tt + u_xx + u - g with g=(1-17π²)sin(πt)sin(4πx). During Adam, vRBA weighting and sampling are used; during SSBroyden, vRBA is used for sampling only and the PDE loss is unweighted mean(residual²). Optimizer workflow matches the strategy: Adam for 10,000 iterations, then SSBroyden for 100,000 iterations with Nchange=100, Hessian initialized to identity at transition, and template-internal SSBroyden mechanics preserved. A dual-curve plateau test is implemented over the trailing 50% of the SSBroyden stage using loss and RL2 band centers with change factors ≤2, extending beyond budget if not flat. Final-iteration RL2 and best-ever RL2 are reported separately, with best-ever explicitly not treated as convergence evidence. Analytical solution matches u_an=sin(πt)sin(4πx). | 1.42e-08 | 35 | 35 | 12 | 14 | **96** |
| 4 | The implementation follows the Run 4 strategy: a PINN for the 2D Helmholtz problem with k=1 and manufactured solution u=sin(pi*x)*sin(4*pi*y). The model applies Periodic Fourier Features with degree 7 and Lx=2.0 to both input coordinates, yielding 28 embedded features, which are passed exclusively into an MLP body of width 30 and depth 3 with tanh activations; no raw coordinates are passed to the MLP. The PDE residual is u_tt + u_xx + u - g with g=(1-17*pi^2)*sin(pi*t)*sin(4*pi*x), corresponding to the strategy's Helmholtz equation under the legacy (t,x) naming. Training uses Adam for 10,000 iterations with vRBA weighting and importance sampling, followed by SSBroyden for 100,000 iterations with Nchange=100, Hessian reinitialization to identity, vRBA sampling only, and an unweighted PDE loss during SSBroyden. The loss assembly contains only L_pde; no L_bc, L_ic, or L_data is used, and the analytical solution is used solely for error metrics. The dual-curve plateau test is implemented over the trailing 50% of the SSBroyden stage, split into early/late halves with change factors <= 2, and the loop extends beyond budget if the plateau is not reached. | 8.20e-09 | 35 | 35 | 12 | 14 | **96** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves 'Helmholtz_predictions.npz' in NumPy .npz format. The npz contains keys: 'u_pred' with shape (100, 100) float64, 'u_analytical' with shape (100, 100) float64, 'X' with shape (100, 100) float64 representing the first coordinate grid, and 'Y' with shape (100, 100) float64 representing the second coordinate grid. It also saves 'Log_files_SSBroyden.mat' with training log arrays. |
| 1 | Saves Helmholtz_predictions.npz containing keys: 'u_pred' (100x100), 'u_analytical' (100x100), 'pointwise_error' (100x100), 'pde_residual' (100x100), 'X' (100x100), 'Y' (100x100), 'all_errors' (1055,), 'all_its' (1055,), and 'all_loss' (1055,). Also saves Log_files_SSBroyden.mat with model_name and histories: all_errors/all_its/all_loss/all_L_infty/all_L_2 length 1055; all_gamma/all_lamB/all_max_RBA length 555. |
| 2 | Saves Helmholtz_predictions.npz containing required keys: 'u_pred' (100x100), 'u_exact' (100x100), 'error' (100x100 pointwise absolute error), 'X' (100x100 coordinate grid, from T meshgrid), and 'Y' (100x100 coordinate grid, from X meshgrid). It also saves extra keys: 'pde_residual' (100x100), 'all_errors' (1D variable length log), 'all_its' (1D variable length log), and 'all_loss' (1D variable length log). Additionally saves Log_files_SSBroyden.mat with training histories and metadata. Required .npz fields are present. |
| 3 | Saves 'Helmholtz_predictions.npz' (NPZ format) containing keys: 'u_pred' (100x100 prediction field), 'u_exact' (100x100 analytical/reference field, equivalent to the strategy's u_analytical), 'error' (100x100 absolute pointwise error), 'pde_residual' (100x100 signed PDE residual), 'X' (100x100 first-coordinate meshgrid), 'Y' (100x100 second-coordinate meshgrid), 'all_errors' (1D logged RL2 history), 'all_its' (1D logged iteration history), and 'all_loss' (1D logged loss history). Also saves 'Log_files_SSBroyden.mat' with keys: model_name, all_errors, all_its, all_loss, all_gamma, all_lamB, all_max_RBA, all_L_infty, all_L_2. |
| 4 | Two artifacts are saved. (1) 'Log_files_SSBroyden.mat' saved via scipy.io.savemat contains keys: 'model_name' (string), 'all_errors' (1D array length N), 'all_its' (1D array length N), 'all_loss' (1D array length N), 'all_gamma' (1D array length N), 'all_lamB' (1D array length N), 'all_max_RBA' (1D array length N), 'all_L_infty' (1D array length N), 'all_L_2' (1D array length N). (2) 'Helmholtz_predictions.npz' saved via np.savez contains keys: 'u_pred' (100x100), 'u_exact' (100x100, the analytical reference), 'error' (100x100, pointwise absolute error), 'pde_residual' (100x100), 'X' (100x100 grid first coordinate), 'Y' (100x100 grid second coordinate), 'all_errors' (1D array length N), 'all_its' (1D array length N), 'all_loss' (1D array length N). N is the variable number of logged training iterations. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run1//Helmholtz2D_Periodic_SinPiX_Sin4PiY/2026_11_06_0252/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run1//Helmholtz2D_Periodic_SinPiX_Sin4PiY/2026_11_06_0300/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run1//Helmholtz2D_Periodic_SinPiX_Sin4PiY/2026_11_06_0312/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run1//Helmholtz2D_Periodic_SinPiX_Sin4PiY/2026_11_06_0324/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run1//Helmholtz2D_Periodic_SinPiX_Sin4PiY/2026_11_06_0335/` |


---
