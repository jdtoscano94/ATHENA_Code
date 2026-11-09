
# Experiment Report
*Last Updated: 2026-11-06 12:26:20*

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
# Strategy: 2D Helmholtz (Refinement Rerun)

## 0. Triage
**Path 2: Advisor Recommendation (MANDATORY).** Previous run scored 91/100, rejected ONLY for non-plateauing final stage. **Cure:** Identical rerun with **SSBroyden stage extended 10,000 → 20,000 fixed iterations**. All architecture, embeddings, sampling, Adam warm-up, and physics UNCHANGED.

## 1. Data Source
None. No new data.

## 2. Problem Statement
- **Governing Equation:** Helmholtz: $u_{xx} + u_{yy} + k^2 u = g(x,y)$, with $k=1$.
- **PDE Residual:** $r = u_{xx} + u_{yy} + u - g$, where $g = (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)$.
- **BCs:** Periodic in both $x$ and $y$ on $[-1,1]^2$ — enforced architecturally (embedding), no loss term.
- **Analytical Solution:** $u_{an} = \sin(\pi x)\sin(4\pi y)$ ($a_1=1$, $a_2=4$).
- **Data Role:** Validation/Error Metrics ONLY. **No L_data.**
- **Problem Type:** Forward. Well-posed for PINNs (smooth strong-form solution).

## 2.1/2.2. Input/Output
- Input: 2D spatial $(x, y) \in [-1,1]^2$. No time.
- Output: scalar $u(x,y)$. Smooth, oscillatory (4 modes in $y$). No output transformation.

## 3. Base Model
**MLP with Hybrid Adam and SSBroyden** (2D problem). Width=30, depth=3, tanh.

## 4. Architecture
**Input → Input Transformation → Main Body → Output**

- **Normalization:** NONE. $x, y \in [-1,1]$ already in nice range — pass raw coordinates unchanged. Any normalization must be differentiable and inside the call function (here: identity).
- **Embedding:** Periodic Fourier Features, **degree=4** (Advisor: retain), applied to both $x$ and $y$.
- **Lx = 2.0, Ly = 2.0** (range of raw coordinates entering embedding: $1-(-1)=2$).
- **Input Composition:** Main body receives ONLY the concatenated $[x\text{-emb}, y\text{-emb}]$. NO bare $x$ or $y$.
- **Output transformation:** None (direct scalar).

## 5. Constraint Strategy
Periodic BCs satisfied exactly by architecture. Count = 1 ($L_{pde}$ only) < 5. No hard constraints, no BC/IC loss terms.

## 6. Loss Function
$$L_{total} = L_{pde}$$

## 6.1. Continuation
Not applicable (no Reynolds/Peclet).

## Iteration Schedule (FIXED BUDGETS)
- **Phase 1 — Adam:** 10,000 iterations (unchanged). vRBA weighting + sampling, exponential decay LR.
- **Phase 2 — SSBroyden:** **20,000 fixed iterations** (2× increase, per Advisor). vRBA lambdas updated for sampling ONLY. Do NOT interfere with SSBroyden internals.
- Early stopping ONLY on NaN/Inf. No open-ended stopping rules.

## 7. Weighting
- **Local:** vRBA. "`lambdas` are used for Weighting & Sampling during Adam, and for Sampling ONLY during SSBroyden."
- **Global:** 1 loss term → no balancing needed.

## 8. Visualization
Do NOT modify legacy visualization callbacks.

## 9. Saving
Save fields as .npz arrays with keys: `u_pred`, `u_analytical`, `error` (plus residual field) for visualization.

## Modules for Librarian
**Main Model:** MLP with Hybrid Adam and SSBroyden
**Enhancements:** Periodic Fourier Features (degree=4, Lx=Ly=2.0)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
REJECTED_FOR_REFINEMENT

## Section 1: Status Decision
* **Status:** **REJECTED_FOR_REFINEMENT**
* **Problem type:** Type A — forward, deterministic Helmholtz problem. The analytical solution is used only for validation, not training.

## Section 2: Quantitative Scorecard
* **Score Integrity:** 35/35
* **Score Accuracy:** 35/35
* **Score Details:** 7/15
* **Score Optimality:** 14/15
* **TOTAL SCORE:** **91/100**

Although the score exceeds 90, the mandatory final-stage plateau check fails: both curves continue descending on average.

## Section 3: Forensic Analysis & Feedback

### Observation
**Loss and error histories**
- The run is stable and readable, with no NaN, Inf, or explosive values.
- Total loss decreases from approximately \(10^{-1}\) to the \(10^{-9}\)–\(10^{-10}\) range. There is a clear optimizer transition near iteration 10,000: the loss spikes briefly, then decreases smoothly throughout the final stage.
- Relative \(L^2\) error begins near \(10^0\), remains around \(10^{-2}\) during much of the warm-up, drops sharply after the transition, and then decreases noisily toward the \(10^{-7}\)–\(10^{-6}\) range.
- Treating the segment after the iteration-10,000 transition as the final stage, with midpoint near iteration 15,000–15,500:
  - **Loss:** midpoint band center \(A \approx 3\times10^{-8}\); ending band center \(B \approx 3\times10^{-10}\).
  - **Relative error:** midpoint band center \(A \approx 5\times10^{-6}\); ending band center \(B \approx 5\times10^{-7}\).
- Thus, \(B<A\) by roughly two orders of magnitude for loss and one order for error. These are **not horizontal plateaus**. Splitting the last half of the final stage into two windows gives the same conclusion: the loss band drops approximately from \(10^{-8}\) to \(10^{-9}\), while the error band drops from roughly \(2\times10^{-6}\) to \(6\times10^{-7}\).

**Residual and pointwise-error fields**
- The PDE residual is broadly distributed and speckled across the domain, without a dominant shock front or localized corner failure.
- The pointwise error is also broadly distributed. Higher-error red regions occupy much of the domain, while lower-error green/cyan islands appear mainly near portions of the left/right boundaries and lower-left region.
- The reported residual range is finite: approximately \(1.21\times10^{-9}\) to \(4.42\times10^{-5}\).

**Reference versus prediction**
- Reference and prediction are visually very similar, reproducing the same oscillatory bands, phase, and amplitude. No substantial boundary mismatch or visible phase distortion is apparent.

**Log confirmation**
- The earlier `RL2 error: 0.00663973` is the Adam warm-up report and is not used for final scoring.
- The exact best-ever value at the bottom of the log is:
  - **`Minimum achieved  RL2 error: 1.0832520801005222e-07`**
- The plot’s rightmost error is approximately **\(5\times10^{-7}\)**, within a noisy \(10^{-7}\)–\(10^{-6}\) band. This final plotted value, rather than the earlier minimum, is used for scoring.

### Diagnosis
**Budget-Limited / Premature Stop.** The formulation is accurate and stable, but the final SSBroyden stage ends while both loss and error are still improving on average. The increased budget improved accuracy, but did not produce a genuine final plateau.

### Targeted Improvements
* **For Integrity:** None. Execution, visualization, and numerical values are valid.
* **For Accuracy:** Increase only the final SSBroyden stage from **20,000 to 40,000 fixed iterations**. Keep the Adam warm-up, periodic Fourier embeddings, MLP, vRBA settings, forcing, and physics unchanged.
* **For Details:** After that fixed-budget rerun, reassess whether both band centers are horizontal over the last 50% of the final stage. Do not interpret the current noisy descent as convergence.
* **For Optimality:** None. Retain periodic Fourier features, vRBA, and SSBroyden. Entropy regularization and shock-formulation changes are not applicable to this smooth Helmholtz problem.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The implementation matches the strategy. It uses a PINN with Periodic Fourier Features of degree 4 and Lx=Ly=2.0 applied to both spatial inputs; only the concatenated embeddings are passed to the MLP. The MLP has 3 hidden layers of width 30 with tanh activation and a direct scalar output, with no output transformation. The PDE residual is r = u_tt + u_xx + u - g with k^2=1 and g=(1-17π^2)sin(πx)sin(4πy), under the code's t/x aliasing for the two physical coordinates. The analytical solution is sin(πx)sin(4πy), used only for validation/error metrics. The loss contains only L_pde: during Adam it is vRBA-weighted MSE on the PDE residual; during SSBroyden it is unweighted MSE, while vRBA lambdas are updated and used for sampling only. The optimizer workflow is Adam for 10,000 iterations with exponential-decay learning rate and vRBA importance sampling (batch size 20,000), followed by SSBroyden for a fixed 10,000-step budget using Nchange=100 per outer loop (100 outer iterations). Early stopping is only on NaN/Inf. No BC or data loss terms are added. | 2.00e-06 | 35 | 35 | 7 | 14 | **91** |
| 2 | Implements the specified 2D Helmholtz PINN. The base model is an MLP with three hidden width-30 tanh layers and scalar output, using Periodic Fourier Features of degree 4 with Lx=Ly=2.0 applied separately to the two raw coordinates. The main body receives only the concatenated embedded coordinates, with no bare coordinates and no normalization. The loss is only L_pde: during Adam it is the vRBA-weighted mean squared Helmholtz residual; during SSBroyden it is the unweighted mean squared residual, with vRBA lambdas used for resampling only. The PDE residual is u_tt + u_xx + u - g with g=(1-17*pi^2)*sin(pi*t)*sin(4*pi*x). No BC/IC or data loss terms are included. Optimizer workflow matches the refinement plan: Adam for 10,000 iterations with exponential-decay LR and vRBA weighting/sampling, followed by SSBroyden for 20,000 fixed iterations (200 outer batches of 100) with vRBA resampling only and NaN/Inf early stopping. The analytical solution used for validation/error is sin(pi*x)*sin(4*pi*y). | 5.00e-07 | 35 | 35 | 7 | 14 | **91** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves `Helmholtz_results.npz` (NumPy .npz) with keys: `x` (100x100 float64 physical-x grid), `y` (100x100 float64 physical-y grid), `u_pred` (100x100 float64 predicted field), `u_analytical` (100x100 float64 exact field), and `relative_l2_error` (1D float64 error-history array; default length 655 = 555 Adam logged entries + 100 SSBroyden outer iterations). It also saves an extra non-required MATLAB file `Log_files_SSBroyden.mat` containing `model_name` plus history keys `all_errors`, `all_its`, `all_loss`, `all_gamma`, `all_lamB`, `all_max_RBA`, `all_L_infty`, and `all_L_2`; with default settings the shared Adam+SSBroyden histories are length 655, while Adam-only histories are length 555. |
| 2 | Saves Helmholtz_results.npz in NumPy .npz format with keys: 'x' (NC_TEST x NC_TEST meshgrid array of the first physical coordinate, default 100x100), 'y' (NC_TEST x NC_TEST meshgrid array of the second physical coordinate), 'u_pred' (same grid shape), 'u_analytical' (same grid shape), 'error' (same grid shape, absolute difference), and 'residual' (same grid shape, PDE residual field). Also saves Log_files_SSBroyden.mat in MAT format containing training logs: 'model_name', 'all_errors', 'all_its', 'all_loss', 'all_gamma', 'all_lamB', 'all_max_RBA', 'all_L_infty', and 'all_L_2'. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run5/results//Helmholtz_2D_Periodic_Sin_pi_x_Sin_4pi_y/2026_11_06_1214/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/Helmholtz/run5/results//Helmholtz_2D_Periodic_Sin_pi_x_Sin_4pi_y/2026_11_06_1222/` |


---
