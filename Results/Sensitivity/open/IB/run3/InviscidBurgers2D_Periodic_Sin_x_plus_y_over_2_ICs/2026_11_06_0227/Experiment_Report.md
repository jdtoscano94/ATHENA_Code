
# Experiment Report
*Last Updated: 2026-11-06 02:45:10*

## User Request
Let's solve the 2D Inviscid Burgers' equation.
Domain: $(x,y) \in [0, 4\pi]^2$, $t \in [0, 2]$.
BCs: Periodic in both x and y.

Initial Condition:
$$ u(x,y,0) = \frac{1}{3} + \frac{2}{3}\sin\left(\frac{x+y}{2}\right) $$

The final plot must be saved as summary_all.png it should include snapshots of several timesteps

---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
**1. Problem Statement**

Solve the scalar 2D inviscid Burgers equation (a nonlinear hyperbolic conservation law) on the doubly periodic square $[0,4\pi]^2$ for $t\in[0,2]$, with diagonal sinusoidal data. Note: the final time $T=2$ **exceeds the gradient-catastrophe time $t_s = 3/2$**, so the target is the **entropy solution** containing a propagating shock. Deliverable: exact ground-truth solution + `summary_all.png` with time snapshots.

**1. Mathematical Specification (Code-Ready Extraction)**

* **Governing Equations:**
$$\frac{\partial u}{\partial t} + \nabla\cdot\mathbf{F}(u) = 0, \qquad \mathbf{F}(u) = \begin{pmatrix} u^2/2 \\ u^2/2 \end{pmatrix}$$
i.e. $u_t + \left(\tfrac{u^2}{2}\right)_x + \left(\tfrac{u^2}{2}\right)_y = 0$ (quasilinear form: $u_t + u\,(u_x + u_y) = 0$).

* **Domain:** $(x,y) \in [0, 4\pi] \times [0, 4\pi]$, $t \in [0, 2]$.

* **Boundary Conditions:** Periodic in $x$ (period $4\pi$); Periodic in $y$ (period $4\pi$).

* **Initial Conditions:**
$$u(x,y,0) = \frac{1}{3} + \frac{2}{3}\sin\!\left(\frac{x+y}{2}\right)$$

* **Constants & Parameters:** Mean $c_0 = 1/3$; amplitude $c_1 = 2/3$; diagonal wavenumber $k = 1/2$; $T_{\text{end}} = 2.0$; $L_x = L_y = 4\pi$. No viscosity, no gravity, no $\gamma$ (scalar law). Derived: shock-formation time $t_s = -1/\min_\sigma V_0'(\sigma) = 3/2$; solution bounds (maximum principle) $u \in [-1/3,\, 1]$.

* **Expected Physics:** Exact diagonal symmetry — $u$ is constant along lines $x - y = \text{const}$; all dynamics propagate along $(1,1)$. Nonlinear steepening on descending flanks ($\pi/2 < \sigma < 3\pi/2$, $\sigma := (x+y)/2$). **Shock formation at $t_s = 1.5$** at $\sigma = \pi + 1/2$ (diagonal line $x+y = 2\pi + 1$); one shock per $2\pi$-period of $\sigma$ (two shock lines in the domain). For $t > 1.5$ the entropy shocks travel along $\sigma_s(t) = \pi + t/3$, i.e. 2D lines $x+y = 2\pi + \tfrac{2t}{3}$ and $x+y = 6\pi + \tfrac{2t}{3}$. At $t=2$: shocks at $x+y \approx 7.62$ and $x+y \approx 20.18$, with jumps $u_L \approx 0.97 \to u_R \approx -0.30$. Smooth rarefaction-like spreading near $x+y = \pi,\ 5\pi$.

**2. Strategic Methodology**

**Can we obtain an exact solution? YES — Method of Characteristics succeeds.**

1. **Characteristic structure (2D MoC):** $du/dt = 0$ along $dx/dt = u,\ dy/dt = u$. Characteristics are straight diagonal lines; $x - y$ is an invariant and $u_0$ is constant along it, so the inversion reduces to the scalar variable $\sigma = (x+y)/2$.
2. **Exact reduction:** $u(x,y,t) = V(\sigma,t)$ satisfies **exactly** the 1D inviscid Burgers equation
$$V_t + V V_\sigma = 0, \qquad V_0(\sigma) = \tfrac13 + \tfrac23\sin\sigma \quad (\text{period } 2\pi).$$
3. **MoC solution:** $V$ is constant along characteristics: $\sigma(t) = \sigma_0 + V_0(\sigma_0)\,t$. Inverting gives the **exact implicit solution**
$$\boxed{\,u(x,y,t) = \frac13 + \frac23\sin(\sigma_0), \qquad \sigma_0 + t\left(\frac13 + \frac23\sin\sigma_0\right) = \frac{x+y}{2}\,}$$
For $t < t_s = 3/2$ the foot map is strictly monotone ($1 + \tfrac{2t}{3}\cos\sigma_0 \ge 1 - \tfrac{2t}{3} > 0$): unique root, classical solution.
4. **Entropy Condition Root Selection (MANDATORY):** For $t > 3/2$ the implicit equation is multi-valued (up to 3 roots per period). The implementation **must select the entropy-satisfying branch** via the Lax–Oleinik (Hopf–Lax) functional:
$$u(\sigma,t) = \frac{\sigma - \sigma_0^*}{t}, \qquad \sigma_0^* = \arg\min_{y \in [\sigma-2\pi,\ \sigma+2\pi]} \left\{ \frac{(\sigma-y)^2}{2t} + \frac{y}{3} - \frac{2}{3}\cos y \right\}$$
(minimize over all characteristic roots; equivalently $u = V_0(\sigma_0^*)$). **Validation cross-check (equal-area rule):** analytic shock path $\sigma_s(t) = \pi + t/3$ for $t \ge 3/2$; left of shock select the largest-$V$ root, right of shock the smallest-$V$ root; shock states $u_{L,R} = \tfrac13 \pm \tfrac23\sin\delta(t)$ with $t = \tfrac{3\delta}{2\sin\delta}$.
5. **Decision:** MoC derivation succeeds → **MANDATORY Main Method**. The implementation is exact for **all** $t \in [0,2]$ including post-shock times; entropy selection handles the shock automatically. This is perfect Ground Truth — no discretization error.

**3. Implementation Requirements (CRITICAL)**

* **Configuration:** Use template defaults. (Analytical MoC evaluation — no DGSEM AMR or Fourier de-aliasing sections apply; no CFL/time-stepper needed; snapshot times are pure evaluation times.)
* **Root-solve specification:** Per query point, bracket all roots of $g(\sigma_0) = \sigma_0 + t(\tfrac13 + \tfrac23\sin\sigma_0) - \sigma$ on $[\sigma-2\pi, \sigma+2\pi]$ (dense sampling + bisection/Brent); apply the Hopf–Lax minimization among the roots for $t > 3/2$; return $u = \tfrac13 + \tfrac23\sin\sigma_0^*$. Handle $t=0$ directly: $u = V_0(\sigma)$.
* **3.1 Advisor-Mandated Adjustments:** None (Advisor report is empty).

**4. Visualization Strategy**

* **Field Selection:** The single primary field $u$ — it is the sole state variable and the driver of all dynamics. No derived quantities.
* **Time steps:** 4 snapshots at $t = 0.0$ (IC), $t = 1.0$ (steepening), $t = 1.5$ (shock formation), $t = 2.0$ (mature entropy shock).
* **Layout:** Grid with 1 row (field $u$) $\times$ 4 columns (time). The diagonal band structure and the sharp diagonal shock lines will be clearly visible.
* **Saving Data:** Save the full solution state ($u$ on the default grid) at frequent intervals, `dt_save = 0.1`, for Ground Truth.
* **Constraints:** Save as `summary_all.png` (strict). **Disable colorbars** (`colorbar=false`) if the code is in Julia. Do not use fixed colormaps/color limits — let scales adjust automatically.

**5. Cross-Validation & Data Integration (Conditional)**

* Not triggered: no external sensor/reference data paths are provided in the problem statement. Internal analytic validation only: confirm the shock lies on $x+y = 2\pi + \tfrac{2t}{3}$ (and its periodic image) for $t \ge 1.5$, and that $u \in [-1/3, 1]$ everywhere.

**REQUIRED COMPONENTS**

**Main Method:**
* Method of Characteristics (Analytical/Exact)

**Enhancements:**
* Entropy Condition Root Selection (Lax–Oleinik / equal-area branch selection for post-shock times $t > 3/2$)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

* **Status:** **APPROVED**
* **Reason:** Perfect Score Achieved. The mandatory exact Method of Characteristics was retained, including post-shock entropy selection. Execution, persistence, and internal analytical checks all passed; no further numerical refinement is warranted.

## Section 2: Quantitative Scorecard

* **Score Integrity:** **50/50**
* **Score Accuracy:** **20/20**
* **Score Details:** **15/15**
* **Score Optimality:** **15/15**
* **TOTAL SCORE:** **100/100**

**Error-log check:** The supplied STDOUT contains **no** `Minimum achieved error:` entry, so no such value can be reported or inferred to be zero. This run evaluates the analytical ground truth rather than a numerical approximation against a separate reference; its quantitative verification consists of solution bounds and independent shock-location checks.

## Section 3: Forensic Analysis & Feedback

### Observation

* **Plot:** The four snapshots show the expected evolution:
  * At \(t=0\), smooth diagonal sinusoidal bands with exact \(x\leftrightarrow y\) symmetry.
  * At \(t=1.0\), nonlinear steepening of the descending flanks and spreading of the opposing flanks.
  * At \(t=1.5\), the expected gradient catastrophe without spurious ringing.
  * At \(t=2.0\), two sharp diagonal entropy-shock segments, with no visible Gibbs oscillations or numerical diffusion. The shock transitions are limited only by image/grid sampling, not by a dissipative numerical scheme.
* **Execution and persistence:** The run completed successfully. `summary_all.png` and `burgers.npz` were saved, with `usol.shape == (21, 256, 256)`, corresponding to 21 times from \(0\) through \(2\).
* **Maximum principle:** The computed bounds,
  \[
  [-0.3333332802324365,\;0.9999999469104073],
  \]
  lie within the analytical interval \([-1/3,1]\).
* **Shock verification:** At \(t=2\), both detected shock locations exceed their analytical predictions by only \(5.0\times10^{-4}\):
  * \(7.617019-7.616519=0.000500\)
  * \(20.183389-20.182889=0.000500\)

  These are shock-detection/localization discrepancies, not a measured global solution error.

### Diagnosis

The solution correctly represents **gradient catastrophe at \(t_s=3/2\)** followed by a **Lax–Oleinik entropy shock**. The analytical reduction to \(\sigma=(x+y)/2\), characteristic inversion, and post-shock entropy selection are appropriate. There is no evidence of numerical dissipation, dispersion, instability, CFL restriction, or non-physical oscillation. The small shock-location offsets are consistent with grid-based post-processing and do not justify rejecting the exact ground-truth construction.

### Targeted Improvements

* **For Integrity:** None.
* **For Accuracy:** None.
* **For Details:** None.
* **For Optimality:** None. Preserve the exact MoC/Hopf–Lax solver; do not replace it with DGSEM, Fourier spectral, or another discretization.

**Recommendation:** Approve `burgers.npz` as high-fidelity PINN ground truth and retain `summary_all.png` as the required snapshot dashboard. No further refinement iterations are needed.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 3 | Implemented an exact analytical Method of Characteristics solver for the 2D inviscid Burgers equation on [0,4π]² with periodic boundaries and initial condition u₀ = 1/3 + (2/3)sin((x+y)/2). The 2D problem is reduced to σ = (x+y)/2, solving the implicit characteristic equation σ₀ + t(1/3 + (2/3)sin σ₀) = σ. Root finding brackets all roots on [σ-2π, σ+2π] using dense sampling plus stationary points and Bisection, with Hopf-Lax entropy selection for t > 1.5. The code evaluates the exact solution on a 256×256 grid at times t = 0:0.1:2, generates a 1×4 heatmap dashboard at t = 0, 1, 1.5, 2 with colorbars disabled, saved as summary_all.png, and saves the full ground-truth tensor to burgers.npz. It also performs internal checks for the maximum principle u ∈ [-1/3, 1] and post-shock shock location at t = 2. | N/A | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 3 | Saves 'summary_all.png' (PNG) containing a 1×4 layout of heatmaps for u at t = 0.0, 1.0, 1.5, 2.0. Saves 'burgers.npz' (NPZ) containing keys: 't' with shape (21,), 'x' with shape (256,), 'y' with shape (256,), and 'usol' with shape (21, 256, 256) ordered as (time, x, y). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/IB/run3//InviscidBurgers2D_Periodic_Sin_x_plus_y_over_2_ICs/2026_11_06_0227/` |


---
