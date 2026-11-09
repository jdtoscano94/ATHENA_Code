
# Experiment Report
*Last Updated: 2026-11-06 02:06:14*

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

Solve the 2D inviscid Burgers equation (scalar hyperbolic conservation law) on the periodic torus $[0,4\pi]^2$ for $t\in[0,2]$. The initial condition depends only on $x+y$, so the field is a diagonal traveling wave that nonlinearly steepens, forms a gradient catastrophe at $t=3/2$, and develops a propagating entropy shock. Deliverable: exact ground-truth fields plus a multi-timestep snapshot figure `summary_all.png`.

---

**1. Mathematical Specification (Code-Ready Extraction)**

* **Governing Equations:**
$$\frac{\partial u}{\partial t} + \frac{\partial}{\partial x}\!\left(\frac{u^{2}}{2}\right) + \frac{\partial}{\partial y}\!\left(\frac{u^{2}}{2}\right) = 0$$
(quasi-linear form: $u_t + u\,(u_x + u_y) = 0$). Single scalar field $u(x,y,t)$.

* **Domain:** $(x,y) \in [0, 4\pi] \times [0, 4\pi]$, $t \in [0, 2]$.

* **Boundary Conditions:** Periodic in $x$ and periodic in $y$ (torus).

* **Initial Conditions:**
$$u(x,y,0) = \frac{1}{3} + \frac{2}{3}\sin\!\left(\frac{x+y}{2}\right)$$

* **Constants & Parameters:**
  * Reduced diagonal coordinate: $\xi = \frac{x+y}{2}$; reduced IC $U_0(\xi) = \frac{1}{3} + \frac{2}{3}\sin\xi$ ($2\pi$-periodic in $\xi$).
  * Velocity range: $u \in [-\tfrac{1}{3},\, 1]$; mean drift speed $= \frac{1}{3}$.
  * Characteristic gradient: $U_0'(\xi) = \frac{2}{3}\cos\xi$; $\min U_0' = -\frac{2}{3}$ at $\xi_0 = \pi$.
  * Breaking time: $t_b = -1/\min U_0' = \frac{3}{2}$; breaking location $\xi_b = \pi + \frac{1}{2}$ (line $x+y = 2\pi + 1$).
  * No $\gamma$, no gravity, no viscosity — pure scalar hyperbolic problem.

* **Expected Physics:**
  * Solution remains exactly diagonal for all time: $u(x,y,t) = U\!\left(\frac{x+y}{2},\,t\right)$ — diagonal stripes advecting along $(1,1)$ with nonlinear steepening.
  * Gradient catastrophe at $t_b = 3/2$ on the line $x + y = 2\pi + 1$.
  * For $t > 3/2$: one compressive entropy shock per $2\pi$-period in $\xi$, propagating at **constant speed $1/3$**: shock trajectory $\xi_s(t) = \pi + \frac{t}{3}$, i.e., the line $x + y = 2\pi + \frac{2t}{3}$.
  * At $t = 2$: shock at $\xi_s = \pi + \frac{2}{3} \approx 3.808$ (line $x+y \approx 7.62$), with left state $u_L \approx 0.971$, right state $u_R \approx -0.305$ (jump $\approx 1.28$).

---

**2. Strategic Methodology**

* **Can we obtain an exact solution? YES — two-level derivation:**
  1. **Exact diagonal reduction.** Both the IC and the fluxes depend only on $x+y$. Substituting $u(x,y,t) = U(\xi,t)$, $\xi = \frac{x+y}{2}$, gives $\partial_x(u^2/2) = \partial_y(u^2/2) = \frac{1}{2}UU_\xi$, so the 2D conservation law collapses **exactly** to the 1D inviscid Burgers equation
  $$U_t + U\,U_\xi = 0, \qquad U(\xi,0) = \tfrac{1}{3} + \tfrac{2}{3}\sin\xi.$$
  The reduction preserves the Rankine–Hugoniot and entropy conditions (a jump along a diagonal line satisfies 2D RH exactly when the 1D shock satisfies 1D RH), so the 2D entropy solution is the 1D entropy solution lifted along the diagonal.
  2. **Method of Characteristics.** Along characteristics $\frac{d\xi}{dt} = U$, $U$ is conserved, giving the implicit characteristic equations
  $$\xi = \xi_0 + \left(\tfrac{1}{3} + \tfrac{2}{3}\sin\xi_0\right)t, \qquad u = U_0(\xi_0) = \tfrac{1}{3} + \tfrac{2}{3}\sin\xi_0.$$
  For $t < t_b = 3/2$ the map is invertible (unique root) → smooth exact solution. For $t > 3/2$ a multivalued wedge (three roots) forms → resolved by the entropy condition.

* **Entropy Condition Root Selection (MANDATORY).** For post-shock multivalued regions the implementation must select the entropy-satisfying branch: determine the shock trajectory from the Rankine–Hugoniot condition $\dot{\xi}_s = \frac{U_L + U_R}{2}$ with footpoints on impinging characteristics (equal-area rule), then take the **maximum-$u$ root for $\xi < \xi_s(t)$** and the **minimum-$u$ root for $\xi > \xi_s(t)$** (never the middle branch). For this IC the entropy shock admits the closed form $\xi_s(t) = \pi + \frac{t}{3}$ with states $u_{L,R} = \frac{1}{3} \pm w(t)$, $w = \frac{2}{3}\sin(t\,w)$ — a built-in validation check.

* MoC derivation succeeds → **Main Method = Method of Characteristics (Analytical/Exact)**, exact for all $t \in [0,2]$ including post-shock. (Fourier spectral is the natural numerical alternative on this periodic domain, but exact ground truth supersedes it.)

---

**3. Implementation Requirements (CRITICAL)**

* **Configuration:** Use template defaults. No time integration is required — the exact solution is evaluated pointwise at each save time.
* **MoC evaluation protocol:**
  * For each grid point and save time: compute $\xi = \frac{x+y}{2}$ (mod $2\pi$); solve the implicit characteristic equation $\xi_0 + \left(\frac{1}{3}+\frac{2}{3}\sin\xi_0\right)t \equiv \xi \pmod{2\pi}$ by bracketed root-finding on $\xi_0 \in [0, 2\pi)$. Max characteristic displacement is $t\max|u| = 2 < 2\pi$, so periodic wrapping is trivial.
  * **Entropy Condition Root Selection:** for $t > 3/2$, select the entropy-satisfying branch exactly as specified in Section 2.
* **3.1 Advisor-Mandated Adjustments:** None.

---

**4. Visualization Strategy**

* **Field Selection:** $u$ is the single (and complete) state field — it carries the entire dynamics (steepening, breaking, shock). Plot only $u$.
* **Time steps:** 4 snapshots: $t = 0.0$ (IC), $t = 0.5$ (smooth steepening + drift), $t = 1.5$ (gradient catastrophe at $t_b$), $t = 2.0$ (developed entropy shock).
* **Layout:** Grid with Rows = Selected Fields (1), Columns = Time (4) → 1×4 panel of 2D fields $u(x,y)$; the shock appears as a sharp diagonal discontinuity.
* **Saving Data:** Save the full solution state at `dt_save = 0.1` for Ground Truth.
* **Constraints:**
  * Save as: `summary_all.png` (strictly).
  * **DISABLE COLORBARS** (`colorbar=false`) if the code is in Julia.
  * Do not use fixed colormaps; let scales adjust automatically.
  * Plot only the primary variable $u$; do not compute derived quantities.

---

**5. Cross-Validation & Data Integration (Conditional)**

* No external data/sensor paths are provided in the problem → None.
* Internal validation: confirm that at $t = 2$ the shock lies on the line $x + y = 2\pi + \frac{4}{3} \approx 7.62$ with jump $\approx 1.28$.

---

**REQUIRED COMPONENTS**

**Main Method:**
Method of Characteristics (Analytical/Exact)

**Enhancements:**
* Entropy Condition Root Selection
* Exact Diagonal Reduction (1D Coordinate Mapping)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

* **Status:** **APPROVED**
* **Reason:** Perfect Score Achieved. The exact Method of Characteristics was correctly prioritized and executed, including post-breaking entropy-shock selection. No further numerical refinement is required.

## Section 2: Quantitative Scorecard

* **Score Integrity:** **50/50**
* **Score Accuracy:** **20/20**
* **Score Details:** **15/15**
* **Score Optimality:** **15/15**
* **TOTAL SCORE:** **100/100**

**Error-log check:** The exact phrase **“Minimum achieved error:” does not appear** in the supplied STDOUT log, so no such numerical value can be reported. This run evaluates the analytical ground truth directly rather than comparing a discretized PDE solution with a separate reference. Consequently, there is no reference-comparison error norm in the log; the quantitative validation instead consists of the shock location, states, jump, and Rankine–Hugoniot speed.

## Section 3: Forensic Analysis & Feedback

### Observation

* The simulation completed successfully. Both `summary_all.png` and `burgers.npz` were generated, with the saved solution having the expected shape `(21, 256, 256)`, corresponding to 21 save times on a \(256\times256\) grid.
* At **\(t=0\)**, the plot shows the expected smooth, diagonally oriented sinusoidal bands with two periods over the square domain.
* At **\(t=0.5\)**, the wave remains smooth while exhibiting nonlinear steepening and advection. The diagonal structure is preserved, as required by the exact reduction to \(\xi=(x+y)/2\).
* At **\(t=1.5\)**, the solution displays the expected gradient catastrophe. The very steep transition is physical: the solution is continuous at breaking, although its spatial gradient becomes unbounded. It should not be mistaken for a numerical oscillation or instability.
* At **\(t=2.0\)**, a sharp diagonal entropy shock is visible, together with its periodic counterpart. There is no visible Gibbs ringing, non-physical oscillation, or dissipative shock smearing beyond the finite image/grid sampling.
* The reported shock validation agrees with the analytical derivation:
  * Shock line: \(x+y=7.616518640512919=2\pi+4/3\).
  * \(u_L=0.971182704670293\), \(u_R=-0.3045160380036263\).
  * Jump: \(1.2756987426739193\).
  * Rankine–Hugoniot speed in the reduced coordinate: \(1/3\), exactly as predicted.

### Diagnosis

**No numerical failure is detected.** The observed evolution is the expected nonlinear steepening, gradient catastrophe, and subsequent entropy-shock formation. Because the solution is evaluated pointwise using characteristics rather than advanced with a numerical flux and time integrator, it introduces no PDE time-stepping error, CFL restriction, numerical dissipation, or dispersion. The entropy-condition root selection correctly resolves the post-breaking multivalued characteristic map.

The method is also optimal for PINN ground-truth generation: the exact diagonal reduction converts the two-dimensional problem to the one-dimensional entropy solution without approximation, while the \(256\times256\) output grid and 21 save times provide adequate sampling at modest storage cost.

### Targeted Improvements

* **For Integrity:** None.
* **For Accuracy:** None.
* **For Details:** None. The sharp transitions are physical shock/breaking features, not resolution artifacts.
* **For Optimality:** None. Retain the exact Method of Characteristics with entropy-condition root selection; switching to DGSEM, spectral, or finite-volume approximation would degrade the ground truth.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | The code implements the exact analytical Method of Characteristics solution for the 2D inviscid Burgers equation on the periodic torus [0,4π]^2 over t∈[0,2]. It uses the exact diagonal reduction ξ=(x+y)/2 with IC U0(ξ)=1/3+2/3 sin ξ. The characteristic equation ξ0+t U0(ξ0)≡ξ (mod 2π) is solved by bracketed root-finding on ξ0∈[0,2π). For t>3/2, the entropy-satisfying branch is selected using the Rankine-Hugoniot shock trajectory ξs(t)=π+t/3: maximum-u root for ξ<ξs and minimum-u root for ξ>ξs. The spatial grid is 256×256, with ground-truth saved at dt_save=0.1 (21 times). Visualization produces a 1×4 heatmap panel at t=0.0,0.5,1.5,2.0 with colorbars disabled. The code includes internal validation of the shock line x+y=2π+4/3 and jump≈1.28 at t=2. | N/A | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves 'burgers.npz' in NPZ format containing keys: 't' with shape (21,), 'x' with shape (256,), 'y' with shape (256,), and 'usol' with shape (21,256,256) stored in (time, x, y) order. Also saves 'summary_all.png' as a 1×4 heatmap dashboard for t=0.0,0.5,1.5,2.0. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/IB/run2//InviscidBurgers_2D_Periodic_Sin_x_plus_y_over_2/2026_11_06_0204/` |


---
