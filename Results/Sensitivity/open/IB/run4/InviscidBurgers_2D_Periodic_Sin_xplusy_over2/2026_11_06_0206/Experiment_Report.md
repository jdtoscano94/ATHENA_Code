
# Experiment Report
*Last Updated: 2026-11-06 02:09:06*

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
Scalar 2D inviscid Burgers' equation (hyperbolic conservation law) on the doubly periodic square $[0,4\pi]^2$ for $t\in[0,2]$, with diagonal sinusoidal initial data. The IC depends only on $x+y$; the profile steepens and forms two diagonal shocks at $t_b = 3/2$. An exact entropy solution is derivable via Method of Characteristics — it will be generated as perfect Ground Truth. Snapshots at several timesteps must be saved to `summary_all.png`.

**1. Mathematical Specification (Code-Ready Extraction)**
* **Governing Equations:**
$$\frac{\partial u}{\partial t} + \frac{\partial}{\partial x}\!\left(\frac{u^2}{2}\right) + \frac{\partial}{\partial y}\!\left(\frac{u^2}{2}\right) = 0 \qquad \text{on } (x,y)\in[0,4\pi]^2,\; t\in[0,2]$$
(equivalently $\partial_t u + u\,(\partial_x u + \partial_y u)=0$; scalar, hyperbolic; fluxes $f_x(u)=f_y(u)=u^2/2$).
* **Domain:** $(x,y)\in[0,4\pi]\times[0,4\pi]$, $t\in[0,2]$.
* **Boundary Conditions:** $x\in\{0,4\pi\}$: Periodic; $y\in\{0,4\pi\}$: Periodic.
* **Initial Conditions:**
$$u(x,y,0) = \frac{1}{3} + \frac{2}{3}\sin\!\left(\frac{x+y}{2}\right)$$
* **Constants & Parameters:** IC mean $\bar u = 1/3$ (conserved), amplitude $2/3$; reduced 1D characteristic speed $a(U)=2U$, flux $f(U)=U^2$ in $s=x+y$; breaking time $t_b = 3/2$; post-shock trajectory $s^*(t)=2\pi+\frac{2t}{3}$ (and $6\pi+\frac{2t}{3}$), shock speed $2/3$. No $\gamma$, no gravity, no viscosity — none applicable.
* **Expected Physics:** Solution stays constant along anti-diagonals ($x-y=$ const); all variation along $s=x+y$; remains $4\pi$-periodic in $x$ and $y$. For $t<1.5$: smooth steepening where $\partial_s u_0<0$ (near $s_0=2\pi,\,6\pi$), slope amplifying as $(1-2t/3)^{-1}$. At $t=1.5$: gradient catastrophe (vertical tangent) at $x+y=2\pi+1$ and $6\pi+1$. For $t>1.5$: two entropy shocks (downward jumps, $u_L>u_R$) propagate at exact speed $2/3$ along $x+y=2\pi+\frac{2t}{3}$ and $x+y=6\pi+\frac{2t}{3}$. **Validation checkpoint at $t=2$:** shocks at $x+y\approx 7.6165$ and $20.1829$ with states $u_L\approx 0.9711$, $u_R\approx -0.3044$ (jump $\approx 1.28$).

**2. Strategic Methodology**
*Physics:* Hyperbolic scalar conservation law → MoC is the highest-priority tool.

*Can we obtain an exact solution? YES — derivation:*
1. **Exact reduction to 1D.** The IC depends only on $s=x+y$, and the ansatz $u(x,y,t)=U(s,t)$ is preserved by the PDE ($\partial_x u=\partial_y u=U_s$), reducing the 2D problem **exactly** to
$$U_t + 2UU_s = 0 \;\Longleftrightarrow\; U_t + (U^2)_s = 0,\qquad U_0(s)=\tfrac13+\tfrac23\sin\tfrac{s}{2},$$
with $U$ $4\pi$-periodic in $s$ — exactly consistent with the 2D periodic BCs (entropy consistency of the reduction is exact, shocks included).
2. **Characteristics.** $\frac{ds}{dt}=2U$, $\frac{dU}{dt}=0$ ⇒
$$s = s_0 + 2t\,U_0(s_0) = s_0 + \frac{2t}{3}\Big(1+2\sin\frac{s_0}{2}\Big),\qquad u = U_0(s_0).$$
3. **Pre-shock ($t<3/2$).** $F(s_0)=s_0+\frac{2t}{3}(1+2\sin\frac{s_0}{2})-s$ has $F'=1+\frac{2t}{3}\cos\frac{s_0}{2}\ge 1-\frac{2t}{3}>0$: a unique root per $(s,t)$ → exact smooth solution by Newton inversion.
4. **Breaking time.** $t_b=-1/\min_s[2U_0'(s)]=-1/\min_s[\frac23\cos\frac{s}{2}]=\frac32$, first crossing at $s_0=2\pi$ (and $6\pi$).
5. **Post-shock ($t>3/2$) — Entropy Condition Root Selection.** The characteristic map becomes multi-valued (up to 3 roots). The implementation **must select the entropy-satisfying branch**: among all roots $\xi_i$ of $s=\xi+2tU_0(\xi)$, choose the minimizer of the Hopf–Lax functional
$$G(\xi)=\frac{(s-\xi)^2}{4t}+\phi_0(\xi),\qquad \phi_0(\xi)=\int_0^\xi U_0=\frac{\xi}{3}+\frac{4}{3}\Big(1-\cos\frac{\xi}{2}\Big),\qquad u(s,t)=U_0(\xi^*)=\frac{s-\xi^*}{2t}.$$
This minimization is exactly the Oleinik entropy condition (equal-area rule); shocks emerge with exact Rankine–Hugoniot speeds. The solution is exact for **all** $t\in[0,2]$ — no Gibbs phenomenon, no numerical viscosity.
6. **Bounded root window.** $U_0\in[-1/3,1]$ ⇒ all roots lie in $\xi\in[s-2t,\;s+\frac{2t}{3}]$ (length $\le 16/3<4\pi$ for $t\le 2$): dense bracketing scan + Newton refinement to machine precision.
7. **2D reconstruction.** $u(x,y,t)=U(x+y,t)$ evaluated pointwise on the grid — no time stepping required.

MoC derivation succeeded (implicit characteristic equations produced) → **MANDATORY selection as Main Method.** This constitutes perfect Ground Truth for any subsequent numerical benchmarking.

**3. Implementation Requirements (CRITICAL)**
* **Configuration:** Use template defaults. (Exact solver: no CFL, no time stepper, no N/K.)
* **Exact solver specification (MoC template):**
  * For each grid point $(x,y)$ and output time $t>0$: set $s=x+y$; find all roots of $F(\xi)=\xi+2t\big(\frac13+\frac23\sin\frac{\xi}{2}\big)-s$ in $[s-2t-\epsilon,\;s+\frac{2t}{3}+\epsilon]$; **select the entropy branch** $\xi^*=\arg\min_i G(\xi_i)$; output $u=\frac13+\frac23\sin\frac{\xi^*}{2}$.
  * $t=0$: output the IC formula directly.
  * Root refinement to machine precision (bracketed Newton); vectorize over the grid.
  * Self-check: at $t=2$ the solution must exhibit shocks at $x+y\approx 7.6165$ and $20.1829$ with $u_L\approx 0.9711$, $u_R\approx -0.3044$.
* **3.1 Advisor-Mandated Adjustments:** None.

**4. Visualization Strategy**
* **Field Selection:** Single primary field $u(x,y,t)$ — the only conserved variable; it fully captures the steepening and shock formation. No derived quantities.
* **Snapshots (4 distinct times):** $t=0.0$ (initial sinusoid), $t=1.0$ (steepened pre-shock), $t=1.5$ (gradient catastrophe), $t=2.0$ (two developed diagonal shocks).
* **Layout:** Grid — 1 row (field $u$) × 4 columns (times).
* **Saving Data:** Save the full solution state ($u$ on the evaluation grid) every `dt_save = 0.1` over $[0,2]$ (21 ground-truth states).
* **Constraints:**
  * Save as: `summary_all.png` (strictly).
  * If the code is in Julia: **disable colorbars** (`colorbar=false`).
  * No fixed colormaps — let scales adjust automatically per snapshot.
  * Plot only $u$.

**5. Cross-Validation & Data Integration (Conditional)**
* Trigger check: [USER_QUESTION] contains no external data paths (no sensors/reference data) → **Not applicable.** The MoC solution is itself the analytical ground truth; use the Section 1 shock checkpoints ($t=2$: shock locations and jump states) for self-validation.

**REQUIRED COMPONENTS**

**Main Method:**
Method of Characteristics (Analytical/Exact)

**Enhancements:**
* Entropy Condition Root Selection

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

* **Status:** **APPROVED**
* **Reason:** Perfect Score Achieved. No further refinement possible. The exact Method of Characteristics was correctly preserved, including entropy-based root selection, and the completed solution passes the analytical shock checkpoints.

## Section 2: Quantitative Scorecard

* **Score Integrity:** **50/50**
* **Score Accuracy:** **20/20**
* **Score Details:** **15/15**
* **Score Optimality:** **15/15**
* **TOTAL SCORE:** **100/100**

## Section 3: Forensic Analysis & Feedback

### Observation

* **Plot:** The four snapshots show the expected qualitative evolution:
  * At \(t=0\), the field is a smooth diagonal sinusoidal pattern.
  * At \(t=1\), the compressive portions have steepened while the solution remains smooth.
  * At \(t=1.5\), the transition is nearly vertical, consistent with gradient catastrophe at the breaking time.
  * At \(t=2\), two sharp diagonal shocks are visible, with no observable Gibbs ringing, numerical oscillations, or diffusive smearing beyond grid/plot sampling.
* The solution retains the expected dependence on \(x+y\), diagonal symmetry, and doubly periodic structure.
* **Execution:** The solver completed all 21 output times, generated `summary_all.png`, and saved `burgers.npz` with `usol.shape = (21,256,256)`. No crash, NaN/Inf warning, or incomplete output is reported.
* **Analytical validation:** At \(t=2\), the logged shock positions are
  \[
  s\approx 7.616519,\qquad s\approx 20.182889,
  \]
  with \(u_L=0.971198\), \(u_R=-0.304532\), and jump \(1.27573\). These agree with the strategist’s analytical checkpoints. Moreover, \(u_L+u_R\approx 0.666666\), consistent with the predicted shock speed \(2/3\) in the reduced coordinate.
* **Error-log check:** The exact phrase **“Minimum achieved error:” does not appear** in the supplied STDOUT tail. Therefore, no numerical error norm can be quoted, and its absence must not be interpreted as a measured zero error. Validation here is instead provided by the exact construction and analytical shock checks.

### Diagnosis

No failure phenomenon is detected. The solution exhibits the correct characteristic steepening, shock formation, and entropy-shock propagation. Because the method evaluates the exact entropy solution rather than marching a discretized PDE, there is no numerical-flux dissipation, CFL restriction, or time-integration error. Accuracy is limited only by floating-point/root-solving tolerance and pointwise grid sampling.

The Hopf–Lax minimization supplies the required entropy branch selection, making the Method of Characteristics both exact and methodologically optimal for PINN ground-truth generation.

### Targeted Improvements

* **For Integrity:** None.
* **For Accuracy:** None.
* **For Details:** None. No additional mesh refinement or polynomial-degree adjustment is applicable or needed.
* **For Optimality:** None. Preserve **Method of Characteristics (Analytical/Exact) + Entropy Condition Root Selection**; do not replace it with DGSEM, finite-volume, or spectral approximation.

**Recommendation:** Approve the run and use `burgers.npz` as high-fidelity PINN training data.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 2 | The code implements the exact Method of Characteristics solution for the 2D inviscid Burgers equation on the doubly periodic domain [0,4π]^2 over t∈[0,2]. It uses the invariant s=x+y to reduce the problem to U_t+2UU_s=0 with U0(s)=1/3+(2/3)sin(s/2). For t>0 it solves the implicit characteristic equation ξ+2tU0(ξ)-s=0 over the bounded window [s-2t-ε, s+2t/3+ε], refines roots by bracketed Newton/bisection, and selects the entropy-satisfying root by minimizing the Hopf–Lax functional G(ξ)=(s-ξ)^2/(4t)+φ0(ξ). It reconstructs u(x,y,t)=U0(ξ*) on a 256×256 periodic grid excluding duplicate endpoints and evaluates 21 output states from t=0 to t=2 with dt_save=0.1. Snapshot times t=0,1,1.5,2 are plotted in a 1×4 dashboard with colorbars disabled and saved as summary_all.png. The t=2 shock checkpoint is included, verifying shock positions and left/right states against the strategy values. | N/A | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 2 | Saves two artifacts. (1) summary_all.png: PNG image containing a 1-row × 4-column dashboard of u(x,y,t) at t=0.0, 1.0, 1.5, and 2.0; each panel is a heatmap and no colorbars are shown. (2) burgers.npz: NPZ archive containing keys: 't' with shape (21,), 'x' with shape (256,), 'y' with shape (256,), and 'usol' with shape (21,256,256), ordered as (time, x, y). The code also reads the NPZ back and verifies these key shapes and finiteness. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/IB/run4//InviscidBurgers_2D_Periodic_Sin_xplusy_over2/2026_11_06_0206/` |


---
