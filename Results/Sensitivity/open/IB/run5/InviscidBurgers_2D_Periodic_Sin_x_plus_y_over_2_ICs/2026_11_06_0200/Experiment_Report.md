
# Experiment Report
*Last Updated: 2026-11-06 02:03:42*

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
Solve the 2D inviscid Burgers equation on the periodic torus $[0,4\pi]^2$ for $t\in[0,2]$ with a diagonal sinusoidal initial condition. The initial data depends only on $x+y$, so the problem is *exactly* reducible to 1D and admits a closed-form characteristic (exact) solution, including rigorous post-shock entropy selection. Deliverable: exact ground-truth solution snapshots saved frequently, with a 4-time-step summary plot saved as `summary_all.png`.

---

**1. Mathematical Specification (Code-Ready Extraction)**

* **Governing Equations:** 2D inviscid Burgers (scalar, conservative form):
$$\partial_t u + \partial_x\!\left(\frac{u^2}{2}\right) + \partial_y\!\left(\frac{u^2}{2}\right) = 0 \;\;\Longleftrightarrow\;\; \partial_t u + u\,\partial_x u + u\,\partial_y u = 0$$

* **Domain:** $(x,y) \in [0, 4\pi] \times [0, 4\pi]$, $t \in [0, 2]$.

* **Boundary Conditions:** Periodic in $x$; Periodic in $y$.

* **Initial Conditions:**
$$u(x,y,0) = \frac{1}{3} + \frac{2}{3}\sin\!\left(\frac{x+y}{2}\right)$$

* **Constants & Parameters:** IC mean $a = \tfrac13$, amplitude $b = \tfrac23$, diagonal wavenumber $k=\tfrac12$ (one full period per $4\pi$ in $x$ and $y$). Derived: $U_0 \in [-\tfrac13,\,1]$; $\min_\xi U_0'(\xi) = -\tfrac23$ at $\xi=\pi$; shock-formation time $t_s = \tfrac{3}{2}$. No $\gamma$, no gravity (not a system — single scalar field).

* **Expected Physics:** Diagonal wave steepening with solution invariant along lines $x-y=\text{const}$. Gradient catastrophe at $t_s = 1.5$ on the line $x+y = 2\pi + 1$. For $1.5 < t \le 2$: two identical diagonal compressive shocks (solution is $2\pi$-periodic in $\xi=(x+y)/2$). At $t=2$: shocks near $x+y \approx 7.6$ and $x+y \approx 20.2$, jumping from $u \approx +0.97$ down to $u \approx -0.31$. Spatial mean $\langle u\rangle = \tfrac13$ conserved.

---

**2. Strategic Methodology**

*Can we obtain an exact solution?* **Yes — fully.**

**Step 1 — Exact diagonal reduction.** The IC depends only on $\xi \equiv \frac{x+y}{2}$. With the ansatz $u(x,y,t) = U(\xi,t)$: $\partial_x u = \partial_y u = \tfrac12 U_\xi$, so the advection terms give $u\,\partial_x u + u\,\partial_y u = U\,U_\xi$. The 2D PDE collapses **exactly** (no approximation) to the 1D inviscid Burgers equation:
$$\partial_t U + U\,\partial_\xi U = 0, \qquad U(\xi,0) = U_0(\xi) = \frac13 + \frac23\sin\xi, \qquad \xi \in [0,4\pi]\ \text{(periodic, data } 2\pi\text{-periodic)}.$$

**Step 2 — Method of Characteristics.** $U$ is constant along characteristics $\dot\xi = U$:
$$\boxed{\;\xi = \xi_0 + t\,U_0(\xi_0), \qquad u = U_0(\xi_0) = \frac13 + \frac23\sin\xi_0\;}$$
Pre-shock, the map $\xi_0 \mapsto \xi_0 + t U_0(\xi_0)$ is a diffeomorphism since $1 + t\,U_0'(\xi_0) = 1 + \tfrac{2t}{3}\cos\xi_0 > 0$; the implicit equation is inverted (Newton) for $\xi_0$ at each query point, then $u = U_0(\xi_0)$.

**Step 3 — Shock formation.** $\partial_\xi U = U_0'(\xi_0)/(1 + t U_0'(\xi_0))$ blows up when $1 + tU_0' = 0$:
$$t_s = -\frac{1}{\min U_0'} = \frac{3}{2}, \quad \text{at } \xi_0 = \pi, \quad \xi_s = \pi + t_s U_0(\pi) = \pi + \frac12 \;\;(\text{line } x+y = 2\pi+1).$$
Since $t_s = 1.5 < 2$, the shock forms inside the time window.

**Step 4 — Post-shock ($t \ge t_s$): Entropy Condition Root Selection (MANDATORY).** The implicit equation becomes multi-valued (three roots in the fold region $\cos\xi_0 < -\tfrac{3}{2t}$). The implementation **must select the entropy-satisfying branch** via the equal-area rule: find $\xi_{sh}(t)$ and feet $\xi_{0,L} < \pi < \xi_{0,R}$ such that
$$\xi_{0,L} + t\,U_0(\xi_{0,L}) = \xi_{0,R} + t\,U_0(\xi_{0,R}) = \xi_{sh}(t), \qquad \int_{\xi_{0,L}}^{\xi_{0,R}}\Big[\xi_0 + t\,U_0(\xi_0) - \xi_{sh}(t)\Big]\,U_0'(\xi_0)\,d\xi_0 = 0,$$
then set $u = U_0(\xi_{0,L})$ for $\xi < \xi_{sh}(t)$ and $u = U_0(\xi_{0,R})$ for $\xi > \xi_{sh}(t)$. This construction automatically satisfies Rankine–Hugoniot $\dot\xi_{sh} = \tfrac12(U_L + U_R)$ and the Lax condition $U_L > \dot\xi_{sh} > U_R$ — the solution is exact (entropy solution) for **all** $t \in [0,2]$, including $t=2$.

Per the critical rules: MoC derivation succeeds → **MoC is the Main Method**; post-shock handling via entropy condition is automatic and exact. No numerical PDE solver is needed.

---

**3. Implementation Requirements (CRITICAL)**

* **Configuration:** Use template defaults.
* **Exact-solution evaluation:** Evaluate $u(x,y,t) = U\!\left(\tfrac{x+y}{2},\,t\right)$ pointwise on the computational grid at every save time. Exploit $2\pi$-periodicity in $\xi$ (two identical shocks per domain). Characteristic feet lie in the window $\xi_0 \in [\xi - t,\ \xi + t/3]$ (since $t\,U_0 \in [-t/3,\, t]$); enumerate all roots of the implicit equation in the periodic window and select per the entropy rule.
* **Entropy Condition Root Selection:** For $t \ge t_s$, in multi-valued (three-root) regions, the implementation must select the entropy-satisfying branch using the equal-area construction in Step 4 above (equivalently: RH shock fitting with Lax condition). Pre-shock: unique root via Newton iteration.
* **3.1 Advisor-Mandated Adjustments:** None.

---

**4. Visualization Strategy**

* **Field Selection:** $u$ is the sole field and fully captures the dynamics (steepening and the shock are discontinuities of $u$ itself). Plot only $u$.
* **Time snapshots:** 4 distinct times $t = 0,\ 0.75,\ 1.5,\ 2.0$ — initial data, smooth steepening, gradient catastrophe at $t_s$, and the developed shock pair.
* **Layout:** Grid with Rows = 1 (field $u$), Columns = 4 (times).
* **Saving Data:** Save the full solution state ($u$ on the grid) at `dt_save = 0.1` for Ground Truth.
* **Constraints:** Save as `summary_all.png` (strictly). **Disable colorbars** (`colorbar=false`) if the code is in Julia. No fixed colormaps — let scales adjust automatically. Plot only the primary variable $u$; do not compute derived quantities.

---

**5. Cross-Validation & Data Integration (Conditional)**

* **Trigger check:** No external data paths (sensors / reference solutions) are provided in the problem statement.
* **Action:** Not applicable. The MoC entropy solution **is** the ground truth; no $L_2$ error metrics against external data are required.

---

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
* **Reason:** Perfect Score Achieved. The exact characteristic/entropy solution was evaluated successfully, all requested artifacts were generated, and the snapshots agree with the expected analytical shock dynamics. No further numerical refinement is warranted.

## Section 2: Quantitative Scorecard

* **Score Integrity:** **50/50**
* **Score Accuracy:** **20/20**
* **Score Details:** **15/15**
* **Score Optimality:** **15/15**
* **TOTAL SCORE:** **100/100**

**Minimum achieved error:** **Not reported.** The supplied STDOUT contains neither the string `Minimum achieved error:` nor any other error norm. No quantitative error value can therefore be extracted, and this absence must not be interpreted as a measured zero error. An external-reference error norm is unnecessary here because the analytical entropy solution itself is the ground truth.

## Section 3: Forensic Analysis & Feedback

* **Observation:**
  * The solver completed normally and produced `summary_all.png` and `burgers.npz`. The saved tensor has shape `(21, 256, 256)`, corresponding to 21 save times at `dt_save = 0.1`, with consistent coordinate arrays and no reported NaNs, infinities, or solver warnings.
  * At \(t=0\), the plot shows the expected smooth diagonal sinusoidal bands. At \(t=0.75\), the compressive side has steepened while the solution remains smooth.
  * At \(t=1.5\), the solution exhibits the expected gradient catastrophe near \(x+y=2\pi+1\). This is the shock-formation instant, so a steep continuous profile—not necessarily a fully developed jump—is appropriate.
  * At \(t=2\), two sharp diagonal shocks are visible, consistent with the predicted locations
    \[
    x+y \approx 7.62,\qquad x+y\approx 20.18.
    \]
    The high-to-low jump orientation is consistent with the entropy solution, with states approximately \(u_L\approx0.97\) and \(u_R\approx-0.31\).
  * The fields preserve the required diagonal structure and periodic edge behavior. There is no visible Gibbs ringing, numerical oscillation, or shock smearing.

* **Diagnosis:**  
  No numerical dissipation, dispersion error, or shock-capturing artifact is evident. The diagonal reduction and characteristic inversion avoid PDE discretization error entirely, while the symmetric equal-area construction supplies the correct post-shock entropy branch. The \(256^2\) grid is used only to sample and store the exact solution, not to approximate its evolution. Any remaining error would be limited to floating-point/root-solver tolerances and spatial sampling, which were not independently quantified.

* **Targeted Improvements:**
  * **For Integrity:** None.
  * **For Accuracy:** None.
  * **For Details:** None. The shocks are sharp and free of oscillations; mesh refinement or added dissipation would not improve the underlying solution.
  * **For Optimality:** None. Method of Characteristics with entropy-condition root selection is the optimal approach for this reducible problem and is preferable to any numerical PDE discretization for PINN ground-truth generation.

* **Recommendation:**  
  Retain `burgers.npz` as the PINN ground-truth dataset and `summary_all.png` as its qualitative verification dashboard.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The code implements the required exact analytical Method of Characteristics solution for the 2D inviscid Burgers equation on the periodic torus [0,4π]^2. It uses the diagonal reduction ξ=(x+y)/2, initial condition u0=1/3+(2/3)sin((x+y)/2), and solves the resulting 1D Burgers problem exactly. Pre-shock characteristics are inverted via safeguarded Newton/bisection with foot window [ξ-t, ξ+t/3]. Post-shock entropy selection is implemented by computing symmetric equal-area shock feet from s=tβ sin s, determining the shock position π+αt, enumerating all characteristic roots, and selecting the leftmost branch for ξ<ξ_sh and rightmost for ξ>ξ_sh. Parameters include α=1/3, β=2/3, shock time t_s=1.5, N_samples=256, dt_save=0.1, t∈[0,2], and snapshot times [0.0,0.75,1.5,2.0]. The visualization saves a 1×4 heatmap dashboard as summary_all.png with colorbars disabled and only u plotted. | Not reported | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves 'burgers.npz' in NPZ format. Keys and shapes: 't' is a Float64 vector of length 21 (t=0.0:0.1:2.0); 'x' is a Float64 vector of length 256 (periodic grid omitting duplicate endpoint); 'y' is a Float64 vector of length 256; 'usol' is a Float64 array of shape (21, 256, 256) with dimensions (time, x, y). Also saves 'summary_all.png' containing four heatmap panels for u at t=0.0, 0.75, 1.5, and 2.0. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/IB/run5//InviscidBurgers_2D_Periodic_Sin_x_plus_y_over_2_ICs/2026_11_06_0200/` |


---
