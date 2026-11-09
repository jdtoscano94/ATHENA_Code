
# Experiment Report
*Last Updated: 2026-11-06 02:09:14*

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
# Scientific Strategy Report — 2D Inviscid Burgers' Equation (Diagonal Planar Wave)

**1. Problem Statement**

Solve the scalar 2D inviscid Burgers equation on the doubly periodic square $[0,4\pi]^2$ for $t\in[0,2]$ with diagonal planar-wave data $u_0 = \tfrac13 + \tfrac23\sin\big((x+y)/2\big)$. The nonlinearity self-steepens the wave; a shock forms at $t_s = 3/2$ and must be resolved as the entropy solution through $t=2$. Deliverable: exact ground-truth solution with time snapshots saved as `summary_all.png`.

---

**2. Mathematical Specification (Code-Ready Extraction)**

* **Governing Equations:** (conservative form; hyperbolic, characteristic velocity $(u,u)$)
$$\partial_t u + \partial_x\!\left(\frac{u^2}{2}\right) + \partial_y\!\left(\frac{u^2}{2}\right) = 0$$
* **Domain:** $(x,y) \in [0, 4\pi] \times [0, 4\pi]$, $t \in [0, 2]$.
* **Boundary Conditions:** x: Periodic; y: Periodic.
* **Initial Conditions:**
$$u(x,y,0) = \frac{1}{3} + \frac{2}{3}\sin\left(\frac{x+y}{2}\right)$$
* **Constants & Parameters:**
  * Flux per direction: $f(u) = u^2/2$; no material constants ($\gamma$, gravity: N/A).
  * Planar coordinate $s := x+y \in [0, 8\pi]$; profile $U_0(s)=\tfrac13+\tfrac23\sin(s/2)$ is $4\pi$-periodic in $s$ (domain = exactly two periods).
  * Conserved mean $\bar u = 1/3$; solution range $u \in [-1/3,\,1]$ for all $t$.
* **Expected Physics:** Diagonal planar wave (constant along $x-y=\text{const}$); crests ($u=1$, $s=\pi \bmod 4\pi$) advance at $\dot s = 2$, troughs ($u=-1/3$, $s=3\pi \bmod 4\pi$) recede at $\dot s = -2/3$ → steepening. Gradient catastrophe at $t_s = 3/2$ on $x+y = 2\pi+1 \pmod{4\pi}$. For $3/2 < t \le 2$: entropy shock along $x+y = 2\pi + \tfrac23 t \pmod{4\pi}$ with $u_L + u_R = 2/3$; at $t=2$: $u_L \approx 0.971$, $u_R \approx -0.304$ (jump $\approx 1.28$).

---

**3. Strategic Methodology**

*Can we obtain an exact solution?* **YES.** The data depends on $x+y$ only, so the 2D problem reduces **exactly** to a 1D conservation law along diagonals:
$$U_t + \partial_s(U^2) = 0, \qquad U(s,0) = \tfrac13 + \tfrac23\sin(s/2).$$

**MoC derivation (exact, all $t$):**
* Characteristics: $\dot x = u,\ \dot y = u,\ \dot u = 0$ → invariants $u = U_0(\sigma)$, $x-y = \text{const}$, with $\sigma = x_0+y_0$.
* **Implicit exact solution:**
$$u(x,y,t) = U_0(\sigma), \qquad x+y \equiv \sigma + 2t\,U_0(\sigma) \pmod{4\pi}, \qquad U_0(\sigma)=\tfrac13+\tfrac23\sin(\sigma/2).$$
* **Pre-shock ($t < 3/2$):** the map $\sigma \mapsto \sigma + 2tU_0(\sigma)$ is strictly monotone since $1 + \tfrac{2t}{3}\cos(\sigma/2) \ge 1 - \tfrac{2t}{3} > 0$ → unique root, machine-precision via bracketed root finding.
* **Shock formation:** $t_s = -1/\min_\sigma[2U_0'(\sigma)] = -1/(2\cdot(-\tfrac13)) = \tfrac32$, at $\sigma = 2\pi$ → $s_* = 2\pi + 1$ (vertical tangent, $u = 1/3$).
* **Post-shock ($t \ge 3/2$):** characteristics become multi-valued. **Entropy Condition Root Selection is REQUIRED** — the implementation must select the entropy-satisfying branch (Oleinik / equal-area rule). Rankine–Hugoniot for flux $U^2$ plus the exact data antisymmetry $U_0(2\pi+\delta) = \tfrac23 - U_0(2\pi-\delta)$ yields the closed-form shock law:
$$s_{sh}(t) = 2\pi + \frac{2}{3}\,t \pmod{4\pi}, \qquad u_L + u_R = \frac{2}{3}.$$

**Selection:** MoC derivation succeeded → **MANDATORY Main Method: Method of Characteristics (Analytical/Exact)** — exact for all $t\in[0,2]$, post-shock handling automatic. Fourier spectral (otherwise natural on this periodic box) is rejected: exact ground truth is available and takes priority, and any spectral run would Gibbs-oscillate for $t > t_s$.

---

**4. Implementation Requirements (CRITICAL)**

* **Configuration:** Use template defaults.
* **Exact-solution evaluation spec:**
  * Per point $(x,y,t)$: reduce $s = (x+y) \bmod 4\pi$; solve the implicit characteristic equation for $\sigma \in [0,4\pi)$ by bracketed bisection/Newton to machine precision.
  * For $t \ge 3/2$: select the entropy-satisfying root among multi-valued branches; the discontinuity must land exactly on $x+y = 2\pi + \tfrac23 t \pmod{4\pi}$.
  * Exploit planar structure: solve once per unique $s$-node, broadcast along diagonals (exact, negligible cost).
  * Verification anchors: conserved mean $1/3$; range $[-1/3, 1]$; shock law above.
* **Advisor-Mandated Adjustments (Overrides):** None.

---

**5. Visualization Strategy**

* **Field Selection:** the single state variable $u$ — it fully captures the steepening and shock dynamics (no other fields exist; no derived quantities).
* **Time steps (4):** $t = 0.0,\ 0.75,\ 1.5,\ 2.0$ — initial planar wave → mid-steepening → gradient catastrophe ($t_s$) → mature entropy shock.
* **Layout:** 1 row × 4 columns (Rows = fields, Columns = time).
* **Saving Data:** save the full solution state ($u$) at `dt_save = 0.1` for Ground Truth.
* **Constraints:** Save as `summary_all.png` (strictly). **Disable colorbars (`colorbar=false`) if the code is in Julia.** Automatic per-snapshot color scaling (no fixed color ranges/cmaps). Plot only $u$.

---

**6. Cross-Validation & Data Integration (Conditional)**

* **Trigger check:** No external data/sensor/reference paths provided → **Not triggered.**
* Internal verification is provided by the analytic anchors in §4 (shock trajectory law, conserved mean, value range).

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

**Status:** **APPROVED**

**Reason:** Perfect Score Achieved. The exact Method of Characteristics with entropy-admissible root selection executed successfully, produced the expected shock dynamics, and generated the requested ground-truth data and visualization. No further numerical refinement is warranted.

## Section 2: Quantitative Scorecard

| Category | Score | Justification |
|---|---:|---|
| **Score Integrity** | **50/50** | Completed without crashes or non-finite values. Both `summary_all.png` and `burgers.npz` were written and verified. |
| **Score Accuracy** | **20/20** | Exact characteristic evaluation with correct value range, shock trajectory, and post-shock states. |
| **Score Details** | **15/15** | Smooth pre-shock evolution, no Gibbs ringing, and a razor-sharp entropy shock at \(t=2\). The planar reduction is highly efficient. |
| **Score Optimality** | **15/15** | An exact analytical method is optimal for PINN ground truth and correctly takes priority over DG, spectral, or finite-volume approximation. |
| **TOTAL SCORE** | **100/100** | |

## Section 3: Forensic Analysis & Feedback

### Observation

- The four snapshots show the correct qualitative progression:
  - **\(t=0\):** smooth diagonal sinusoidal planar wave with two periods over the domain.
  - **\(t=0.75\):** nonlinear steepening without oscillations.
  - **\(t=1.5\):** gradient catastrophe at shock-formation time.
  - **\(t=2\):** a sharp diagonal entropy shock, including its periodic counterpart, with no visible numerical smearing or ringing.
- Diagonal planar symmetry and doubly periodic behavior are preserved.
- The logged value range is exactly
  \[
  u\in[-0.3333333333333333,\;1.0],
  \]
  matching the theoretical range \([-1/3,1]\).
- At \(t=2\), the reported shock location is
  \[
  s_{\mathrm{sh}}=7.616518640512919
  =2\pi+\frac{2}{3}(2),
  \]
  with maximum shock-position error \(8.881784197001252\times10^{-16}\), effectively machine precision.
- The shock states \(u_L=0.9711823879738963\) and \(u_R=-0.30451572130722976\) satisfy \(u_L+u_R\simeq2/3\), and their jump is approximately \(1.276\), consistent with the analytical prediction.
- The saved tensor has the requested 21 time slices and shape `(21, 256, 256)`.

### Requested Error-Log Check

The exact string **“Minimum achieved error:” does not appear** in the supplied stdout tail, and no \(L^2\) or \(L^\infty\) solution-error norm is reported. It should therefore **not** be assigned a fabricated numerical value.

The available diagnostics are instead:

- **Maximum discrete mean error:** `0.0016822094254394515`
- **Maximum shock-position error:** `8.881784197001252e-16`

The small discrete mean discrepancy is consistent with sampling/quadrature of a discontinuous profile on the finite output grid. It is not evidence of time-integration drift or conservation error in a numerical PDE evolution, because the solution is evaluated analytically at each output time.

### Diagnosis

The run exhibits the correct characteristic steepening and entropy-shock solution. There is no numerical dissipation, dispersion error, or Gibbs phenomenon to diagnose. The \(256\times256\) grid determines output sampling and visual resolution, not the underlying analytical solution’s accuracy.

### Targeted Improvements

- **For Integrity:** None.
- **For Accuracy:** None.
- **For Details:** None.
- **For Optimality:** None. Do not replace the exact method with a numerical shock-capturing scheme.

**Recommendation:** Approve this run and use `burgers.npz`/`usol` as the PINN ground-truth dataset.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 2 | The implementation uses the exact Method of Characteristics for the 2D inviscid Burgers diagonal planar-wave problem. It defines U0(s)=1/3+2/3 sin(s/2), builds the characteristic map F_t(s)=s+2tU0(s), and solves the implicit characteristic relation by bisection. Pre-shock it inverts the monotone map; post-shock it selects the entropy-admissible branch using the equal-area shock half-width equation sin(q)/q=3/(2t) with q=δ/2 and shock location s_sh=2π+(2/3)t mod 4π. The grid is 256×256 on the periodic domain [0,4π]^2, time span [0,2], with dt_save=0.1. Exact snapshots are generated at t=0.0, 0.75, 1.5, and 2.0. Visualization saves summary_all.png as a 1×4 heatmap dashboard of u with colorbar=false. Internal checks verify value range, conserved mean, shock-position law, and u_L+u_R=2/3. | N/A (no solution-error norm reported; max discrete mean error 0.0016822094254394515; max shock-position error 8.881784197001252e-16) | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 2 | Saves burgers.npz in NPZ format with keys: 't' shape (21,), 'x' shape (256,), 'y' shape (256,), and 'usol' shape (21,256,256) ordered as (time,x,y). Also saves summary_all.png containing four u-field snapshots at the specified visualization times. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/IB/run1//InviscidBurgers_Periodic_Sin_x_plus_y_over_2/2026_11_06_0206/` |


---
