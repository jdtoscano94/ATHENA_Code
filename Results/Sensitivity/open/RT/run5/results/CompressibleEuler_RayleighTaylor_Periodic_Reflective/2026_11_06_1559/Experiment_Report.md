
# Experiment Report
*Last Updated: 2026-11-06 16:23:23*

## User Request
Let's solve the 2D Compressible Euler equations for the Rayleigh-Taylor Instability.

**Physics Parameters:**
* **Equations:** Compressible Euler ($\gamma = 1.4$).
* **Gravity:** $g = 1.0$ acting in the **positive $y$-direction** (The source term adds momentum in $+y$).
* **Inviscid:** $\mu = 0$, thermal conductivity $k = 0$.

**Domain:**
$(x,y) \in [0, 0.25] \times [0, 1.0]$
Time interval: $t \in [0, 3.0]$

**Boundary Conditions:**
* $x$-direction: **Periodic**.
* $y$-direction: **Reflective / Free-slip walls** at $y=0$ and $y=1$ (No-penetration condition $v=0$).

**Initial Conditions:**
The domain is split at the interface $y=0.5$.
* **Density ($\rho$):** A smoothed interface between $\rho_{heavy} = 2.0$ (bottom) and $\rho_{light} = 1.0$ (top).
    $$ \rho(y) = \rho_{heavy} + (\rho_{light} - \rho_{heavy}) \cdot \frac{1}{2}\left(1 + \tanh\left(50(y - 0.5)\right)\right) $$
* **Pressure ($p$):** Hydrostatic balance ($\frac{dp}{dy} = \rho g$).
    $$ p(y) = \begin{cases} 2y + 1 & \text{if } y < 0.5 \\ y + 1.5 & \text{if } y \geq 0.5 \end{cases} $$
* **Velocity:** Initialized with zero horizontal velocity ($u=0$) and a perturbation in vertical velocity ($v$):
    $$ v(x,y) = -0.025 \cdot c \cdot \cos(8\pi x) \cdot \sin(\pi y)^6 $$
    Where $c = \sqrt{\gamma p / \rho}$ is the local speed of sound.

**Output:**
Create an appropiate Figure and saved it as summary_all.png

---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
# Scientific Strategy Report — Run 7 (Verification-Only Refinement)

**1. Problem Statement**

This is a **refinement iteration** of the single-mode compressible Rayleigh–Taylor Instability (RTI). We solve the 2D inviscid compressible Euler equations with a gravitational source ($g = +1.0$ in $y$) on $[0,0.25]\times[0,1]$, $t\in[0,3]$. The heavy fluid ($\rho=2$) occupies $y<0.5$, which lies "above" the light fluid ($\rho=1$) in the effective gravity frame (gravity points $+y$) — an unstable configuration with Atwood number $A = 1/3$ — seeded by a single-mode vertical velocity perturbation.

The prior run scored **99/100** and the Advisor confirmed **no physical, stability, or resolution defect**. The sole outstanding deficiency is an **incomplete verification/audit trail** (missing raw accepted-step audit output; skipped direct Stage B field comparison due to absent archives). Therefore this run is a **verification-only refinement**: identical physics, identical frozen Stage B discretization, with a completed timestep audit and archived baseline states.

**1. Mathematical Specification (Code-Ready Extraction)**

* **Governing Equations:** 2D Compressible Euler with gravitational source:
$$
\frac{\partial}{\partial t}\begin{pmatrix} \rho \\ \rho u \\ \rho v \\ \rho E \end{pmatrix}
+\frac{\partial}{\partial x}\begin{pmatrix} \rho u \\ \rho u^2 + p \\ \rho u v \\ u(\rho E + p) \end{pmatrix}
+\frac{\partial}{\partial y}\begin{pmatrix} \rho v \\ \rho u v \\ \rho v^2 + p \\ v(\rho E + p) \end{pmatrix}
=\begin{pmatrix} 0 \\ 0 \\ \rho g \\ \rho g v \end{pmatrix},
\qquad g = 1.0
$$
with equation of state and total energy:
$$
\rho E = \frac{p}{\gamma - 1} + \frac{1}{2}\rho\left(u^2 + v^2\right), \qquad \gamma = 1.4
$$

* **Domain:** $(x,y) \in [0,\, 0.25] \times [0,\, 1.0]$, $\quad t \in [0,\, 3.0]$

* **Boundary Conditions:**
  * $x$-direction ($x=0$, $x=0.25$): **Periodic**
  * $y$-direction ($y=0$, $y=1$): **Reflective / free-slip walls** (no-penetration $v=0$; mirror reflection of normal momentum)

* **Initial Conditions** (exact, every field):
$$
\rho(x,y) = 2.0 + (1.0 - 2.0)\cdot\frac{1}{2}\Big(1 + \tanh\big(50\,(y - 0.5)\big)\Big)
$$
$$
p(y) = \begin{cases} 2y + 1, & y < 0.5 \\ y + 1.5, & y \geq 0.5 \end{cases}
$$
$$
u(x,y) = 0
$$
$$
v(x,y) = -0.025 \cdot c(y) \cdot \cos(8\pi x)\cdot\big[\sin(\pi y)\big]^6,
\qquad c(y) = \sqrt{\frac{\gamma\, p(y)}{\rho(y)}}
$$
$$
\rho E(x,y) = \frac{p(y)}{\gamma - 1} + \frac{1}{2}\,\rho(y)\left(u^2 + v^2\right) = \frac{p(y)}{0.4} + \frac{1}{2}\,\rho(y)\,v(x,y)^2
$$

* **Constants & Parameters:**
  * $\gamma = 1.4$
  * Gravity vector $\mathbf{g} = [0,\, +1.0]^T$ (source adds momentum in $+y$; energy source $\rho g v$)
  * $\rho_{heavy} = 2.0$, $\rho_{light} = 1.0$; interface center $y_i = 0.5$; smoothing steepness $50$
  * Perturbation amplitude $0.025$; mode $\cos(8\pi x)$ (single mode, wavelength $\lambda = 0.25$ = domain width)
  * $\mu = 0$, thermal conductivity $k = 0$ (inviscid — no viscous/heat fluxes)
  * Atwood number $A = (\rho_h - \rho_l)/(\rho_h + \rho_l) = 1/3$
  * Hydrostatic consistency: $\frac{dp}{dy} = \rho g$ holds exactly in each half-domain ($2 = 2\cdot 1$ below, $1 = 1\cdot 1$ above); $p$ continuous at $y=0.5$ ($p=2.0$)

* **Expected Physics:** Effective gravity points $+y$; the heavy fluid is gravitationally "on top" → unstable. Spike descends toward $+y$ (upper wall), bubbles rise toward $-y$. Timeline (confirmed by prior runs): $t\approx0.75$ smooth single-mode interface displacement (linear growth); $t\approx1.5$ developed heavy-fluid spike with adjacent light-fluid bubbles; $t\approx2.25$ mushroom-head formation and vortex roll-up; $t=3.0$ secondary Kelvin–Helmholtz filaments, extensive mixing, wall interaction, possible weak compressive shocks (spike/wall interaction). Pressure remains predominantly smooth hydrostatic stratification ($p: 1 \to 2.5$).

**2. Strategic Methodology**

* **Can we obtain an exact solution? No.**
  * **Method of Characteristics:** Not feasible. MoC yields closed-form/implicit solutions only for 1D hyperbolic systems or simple-wave structures. This problem is genuinely multi-dimensional and nonlinear: the perturbed interface deposits baroclinic vorticity ($\nabla\rho \times \nabla p \neq 0$), undergoes vortex roll-up, and develops secondary KH instabilities and thin filaments. Characteristics intersect and entangle in 2D; no reducible implicit characteristic system exists past $t=0$. The only exact fact is the unperturbed hydrostatic equilibrium ($dp/dy = \rho g$), which serves as a $t=0$ consistency check, not a solution.
  * **Fourier Spectral:** Inappropriate — non-periodic in $y$ (slip walls) and the solution develops contact discontinuities and filaments → global Gibbs phenomenon.
  * **Separation of variables / Green's functions:** Not tractable for nonlinear multi-D evolution.
  * **Conclusion:** A numerical method is mandatory. The Advisor explicitly confirmed no architecture override is warranted.

* **Selection: Discontinuous Galerkin Spectral Element Method in 2D plus time.** It is the only toolbox method that simultaneously provides: (i) mixed periodic/wall boundary handling, (ii) hybrid DG–FV shock capturing for late-time compressive features, (iii) local AMR for the thin KH filaments, (iv) high-order accuracy in the smooth stratified bulk, and (v) robust treatment of the gravitational source on the hydrostatic background.

* **Run context:** The Advisor's diagnosis is definitive — *no physical stability or resolution defect*; the discretization is **frozen** (Stage B). This run changes **nothing** in the physics or numerics; it only completes the audit trail and archives the baseline.

**3. Implementation Requirements (CRITICAL)**

* **Configuration:** Use the **frozen Stage B configuration exactly** (Advisor-mandated retention — do NOT alter the discretization): `polydeg=6`, CFL $= 0.30$, HLLC surface flux, pressure-based Hennemann–Gassner shock capturing (DG volume flux `ranocha`, FV flux `lax_friedrichs`), density-based Löhner AMR with `ControllerThreeLevel` (base=4, med=5 @ 0.1, max=6 @ 0.3, interval=20, `adapt_initial_condition=true`), $t\in[0,3]$. Retain the existing time-integration configuration unchanged. No resolution changes are justified.

* **3.0.1 Adaptive Mesh Refinement & Shock Capturing — Variable Selection Analysis:**
  * **AMR sensor: Density ($\rho$).** *Signal-to-noise reasoning:* Every structural feature of interest in RTI — the material interface, descending spike, rising bubbles, mushroom roll-up, and KH filaments — **is** a density feature. Density remains sharp and localized at these features throughout the evolution (bounded $1 \to 2$), giving a high signal-to-noise ratio for refinement. Velocity, in contrast, becomes globally chaotic in the mixing zone at late times (vortex roll-up fills the domain with broad velocity gradients) — refining on it would trigger near-uniform refinement and waste resources. Pressure is dominated by the smooth hydrostatic background and under-tracks the interface. **Density is strictly superior for AMR on this PDE.**
  * **Shock-capturing sensor: Pressure ($p$).** The features requiring FV stabilization (late-time compression, spike–wall interaction, acoustic transients) are pressure-driven shocks. Using pressure avoids falsely flagging the density contact (a slip/contact discontinuity, not a shock), preserving high-order DG along the interface. This division of labor — density → AMR (structure tracking), pressure → shock capturing (compressive stabilization) — is retained per the Advisor.

* **3.1 Advisor-Mandated Adjustments (Overrides):**
  * **Verification-only refinement — do NOT alter the discretization.** The final validation block MUST print:
    1. Total accepted steps.
    2. Minimum non-final accepted $dt$.
    3. `count_nonfinal_dt_lt_1e-8` — must be **exactly zero**.
    4. Final-step $dt$, separately identified.
    5. An explicit **`TIMESTEP_AUDIT: PASS/FAIL`** line.
  * **Baseline archiving:** Archive the raw accepted-step times, the **full Stage B states** (all conservative variables at the scheduled snapshot times), and the rendered fields, so the next run can perform a **direct field comparison** rather than relying on approximate scalar references.
  * **Direct comparison if available:** If archived Stage B states from a prior run exist, load them and perform the direct rendered-field comparison (this was skipped last run due to absent archives).
  * Retain `polydeg=6`, CFL $=0.30$, HLLC, and the existing AMR/shock-capturing settings. Retain the existing diagnostic suite (mass drift, positivity $\min\rho,\min p$, wall $|v|$, symmetry residual) unchanged.

**4. Visualization Strategy**

* **Field Selection (Analysis):** Plot exactly **2 primary fields**:
  * **Density ($\rho$)** — the primary driver: RTI dynamics are entirely defined by the density contrast; interface, spike, bubbles, mushrooms, and filaments are all density structures.
  * **Pressure ($p$)** — secondary: captures the hydrostatic stratification and late-time compressible structure.
  * Do **not** plot $u, v, E$ (subplots become unreadable; velocity is noisy at late times). Do not compute derived quantities (no vorticity/Schlieren).
* **Task:** Plot $\rho$ (top row) and $p$ (bottom row) at **4 time steps**: $t = 0.75,\ 1.5,\ 2.25,\ 3.0$.
* **Layout:** Grid — Rows = Selected Fields ($\rho$, $p$), Columns = Time (2 × 4).
* **Saving Data:** Save the **full solution state** (all variables) at frequent intervals (`dt_save = 0.1`) for Ground Truth, and archive full conservative states at the scheduled snapshot times ($t = 0, 0.75, 1.5, 2.25, 3.0$) per the Advisor.
* **Constraints:**
  * Save as: `summary_all.png` (strictly).
  * **CRITICAL: DISABLE COLORBARS** (`colorbar=false`) — Julia.
  * Automatic per-panel color scaling; no fixed colormaps or clims.

**5. Cross-Validation & Data Integration (Conditional)**

* **Trigger check:** [USER_QUESTION] contains **no external data paths** (no sensors, no reference `.npz` files) → standard cross-validation **not triggered**.
* **Advisor-directed substitute:** If archived Stage B states from a prior run are found on disk, load them and compute the direct field comparison (relative $L_2$ error on the rendered fields vs. the current run). If absent (as in the last run), run the simulation and **archive** the current full states + rendered fields as the baseline for the next run's direct comparison.

**REQUIRED COMPONENTS**

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
* Shock Capturing Indicator (pressure-based)
* Adaptive Mesh Refinement (density-based)
* Accepted-Step Timestep Audit (Advisor-mandated, verification-only)
* Baseline State Archiving (Advisor-mandated)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

* **Status:** **APPROVED**
* **Reason:** Perfect score achieved. The simulation completed successfully, the requested figure and archives were produced, all reported validation checks passed, and the plots show the expected nonlinear Rayleigh–Taylor evolution without a visible numerical defect. Further refinement would add computational cost without evidence of improved accuracy.

## Section 2: Quantitative Scorecard

* **Score Integrity:** **50/50**
* **Score Accuracy:** **20/20**
* **Score Details:** **15/15**
* **Score Optimality:** **15/15**
* **TOTAL SCORE:** **100/100**

## Section 3: Forensic Analysis & Feedback

### Observation

**Qualitative plot assessment:**

- At \(t=0.75\), the density interface shows a smooth, single-mode displacement while pressure remains predominantly hydrostatically stratified.
- At \(t=1.5\), a coherent heavy-fluid spike and adjacent bubble structure have developed.
- At \(t=2.25\), the spike forms a mushroom-shaped head with visible vortex roll-up.
- At \(t=3.0\), secondary Kelvin–Helmholtz-like filaments and extensive mixing are visible, together with interaction near the upper wall. These small-scale structures remain well defined; there is no obvious domain-wide ringing, blank panel, or non-physical blow-up.
- Pressure remains mostly stratified, with localized late-time variations consistent with compressible motion and mixing. Its mottled late-time appearance is not, by itself, evidence of numerical oscillation.

**Log assessment:**

- The run reached \(t=3\) and ended with `Done.`; both `TIMESTEP_AUDIT` and `STAGE B VERIFICATION` report **PASS**.
- The raw accepted-step audit is complete:
  - Total accepted steps: **38,087**
  - Minimum non-final accepted \(dt\): **\(3.9336147810153932\times10^{-6}\)**
  - Non-final steps with \(dt<10^{-8}\): **0**
  - Final-step \(dt\): **\(2.8361614368588306\times10^{-5}\)**
- Relative mass drift is only **\(6.804631\times10^{-12}\)** in magnitude, indicating excellent discrete mass conservation.
- Positivity is maintained: \(\min\rho=0.3444687\) and \(\min p=1.207574\). The density reduction below its initial minimum is compatible with compressible expansion and is not a positivity failure.
- Wall-normal velocity maxima are small: \(5.69\times10^{-5}\) at \(y=0\) and \(2.10\times10^{-4}\) at \(y=1\).
- The largest corrected parity-aware symmetry residual is **\(1.386427\times10^{-3}\)**, below the \(10^{-2}\) threshold; the fixed-mesh symmetry contingency was therefore correctly not triggered.
- Thirty-three full-state snapshots and four rendered field sets were archived. A prior baseline was unavailable, so a direct baseline-to-baseline field comparison could not be performed; the conditional requirement was handled by archiving this run as the baseline.

### Error Check

The supplied stdout tail contains **no** line labeled `Minimum achieved error:`, so no such value can be extracted.

**Quantitative error is not applicable as no ground truth is available.** The mass drift and symmetry residual are conservation and symmetry diagnostics, not solution errors against an analytical reference. The unperturbed hydrostatic state provides only an initial-equilibrium check, not an exact solution of the nonlinear time-dependent RTI problem.

### Diagnosis

No remaining failure phenomenon is identified. The previous deficiency was an incomplete verification/audit trail rather than a physical or numerical defect; the present run supplies the raw timestep audit and baseline archives.

The frozen Stage B method remains methodologically appropriate:

- **HLLC** is suitable for Euler contact and shock structures.
- **Pressure-based shock capturing** targets compressive features without unnecessarily applying first-order stabilization to every density interface.
- **Density-based AMR** efficiently tracks spikes, bubbles, and filaments.
- **DGSEM with \(N=6\)** provides high-order accuracy in smooth regions, while the hybrid DG–FV formulation supplies robustness where needed.
- The observed 38,087 steps and small late-time steps are consistent with the cost of resolving the developing small-scale dynamics, not evidence of a timestep collapse.

An exact analytical method is not feasible for the full nonlinear, multidimensional evolution with baroclinic vorticity generation, mushroom roll-up, and secondary instabilities. Consequently, no numerical-to-exact architecture override is warranted.

### Targeted Improvements

- **For Integrity:** None.
- **For Accuracy:** None.
- **For Details:** None. No further mesh refinement or polynomial-degree increase is justified by the evidence.
- **For Optimality:** None.

**Recommendation:** Freeze the Stage B configuration and use the archived full conservative-state snapshots—not the rendered PNG—as the PINN ground-truth dataset. Further refinement iterations would be redundant.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | The code implements the 2D compressible Euler equations with gravitational source using Trixi.jl. The spatial discretization is DGSEM with polynomial degree 3 on a P4estMesh (domain [0,0.25]×[0,1], periodic in x, reflective slip walls in y via boundary_condition_slip_wall). Shock capturing is density-based using IndicatorHennemannGassner with a hybrid DG-FV volume integral (flux_ranocha for DG, flux_lax_friedrichs for FV) and HLL surface flux. AMR is density-based using IndicatorLöhner with ControllerThreeLevel (base_level=4, med_level=5, max_level=6) and AMRCallback interval=20. The initial condition exactly matches the strategy: tanh density profile, piecewise hydrostatic pressure, zero u, and the specified single-mode v perturbation. Gravity source is (0,0,ρg,ρgv) with g=1.0. Time integration uses RDPK3SpFSAL49 with adaptive=false, StepsizeCallback cfl=0.5, and tspan=(0.0,3.0). Visualization produces summary_all.png with density (top row) and pressure (bottom row) at t=0.75, 1.5, 2.25, 3.0, with colorbars disabled and automatic color scaling. | N/A | 47 | 18 | 11 | 13 | **89** |
| 2 | The implementation faithfully reproduces the DGSEM solution of the 2D inviscid compressible Euler equations with a gravitational source for the single-mode Rayleigh–Taylor instability. Governing equations use CompressibleEulerEquations2D(γ=1.4); the initial condition is the verbatim strategy profile (smoothed density 2→1 via tanh(50(y-0.5))·0.5, piecewise pressure 2y+1 / y+1.5 at y=0.5, u=0, v=-0.025·c·cos(8πx)·sin^6(πy) with c=sqrt(γp/ρ), and energy via prim2cons giving E=p/(γ-1)+0.5ρv²). The gravity source term returns SVector(0,0,g·ρ, g·ρv), i.e. (0,0,ρg,ρgv). Discretization: P4estMesh on [0,0.25]×[1.0] with x-periodic (periodicity=(true,false)) and slip-wall BCs on :y_neg/:y_pos; DGSEM with polydeg=3, LobattoLegendreBasis. Advisor-mandated upgrades are present: surface_flux=flux_hllc, pressure-based IndicatorHennemannGassner shock capture (pressure_indicator derived from conservative state) feeding VolumeIntegralShockCapturingHG with volume_flux_dg=flux_ranocha and volume_flux_fv=flux_lax_friedrichs, while AMR uses a separate density-based IndicatorLöhner (variable=density_indicator=u[1]) with ControllerThreeLevel (base 4 / med 5 @0.1 / max 6 @0.3), interval=20, adapt_initial_condition=true. Held-fixed config is respected: RDPK3SpFSAL49 time integrator, adaptive=false, StepsizeCallback(cfl=0.5), tspan=(0.0,3.0). The full diagnostic suite is implemented and printed at every save: total mass (Trixi.integrate) with absolute/relative drift vs t=0, min/max ρ and p (positivity), dt_min/dt_max/current dt (CFL), max|v| at y=0 and y=1 (wall no-penetration) and an approximate symmetry residual ‖q-q(0.25-x)‖/‖q‖ about x=0.125. Visualization renders a clean 2×4 grid (density top row, pressure bottom row) at t=0.75,1.5,2.25,3.0 with colorbar=false, no mesh lines (framestyle=:none, axis=nothing), per-panel automatic scaling, and higher resolution size=(3200,2400) saved as summary_all.png. | N/A | 47 | 19 | 12 | 15 | **93** |
| 3 | Implementation is a numerical DGSEM solver for the 2D compressible Euler equations with gamma=1.4 and the prescribed gravity source. Domain is [0,0.25] x [0,1] with t in [0,3], periodic x-boundaries, and slip walls at y=0 and y=1. Initial conditions match the strategy: smoothed tanh density interface at y=0.5, piecewise hydrostatic pressure, u=0, v=-0.025*c*cos(8*pi*x)*sin(pi*y)^6, and total energy through prim2cons. Spatial discretization uses polydeg=5 Lobatto basis, HLLC surface flux, pressure-based Hennemann-Gassner shock capturing with flux_ranocha DG and flux_lax_friedrichs FV, and density-based Lohner AMR with ControllerThreeLevel base=4/med=5/max=6, interval=20, adapt_initial_condition=true. Time integration uses RDPK3SpFSAL49 with adaptive=false, tspan=(0.0,3.0), and CFL=0.35 via StepsizeCallback. Visualization samples the DG solution onto a uniform Cartesian grid and saves summary_all.png as a 2x4 heatmap grid of density and pressure at t=0.75,1.5,2.25,3.0 with colorbar=false. The code also performs the required t=0 renderer x-independence test and prints the final diagnostic summary for mass drift, positivity, dt extrema, wall |v|, and symmetry residual. | N/A | 49 | 19 | 14 | 15 | **97** |
| 4 | Numerical implementation of the 2D inviscid compressible Euler Rayleigh-Taylor instability using DGSEM. Stage A uses polydeg=5, CFL=0.35, HLLC surface flux, pressure-based Hennemann-Gassner shock capturing with flux_ranocha DG and flux_lax_friedrichs FV, density-based Löhner AMR with ControllerThreeLevel 4/5/6 at thresholds 0.1/0.3 and interval=20, RDPK3SpFSAL49 with adaptive=false, and tspan=(0.0,3.0) on [0,0.25]x[0,1] with periodic x and slip y walls. Stage B repeats the same configuration with polydeg=6 and CFL=0.30 as the single non-repeating resolution check. The code regenerates the polydeg=5 baseline, computes both stages on the identical 512x2048 cell-centered comparison grid at t=2.25 and t=3.0, evaluates the M1/M2/M3 materiality gate, prints the explicit RESOLUTION CHECK verdict, and selects summary_all.png from the already-rendered Stage A or Stage B 2x4 rho/p panel figure. It also implements the raw accepted-step timestep audit, per-variable mirror-indexed symmetry residuals at t=0,0.75,1.5,2.25,3.0, mass drift/positivity/wall diagnostics, and the conditional fixed-mesh symmetry control run. The initial conditions, pressure law, gravity source term, boundary topology, and required diagnostics match the strategy. | N/A | 50 | 19 | 15 | 15 | **99** |
| 5 | Implementation is a 2D DGSEM plus explicit time integration for the compressible Euler equations with gamma=1.4 and a +y gravitational source. Domain is [0,0.25]x[0,1], t in [0,3]. Stage B is selected: polydeg=6, CFL=0.30, HLLC surface flux, pressure-based Hennemann-Gassner shock capturing, density-based Löhner AMR with ControllerThreeLevel base=4 med=5 max=6, thresholds 0.1/0.3, interval=20. Initial conditions match the strategy exactly: tanh density interface at y=0.5, piecewise hydrostatic pressure, zero u, v=-0.025 sqrt(gamma p/rho) cos(8*pi*x) sin(pi*y)^6, and prim2cons energy. y-walls are slip/reflective; x is periodic. The code first attempts to load archived Stage B npz full states at t=0,0.75,1.5,2.25,3.0; if absent it reproduces the identical Stage B configuration. It computes parity-aware reflection residuals on a cell-centered 512x2048 grid: even for rho, rho v, rho E; odd for rho u, with zero-norm absolute fallback. It prints residuals and standard diagnostics (mass drift, min rho/p, wall |v|), applies a connected-component minimum-area filter, and triggers a fixed-mesh control only if corrected t=3 max residual exceeds 1e-2. It renders rho (top row) and p (bottom row) at the four requested times with colorbar=false and copies the Stage B figure to summary_all.png. | N/A | 49 | 20 | 15 | 15 | **99** |
| 12 | The code implements the prescribed 2D inviscid compressible Euler RTI simulation with DGSEM (Lobatto-Legendre, polydeg=6 in the executed Stage B run), HLLC surface flux, pressure-based Hennemann-Gassner shock capturing (DG flux ranocha, FV flux lax_friedrichs), density-based Löhner AMR with ControllerThreeLevel (base=4, med=5@0.1, max=6@0.3, interval=20, adapt_initial_condition=true), RDPK3SpFSAL49 with adaptive=false and CFL=0.30 over tspan=(0,3). The initial conditions match the strategy exactly: tanh density interface at y=0.5, piecewise hydrostatic pressure, u=0, v=-0.025*c*cos(8*pi*x)*sin(pi*y)^6, and gravity source (0,0,rho*g,rho*g*v). Boundary conditions are periodic in x and slip walls in y. Event-time de-duplication and roundoff-safe step scheduling are implemented, and the raw accepted-step audit enforces zero non-final steps with dt<1e-8 and prints PASS/FAIL. Diagnostics include mass drift, positivity, wall no-penetration, parity-aware symmetry residual, and baseline comparison. The visualization produces the required 2x4 rho/p panels at t=0.75,1.5,2.25,3.0 with colorbars disabled and saves summary_all.png. | N/A | 49 | 20 | 15 | 15 | **99** |
| 14 | The implementation is a numerical 2D DGSEM solver for the inviscid compressible Euler equations with γ=1.4, solved on [0,0.25]×[0,1] over t∈[0,3]. Stage B uses polydeg=6, CFL=0.30, HLLC surface flux, pressure-based Hennemann–Gassner shock capturing with DG volume flux_ranocha and FV flux_lax_friedrichs, and density-based Löhner AMR with ControllerThreeLevel base=4, med=5 at 0.1, max=6 at 0.3, interval=20, adapt_initial_condition=true. Time integration uses RDPK3SpFSAL49 with adaptive=false and StepsizeCallback. The initial conditions exactly match the strategy: smoothed density interface, piecewise hydrostatic pressure, zero u, single-mode vertical velocity perturbation with c(y)=sqrt(γp/ρ), and prim2cons energy. Gravity source is (0,0,ρg,ρgv) with g=+1.0. x-boundaries are periodic and y-boundaries are slip walls. The code includes the advisor-mandated accepted-step audit, raw accepted-step history archiving, full conservative state snapshots at 0.1 intervals plus scheduled snapshot times, rendered ρ/p field archiving, baseline comparison when archives exist, and a 2×4 summary_all.png figure with colorbar disabled. | N/A | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | NPZ files are saved in directory `data_npz` with filenames `time_%08.5f.npz` at dt_save=0.1 and additionally at the visualization times 0.75, 1.5, 2.25, 3.0. Each NPZ file contains keys: `t` (scalar Float64), `x` (1D array of node coordinates, length N_nodes(t)), `y` (1D array, length N_nodes(t)), `rho` (1D array, length N_nodes(t)), `rho_u` (1D array), `rho_v` (1D array), `rho_e` (1D array), and `p` (1D array, length N_nodes(t)). N_nodes varies with AMR. Additionally, `SaveSolutionCallback` writes HDF5 solution files to `trixi_output` at dt=0.1 with full conservative state. The plot `summary_all.png` is saved in the working directory, containing a 2×4 grid: top row density, bottom row pressure at the four specified times, with colorbars disabled. |
| 2 | Per-snapshot .npz files (e.g. time_00.75000.npz, time_01.50000.npz, time_02.25000.npz, time_03.00000.npz) written to <script_dir>/data_npz/ via npzwrite. Each file contains a Dict with keys: 't' (scalar Float64), 'x' (Float64 array of DG solution-node x-coordinates from PlotData2D), 'y' (Float64 array of node y-coordinates), 'rho' (Float64 array of DG-node conserved density), 'rho_u' (Float64 array, x-momentum ρu), 'rho_v' (Float64 array, y-momentum ρv), 'rho_e' (Float64 array, total energy ρE), 'p' (Float64 array, pressure from cons2prim). All field arrays (rho, rho_u, rho_v, rho_e, p, x, y) are structured on the same DG-node grid (per-element nodes × elements; flattened with vec() by the rasterizer during plotting). Save times = union of {0.0,0.1,...,3.0} plus the four visualization times {0.75,1.5,2.25,3.0}, ensuring dt_save=0.1 cadence and exact plot snapshots. Additionally, SaveSolutionCallback writes full conservative state (cons2cons) HDF5 snapshots at dt=0.1 (plus initial) into <script_dir>/trixi_output/. Final figure summary_all.png (2 rows × 4 columns, 3200×2400) is saved to <script_dir>/. |
| 3 | Saves NPZ files under data_npz named time_%08.5f.npz (e.g., time_00.00000.npz, time_00.10000.npz, ..., time_03.00000.npz) at times formed from 0.0:0.1:3.0 plus [0.75,1.5,2.25,3.0]. Each NPZ contains keys: 't' as a scalar Float64; 'x' and 'y' as Trixi PlotData2D element-block arrays; and 'rho','rho_u','rho_v','rho_e','p' as PlotData2D element-block arrays matching the x/y layout. In expected block form these are (polydeg+1, polydeg+1, n_elements) = (6,6,n_elements) or a flattened 2D equivalent, with n_elements varying under AMR. The full conservative state is saved via rho, rho_u, rho_v, rho_e, and pressure p is also exported. Additionally, SaveSolutionCallback writes full conservative snapshots to trixi_output at dt=0.1 including the initial solution. Summary visualization is saved as summary_all.png: a 2x4 heatmap grid with density in the top row and pressure in the bottom row, colorbars disabled. |
| 4 | Stage A writes into data_npz/ and trixi_output/; Stage B writes into stage_b_polydeg6/data_npz/ and stage_b_polydeg6/trixi_output/. NPZ full-state snapshots are written as data_npz/time_%08.5f.npz for Stage A and stage_b_polydeg6/data_npz/time_%08.5f.npz for Stage B, containing keys: t (scalar), polydeg (scalar Int), x, y, rho, rho_u, rho_v, rho_e, p. The x/y/rho/rho_u/rho_v/rho_e/p arrays are DG block arrays with shape normally (polydeg+1, polydeg+1, n_elements), where n_elements is AMR/time-dependent; Stage A has first two dimensions 6x6 and Stage B has 7x7. Fixed-grid comparison files are data_npz/fixed_grid_time_02.25000.npz, data_npz/fixed_grid_time_03.00000.npz, and the same under stage_b_polydeg6/data_npz/. They contain keys: t (scalar), x (vector length 512), y (vector length 2048), rho (2048x512 matrix), p (2048x512 matrix), mixing_zone_count (scalar Int), mixing_zone_area (scalar Float64), sharpness (scalar Float64), structural_count (scalar Int), upper_mixing_component_count (scalar Int). Raw timestep histories are saved as data_npz/accepted_step_history.npz and stage_b_polydeg6/data_npz/accepted_step_history.npz when steps exist, with keys: t, tprev, dt, iter, amr_event (all vectors of length N accepted steps; amr_event is Bool), initialization_t (scalar), initialization_dt (scalar). Trixi SaveSolutionCallback also writes full conservative-state HDF5 solution snapshots at dt=0.1 into trixi_output/ and stage_b_polydeg6/trixi_output/. Figures saved are summary_stage_a_polydeg5.png for Stage A, stage_b_polydeg6/summary.png for Stage B, and the selected production panel copied to summary_all.png; all are 2x4 rho/p PNG panels at t=0.75,1.5,2.25,3.0 with size 3200x2400. If the conditional symmetry control triggers, it additionally writes the control run's time_*.npz snapshots and Trixi HDF5 output under symmetric_fixed_control_polydeg<deg>_level<level>/. |
| 5 | Saves under stage_b_polydeg6/data_npz. For a live/reproduced Stage B, time_%08.5f.npz at t=0.0:0.1:3.0 plus 0.75,1.5,2.25,3.0 contains keys 't' scalar, 'polydeg' scalar, and 'x','y','rho','rho_u','rho_v','rho_e','p' as matching 3D DG element-block arrays, typically shape (polydeg+1, polydeg+1, n_elements) = (7,7,n_elements) for Stage B; n_elements varies with AMR. At t=2.25 and 3.0 it also writes fixed_grid_time_%08.5f.npz with keys 't' scalar, 'x' length 512, 'y' length 2048, 'rho' 2048x512, 'p' 2048x512, and scalar metrics 'mixing_zone_count','mixing_zone_area','sharpness','structural_count','upper_mixing_component_count'. If accepted steps are available it writes accepted_step_history.npz with vector keys 't','tprev','dt','iter','amr_event' (length N accepted steps) and scalar keys 'initialization_t','initialization_dt'. Trixi SaveSolutionCallback also writes full cons2cons state at dt=0.1 to stage_b_polydeg6/trixi_output. The 2x4 rho/p figure is saved to stage_b_polydeg6/summary.png and copied to summary_all.png. When using the archived Stage B, existing time_*.npz snapshots are read for post-processing; new time_*.npz files are not rewritten, but the fixed-grid metric npz files and figures are regenerated. |
| 12 | Primary snapshots are written to stage_b_polydeg6_event_fixed/data_npz as time_%08.5f.npz (NPZ) at t=0, 0.1, 0.2, ..., 3.0 merged/de-duplicated with 0.75,1.5,2.25,3.0. Each snapshot contains keys: 't' (scalar Float64), 'polydeg' (scalar Int), 'x' (DG element-block array, shape (7,7,N_t) for polydeg=6), 'y' (same shape as x), 'rho' (same shape), 'rho_u' (same shape), 'rho_v' (same shape), 'rho_e' (same shape), and 'p' (same shape). N_t is the number of active elements at that save time and varies with AMR. The timestep audit writes accepted_step_history.npz with keys 't', 'tprev', 'dt', 'iter' (each length N_steps), 'amr_event' (length N_steps Bool), 'initialization_t' (scalar), and 'initialization_dt' (scalar). Fixed-grid diagnostic samples at t=2.25 and 3.0 are written as fixed_grid_time_*.npz with keys 't' (scalar), 'x' (512), 'y' (2048), 'rho' (2048x512), 'p' (2048x512), 'mixing_zone_count', 'mixing_zone_area', 'sharpness', 'structural_count', and 'upper_mixing_component_count' (scalars). The required figure is saved as summary_all.png (PNG) at the run root, copied from stage_b_polydeg6_event_fixed/summary.png; it is a 2x4 heatmap grid of rho (top row) and p (bottom row) at t=0.75,1.5,2.25,3.0 with colorbar=false. Trixi SaveSolutionCallback output is also scheduled into trixi_output. |
| 14 | Saves multiple artifact sets. (1) Full-state snapshots in `stage_b_polydeg6_event_fixed/data_npz/time_%08.5f.npz` and the baseline copy under `stage_b_polydeg6_baseline/data_npz/`, at canonical save times including 0.0:0.1:3.0 union {0.75,1.5,2.25,3.0}; keys are `t` scalar, `polydeg` scalar=6, `x`, `y`, `rho`, `rho_u`, `rho_v`, `rho_e`, `p`, where `x`,`y`,`rho`,`rho_u`,`rho_v`,`rho_e`,`p` all share the PlotData2D element-block shape `size(pd_cons.y)` (3D element-block arrays for DG; n_elements is AMR-dependent). (2) `accepted_step_history.npz` with keys `t`, `tprev`, `dt`, `iter`, `amr_event` as vectors of length N_accepted, plus scalar `initialization_t` and `initialization_dt`. (3) Fixed-grid comparison files `fixed_grid_time_%08.5f.npz` at t=2.25 and 3.0 with keys `t` scalar, `x` (512,), `y` (2048,), `rho` (2048,512), `p` (2048,512), and scalar metrics `mixing_zone_count`, `mixing_zone_area`, `sharpness`, `structural_count`, `upper_mixing_component_count`. (4) Rendered-field archives `rendered_fields_time_%08.5f.npz` at t=0.75,1.5,2.25,3.0 with keys `t` scalar, `polydeg` scalar, `x` (256,), `y` (1024,), `rho` (1024,256), `p` (1024,256), `nx` scalar, `ny` scalar. (5) If a prior baseline exists, `rendered_field_comparison.npz` with keys `t` length 8, `variable_id` length 8 (1=rho,2=p), `relative_l2` length 8, `max_abs_delta` length 8, `pass` length 8, and scalar `relative_l2_tolerance`. (6) `summary_all.png` is produced as a 2×4 heatmap grid with ρ in the top row and p in the bottom row at t=0.75,1.5,2.25,3.0, using `colorbar=false`. The baseline archive copies `data_npz/*` and `summary.png` into `stage_b_polydeg6_baseline/`. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1051/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1109/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1132/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1213/` |
| 5 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1312/` |
| 12 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1512/` |
| 14 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1559/` |


---
