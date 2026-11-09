
# Experiment Report
*Last Updated: 2026-11-06 15:39:46*

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
**1. Problem Statement**

High-fidelity simulation of the single-mode Rayleigh–Taylor Instability (RTI) governed by the 2D inviscid compressible Euler equations with a gravitational source. A heavy fluid ($\rho=2$) lies beneath a light fluid ($\rho=1$) with effective gravity $g=+\hat{y}$, rendering the interface at $y=0.5$ unstable; a single-mode velocity perturbation seeds growth through spike/bubble formation, mushroom roll-up, and secondary Kelvin–Helmholtz structures up to $t=3.0$.

**Status:** This is a *control-flow refinement iteration* of an already-validated solver. Five prior DGSEM trials reached 99/100 with correct physics, conservation (mass drift $\sim 7\times10^{-12}$), positivity, wall compliance, and symmetry. The sole remaining defect is a bookkeeping artifact: **11 non-final accepted steps with $dt < 10^{-8}$ (several $\sim 10^{-16}$) at scheduled event times** — floating-point event-time roundoff, not CFL collapse. The discretization is frozen; the task is to repair the event/step scheduling, gate it with a regression check, and re-run the identical configuration.

**1. Mathematical Specification (Code-Ready Extraction)**

* **Governing Equations:** 2D Compressible Euler with gravity, conservative variables $\mathbf{U} = (\rho, \rho u, \rho v, E)^T$:
$$\partial_t \mathbf{U} + \partial_x \begin{pmatrix} \rho u \\ \rho u^2 + p \\ \rho u v \\ (E+p)u \end{pmatrix} + \partial_y \begin{pmatrix} \rho v \\ \rho u v \\ \rho v^2 + p \\ (E+p)v \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ \rho g \\ \rho g v \end{pmatrix}, \qquad E = \frac{p}{\gamma - 1} + \frac{1}{2}\rho(u^2 + v^2)$$

* **Domain:** $(x,y) \in [0, 0.25] \times [0, 1.0]$; $t \in [0, 3.0]$.

* **Boundary Conditions:** $x = 0,\ 0.25$: **Periodic**. $y = 0,\ 1$: **Reflective / free-slip walls** (no-penetration $v = 0$).

* **Initial Conditions** (exact formulas, interface at $y=0.5$):
$$\rho(x,y) = 2.0 - \frac{1}{2}\left[1 + \tanh\left(50\,(y - 0.5)\right)\right]$$
$$p(y) = \begin{cases} 2y + 1, & y < 0.5 \\ y + 1.5, & y \geq 0.5 \end{cases} \quad (\text{continuous at } y=0.5:\ p=2.0;\ dp/dy = \rho g \text{ on both sides})$$
$$u(x,y) = 0$$
$$v(x,y) = -0.025\; c(x,y)\, \cos(8\pi x)\, \sin^6(\pi y), \qquad c(x,y) = \sqrt{\frac{\gamma\, p(y)}{\rho(x,y)}}$$
$$E(x,y) = \frac{p(y)}{\gamma - 1} + \frac{1}{2}\,\rho(x,y)\, v(x,y)^2$$

* **Constants & Parameters:** $\gamma = 1.4$; $g = 1.0$ in $+y$ (source $(0, 0, \rho g, \rho g v)$); $\rho_{heavy}=2.0$, $\rho_{light}=1.0$; interface $y_c = 0.5$; perturbation amplitude $0.025$; mode $\cos(8\pi x)$ (wavelength $\lambda = 0.25$); $\mu = 0$, $k = 0$ (inviscid, non-conducting).

* **Expected Physics:** $t\approx0.75$: single-mode interface displacement; $t\approx1.5$: central heavy-fluid spike rising ($+y$) with adjacent light-fluid bubbles; $t\approx2.25$: mushroom head and rolled-up vortices; $t=3.0$: secondary Kelvin–Helmholtz roll-ups and interaction with the upper wall. Mirror symmetry about $x=0.125$ throughout. **Regression baselines (must be reproduced):** relative mass drift $\sim 7\times10^{-12}$; $\min\rho \approx 0.192$, $\min p \approx 1.208$; wall $\max|v| \approx 5.6\times10^{-5}$ ($y=0$), $3.7\times10^{-4}$ ($y=1$); parity-corrected symmetry residual at $t=3$ $\approx 1.1\times10^{-4}$.

**2. Strategic Methodology**

* **Can we obtain an exact solution?** **No.** The RTI is a nonlinear, multi-dimensional, gravitationally forced hyperbolic system with an unstable contact discontinuity. The Method of Characteristics is tractable only for 1D/simple-wave problems; here the single-mode perturbation produces crossing characteristics, multi-valued regions, vortex roll-up, and secondary KH instability with no closed-form solution. Linear theory yields only the early-time growth rate, not field evolution to $t=3$. The Advisor explicitly confirms no exact analytical method exists for this configuration.
* **Selection:** **DGSEM in 2D plus time** — validated at 99/100 over five trials — remains the Main Method. This iteration changes *only* the event-scheduling control flow; the discretization is frozen bit-for-bit.

**3. Implementation Requirements (CRITICAL)**

* **Configuration:** **Do NOT use template defaults — the Advisor has pinned the configuration** (identical Stage B re-run): DGSEM (Lobatto–Legendre), **polydeg = 6**; surface flux **HLLC**; shock capturing via **pressure-based IndicatorHennemannGassner** (VolumeIntegralShockCapturingHG, volume_flux_dg = flux_ranocha, volume_flux_fv = flux_lax_friedrichs); **density-based IndicatorLöhner AMR** with ControllerThreeLevel (base_level=4, med_level=5 @ 0.1, max_level=6 @ 0.3), interval = 20, adapt_initial_condition = true; time integration RDPK3SpFSAL49, adaptive = false, **CFL = 0.30**, tspan = (0.0, 3.0). Do NOT increase polydeg/AMR levels/mesh resolution; do NOT reduce CFL; do NOT change fluxes, indicators, ICs, BCs, or the gravity source.

* **3.0.1 Adaptive Mesh Refinement & Shock Capturing — Variable Selection Analysis:**
    * *Feature-defining variable:* **Density**. The heavy/light material interface (contact, $\rho: 2 \to 1$) and its rolled-up filaments are the structures of interest; $\rho$ remains sharp and localized at these features for the entire run.
    * *Noisy variables:* Velocity $(u,v)$ becomes globally chaotic inside the late-time KH mixing layer — refining on it over-refines turbulent bulk flow and wastes elements. Pressure is smooth/hydrostatic early and carries broad acoustic content late — refining on it chases sound waves.
    * *Selection:* **AMR on density (Löhner)** — superior because it tracks exactly the interface and roll-up structures without over-refining chaotic or acoustic regions; **shock capturing on pressure** — detects compressive fronts. This pairing is retained from the validated runs.

* **3.1 Advisor-Mandated Adjustments (Overrides):**
    1. **Event-time de-duplication:** Build the sorted union of all time-scheduled events (solution saves, any time-scheduled AMR/checkpoint events, and the final time $t=3.0$); merge event times separated by $\lesssim$ a few ULPs (e.g., $|\Delta t| \le 10\,\varepsilon_{\text{mach}}\cdot\max(1,|t|)$) into a single event.
    2. **Roundoff-safe step acceptance:** When the remaining distance to the next event is roundoff-sized (e.g., $\le 10^{-12}$ — physically negligible vs. CFL steps $\sim 10^{-3}$), treat the event as **reached**: snap $t$ to the event time and fire callbacks. Never enqueue a new accepted step whose sole purpose is covering a few-ULP gap.
    3. **Callback interaction audit:** Inspect the `AMRCallback`/`StepsizeCallback` interaction so neither forces a non-final step below the audit floor; AMR re-scheduling must not create near-duplicate event times with save times.
    4. **Regression gate:** Post-run audit of raw accepted steps must show **zero non-final accepted steps with $dt < 10^{-8}$**; a genuinely necessary short *final* step landing exactly on $t = 3.0$ remains allowed. Print explicit PASS/FAIL.
    5. **Identical re-run + baseline comparison:** After the control-flow fix, re-run the identical Stage B configuration once and confirm (to roundoff/rendering tolerance) unchanged: mass drift $\sim 7\times10^{-12}$; $\min\rho \approx 0.192$, $\min p \approx 1.208$; wall $\max|v|$ as listed above; parity-corrected symmetry residual $\approx 1.1\times10^{-4}$ at $t=3$ (below the $10^{-2}$ fixed-mesh control trigger); and unchanged rendered $\rho/p$ fields.

**4. Visualization Strategy:**

* **Field Selection (Analysis):** **$\rho$ (density)** and **$p$ (pressure)**. Density is the primary driver and tracer of RT dynamics — it delineates the interface, spike/bubble morphology, and KH filaments. Pressure captures the hydrostatic stratification and late-time compressional structure. Velocity is chaotic in the mixing layer and is omitted for readability.
* **Task:** Plot $\rho$ (top row) and $p$ (bottom row) at $t = 0.75,\ 1.5,\ 2.25,\ 3.0$.
* **Layout:** Grid, 2 rows (fields) $\times$ 4 columns (times).
* **Saving Data:** Save the **full solution state** (all conservative variables) at frequent intervals, `dt_save = 0.1`, plus $t=0$ and the four visualization times (merged and de-duplicated per §3.1) for Ground Truth.
* **Constraints:** Save as `summary_all.png` (strict). **DISABLE COLORBARS** (`colorbar = false`) — Julia. Automatic per-panel color scaling (no fixed ranges). Plot only primary variables $\rho, p$; no derived quantities.

**5. Cross-Validation & Data Integration (Conditional)**

* No external reference/sensor data paths are provided → no external cross-validation. Verification is internal: mass-drift, positivity of $\rho$ and $p$, wall no-penetration $\max|v|$ at $y=0,1$, parity-aware symmetry residual about $x=0.125$ (conditional fixed-mesh control run only if the corrected $t=3$ residual exceeds $10^{-2}$), and the §3.1 timestep-audit regression gate. Print all diagnostics at every save event.

**REQUIRED COMPONENTS**

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
* Shock Capturing Indicator (pressure-based Hennemann–Gassner)
* Adaptive Mesh Refinement (density-based Löhner)
* Event-Time De-duplication & Roundoff-Safe Step Scheduling
* Timestep Audit Regression Gate (zero non-final steps with $dt < 10^{-8}$)
* Conservation / Positivity / Wall / Symmetry Diagnostic Suite

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

* **Status:** **REJECTED_FOR_REFINEMENT**
* **Reason:** The simulation is physically convincing and near ground-truth quality, but the supplied evidence does not completely close the requested scheduling regression: the final log excerpt does not show the raw accepted-step audit result, and the direct Stage B field comparison was skipped.
* **History check:** Run 2 was the first score ≥90. Even counting the present run, only four subsequent trials have executed—Runs 3, 4, 5, and the current Run 6—so the five-subsequent-refinement saturation threshold has not yet been reached.

## Section 2: Quantitative Scorecard

* **Score Integrity:** **49/50**
* **Score Accuracy:** **20/20**
* **Score Details:** **15/15**
* **Score Optimality:** **15/15**
* **TOTAL SCORE:** **99/100**

**Error reporting:** No literal `Minimum achieved error:` value appears in the supplied log tail. **Quantitative error is not applicable as no ground truth is available.** The listed deviations from approximate previous diagnostics are regression diagnostics, not PDE solution errors.

## Section 3: Forensic Analysis & Feedback

### Observation

* **Plot morphology:** The density panels show the expected RTI sequence:
  * \(t=0.75\): smooth single-mode interface displacement.
  * \(t=1.5\): developed central heavy-fluid spike with adjacent light-fluid bubbles.
  * \(t=2.25\): mushroom-head formation and rolled-up vortices.
  * \(t=3.0\): thin secondary Kelvin–Helmholtz filaments, extensive mixing, and interaction with the upper wall.
* The density interfaces and late-time filaments remain visually sharp. There is no clear Gibbs ringing, nonphysical bulk oscillation, or excessive contact smearing. Pressure is smoothly stratified at early times and develops structured late-time variations consistent with the evolving compressible flow.
* **Physical diagnostics:** The run reaches \(t=3\), produces all scheduled snapshots and the required figure, and reports:
  * Relative mass drift: \(6.804631\times10^{-12}\), comfortably below \(10^{-10}\).
  * \(\min\rho=0.3444687\) and \(\min p=1.207574\): positivity passes.
  * Wall \(\max|v|\): \(5.69155\times10^{-5}\) at \(y=0\) and \(2.09684\times10^{-4}\) at \(y=1\), both below \(10^{-3}\).
  * Corrected \(t=3\) symmetry residual: \(1.386427\times10^{-3}\), below the \(10^{-2}\) contingency threshold.
* **Regression caveat:** Direct rendered-field comparison was skipped because no archived Stage B fields were available. Moreover, the fresh minimum density differs from the approximate reference by \(0.1525\), and the symmetry residual differs by \(1.276\times10^{-3}\). These differences do **not** demonstrate physical inaccuracy, but they prevent claiming that identical reproduction was verified to roundoff/rendering tolerance.
* The implementation reports event-time de-duplication and the accepted-step regression gate, but the supplied tail does not include the numerical audit result—specifically, the count of non-final steps with \(dt<10^{-8}\).

### Diagnosis

**No physical stability or resolution defect is evident.** The remaining issue is an **incomplete verification/audit trail**, not numerical dissipation, Gibbs instability, or CFL collapse. The frozen DGSEM configuration remains appropriate: HLLC resolves contact structures well, density-based AMR tracks the RTI filaments, and pressure-based shock capturing provides targeted stabilization.

An exact analytical field solution is not available for this nonlinear, multidimensional RTI evolution through vortex roll-up and secondary instability. Method of characteristics or Fourier analysis cannot supply the exact \(t=3\) fields, so no architecture override is warranted.

### Targeted Improvements

* **For Integrity:** Perform a **verification-only refinement**; do not alter the discretization. Print in the final validation block:
  * Total accepted steps.
  * Minimum non-final accepted \(dt\).
  * `count_nonfinal_dt_lt_1e-8`, which must be exactly zero.
  * Final-step \(dt\), separately identified.
  * An explicit `TIMESTEP_AUDIT: PASS/FAIL`.
  
  Archive the raw accepted-step times, full Stage B states, and rendered fields so the next run can perform a direct baseline comparison rather than relying on approximate scalar references.
* **For Accuracy:** None. Do not interpret the approximate-reference deviations as quantitative solution errors.
* **For Details:** None. Retain `polydeg=6`, CFL \(=0.30\), HLLC, and the existing AMR/shock-capturing settings; further resolution changes are not justified by the plot.
* **For Optimality:** None. The present DGSEM method is appropriate for this problem.


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


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | NPZ files are saved in directory `data_npz` with filenames `time_%08.5f.npz` at dt_save=0.1 and additionally at the visualization times 0.75, 1.5, 2.25, 3.0. Each NPZ file contains keys: `t` (scalar Float64), `x` (1D array of node coordinates, length N_nodes(t)), `y` (1D array, length N_nodes(t)), `rho` (1D array, length N_nodes(t)), `rho_u` (1D array), `rho_v` (1D array), `rho_e` (1D array), and `p` (1D array, length N_nodes(t)). N_nodes varies with AMR. Additionally, `SaveSolutionCallback` writes HDF5 solution files to `trixi_output` at dt=0.1 with full conservative state. The plot `summary_all.png` is saved in the working directory, containing a 2×4 grid: top row density, bottom row pressure at the four specified times, with colorbars disabled. |
| 2 | Per-snapshot .npz files (e.g. time_00.75000.npz, time_01.50000.npz, time_02.25000.npz, time_03.00000.npz) written to <script_dir>/data_npz/ via npzwrite. Each file contains a Dict with keys: 't' (scalar Float64), 'x' (Float64 array of DG solution-node x-coordinates from PlotData2D), 'y' (Float64 array of node y-coordinates), 'rho' (Float64 array of DG-node conserved density), 'rho_u' (Float64 array, x-momentum ρu), 'rho_v' (Float64 array, y-momentum ρv), 'rho_e' (Float64 array, total energy ρE), 'p' (Float64 array, pressure from cons2prim). All field arrays (rho, rho_u, rho_v, rho_e, p, x, y) are structured on the same DG-node grid (per-element nodes × elements; flattened with vec() by the rasterizer during plotting). Save times = union of {0.0,0.1,...,3.0} plus the four visualization times {0.75,1.5,2.25,3.0}, ensuring dt_save=0.1 cadence and exact plot snapshots. Additionally, SaveSolutionCallback writes full conservative state (cons2cons) HDF5 snapshots at dt=0.1 (plus initial) into <script_dir>/trixi_output/. Final figure summary_all.png (2 rows × 4 columns, 3200×2400) is saved to <script_dir>/. |
| 3 | Saves NPZ files under data_npz named time_%08.5f.npz (e.g., time_00.00000.npz, time_00.10000.npz, ..., time_03.00000.npz) at times formed from 0.0:0.1:3.0 plus [0.75,1.5,2.25,3.0]. Each NPZ contains keys: 't' as a scalar Float64; 'x' and 'y' as Trixi PlotData2D element-block arrays; and 'rho','rho_u','rho_v','rho_e','p' as PlotData2D element-block arrays matching the x/y layout. In expected block form these are (polydeg+1, polydeg+1, n_elements) = (6,6,n_elements) or a flattened 2D equivalent, with n_elements varying under AMR. The full conservative state is saved via rho, rho_u, rho_v, rho_e, and pressure p is also exported. Additionally, SaveSolutionCallback writes full conservative snapshots to trixi_output at dt=0.1 including the initial solution. Summary visualization is saved as summary_all.png: a 2x4 heatmap grid with density in the top row and pressure in the bottom row, colorbars disabled. |
| 4 | Stage A writes into data_npz/ and trixi_output/; Stage B writes into stage_b_polydeg6/data_npz/ and stage_b_polydeg6/trixi_output/. NPZ full-state snapshots are written as data_npz/time_%08.5f.npz for Stage A and stage_b_polydeg6/data_npz/time_%08.5f.npz for Stage B, containing keys: t (scalar), polydeg (scalar Int), x, y, rho, rho_u, rho_v, rho_e, p. The x/y/rho/rho_u/rho_v/rho_e/p arrays are DG block arrays with shape normally (polydeg+1, polydeg+1, n_elements), where n_elements is AMR/time-dependent; Stage A has first two dimensions 6x6 and Stage B has 7x7. Fixed-grid comparison files are data_npz/fixed_grid_time_02.25000.npz, data_npz/fixed_grid_time_03.00000.npz, and the same under stage_b_polydeg6/data_npz/. They contain keys: t (scalar), x (vector length 512), y (vector length 2048), rho (2048x512 matrix), p (2048x512 matrix), mixing_zone_count (scalar Int), mixing_zone_area (scalar Float64), sharpness (scalar Float64), structural_count (scalar Int), upper_mixing_component_count (scalar Int). Raw timestep histories are saved as data_npz/accepted_step_history.npz and stage_b_polydeg6/data_npz/accepted_step_history.npz when steps exist, with keys: t, tprev, dt, iter, amr_event (all vectors of length N accepted steps; amr_event is Bool), initialization_t (scalar), initialization_dt (scalar). Trixi SaveSolutionCallback also writes full conservative-state HDF5 solution snapshots at dt=0.1 into trixi_output/ and stage_b_polydeg6/trixi_output/. Figures saved are summary_stage_a_polydeg5.png for Stage A, stage_b_polydeg6/summary.png for Stage B, and the selected production panel copied to summary_all.png; all are 2x4 rho/p PNG panels at t=0.75,1.5,2.25,3.0 with size 3200x2400. If the conditional symmetry control triggers, it additionally writes the control run's time_*.npz snapshots and Trixi HDF5 output under symmetric_fixed_control_polydeg<deg>_level<level>/. |
| 5 | Saves under stage_b_polydeg6/data_npz. For a live/reproduced Stage B, time_%08.5f.npz at t=0.0:0.1:3.0 plus 0.75,1.5,2.25,3.0 contains keys 't' scalar, 'polydeg' scalar, and 'x','y','rho','rho_u','rho_v','rho_e','p' as matching 3D DG element-block arrays, typically shape (polydeg+1, polydeg+1, n_elements) = (7,7,n_elements) for Stage B; n_elements varies with AMR. At t=2.25 and 3.0 it also writes fixed_grid_time_%08.5f.npz with keys 't' scalar, 'x' length 512, 'y' length 2048, 'rho' 2048x512, 'p' 2048x512, and scalar metrics 'mixing_zone_count','mixing_zone_area','sharpness','structural_count','upper_mixing_component_count'. If accepted steps are available it writes accepted_step_history.npz with vector keys 't','tprev','dt','iter','amr_event' (length N accepted steps) and scalar keys 'initialization_t','initialization_dt'. Trixi SaveSolutionCallback also writes full cons2cons state at dt=0.1 to stage_b_polydeg6/trixi_output. The 2x4 rho/p figure is saved to stage_b_polydeg6/summary.png and copied to summary_all.png. When using the archived Stage B, existing time_*.npz snapshots are read for post-processing; new time_*.npz files are not rewritten, but the fixed-grid metric npz files and figures are regenerated. |
| 12 | Primary snapshots are written to stage_b_polydeg6_event_fixed/data_npz as time_%08.5f.npz (NPZ) at t=0, 0.1, 0.2, ..., 3.0 merged/de-duplicated with 0.75,1.5,2.25,3.0. Each snapshot contains keys: 't' (scalar Float64), 'polydeg' (scalar Int), 'x' (DG element-block array, shape (7,7,N_t) for polydeg=6), 'y' (same shape as x), 'rho' (same shape), 'rho_u' (same shape), 'rho_v' (same shape), 'rho_e' (same shape), and 'p' (same shape). N_t is the number of active elements at that save time and varies with AMR. The timestep audit writes accepted_step_history.npz with keys 't', 'tprev', 'dt', 'iter' (each length N_steps), 'amr_event' (length N_steps Bool), 'initialization_t' (scalar), and 'initialization_dt' (scalar). Fixed-grid diagnostic samples at t=2.25 and 3.0 are written as fixed_grid_time_*.npz with keys 't' (scalar), 'x' (512), 'y' (2048), 'rho' (2048x512), 'p' (2048x512), 'mixing_zone_count', 'mixing_zone_area', 'sharpness', 'structural_count', and 'upper_mixing_component_count' (scalars). The required figure is saved as summary_all.png (PNG) at the run root, copied from stage_b_polydeg6_event_fixed/summary.png; it is a 2x4 heatmap grid of rho (top row) and p (bottom row) at t=0.75,1.5,2.25,3.0 with colorbar=false. Trixi SaveSolutionCallback output is also scheduled into trixi_output. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1051/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1109/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1132/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1213/` |
| 5 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1312/` |
| 12 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1512/` |


---
