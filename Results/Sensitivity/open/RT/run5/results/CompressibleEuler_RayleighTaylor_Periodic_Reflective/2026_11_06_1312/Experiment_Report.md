
# Experiment Report
*Last Updated: 2026-11-06 13:34:44*

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

Single-mode inviscid compressible Rayleigh–Taylor instability: a heavy fluid ($\rho=2$) rests *beneath* a light fluid ($\rho=1$) while a body force $g=+1$ acts in $+y$, pushing heavy fluid into light fluid — the mirror image of the classical configuration, hence unstable. A single-mode $\cos(8\pi x)$ perturbation seeds the interface at $y=0.5$.

**This iteration is a verification-closure, NOT a refinement.** The Stage B production configuration (polydeg=6, CFL=0.30, HLLC, AMR 4/5/6) is **retained verbatim**. The single required change is the correction of the symmetry diagnostic to **parity-aware reflection residuals**, recomputed from the archived Stage B state. The reported residual of exactly $2.0$ on $\rho u$ is the mathematical signature of applying an even-field formula to an odd field ($\|q-(-q)\|/\|q\|=2$), i.e., evidence of *perfect* symmetry, not symmetry loss.

---

**1. Mathematical Specification (Code-Ready Extraction)**

* **Governing Equations:** 2D Compressible Euler with gravitational source:
$$\partial_t \mathbf{U} + \partial_x \mathbf{F}(\mathbf{U}) + \partial_y \mathbf{G}(\mathbf{U}) = \mathbf{S}(\mathbf{U}), \qquad \mathbf{U} = \begin{pmatrix}\rho \\ \rho u \\ \rho v \\ \rho E\end{pmatrix}$$
$$\mathbf{F} = \begin{pmatrix}\rho u \\ \rho u^2 + p \\ \rho u v \\ (\rho E + p)\,u\end{pmatrix}, \qquad \mathbf{G} = \begin{pmatrix}\rho v \\ \rho u v \\ \rho v^2 + p \\ (\rho E + p)\,v\end{pmatrix}, \qquad \mathbf{S} = \begin{pmatrix}0 \\ 0 \\ \rho g \\ \rho g v\end{pmatrix}$$
with $p = (\gamma-1)\left(\rho E - \tfrac{1}{2}\rho(u^2+v^2)\right)$, $\gamma = 1.4$, $g = 1.0$ (positive $y$).

* **Domain:** $(x,y) \in [0,\,0.25] \times [0,\,1.0]$, $t \in [0,\,3.0]$.

* **Boundary Conditions:** $x$: Periodic. $y=0$ and $y=1$: Reflective / free-slip walls (no-penetration $v=0$; mirror reflection of tangential velocity).

* **Initial Conditions** (interface at $y=0.5$):
$$\rho(x,y,0) = 2.0 + (1.0 - 2.0)\cdot\tfrac{1}{2}\left[1 + \tanh\left(50(y-0.5)\right)\right]$$
$$p(y,0) = \begin{cases} 2y + 1, & y < 0.5 \\ y + 1.5, & y \geq 0.5 \end{cases} \qquad (\text{continuous at } y=0.5:\ p=2.0)$$
$$u(x,y,0) = 0$$
$$v(x,y,0) = -0.025\,\sqrt{\frac{\gamma\, p(y)}{\rho(y)}}\;\cos(8\pi x)\,\left[\sin(\pi y)\right]^6$$
$$\rho E = \frac{p}{\gamma - 1} + \frac{1}{2}\rho\left(u^2 + v^2\right)$$

* **Constants & Parameters:** $\gamma = 1.4$; gravity vector $\mathbf{g} = [0,\,1.0]^T$; $\rho_{heavy}=2.0$, $\rho_{light}=1.0$; perturbation amplitude $0.025$; mode $\cos(8\pi x)$ (wavelength $=0.25$, single mode across periodic width); $\mu = 0$, $k = 0$.

* **Expected Physics:** Heavy-fluid **spike rises at $x=0.125$** (where $v>0$ initially); light-fluid **bubbles descend at $x=0 \equiv 0.25$**. $t\approx0.75$: visible modal displacement; $t\approx1.5$: spike + bubbles developed; $t\approx2.25$: mushroom head with rolled-up side structures; $t\approx3.0$: secondary Kelvin–Helmholtz roll-ups and spike interaction with the upper wall $y=1$. The exact solution is mirror-symmetric about $x=0.125$: under $x \to 0.25 - x$, $(\rho, \rho u, \rho v, \rho E) \mapsto (\rho, -\rho u, \rho v, \rho E)$.

---

**2. Strategic Methodology**

* **Can we obtain an exact solution?** **No.** This is nonlinear multi-dimensional compressible Euler with a gravitational source, wall boundaries, and vortex roll-up. The Method of Characteristics is restricted to 1D/simple-wave structures — in 2D the characteristic families interact non-trivially, gravity destroys Riemann-invariant reductions, and KH roll-up generates multi-valued caustics immediately. Fourier methods require periodicity in *all* directions (violated by the slip walls in $y$) and global smoothness (violated by contacts, shocks, and roll-up). No Green's function / separation-of-variables reduction exists for the nonlinear system. The Advisor explicitly confirms: no exact-method override is warranted.
* **Selection:** **DGSEM 2D plus time**, with the Stage B configuration retained verbatim. This iteration modifies only the *verification diagnostic*, not the discretization.

---

**3. Implementation Requirements (CRITICAL)**

* **Configuration:** Use template defaults, **except** the Advisor-mandated retentions in §3.1. Do not change the time stepper or any parameter not listed there.

* **3.0.1 Adaptive Mesh Refinement & Shock Capturing:**
  * **Variable Selection Analysis:** The structural features of RT (interface, spikes, bubbles, mushroom roll-ups) are **contact discontinuities**, across which pressure is *continuous* — a pressure-based AMR indicator would be blind to the dominant feature. Velocity $(u,v)$ becomes globally chaotic in the late-time mixing zone (KH turbulence) and would scatter refinement elements everywhere. **Density $\rho$** remains sharp and strictly localized at the features of interest for the entire run → **AMR indicator: density** (Löhner). This is strictly superior for this PDE.
  * **Shock Capturing indicator: pressure** (retained per Advisor) — compressive features (wall impact, late-time acoustic compression) are pressure phenomena, while contact-rich interface elements remain pure-DG with HLLC contact resolution.

* **3.1 Advisor-Mandated Adjustments (Overrides):**
  1. **RETAIN Stage B verbatim:** polydeg=6, CFL=0.30, HLLC surface flux, AMR `ControllerThreeLevel(base=4, med=5, max=6)` with thresholds 0.1/0.3, interval=20. **No further degree, mesh, or CFL increases.**
  2. **PARITY-AWARE SYMMETRY DIAGNOSTIC (the core fix).** With $M$ = reflection about $x=0.125$ ($x \to 0.25-x$):
     $$R_{\rho u} = \frac{\|\rho u + M(\rho u)\|}{\|\rho u\|} \quad (\text{odd field}), \qquad R_q = \frac{\|q - M(q)\|}{\|q\|} \quad \text{for } q = \rho,\ \rho v,\ \rho E \ (\text{even fields}).$$
     Guard the initially-zero $\rho u$ norm: use $\max(\|\rho u\|, \varepsilon)$ or fall back to the absolute residual $\|\rho u + M(\rho u)\|$ when $\|\rho u\| < \varepsilon$. The value $R=2.0$ from the old formula on $\rho u$ is the signature of *perfect* symmetry, not failure.
  3. **RECOMPUTE, DON'T RE-SOLVE:** Recompute the corrected residuals at $t = 0,\,0.75,\,1.5,\,2.25,\,3.0$ plus the standard suite (mass drift, min $\rho/p$ positivity, wall $|v|$) **from the archived Stage B full state**. A new PDE solve is unnecessary unless corrected residuals reveal genuine asymmetry. If the archive is unavailable in the execution environment, re-run the *identical* Stage B configuration (a reproduction, not a refinement).
  4. **NO fixed-mesh control runs** may be triggered from the uncorrected maximum residual. Any control-run trigger must key on the corrected parity-aware residuals only (genuine asymmetry $\equiv$ corrected $R \gtrsim 10^{-2}$, far above the $\sim10^{-5}$ discretization noise floor).
  5. **Connected-component census filter:** any mixing-zone/vortex component counting must apply a **minimum physical-area threshold** so grid-scale fragments are not counted as newly resolved vortices.

---

**4. Visualization Strategy**

* **Field Selection (Analysis):** **Density $\rho$** — the primary driver; the interface, spikes, bubbles, and mixing zone are density features. **Pressure $p$** — secondary; hydrostatic stratification and late-time wall compression. Do **not** plot $u, v, E$ or derived quantities.
* **Task:** Plot $\rho$ (top row) and $p$ (bottom row) at $t = 0.75,\ 1.5,\ 2.25,\ 3.0$.
* **Layout:** Grid, 2 rows (fields) × 4 columns (time).
* **Saving Data:** Save the **full solution state** (all conserved variables) at frequent intervals (`dt_save=0.1`) — required so the parity-aware diagnostics can be recomputed post hoc from the archive.
* **Constraints:** Save as `summary_all.png` (strictly), rendered from the **Stage B (polydeg=6)** solution. **Julia: `colorbar=false` (explicitly disable colorbars).** Automatic per-panel color scaling; no fixed colormaps/clims; no mesh lines.

---

**5. Cross-Validation & Data Integration (Conditional)**

* **Trigger check:** No external data paths (sensors / true data / reference solution) are provided in [USER_QUESTION] → **Not applicable.**
* The parity-aware symmetry residual is an *internal verification* diagnostic. Print the corrected values with explicit labels; expected result: corrected $R_{\rho u} \approx 0$ (not 2.0) and even-field residuals consistent with the previously reported $\approx 5.2\times10^{-5}$ ($\rho v$) and $\approx 1.4\times10^{-6}$ ($\rho E$), confirming symmetry and closing the verification loop.

---

**REQUIRED COMPONENTS**

**Main Method:**
* Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
* Shock Capturing (pressure-based indicator)
* Adaptive Mesh Refinement (density-based indicator)
* Parity-Aware Symmetry Diagnostic (Advisor-mandated)
* Connected-Component Minimum-Area Filter (Advisor-mandated)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

* **Status:** **REJECTED_FOR_REFINEMENT**
* **Rationale:** This is a successful, high-fidelity verification run, not a fatal failure. However, the total score is below 100 because the timestep audit still reports recurring non-final, roundoff-sized steps. History saturation also does **not** yet apply: Run 2 was the first score ≥90, and fewer than five subsequent refinement/verification trials have been completed.

## Section 2: Quantitative Scorecard

* **Score Integrity:** **49/50**
* **Score Accuracy:** **20/20**
* **Score Details:** **15/15**
* **Score Optimality:** **15/15**
* **TOTAL SCORE:** **99/100**

**Ground-truth error:** Quantitative error is not applicable as no ground truth is available. The supplied log tail contains no `Minimum achieved error:` entry; the reported symmetry and conservation quantities are internal verification diagnostics, not errors against an analytical solution.

## Section 3: Forensic Analysis & Feedback

### Observation

* The simulation completed through \(t=3\), produced a readable `summary_all.png`, and retained positive density and pressure:
  * \(\min\rho=0.1920849\), \(\min p=1.208287\).
  * Relative mass drift is only \(7.25\times10^{-12}\).
  * Wall-normal velocity maxima are small: \(5.63\times10^{-5}\) at \(y=0\) and \(3.73\times10^{-4}\) at \(y=1\).
* The density evolution is physically consistent with the prescribed unstable configuration:
  * At \(t=0.75\), the single-mode interface displacement is visible.
  * At \(t=1.5\), the central heavy-fluid spike and adjacent light-fluid bubbles are developed.
  * At \(t=2.25\), the mushroom head and rolled-up side structures appear.
  * At \(t=3\), secondary Kelvin–Helmholtz structures and interaction with the upper wall are visible.
* The structures remain visually symmetric about \(x=0.125\). The corrected parity-aware residual at \(t=3\), \(1.126302\times10^{-4}\), supports this assessment and correctly does not trigger the fixed-mesh control run. The former residual of \(2.0\) on \(\rho u\) was indeed a diagnostic-parity artifact, not evidence of symmetry loss.
* Density interfaces and vortex roll-ups are well resolved, without conspicuous Gibbs ringing or excessive large-scale smearing. Pressure remains predominantly stratified at early times and develops plausible late-time compressional structure.
* The only remaining integrity concern is the timestep audit: **11 non-final steps with \(dt<10^{-8}\)**, several with \(dt\sim10^{-16}\), occur at scheduled event times. Their magnitude indicates floating-point event-time roundoff rather than a physical CFL collapse, but the log itself correctly requests inspection of the AMR/stepsize callback interaction.

### Diagnosis

**Solver event-scheduling roundoff**, not a PDE stability or accuracy failure. Duplicate or nearly coincident save/AMR event times appear to leave a remaining interval of a few machine epsilons, which is then taken as a separate accepted step. No further mesh, polynomial-degree, or flux refinement is justified by the present plots or diagnostics.

### Targeted Improvements

* **For Integrity:** Fix the event/timestep bookkeeping while retaining the Stage B discretization:
  * De-duplicate scheduled save and AMR event times and treat event-time differences within a few ULPs as coincident.
  * When the remaining distance to an event is roundoff-sized, treat the event as reached rather than enqueueing another step.
  * Inspect `AMRCallback` and `StepsizeCallback` interaction so neither forces a non-final step below the audit floor.
  * Add a regression check requiring **zero non-final accepted steps with \(dt<10^{-8}\)**; continue to allow a genuinely necessary short final step to reach \(t=3\).
  * Re-run the identical configuration once after this control-flow fix and confirm that conservation, positivity, wall behavior, symmetry, and the rendered fields remain unchanged.
* **For Accuracy:** **None.** Retain `polydeg=6`, CFL \(=0.30\), HLLC, and the existing initial/boundary conditions and gravity source.
* **For Details:** **None.** Retain density-based AMR and pressure-based shock capturing; the present interfaces and roll-ups do not justify more aggressive refinement or dissipation.
* **For Optimality:** **None.** No exact analytical method is available for this nonlinear, multidimensional, gravity-driven Euler problem with wall interaction and vortex roll-up. DGSEM with HLLC, contact-resolving AMR, and localized shock capturing remains an appropriate high-fidelity fallback.

**Recommendation:** Address only the roundoff-sized event steps. Do not increase polynomial degree, AMR levels, or mesh resolution, and do not reduce CFL merely to suppress this bookkeeping artifact.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | The code implements the 2D compressible Euler equations with gravitational source using Trixi.jl. The spatial discretization is DGSEM with polynomial degree 3 on a P4estMesh (domain [0,0.25]×[0,1], periodic in x, reflective slip walls in y via boundary_condition_slip_wall). Shock capturing is density-based using IndicatorHennemannGassner with a hybrid DG-FV volume integral (flux_ranocha for DG, flux_lax_friedrichs for FV) and HLL surface flux. AMR is density-based using IndicatorLöhner with ControllerThreeLevel (base_level=4, med_level=5, max_level=6) and AMRCallback interval=20. The initial condition exactly matches the strategy: tanh density profile, piecewise hydrostatic pressure, zero u, and the specified single-mode v perturbation. Gravity source is (0,0,ρg,ρgv) with g=1.0. Time integration uses RDPK3SpFSAL49 with adaptive=false, StepsizeCallback cfl=0.5, and tspan=(0.0,3.0). Visualization produces summary_all.png with density (top row) and pressure (bottom row) at t=0.75, 1.5, 2.25, 3.0, with colorbars disabled and automatic color scaling. | N/A | 47 | 18 | 11 | 13 | **89** |
| 2 | The implementation faithfully reproduces the DGSEM solution of the 2D inviscid compressible Euler equations with a gravitational source for the single-mode Rayleigh–Taylor instability. Governing equations use CompressibleEulerEquations2D(γ=1.4); the initial condition is the verbatim strategy profile (smoothed density 2→1 via tanh(50(y-0.5))·0.5, piecewise pressure 2y+1 / y+1.5 at y=0.5, u=0, v=-0.025·c·cos(8πx)·sin^6(πy) with c=sqrt(γp/ρ), and energy via prim2cons giving E=p/(γ-1)+0.5ρv²). The gravity source term returns SVector(0,0,g·ρ, g·ρv), i.e. (0,0,ρg,ρgv). Discretization: P4estMesh on [0,0.25]×[1.0] with x-periodic (periodicity=(true,false)) and slip-wall BCs on :y_neg/:y_pos; DGSEM with polydeg=3, LobattoLegendreBasis. Advisor-mandated upgrades are present: surface_flux=flux_hllc, pressure-based IndicatorHennemannGassner shock capture (pressure_indicator derived from conservative state) feeding VolumeIntegralShockCapturingHG with volume_flux_dg=flux_ranocha and volume_flux_fv=flux_lax_friedrichs, while AMR uses a separate density-based IndicatorLöhner (variable=density_indicator=u[1]) with ControllerThreeLevel (base 4 / med 5 @0.1 / max 6 @0.3), interval=20, adapt_initial_condition=true. Held-fixed config is respected: RDPK3SpFSAL49 time integrator, adaptive=false, StepsizeCallback(cfl=0.5), tspan=(0.0,3.0). The full diagnostic suite is implemented and printed at every save: total mass (Trixi.integrate) with absolute/relative drift vs t=0, min/max ρ and p (positivity), dt_min/dt_max/current dt (CFL), max|v| at y=0 and y=1 (wall no-penetration) and an approximate symmetry residual ‖q-q(0.25-x)‖/‖q‖ about x=0.125. Visualization renders a clean 2×4 grid (density top row, pressure bottom row) at t=0.75,1.5,2.25,3.0 with colorbar=false, no mesh lines (framestyle=:none, axis=nothing), per-panel automatic scaling, and higher resolution size=(3200,2400) saved as summary_all.png. | N/A | 47 | 19 | 12 | 15 | **93** |
| 3 | Implementation is a numerical DGSEM solver for the 2D compressible Euler equations with gamma=1.4 and the prescribed gravity source. Domain is [0,0.25] x [0,1] with t in [0,3], periodic x-boundaries, and slip walls at y=0 and y=1. Initial conditions match the strategy: smoothed tanh density interface at y=0.5, piecewise hydrostatic pressure, u=0, v=-0.025*c*cos(8*pi*x)*sin(pi*y)^6, and total energy through prim2cons. Spatial discretization uses polydeg=5 Lobatto basis, HLLC surface flux, pressure-based Hennemann-Gassner shock capturing with flux_ranocha DG and flux_lax_friedrichs FV, and density-based Lohner AMR with ControllerThreeLevel base=4/med=5/max=6, interval=20, adapt_initial_condition=true. Time integration uses RDPK3SpFSAL49 with adaptive=false, tspan=(0.0,3.0), and CFL=0.35 via StepsizeCallback. Visualization samples the DG solution onto a uniform Cartesian grid and saves summary_all.png as a 2x4 heatmap grid of density and pressure at t=0.75,1.5,2.25,3.0 with colorbar=false. The code also performs the required t=0 renderer x-independence test and prints the final diagnostic summary for mass drift, positivity, dt extrema, wall |v|, and symmetry residual. | N/A | 49 | 19 | 14 | 15 | **97** |
| 4 | Numerical implementation of the 2D inviscid compressible Euler Rayleigh-Taylor instability using DGSEM. Stage A uses polydeg=5, CFL=0.35, HLLC surface flux, pressure-based Hennemann-Gassner shock capturing with flux_ranocha DG and flux_lax_friedrichs FV, density-based Löhner AMR with ControllerThreeLevel 4/5/6 at thresholds 0.1/0.3 and interval=20, RDPK3SpFSAL49 with adaptive=false, and tspan=(0.0,3.0) on [0,0.25]x[0,1] with periodic x and slip y walls. Stage B repeats the same configuration with polydeg=6 and CFL=0.30 as the single non-repeating resolution check. The code regenerates the polydeg=5 baseline, computes both stages on the identical 512x2048 cell-centered comparison grid at t=2.25 and t=3.0, evaluates the M1/M2/M3 materiality gate, prints the explicit RESOLUTION CHECK verdict, and selects summary_all.png from the already-rendered Stage A or Stage B 2x4 rho/p panel figure. It also implements the raw accepted-step timestep audit, per-variable mirror-indexed symmetry residuals at t=0,0.75,1.5,2.25,3.0, mass drift/positivity/wall diagnostics, and the conditional fixed-mesh symmetry control run. The initial conditions, pressure law, gravity source term, boundary topology, and required diagnostics match the strategy. | N/A | 50 | 19 | 15 | 15 | **99** |
| 5 | Implementation is a 2D DGSEM plus explicit time integration for the compressible Euler equations with gamma=1.4 and a +y gravitational source. Domain is [0,0.25]x[0,1], t in [0,3]. Stage B is selected: polydeg=6, CFL=0.30, HLLC surface flux, pressure-based Hennemann-Gassner shock capturing, density-based Löhner AMR with ControllerThreeLevel base=4 med=5 max=6, thresholds 0.1/0.3, interval=20. Initial conditions match the strategy exactly: tanh density interface at y=0.5, piecewise hydrostatic pressure, zero u, v=-0.025 sqrt(gamma p/rho) cos(8*pi*x) sin(pi*y)^6, and prim2cons energy. y-walls are slip/reflective; x is periodic. The code first attempts to load archived Stage B npz full states at t=0,0.75,1.5,2.25,3.0; if absent it reproduces the identical Stage B configuration. It computes parity-aware reflection residuals on a cell-centered 512x2048 grid: even for rho, rho v, rho E; odd for rho u, with zero-norm absolute fallback. It prints residuals and standard diagnostics (mass drift, min rho/p, wall |v|), applies a connected-component minimum-area filter, and triggers a fixed-mesh control only if corrected t=3 max residual exceeds 1e-2. It renders rho (top row) and p (bottom row) at the four requested times with colorbar=false and copies the Stage B figure to summary_all.png. | N/A | 49 | 20 | 15 | 15 | **99** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | NPZ files are saved in directory `data_npz` with filenames `time_%08.5f.npz` at dt_save=0.1 and additionally at the visualization times 0.75, 1.5, 2.25, 3.0. Each NPZ file contains keys: `t` (scalar Float64), `x` (1D array of node coordinates, length N_nodes(t)), `y` (1D array, length N_nodes(t)), `rho` (1D array, length N_nodes(t)), `rho_u` (1D array), `rho_v` (1D array), `rho_e` (1D array), and `p` (1D array, length N_nodes(t)). N_nodes varies with AMR. Additionally, `SaveSolutionCallback` writes HDF5 solution files to `trixi_output` at dt=0.1 with full conservative state. The plot `summary_all.png` is saved in the working directory, containing a 2×4 grid: top row density, bottom row pressure at the four specified times, with colorbars disabled. |
| 2 | Per-snapshot .npz files (e.g. time_00.75000.npz, time_01.50000.npz, time_02.25000.npz, time_03.00000.npz) written to <script_dir>/data_npz/ via npzwrite. Each file contains a Dict with keys: 't' (scalar Float64), 'x' (Float64 array of DG solution-node x-coordinates from PlotData2D), 'y' (Float64 array of node y-coordinates), 'rho' (Float64 array of DG-node conserved density), 'rho_u' (Float64 array, x-momentum ρu), 'rho_v' (Float64 array, y-momentum ρv), 'rho_e' (Float64 array, total energy ρE), 'p' (Float64 array, pressure from cons2prim). All field arrays (rho, rho_u, rho_v, rho_e, p, x, y) are structured on the same DG-node grid (per-element nodes × elements; flattened with vec() by the rasterizer during plotting). Save times = union of {0.0,0.1,...,3.0} plus the four visualization times {0.75,1.5,2.25,3.0}, ensuring dt_save=0.1 cadence and exact plot snapshots. Additionally, SaveSolutionCallback writes full conservative state (cons2cons) HDF5 snapshots at dt=0.1 (plus initial) into <script_dir>/trixi_output/. Final figure summary_all.png (2 rows × 4 columns, 3200×2400) is saved to <script_dir>/. |
| 3 | Saves NPZ files under data_npz named time_%08.5f.npz (e.g., time_00.00000.npz, time_00.10000.npz, ..., time_03.00000.npz) at times formed from 0.0:0.1:3.0 plus [0.75,1.5,2.25,3.0]. Each NPZ contains keys: 't' as a scalar Float64; 'x' and 'y' as Trixi PlotData2D element-block arrays; and 'rho','rho_u','rho_v','rho_e','p' as PlotData2D element-block arrays matching the x/y layout. In expected block form these are (polydeg+1, polydeg+1, n_elements) = (6,6,n_elements) or a flattened 2D equivalent, with n_elements varying under AMR. The full conservative state is saved via rho, rho_u, rho_v, rho_e, and pressure p is also exported. Additionally, SaveSolutionCallback writes full conservative snapshots to trixi_output at dt=0.1 including the initial solution. Summary visualization is saved as summary_all.png: a 2x4 heatmap grid with density in the top row and pressure in the bottom row, colorbars disabled. |
| 4 | Stage A writes into data_npz/ and trixi_output/; Stage B writes into stage_b_polydeg6/data_npz/ and stage_b_polydeg6/trixi_output/. NPZ full-state snapshots are written as data_npz/time_%08.5f.npz for Stage A and stage_b_polydeg6/data_npz/time_%08.5f.npz for Stage B, containing keys: t (scalar), polydeg (scalar Int), x, y, rho, rho_u, rho_v, rho_e, p. The x/y/rho/rho_u/rho_v/rho_e/p arrays are DG block arrays with shape normally (polydeg+1, polydeg+1, n_elements), where n_elements is AMR/time-dependent; Stage A has first two dimensions 6x6 and Stage B has 7x7. Fixed-grid comparison files are data_npz/fixed_grid_time_02.25000.npz, data_npz/fixed_grid_time_03.00000.npz, and the same under stage_b_polydeg6/data_npz/. They contain keys: t (scalar), x (vector length 512), y (vector length 2048), rho (2048x512 matrix), p (2048x512 matrix), mixing_zone_count (scalar Int), mixing_zone_area (scalar Float64), sharpness (scalar Float64), structural_count (scalar Int), upper_mixing_component_count (scalar Int). Raw timestep histories are saved as data_npz/accepted_step_history.npz and stage_b_polydeg6/data_npz/accepted_step_history.npz when steps exist, with keys: t, tprev, dt, iter, amr_event (all vectors of length N accepted steps; amr_event is Bool), initialization_t (scalar), initialization_dt (scalar). Trixi SaveSolutionCallback also writes full conservative-state HDF5 solution snapshots at dt=0.1 into trixi_output/ and stage_b_polydeg6/trixi_output/. Figures saved are summary_stage_a_polydeg5.png for Stage A, stage_b_polydeg6/summary.png for Stage B, and the selected production panel copied to summary_all.png; all are 2x4 rho/p PNG panels at t=0.75,1.5,2.25,3.0 with size 3200x2400. If the conditional symmetry control triggers, it additionally writes the control run's time_*.npz snapshots and Trixi HDF5 output under symmetric_fixed_control_polydeg<deg>_level<level>/. |
| 5 | Saves under stage_b_polydeg6/data_npz. For a live/reproduced Stage B, time_%08.5f.npz at t=0.0:0.1:3.0 plus 0.75,1.5,2.25,3.0 contains keys 't' scalar, 'polydeg' scalar, and 'x','y','rho','rho_u','rho_v','rho_e','p' as matching 3D DG element-block arrays, typically shape (polydeg+1, polydeg+1, n_elements) = (7,7,n_elements) for Stage B; n_elements varies with AMR. At t=2.25 and 3.0 it also writes fixed_grid_time_%08.5f.npz with keys 't' scalar, 'x' length 512, 'y' length 2048, 'rho' 2048x512, 'p' 2048x512, and scalar metrics 'mixing_zone_count','mixing_zone_area','sharpness','structural_count','upper_mixing_component_count'. If accepted steps are available it writes accepted_step_history.npz with vector keys 't','tprev','dt','iter','amr_event' (length N accepted steps) and scalar keys 'initialization_t','initialization_dt'. Trixi SaveSolutionCallback also writes full cons2cons state at dt=0.1 to stage_b_polydeg6/trixi_output. The 2x4 rho/p figure is saved to stage_b_polydeg6/summary.png and copied to summary_all.png. When using the archived Stage B, existing time_*.npz snapshots are read for post-processing; new time_*.npz files are not rewritten, but the fixed-grid metric npz files and figures are regenerated. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1051/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1109/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1132/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1213/` |
| 5 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1312/` |


---
