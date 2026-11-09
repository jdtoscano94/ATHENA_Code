
# Experiment Report
*Last Updated: 2026-11-06 11:44:54*

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
# Scientific Strategy Report — Rayleigh–Taylor Instability (2D Compressible Euler)

---

## 1. Problem Statement

Solve the 2D inviscid compressible Euler equations with gravitational source for the single-mode Rayleigh–Taylor Instability (RTI): a heavy fluid ($\rho=2$) sits below a light fluid ($\rho=1$) with effective gravity pointing in $+y$, making the configuration buoyantly unstable. A single-mode velocity perturbation seeds the growth of a rising spike and descending bubbles, culminating in mushroom roll-up, secondary Kelvin–Helmholtz vortices, and interaction with the upper wall by $t=3.0$.

This is **Run 3**. The DGSEM/HLLC/decoupled-indicator architecture (Run 2, score 93) is physically correct and is retained. The remaining defects are (a) a **visualization striping artifact** from element-wise rendering and (b) insufficient resolution of the smallest late-time vortices. The Advisor mandates a renderer fix plus a polynomial-degree/CFL refinement — no architectural changes.

---

## 1. Mathematical Specification (Code-Ready Extraction)

* **Governing Equations:** 2D compressible Euler with gravity source ($\gamma = 1.4$):

$$
\frac{\partial}{\partial t}\begin{pmatrix}\rho\\ \rho u\\ \rho v\\ E\end{pmatrix}
+ \frac{\partial}{\partial x}\begin{pmatrix}\rho u\\ \rho u^2 + p\\ \rho u v\\ u(E+p)\end{pmatrix}
+ \frac{\partial}{\partial y}\begin{pmatrix}\rho v\\ \rho u v\\ \rho v^2 + p\\ v(E+p)\end{pmatrix}
= \begin{pmatrix}0\\ 0\\ \rho g\\ \rho g\, v\end{pmatrix},
\qquad
E = \frac{p}{\gamma - 1} + \frac{1}{2}\rho\left(u^2 + v^2\right)
$$

* **Domain:** $(x,y) \in [0,\,0.25] \times [0,\,1.0]$, $\quad t \in [0,\,3.0]$
* **Boundary Conditions:**
  * $x$-direction: **Periodic** ($x=0 \leftrightarrow x=0.25$).
  * $y=0$ and $y=1$: **Reflective / free-slip walls** (no-penetration $v=0$; slip-wall boundary condition).
* **Initial Conditions** (interface at $y=0.5$):

$$
\rho(x,y) = 2.0 + (1.0 - 2.0)\cdot\frac{1}{2}\left(1 + \tanh\left(50\,(y - 0.5)\right)\right)
$$

$$
p(x,y) = \begin{cases} 2y + 1 & \text{if } y < 0.5 \\ y + 1.5 & \text{if } y \geq 0.5 \end{cases}
\qquad \text{(hydrostatic: } \tfrac{dp}{dy} = \rho g,\ \text{continuous at } y=0.5\text{)}
$$

$$
u(x,y) = 0, \qquad
v(x,y) = -0.025\; c(x,y)\, \cos(8\pi x)\, \sin^6(\pi y), \qquad c = \sqrt{\gamma\, p / \rho}
$$

$$
E(x,y) = \frac{p}{\gamma - 1} + \frac{1}{2}\rho\, v^2
$$

* **Constants & Parameters:** $\gamma = 1.4$; gravity $g = 1.0$ in $+y$ (source $S = (0,\,0,\,\rho g,\,\rho g v)^T$); $\rho_{heavy}=2.0$, $\rho_{light}=1.0$; interface smoothing factor $50$; perturbation amplitude $0.025$; mode $\cos(8\pi x)$; envelope $\sin^6(\pi y)$; $\mu = 0$, $k = 0$.
* **Expected Physics:** Upward heavy-fluid spike at $x=0.125$, downward light-fluid bubbles at periodic edges; spike formation by $t\approx0.75$, narrow rising spike at $t\approx1.5$, mushroom roll-up at $t\approx2.25$, complex secondary vortices interacting with the upper wall at $t=3.0$. Left–right symmetry about $x=0.125$; pressure remains predominantly hydrostatic with late-time compression near $y=1$.

---

## 2. Strategic Methodology

**Can an exact solution be obtained? No.**
* **Method of Characteristics** is inapplicable: the system is nonlinear, two-dimensional, and source-terms-driven. Characteristics from the single-mode perturbation intersect (caustics) within the spike formation, and the late-time mushroom roll-up is inherently multi-valued — no implicit characteristic system can describe the full evolution. No entropy-condition branch selection can rescue a 2D multi-dimensional roll-up.
* **Fourier Spectral** is inapplicable: the $y$-direction has reflective walls (non-periodic), and the solution develops sharp material contacts and vortex sheets — global Gibbs contamination.
* **Separation of variables / Green's functions** fail for the same reasons (nonlinear advection + discontinuous interface evolution).
* The Advisor explicitly confirms: no exact-method override is warranted for this problem.

**Selection: Discontinuous Galerkin Spectral Element Method in 2D plus time (DGSEM).**
This architecture has been validated across two runs (89 → 93) and correctly reproduces the full RTI sequence. The strategy is surgical: retain the architecture, apply the Advisor-mandated resolution refinement (polydeg 3 → 5, CFL 0.5 → 0.35), and eliminate the visualization artifact via uniform-grid sampling. No additional mesh refinement, no additional stabilization.

---

## 3. Implementation Requirements (CRITICAL)

* **Configuration:** Use template defaults, **except** for the Advisor-mandated overrides in §3.1.

### 3.0.1 Adaptive Mesh Refinement & Shock Capturing — Variable Selection Analysis

* **AMR variable: Density ($\rho$).** The structural feature of interest is the **material interface** (heavy/light contact, spike, mushroom rim). Density is sharp exactly at this interface and smooth everywhere else — a high signal-to-noise tracer. **Velocity** is rejected: the late-time Kelvin–Helmholtz roll-up makes the velocity field globally chaotic, which would trigger refinement across the entire domain and waste elements. **Pressure** is rejected for AMR: it is dominated by the smooth hydrostatic stratification $\partial p/\rho = g\,dy$ spanning the whole domain and would not localize the interface. → Density-based Löhner indicator with `ControllerThreeLevel` (base 4 / med 5 / max 6), interval 20, `adapt_initial_condition=true` — **retained unchanged**.
* **Shock-capturing variable: Pressure ($p$).** Compressive features (acoustic waves, wall compression at late time) manifest as pressure jumps, while the material contact is in approximate pressure equilibrium. A pressure-based Hennemann–Gassner indicator therefore captures genuine compression **without** over-diffusing the density contact. The **decoupling** of the two indicators (pressure → shock capturing, density → AMR) is the endorsed architecture and is retained.

### 3.1 Advisor-Mandated Adjustments (Overrides)

These supersede template defaults and the Run 2 configuration:

1. **`polydeg = 5`** (replaces 3) — final resolution refinement; more efficient for smooth vortex cores than added mesh refinement. **Do not increase AMR levels.**
2. **`CFL ≈ 0.35`** (replaces 0.5) — stability margin for the higher polynomial degree.
3. **Renderer replacement (Integrity fix):** Replace the element-wise plotting path with **uniformly sampled solution arrays rendered as one heatmap/image per panel** — no element edge strokes, no inter-element gaps, no per-element sub-rectangles.
4. **Decisive rendering test at $t=0$:** Sample the initial condition through the *same* rendering path. Density and pressure must be **exactly $x$-independent**; print the max deviation over $x$ to stdout. Any vertical stripes at $t=0$ are conclusively renderer artifacts and must be eliminated before attributing stripes to numerics.
5. **Final compact diagnostic summary in stdout** (auditable values, printed at end of run):
   * Relative mass drift $\left|\,M(t)-M(0)\right|/M(0)$ (must be near conservation/AMR tolerance),
   * $\min\rho$ and $\min p$ (positivity),
   * $dt_{\min}/dt_{\max}$ (CFL behavior),
   * Max wall-normal velocity $|v|$ at $y=0$ and $y=1$ (no-penetration),
   * Symmetry residual $\|q - q(0.25-x)\|/\|q\|$ about $x=0.125$.
6. **Retained unchanged (do NOT modify):** HLLC surface flux; pressure-based HG shock capturing (DG volume flux `flux_ranocha`, FV flux `flux_lax_friedrichs`); density-based Löhner AMR with current limits; initial conditions; gravity source $(0,0,\rho g,\rho g v)$; boundary conditions; time integrator `RDPK3SpFSAL49`, `adaptive=false`, `tspan=(0.0, 3.0)`.
7. **No additional stabilization.** Inspect the resulting contacts for ringing; only act if ringing is demonstrated.

---

## 4. Visualization Strategy

* **Field Selection (2 fields):**
  * **Density $\rho$** — the primary driver and tracer of RTI: interface displacement, spike, mushroom, and mixing are all density features.
  * **Pressure $p$** — verifies hydrostatic balance, gravity direction, and late-time wall compression.
  * Velocity and energy are excluded: they add clutter without new physical information, and no derived quantities (e.g., vorticity) are requested.
* **Task:** Plot $\rho$ (top row) and $p$ (bottom row) at **4 time steps: $t = 0.75,\ 1.5,\ 2.25,\ 3.0$**.
* **Layout:** 2 rows × 4 columns grid.
* **Rendering (CRITICAL):** Each panel is a **single heatmap from a uniformly sampled Cartesian array** (interpolate the DG solution onto a uniform grid, or use the solver's uniform-sampling plot data). No mesh lines, no element boundaries, no gaps. Apply the $t=0$ $x$-independence test of §3.1(4) through this exact path.
* **Saving Data:** Save the **full solution state** (all conservative variables) at frequent intervals (`dt_save = 0.1`) for Ground Truth. Plotting selection affects only `summary_all.png`.
* **Constraints:**
  * Save as: `summary_all.png` (strictly).
  * **DISABLE COLORBARS** (`colorbar=false`) — Julia.
  * Automatic per-panel color scaling; no fixed colormaps or manual limits.

---

## 5. Cross-Validation & Data Integration (Conditional)

* **Trigger check:** No external data paths (sensors, reference solutions, `.npz` files) are provided in [USER_QUESTION]. → **Not applicable.**
* Internal validation is performed via the §3.1(5) diagnostic suite (mass drift, positivity, wall no-penetration, symmetry) printed to stdout, and the $t=0$ renderer test.

---

## REQUIRED COMPONENTS

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
* Shock Capturing Indicator (pressure-based)
* Adaptive Mesh Refinement (density-based)
* Advisor-mandated resolution refinement: `polydeg=5`, `CFL=0.35`
* Uniform-grid sampling renderer (single heatmap per panel, no element edges/gaps)
* Final diagnostic summary printout (mass drift, positivity, wall $|v|$, symmetry residual)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

**Status:** `REJECTED_FOR_REFINEMENT`

The run is scientifically strong and completes successfully, but it does not achieve a perfect score. Run 2 previously scored **93**, so the history-saturation rule is active. However, **Run 3 is only the first trial executed since that success**—the count is **1 < 5**. The verification phase must therefore continue unless a subsequent run scores 100.

## Section 2: Quantitative Scorecard

- **Score Integrity:** 49/50
- **Score Accuracy:** 19/20
- **Score Details:** 14/15
- **Score Optimality:** 15/15
- **TOTAL SCORE:** **97/100**

**Error-norm check:** The supplied STDOUT tail contains **no** `Minimum achieved error:` line and no solution error norm. This is a forward nonlinear RTI simulation without an analytical reference. **Quantitative error is not applicable as no ground truth is available.** The mass drift and symmetry residual are diagnostics, not solution errors against a reference.

## Section 3: Forensic Analysis & Feedback

### Observation

- **Execution and conservation:** The solver reaches \(t=3.0\), generates `summary_all.png`, and reports finite, positive fields:
  - Relative mass drift: \(4.9155\times10^{-12}\), excellent for an AMR calculation.
  - \(\min\rho=0.35441>0\) and \(\min p=1.20991>0\): positivity is maintained.
  - Maximum wall-normal velocities are approximately \(9.77\times10^{-5}\) at \(y=0\) and \(3.08\times10^{-4}\) at \(y=1\), indicating good no-penetration behavior.
- **Renderer correction:** The previous element-striping artifact is no longer visible. The \(t=0\) test gives maximum \(x\)-deviations of \(1.44\times10^{-14}\) in density and \(1.38\times10^{-8}\) in pressure, confirming effective initial \(x\)-independence through the plotting path.
- **Density evolution:** The panels reproduce the expected RTI sequence:
  - \(t=0.75\): single-mode interface displacement.
  - \(t=1.5\): narrow rising heavy-fluid spike with descending light-fluid regions.
  - \(t=2.25\): mushroom formation and roll-up.
  - \(t=3.0\): secondary Kelvin–Helmholtz structures and interaction with the upper wall.
- **Pressure:** The predominantly hydrostatic vertical stratification is retained, with late-time compression near the upper wall. No obvious nonphysical pressure oscillation is visible.
- **Fine-scale quality:** Contacts and mushroom rims appear sharp, with no clear Gibbs ringing or gross interface smearing. Nevertheless, visual sharpness alone does not establish convergence of the smallest \(t=3\) vortices.
- **Remaining diagnostic concerns:**
  - The **symmetry residual is \(3.4052\times10^{-2}\)**, or approximately **3.4%**. For this formally symmetric problem, that is large enough to investigate, although late-time chaotic dynamics, non-symmetric AMR, or an interpolation/indexing error in the diagnostic could explain it.
  - `dt_min=5.551115e-17` is suspicious. Given the successful completion and otherwise reasonable `dt_max=3.89756e-4`, it is likely a final-step clipping or diagnostic artifact, but this must be verified rather than assumed.

### Diagnosis

This is a **high-quality, stable DGSEM RTI calculation**, with the remaining deficiencies primarily in **verification and diagnostics**, not in the physical architecture. There is no demonstrated need for a more dissipative flux or additional shock capturing; either would risk unnecessarily smearing the material interface.

The exact-method check also favors retaining the current approach. The full nonlinear, two-dimensional compressible Euler system with gravity, wall interaction, and vortex roll-up has no feasible closed-form analytical solution. Linearized Fourier analysis or characteristics could describe only restricted early-time behavior, not the requested evolution through \(t=3\).

### Targeted Improvements

- **For Integrity:**
  - Audit the timestep diagnostic using the raw accepted-step history. Report the minimum over ordinary integration steps separately from any final `tspan`-clipping step and from initialization values.
  - If genuinely tiny steps occur repeatedly, inspect the interaction between AMR reinitialization and `StepsizeCallback`; only then consider reducing CFL to **0.30**. Do not blindly tighten CFL based solely on the current summary.

- **For Accuracy:**
  - Recompute the symmetry residual on a **cell-centered symmetric uniform grid without a duplicated periodic endpoint**, using the exact index pairing \(i\leftrightarrow N_x-1-i\). Report residuals separately for each conservative variable and at several times to identify when asymmetry first grows.
  - If the residual remains significant after correcting the diagnostic, use **mirror-paired AMR refinement/coarsening**, or a symmetric fixed-mesh control run at comparable degrees of freedom, to distinguish AMR-induced symmetry breaking from physical chaotic amplification.

- **For Details:**
  - Perform one non-repeating resolution check with **`polydeg=6`, `CFL=0.30`**, retaining HLLC, the present pressure-based HG indicator, and the current AMR level limits. Compare the density interface and vortex cores at \(t=2.25\) and \(t=3.0\) on an identical sampling grid.
  - Use the higher-degree result as the production configuration only if it materially changes those features. Otherwise, the present `polydeg=5` run is the more cost-effective choice.
  - **Do not** revert to HLL/Lax–Friedrichs surface fluxes or add indiscriminate stabilization; no ringing is demonstrated, and those changes would increase contact dissipation.

- **For Optimality:** None. DGSEM with HLLC, pressure-based shock capturing, density-based AMR, and the current explicit low-storage integrator remains an appropriate architecture for this problem.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | The code implements the 2D compressible Euler equations with gravitational source using Trixi.jl. The spatial discretization is DGSEM with polynomial degree 3 on a P4estMesh (domain [0,0.25]×[0,1], periodic in x, reflective slip walls in y via boundary_condition_slip_wall). Shock capturing is density-based using IndicatorHennemannGassner with a hybrid DG-FV volume integral (flux_ranocha for DG, flux_lax_friedrichs for FV) and HLL surface flux. AMR is density-based using IndicatorLöhner with ControllerThreeLevel (base_level=4, med_level=5, max_level=6) and AMRCallback interval=20. The initial condition exactly matches the strategy: tanh density profile, piecewise hydrostatic pressure, zero u, and the specified single-mode v perturbation. Gravity source is (0,0,ρg,ρgv) with g=1.0. Time integration uses RDPK3SpFSAL49 with adaptive=false, StepsizeCallback cfl=0.5, and tspan=(0.0,3.0). Visualization produces summary_all.png with density (top row) and pressure (bottom row) at t=0.75, 1.5, 2.25, 3.0, with colorbars disabled and automatic color scaling. | N/A | 47 | 18 | 11 | 13 | **89** |
| 2 | The implementation faithfully reproduces the DGSEM solution of the 2D inviscid compressible Euler equations with a gravitational source for the single-mode Rayleigh–Taylor instability. Governing equations use CompressibleEulerEquations2D(γ=1.4); the initial condition is the verbatim strategy profile (smoothed density 2→1 via tanh(50(y-0.5))·0.5, piecewise pressure 2y+1 / y+1.5 at y=0.5, u=0, v=-0.025·c·cos(8πx)·sin^6(πy) with c=sqrt(γp/ρ), and energy via prim2cons giving E=p/(γ-1)+0.5ρv²). The gravity source term returns SVector(0,0,g·ρ, g·ρv), i.e. (0,0,ρg,ρgv). Discretization: P4estMesh on [0,0.25]×[1.0] with x-periodic (periodicity=(true,false)) and slip-wall BCs on :y_neg/:y_pos; DGSEM with polydeg=3, LobattoLegendreBasis. Advisor-mandated upgrades are present: surface_flux=flux_hllc, pressure-based IndicatorHennemannGassner shock capture (pressure_indicator derived from conservative state) feeding VolumeIntegralShockCapturingHG with volume_flux_dg=flux_ranocha and volume_flux_fv=flux_lax_friedrichs, while AMR uses a separate density-based IndicatorLöhner (variable=density_indicator=u[1]) with ControllerThreeLevel (base 4 / med 5 @0.1 / max 6 @0.3), interval=20, adapt_initial_condition=true. Held-fixed config is respected: RDPK3SpFSAL49 time integrator, adaptive=false, StepsizeCallback(cfl=0.5), tspan=(0.0,3.0). The full diagnostic suite is implemented and printed at every save: total mass (Trixi.integrate) with absolute/relative drift vs t=0, min/max ρ and p (positivity), dt_min/dt_max/current dt (CFL), max|v| at y=0 and y=1 (wall no-penetration) and an approximate symmetry residual ‖q-q(0.25-x)‖/‖q‖ about x=0.125. Visualization renders a clean 2×4 grid (density top row, pressure bottom row) at t=0.75,1.5,2.25,3.0 with colorbar=false, no mesh lines (framestyle=:none, axis=nothing), per-panel automatic scaling, and higher resolution size=(3200,2400) saved as summary_all.png. | N/A | 47 | 19 | 12 | 15 | **93** |
| 3 | Implementation is a numerical DGSEM solver for the 2D compressible Euler equations with gamma=1.4 and the prescribed gravity source. Domain is [0,0.25] x [0,1] with t in [0,3], periodic x-boundaries, and slip walls at y=0 and y=1. Initial conditions match the strategy: smoothed tanh density interface at y=0.5, piecewise hydrostatic pressure, u=0, v=-0.025*c*cos(8*pi*x)*sin(pi*y)^6, and total energy through prim2cons. Spatial discretization uses polydeg=5 Lobatto basis, HLLC surface flux, pressure-based Hennemann-Gassner shock capturing with flux_ranocha DG and flux_lax_friedrichs FV, and density-based Lohner AMR with ControllerThreeLevel base=4/med=5/max=6, interval=20, adapt_initial_condition=true. Time integration uses RDPK3SpFSAL49 with adaptive=false, tspan=(0.0,3.0), and CFL=0.35 via StepsizeCallback. Visualization samples the DG solution onto a uniform Cartesian grid and saves summary_all.png as a 2x4 heatmap grid of density and pressure at t=0.75,1.5,2.25,3.0 with colorbar=false. The code also performs the required t=0 renderer x-independence test and prints the final diagnostic summary for mass drift, positivity, dt extrema, wall |v|, and symmetry residual. | N/A | 49 | 19 | 14 | 15 | **97** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | NPZ files are saved in directory `data_npz` with filenames `time_%08.5f.npz` at dt_save=0.1 and additionally at the visualization times 0.75, 1.5, 2.25, 3.0. Each NPZ file contains keys: `t` (scalar Float64), `x` (1D array of node coordinates, length N_nodes(t)), `y` (1D array, length N_nodes(t)), `rho` (1D array, length N_nodes(t)), `rho_u` (1D array), `rho_v` (1D array), `rho_e` (1D array), and `p` (1D array, length N_nodes(t)). N_nodes varies with AMR. Additionally, `SaveSolutionCallback` writes HDF5 solution files to `trixi_output` at dt=0.1 with full conservative state. The plot `summary_all.png` is saved in the working directory, containing a 2×4 grid: top row density, bottom row pressure at the four specified times, with colorbars disabled. |
| 2 | Per-snapshot .npz files (e.g. time_00.75000.npz, time_01.50000.npz, time_02.25000.npz, time_03.00000.npz) written to <script_dir>/data_npz/ via npzwrite. Each file contains a Dict with keys: 't' (scalar Float64), 'x' (Float64 array of DG solution-node x-coordinates from PlotData2D), 'y' (Float64 array of node y-coordinates), 'rho' (Float64 array of DG-node conserved density), 'rho_u' (Float64 array, x-momentum ρu), 'rho_v' (Float64 array, y-momentum ρv), 'rho_e' (Float64 array, total energy ρE), 'p' (Float64 array, pressure from cons2prim). All field arrays (rho, rho_u, rho_v, rho_e, p, x, y) are structured on the same DG-node grid (per-element nodes × elements; flattened with vec() by the rasterizer during plotting). Save times = union of {0.0,0.1,...,3.0} plus the four visualization times {0.75,1.5,2.25,3.0}, ensuring dt_save=0.1 cadence and exact plot snapshots. Additionally, SaveSolutionCallback writes full conservative state (cons2cons) HDF5 snapshots at dt=0.1 (plus initial) into <script_dir>/trixi_output/. Final figure summary_all.png (2 rows × 4 columns, 3200×2400) is saved to <script_dir>/. |
| 3 | Saves NPZ files under data_npz named time_%08.5f.npz (e.g., time_00.00000.npz, time_00.10000.npz, ..., time_03.00000.npz) at times formed from 0.0:0.1:3.0 plus [0.75,1.5,2.25,3.0]. Each NPZ contains keys: 't' as a scalar Float64; 'x' and 'y' as Trixi PlotData2D element-block arrays; and 'rho','rho_u','rho_v','rho_e','p' as PlotData2D element-block arrays matching the x/y layout. In expected block form these are (polydeg+1, polydeg+1, n_elements) = (6,6,n_elements) or a flattened 2D equivalent, with n_elements varying under AMR. The full conservative state is saved via rho, rho_u, rho_v, rho_e, and pressure p is also exported. Additionally, SaveSolutionCallback writes full conservative snapshots to trixi_output at dt=0.1 including the initial solution. Summary visualization is saved as summary_all.png: a 2x4 heatmap grid with density in the top row and pressure in the bottom row, colorbars disabled. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1051/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1109/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1132/` |


---
