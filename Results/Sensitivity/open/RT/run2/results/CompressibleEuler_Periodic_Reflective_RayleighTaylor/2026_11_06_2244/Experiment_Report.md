
# Experiment Report
*Last Updated: 2026-11-06 22:54:40*

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
# Scientific Strategy Report: 2D Compressible Euler Rayleigh-Taylor Instability

**1. Problem Statement**

A gravitationally unstable, single-mode Rayleigh-Taylor instability (heavy fluid $\rho=2$ below light fluid $\rho=1$, effective gravity $+y$ pointing from heavy to light) is evolved to $t=3.0$ through linear growth, nonlinear spike/bubble formation, mushroom-cap rollup, and Kelvin-Helmholtz filament generation. The system is inviscid, compressible, and discontinuous — a shock-capturing, locally adaptive high-order method is mandatory. This is a **refinement trial**: the previous run scored 98/100; the Advisor mandates a fresh standalone run with tightened AMR tracking and an auditable end-of-log verification table.

---

**1. Mathematical Specification (Code-Ready Extraction)**

* **Governing Equations:** 2D Compressible Euler with gravitational source:
$$\partial_t \begin{pmatrix} \rho \\ \rho u \\ \rho v \\ E \end{pmatrix} + \partial_x \begin{pmatrix} \rho u \\ \rho u^2 + p \\ \rho u v \\ u(E+p) \end{pmatrix} + \partial_y \begin{pmatrix} \rho v \\ \rho u v \\ \rho v^2 + p \\ v(E+p) \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ \rho g \\ \rho g v \end{pmatrix}, \qquad E = \frac{p}{\gamma - 1} + \frac{1}{2}\rho(u^2 + v^2)$$

* **Domain:** $(x,y) \in [0, 0.25] \times [0, 1.0]$, $t \in [0, 3.0]$

* **Boundary Conditions:** $x$: Periodic. $y=0$ and $y=1$: Reflective / free-slip walls (no-penetration $v = 0$; tangential velocity, $\rho$, $p$ mirrored).

* **Initial Conditions** (exact, every field):
$$\rho(x,y) = 2.0 + (1.0 - 2.0)\cdot \frac{1}{2}\left(1 + \tanh\left(50(y - 0.5)\right)\right)$$
$$p(x,y) = \begin{cases} 2y + 1 & \text{if } y < 0.5 \\ y + 1.5 & \text{if } y \geq 0.5 \end{cases}$$
$$u(x,y) = 0$$
$$v(x,y) = -0.025 \cdot \sqrt{\frac{\gamma p}{\rho}} \cdot \cos(8\pi x) \cdot \sin^6(\pi y)$$
$$E(x,y) = \frac{p}{\gamma - 1} + \frac{1}{2}\rho\, v(x,y)^2$$

* **Constants & Parameters:** $\gamma = 1.4$; gravity $\mathbf{g} = [0, 1.0]^T$ (source adds momentum in $+y$); $\rho_{heavy} = 2.0$, $\rho_{light} = 1.0$; interface at $y = 0.5$; local sound speed $c = \sqrt{\gamma p / \rho}$; $\mu = 0$, $k = 0$.

* **Expected Physics:** Unstable stratification ($\mathbf{g}$ points from heavy to light) → exponential single-mode growth ($t \lesssim 0.75$) → central heavy spike rising in $+y$ with descending side bubbles ($t \approx 1.5$) → mushroom cap with rolled KH shear layers ($t \approx 2.25$) → elongated spike approaching the $y=1$ wall with thin filaments ($t = 3.0$).

---

**2. Strategic Methodology**

* **Exact solution feasible?** **No.** The unperturbed state is an exact hydrostatic equilibrium ($dp/dy = \rho g$, $v = 0$), but the imposed single-mode perturbation triggers genuinely nonlinear multi-dimensional evolution: mode coupling, secondary KH instabilities on the spike shear layers, and shocklets from compressibility. The Method of Characteristics is restricted to 1D hyperbolic systems/simple waves and cannot treat 2D instability growth with shocks. No closed-form solution exists through $t=3$ (Advisor concurs). Numerical simulation is mandatory.
* **Selection:** **DGSEM in 2D plus time.** The system is hyperbolic with contact discontinuities (the density interface), shear layers, and shocks — requiring local conservation, high order in smooth regions, robust shock capturing, and hp-adaptivity for multi-scale filaments. Fourier fails (non-periodic in $y$; Gibbs at contacts); continuous Galerkin lacks shock robustness; 1D methods are dimensionally inapplicable. This choice is validated by the trial history (96 → 98 scores).

---

**3. Implementation Requirements (CRITICAL)**

* **Configuration:** Advisor-mandated values below **supersede** template defaults. All other solver settings: retain the previously validated configuration (this is a controlled refinement, not a redesign).

* **3.0.1 Adaptive Mesh Refinement & Shock Capturing — Variable Selection Analysis:**
  * *Structural variable:* **Density $\rho$**. Every RTI feature of interest — the interface, spike, mushroom cap, and filaments — is a density discontinuity. Density remains sharp *only* at these features, giving maximal signal-to-noise for refinement and for the HG shock-capturing indicator (dissipation is injected exactly at the material interface where it is needed).
  * *Rejected variables:* Velocity $(u, v)$ becomes globally chaotic throughout the mixing layer by $t \approx 2$ — refining on it wastes resources on turbulence rather than tracking structures. Pressure carries acoustic noise and shocklets uncorrelated with the buoyancy-driven features.
  * **AMR/Shock-capturing sensor: Density $\rho$.**

* **3.1 Advisor-Mandated Adjustments (Overrides):**
  1. **Standalone trial:** Completely fresh run in a new folder. Regenerate all solution data and `summary_all.png` from scratch; no loading, re-rendering, or comparison against prior trial outputs.
  2. **Retain exactly:** `polydeg=5`; HLLC surface flux; Hennemann-Gassner shock capturing on density with `alpha_max=0.25` (do **not** increase dissipation unless the run exhibits ringing or non-positive states); AMR `max_level=6`; `saveat=0.1`; the specified ICs, BCs, and gravity source.
  3. **Controlled local refinement (the only solver change):** AMR `interval`: $20 \to 10$; `med_threshold`: $0.05 \to 0.03$; `max_threshold`: $0.15 \to 0.10$. This tracks emerging thin filaments more promptly without global refinement or added stiffness.
  4. **Verification table — print at the very END of the log so it appears in the STDOUT tail:**
     * Finiteness of all conserved variables (no NaN/Inf);
     * $\min\rho$, $\max\rho$, $\min p$, $\max p$ (positivity certification);
     * Relative mass drift $\left|M(t) - M(0)\right|/M(0)$;
     * Relative drift of modified energy $\mathcal{E}^* = \int (E - \rho g y)\, dA$ (exactly conserved analytically under slip walls + periodic $x$);
     * Maximum wall $|v|$ and wall-normal mass flux at both $y=0$ and $y=1$;
     * Optional: left-right symmetry residual about $x = 0.125$.

---

**4. Visualization Strategy**

* **Field Selection (Analysis):** **Density $\rho$** — the primary driver; buoyancy is set by the density contrast and all RTI structures are density features. **Vertical velocity $v$** — the dominant motion (spike rise in $+y$, bubble descent); $u$ is secondary. Two fields only; do not plot the full primitive set.
* **Task:** Plot $\rho$ (row 1) and $v$ (row 2) at $t = 0.75,\ 1.5,\ 2.25,\ 3.0$.
* **Layout:** Grid, 2 rows (fields) × 4 columns (time).
* **Saving Data:** Save the **full solution state** (all conserved variables) at `saveat = 0.1` for Ground Truth.
* **Constraints:** Save as `summary_all.png` (strictly). **DISABLE COLORBARS** (`colorbar=false`) — Julia. No fixed colormaps; let scales adjust automatically. Plot only primary variables; no derived quantities.

---

**5. Cross-Validation & Data Integration (Conditional)**

* **Trigger check:** No external data paths (sensors, reference solutions) are provided in the problem statement. **Not applicable.** No analytical ground truth exists; quantitative validation is carried by the invariants table (mass, modified energy $\mathcal{E}^*$, positivity, wall fluxes) mandated in §3.1.

---

**REQUIRED COMPONENTS**

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
* Shock Capturing Indicator (Hennemann-Gassner, density-based)
* Adaptive Mesh Refinement (density-based Löhner, three-level controller)
* Advisor-mandated AMR retuning (`interval=10`, `med_threshold=0.03`, `max_threshold=0.10`)
* End-of-log conservation / positivity / wall-behavior verification table

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

* **Status:** **REJECTED_FOR_REFINEMENT**

The simulation is stable and visually plausible, but the reported **42.26% mass drift** and **55.11% modified-energy drift** prevent certification as PINN ground truth. The history-saturation stopping rule is also not met: Run 4 was the first score ≥90, and only Runs 5 and the present trial have followed it, including this trial—fewer than five refinement attempts.

## Section 2: Quantitative Scorecard

* **Score Integrity:** **50/50**
* **Score Accuracy:** **6/20**
* **Score Details:** **13/15**
* **Score Optimality:** **15/15**
* **TOTAL SCORE:** **84/100**

Integrity receives full credit because the run completed, all fields are finite, positivity was maintained, and the figure is readable. Accuracy is heavily penalized because global conservation violations make the quantitative solution untrustworthy despite its plausible appearance.

## Section 3: Forensic Analysis & Feedback

### Observation

**Ground-truth error:** The string `Minimum achieved error:` does **not** appear in the supplied log. **Quantitative error is not applicable as no ground truth is available.** The conservation drifts below are invariant diagnostics, not errors against an analytical solution.

The density sequence shows the expected qualitative Rayleigh–Taylor development:

- At \(t=0.75\), the initially smooth interface develops a broad single-mode deformation.
- At \(t=1.5\), a central heavy spike rises while light-fluid regions descend along the sides.
- At \(t=2.25\), the spike forms a mushroom cap with rolled shear layers.
- At \(t=3.0\), elongated filaments and secondary Kelvin–Helmholtz structures are visible.

The interfaces and filaments are comparatively sharp, with no obvious large-scale Gibbs ringing or blank/NaN regions. The velocity panels are consistent with a strong central rising flow and shear around the rolled structures.

The log confirms:

- All conserved and derived fields are finite.
- Density and pressure remain positive:
  \[
  \rho\in[0.7817708,\;2.7542183],\qquad
  p\in[1.2116414,\;2.6497854].
  \]
- Wall-normal velocities and mass fluxes are small, with maximum wall \(|v|\approx1.42\times10^{-4}\).

However, the final verification table reports:

\[
\frac{|M(t)-M(0)|}{M(0)}=0.4226332329,
\qquad
\frac{|\mathcal E^*(t)-\mathcal E^*(0)|}{|\mathcal E^*(0)|}
=0.5510555284.
\]

For periodic \(x\)-boundaries, impermeable walls, and no mass source,
\[
\frac{d}{dt}\int_\Omega \rho\,dA=0.
\]
Thus, a genuine 42% mass change is impossible for the stated physics. Likewise, \(\mathcal E^*=\int(E-\rho gy)\,dA\) is the appropriate analytical energy invariant for this configuration.

### Diagnosis

**Severe global invariant violation, potentially caused by either faulty conservation diagnostics or an AMR-related conservation regression.**

A conservative DGSEM/HLLC calculation should not exhibit this magnitude of mass drift. Two possibilities must be distinguished:

1. **Diagnostic artifact:** Global integrals may have been computed using unweighted nodal sums, rasterized plotting data, inconsistent meshes, or an initial mass evaluated inconsistently with initial AMR. Changing element populations during AMR can then produce large apparent drifts.
2. **Actual solver defect:** If the integrals are correct, mass jumps around adaptation events would indicate non-conservative AMR transfer/projection or another mesh-update problem. The more aggressive AMR settings are the principal solver change from the previous trial and therefore warrant scrutiny.

This is **not** primarily a CFL, polynomial-degree, or shock-capturing problem. Reducing CFL or adding dissipation cannot justify or repair a 42% mass error. The visually plausible plot cannot override the failed invariants, particularly because independent color scaling can conceal global amplitude changes.

### Targeted Improvements

* **For Integrity:** None for execution stability. The run completed successfully and remained finite and positive.

* **For Accuracy:**
  1. In the next **fresh standalone run**, compute all global integrals using the solver’s current-mesh, Jacobian-weighted Legendre quadrature:
     \[
     M_h=\sum_e\sum_{i,j}\omega_i\omega_jJ_{e,ij}\rho_{e,ij},
     \]
     with an analogous quadrature for \(\mathcal E^*\). Do not use nodal sums or interpolated image/raster data.
  2. Establish \(M(0)\) and \(\mathcal E^*(0)\) consistently **after initialization and initial AMR**, and report both signed and absolute drift.
  3. Save these invariants at every `saveat=0.1` and immediately before and after each AMR event. Audit the actual numerical wall boundary fluxes, not merely interior near-wall values of \(\rho v\).
  4. Keep `polydeg=5`, HLLC, and `CFL=0.3`. As a new, less aggressive adaptation configuration, use `interval=15`, `med_threshold=0.04`, and `max_threshold=0.12`, retaining `max_level=6`.
  5. Acceptance should require mass conservation near solver/quadrature precision—target relative drift \(\lesssim10^{-8}\)—and a small, smoothly accumulating modified-energy drift, target \(\lesssim10^{-3}\), with no jumps at AMR events. If mass jumps persist at adaptation, perform a fresh control run with time-dependent AMR frozen after initial adaptation to isolate the mesh-transfer defect.

* **For Details:** Retain `alpha_max=0.25`; additional dissipation is not supported by the plots. Add a 10–90% density-interface thickness diagnostic to verify that the extra AMR cost improves filament resolution. Do not increase polynomial degree or refine further until conservation passes.

* **For Optimality:** None. DGSEM with HLLC, density-based shock capturing, and AMR remains appropriate. No exact-method override is warranted: the nonlinear two-dimensional RTI evolution with vortex rollup has no feasible closed-form analytical solution.

All corrections, solution data, diagnostics, and `summary_all.png` must be regenerated from scratch in the next standalone trial.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 2 | Numerical implementation of 2D inviscid compressible Euler RTI using DGSEM in Trixi. Uses CompressibleEulerEquations2D with gamma=1.4, P4estMesh on [0,0.25]x[0,1] with periodicity=(true,false), initial_refinement_level=3, polydeg=3. DGSEM has surface_flux=flux_hll and VolumeIntegralShockCapturingHG with IndicatorHennemannGassner on density (alpha_max=1.0, alpha_min=0.001, alpha_smooth=true) blending flux_ranocha and flux_lax_friedrichs. Slip-wall boundary conditions are set on y_neg/y_pos; x is periodic. Gravity source term implements S=(0,0,rho*g,g*rho*v) with g=1.0 in +y. Initial conditions match the strategy: smoothed tanh density interface at y=0.5 (rho=2 below, rho=1 above), piecewise hydrostatic pressure (2y+1 for y<0.5, y+1.5 for y>=0.5), u=0, v=-0.025*c*cos(8*pi*x)*sin(pi*y)^6 with c=sqrt(gamma*p/rho), and energy via prim2cons. AMR uses IndicatorLöhner on density with ControllerThreeLevel base_level=4, med_level=5, med_threshold=0.1, max_level=6, max_threshold=0.3, interval=20, adapt_initial_condition=true. Time integration is RDPK3SpFSAL49, adaptive=false, tspan=(0.0,3.0), CFL=0.5 via StepsizeCallback, saveat=0.1. Visualization plots density rho and vertical velocity v at t=0.75,1.5,2.25,3.0 in a 2x4 layout, colorbar=false, no fixed colormap, saving summary_all.png. | N/A | 40 | 15 | 8 | 12 | **75** |
| 4 | The generated code implements the requested 2D inviscid compressible Euler RTI with gravity using Trixi.jl DGSEM. Domain [0,0.25]x[0,1], t in [0,3]. Equations use CompressibleEulerEquations2D(1.4); gravity source S=(0,0,rho*g,rho*g*v) with g=1. IC matches: smoothed density interface at y=0.5 via tanh(50(y-0.5)), piecewise hydrostatic pressure, u=0, v=-0.025*sqrt(gamma p/rho)*cos(8πx)*sin^6(πy). BCs: periodic x, slip walls y. Solver: LobattoLegendreBasis polydeg=5, DGSEM with surface_flux=flux_hllc; shock capturing via IndicatorHennemannGassner on density with alpha_max=0.5, alpha_min=0.001, alpha_smooth=true, volume flux blend flux_ranocha/flux_lax_friedrichs. AMR is density-based Löhner with ControllerThreeLevel max_level=6, adapt_initial_condition=true. CFL=0.3. Time integration RDPK3SpFSAL49, adaptive=false, tspan=(0,3). Runtime diagnostics print finiteness, positivity, rho/p extrema, mass drift, modified energy drift, wall max|v|. Visualization plots only rho and v at t=0.75,1.5,2.25,3.0 in a 2x4 layout with colorbar=false, AMR-consistent rasterization to 256x1024, and saves summary_all.png. | N/A | 50 | 18 | 13 | 15 | **96** |
| 5 | The implementation solves the 2D inviscid compressible Euler equations with gravity using Trixi.jl DGSEM on a P4estMesh over [0,0.25] x [0,1.0]. It uses polydeg=5, HLLC surface flux, Hennemann-Gassner shock capturing on density with alpha_max=0.25, alpha_min=0.001, alpha_smooth=true, and a volume flux blend using flux_ranocha and flux_lax_friedrichs. The mesh is periodic in x and uses slip walls at y=0 and y=1. AMR is density-based with a Löhner indicator and a three-level controller using base_level=4, med_level=5, med_threshold=0.05, max_level=6, max_threshold=0.15, adapt_initial_condition=true, and interval=20. Time integration uses RDPK3SpFSAL49 with CFL=0.3 over t in [0,3]. The initial conditions exactly match the strategy: tanh-smoothed density interface, piecewise hydrostatic pressure, zero u, and the prescribed single-mode vertical velocity perturbation. The gravity source is (0,0,rho,rho*v) with g=1. Runtime diagnostics compute and print finiteness, min rho and min p, relative mass drift, relative modified-energy drift E* = integral(E - rho*y)dA, maximum wall |v|, wall-normal mass flux at both walls, density and pressure extrema at the four plotting times, and a final verification table. Visualization plots density and vertical velocity at t=0.75,1.5,2.25,3.0 in a 2x4 grid with no colorbars, automatic scales, equal aspect ratio, and saves summary_all.png. | N/A | 50 | 19 | 14 | 15 | **98** |
| 6 | The code implements the required 2D compressible Euler Rayleigh-Taylor simulation using DGSEM on a P4est mesh over [0,0.25] x [0,1] with t in [0,3]. The equations are CompressibleEulerEquations2D(gamma=1.4) with gravity source [0,0,rho*g,rho*g*v], g=1.0. Initial conditions match the strategy: smoothed density tanh profile, piecewise hydrostatic pressure, u=0, v=-0.025*sqrt(gamma*p/rho)*cos(8*pi*x)*sin(pi*y)^6, and energy via prim2cons. Boundary conditions are periodic in x and slip walls at y=0 and y=1. The discretization uses polydeg=5, HLLC surface flux, Hennemann-Gassner shock capturing on density with alpha_max=0.25, and density-based Loehner AMR with interval=10, med_threshold=0.03, max_threshold=0.10, max_level=6. Time integration uses RDPK3SpFSAL49 with CFL=0.3 and adaptive=false. The code saves full conservative data at saveat=0.1, plots rho and v at t=0.75,1.5,2.25,3.0 in a 2x4 layout with colorbar disabled, saves summary_all.png, and prints an end-of-log verification table covering finiteness, positivity, min/max rho and p, relative mass drift, relative modified-energy drift, and wall |v| / wall-normal mass flux. | N/A | 50 | 6 | 13 | 15 | **84** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 2 | Creates a fresh rti_run/ directory and rti_run/data_npz/. NPZ snapshots are written with npzwrite to files named time_%08.5f.npz (e.g., time_00.75000.npz) at t=0.0,0.1,0.2,...,3.0 plus 0.75,1.5,2.25. Each NPZ file contains keys: 't' (scalar Float64), 'x' (1D vector length N_points), 'y' (1D vector length N_points), 'rho' (1D vector length N_points), 'rho_u' (1D vector length N_points), 'rho_v' (1D vector length N_points), 'rho_e' (1D vector length N_points), 'u' (1D vector length N_points), 'v' (1D vector length N_points), 'p' (1D vector length N_points). N_points is the number of PlotData2D visualization points from the adaptive DG mesh and varies with AMR. Additionally, SaveSolutionCallback writes Trixi HDF5 solution files to rti_run/trixi_output/ at dt=0.1 with initial and final solutions saved, using solution_variables=cons2cons. The final 2x4 visualization is saved as summary_all.png in the script directory. |
| 4 | NPZ snapshots are written to rti_run/data_npz/time_%08.5f.npz at times 0.0:0.1:3.0 plus 0.75 and 2.25. Each NPZ contains keys: t (scalar Float64); x, y, rho, rho_u, rho_v, rho_e, u, v, p (arrays with the raw Trixi PlotData2D element-patch shape; the code does not reshape before writing). With default PlotData2D sampling this is the AMR element-patch layout, typically nvis=2*polydeg+1=11 per element and n_elements AMR-dependent. The code also uses SaveSolutionCallback(dt=0.1, solution_variables=cons2cons) writing full DG conservative-state snapshots to rti_run/trixi_output. The final figure summary_all.png is saved at @__DIR__/summary_all.png with size 1600x2200, layout 2 rows (rho, v) by 4 columns (t=0.75,1.5,2.25,3.0), and colorbar=false. |
| 5 | Creates a fresh standalone run directory with mktempdir. NPZ snapshots are written to run_dir/data_npz as time_%08.5f.npz (e.g., time_00.75000.npz) at t=0.0,0.1,...,3.0 plus the visualization times 0.75,1.5,2.25,3.0. Each .npz contains keys: 't' (scalar Float64); 'x' and 'y' (PlotData2D node coordinates, identical shape); 'rho', 'rho_u', 'rho_v', 'rho_e' (conserved variables, identical shape to x/y); and 'u', 'v', 'p' (primitive fields, identical shape). These arrays are not resized to a uniform global grid; their shapes are inherited from Trixi.PlotData2D and are AMR-dependent. The code supports multiple PlotData2D layouts (3D element arrays, 2D matrices, and flattened/vector forms) for later rasterization. In addition, Trixi.SaveSolutionCallback writes full DG solution state HDF5 files at dt=0.1 to run_dir/trixi_output. The final visualization is saved as run_dir/summary_all.png (PNG). |
| 6 | Two saving paths are implemented. (1) Trixi SaveSolutionCallback writes full conservative-state HDF5 snapshots to run_dir/trixi_output at dt=0.1 (saveat=0.1), including initial and final states, using solution_variables=cons2cons. (2) A separate NPZ callback writes per-time files to run_dir/data_npz/time_%08.5f.npz (e.g., time_00.00000.npz ... time_03.00000.npz) at the 0.0:0.1:3.0 schedule plus the visualization times 0.75, 1.5, 2.25, 3.0. Each NPZ contains keys: 't' scalar Float64; 'x', 'y', 'rho', 'rho_u', 'rho_v', 'rho_e', 'u', 'v', 'p'. The arrays 'x', 'y', and all field arrays have identical dimensions as returned by Trixi PlotData2D; the code does not reshape them before NPZ writing. The plotting artifact is run_dir/summary_all.png, a 2x4 heatmap layout of rho and v at the four requested times with colorbar=false. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run2/results//CompressibleEuler_Periodic_Reflective_RayleighTaylor/2026_11_06_2147/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run2/results//CompressibleEuler_Periodic_Reflective_RayleighTaylor/2026_11_06_2218/` |
| 5 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run2/results//CompressibleEuler_Periodic_Reflective_RayleighTaylor/2026_11_06_2230/` |
| 6 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run2/results//CompressibleEuler_Periodic_Reflective_RayleighTaylor/2026_11_06_2244/` |


---
