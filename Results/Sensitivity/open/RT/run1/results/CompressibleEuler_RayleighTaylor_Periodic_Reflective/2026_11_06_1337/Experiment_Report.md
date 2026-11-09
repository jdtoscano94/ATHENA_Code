
# Experiment Report
*Last Updated: 2026-11-06 13:45:37*

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
# Scientific Strategy Report — RTI Refinement Iteration (Post-98/100)

## 1. Problem Statement

We solve the single-mode 2D compressible Rayleigh–Taylor instability (heavy fluid $\rho=2$ below light fluid $\rho=1$, effective gravity in $+y$) on $[0,0.25]\times[0,1]$ up to $t=3$. The physics configuration is **frozen** at the Advisor-endorsed 98/100 setup (DGSEM $p=5$, HLLC/Ranocha, density-based AMR). The **sole methodological change** this iteration, per the Advisor, is elimination of the late-time symmetry error ($6.27\times10^{-3}$) via **mirror-paired AMR decisions** and **verified-symmetric setup** (initial projection + source quadrature). All states are archived at common save times to serve as the degree-5 reference for the Advisor-mandated degree-6 confirmation run (next iteration).

## 2. Mathematical Specification (Code-Ready Extraction)

* **Governing Equations:** 2D compressible Euler with gravitational source:
$$\partial_t \rho + \partial_x(\rho u) + \partial_y(\rho v) = 0$$
$$\partial_t(\rho u) + \partial_x(\rho u^2 + p) + \partial_y(\rho u v) = 0$$
$$\partial_t(\rho v) + \partial_x(\rho u v) + \partial_y(\rho v^2 + p) = \rho g$$
$$\partial_t E + \partial_x\big(u(E+p)\big) + \partial_y\big(v(E+p)\big) = \rho g v$$
$$E = \frac{p}{\gamma-1} + \frac{1}{2}\rho(u^2+v^2), \qquad \gamma = 1.4, \qquad g = 1.0$$

* **Domain:** $(x,y) \in [0, 0.25] \times [0, 1.0]$, $t \in [0, 3.0]$.

* **Boundary Conditions:** $x$: Periodic. $y=0$ and $y=1$: Reflective/free-slip walls (no-penetration $v=0$, $u$ mirrored).

* **Initial Conditions** (exact, all fields):
$$\rho(x,y) = 2.0 + (1.0 - 2.0)\cdot\frac{1}{2}\Big(1 + \tanh\big(50\,(y - 0.5)\big)\Big)$$
$$p(x,y) = \begin{cases} 2y + 1, & y < 0.5 \\ y + 1.5, & y \geq 0.5 \end{cases}$$
$$u(x,y) = 0$$
$$v(x,y) = -0.025\,\sqrt{\frac{\gamma\, p(y)}{\rho(y)}}\,\cos(8\pi x)\,\big(\sin(\pi y)\big)^{6}$$
$$E(x,y) = \frac{p(x,y)}{\gamma - 1} + \frac{1}{2}\,\rho(x,y)\,v(x,y)^2$$

* **Constants & Parameters:** $\gamma = 1.4$; gravity vector $\mathbf{g} = [0, 1.0]^T$ (source $\mathbf{S} = [0,\,0,\,\rho g,\,\rho g v]^T$); $\rho_{heavy}=2.0$, $\rho_{light}=1.0$; interface $y^\*=0.5$; tanh steepness $50$; perturbation amplitude $0.025$; single mode $\cos(8\pi x)$ ($\lambda = 0.25$); $\mu = 0$, $k = 0$; **mirror map** $\mathcal{M}: (x,y)\mapsto(0.25-x,\,y)$ (symmetry plane $x_s = 0.125$; under $\mathcal{M}$: $\rho, p, E, v$ even, $u$ odd).

* **Expected Physics:** Exponential single-mode growth ($t \lesssim 1$) → nonlinear central heavy-fluid spike with mushroom cap and two Kelvin–Helmholtz roll-ups ($t \approx 2$) → secondary plumes, spike approaching the $y=1$ wall ($t \approx 3$). Exact mirror symmetry about $x=0.125$ at all times. No shocks; steep compressible gradients at the interface/roll-ups handled by local blending.

## 3. Strategic Methodology

**Exact solution: NOT feasible.** The system is a nonlinear, coupled 2D hyperbolic system with a gravitational source whose solution is a developing instability with KH roll-up. MoC fails (multi-D characteristics form caustics; no closed form); Fourier fails (non-periodic in $y$, steep gradients); separation of variables/Green's functions fail (nonlinear, time-developing). The Advisor explicitly confirmed the exact-method override is not triggered. No entropy-condition root selection is required.

**Selection: Discontinuous Galerkin Spectral Element Method in 2D plus time.** High-order accuracy for the smooth growth phase, $h$-adaptivity for the moving interface, entropy-stable fluxes, local conservation, and robust slip-wall/periodic BCs — the benchmark-appropriate architecture per the Advisor ("DGSEM ... remains an appropriate architecture").

**Iteration delta:** Only the symmetry enforcement changes. The degree-6 confirmation run (polydeg=6, CFL=0.25, same base mesh and AMR policy) is **scheduled for the next iteration**; this run must only produce the archived degree-5 reference.

## 4. Implementation Requirements (CRITICAL)

### 4.0.1 AMR & Shock-Capturing — Variable Selection (Signal-to-Noise Analysis)

**AMR and shock-capturing variable: Density $\rho$.** $\rho$ is the material-interface tracer: piecewise-smooth with a sharp jump localized exactly at the RTI front and roll-ups, smooth elsewhere, never globally chaotic. By contrast, $v$ becomes sign-chaotic throughout the mixing zone (KH turbulence) — refining on it wastes elements and produces asymmetric decisions; $p$ is dominated by the smooth hydrostatic background ($\partial_y p = \rho g$), giving poor feature localization; entropy is noisy in the absence of shocks. Density is strictly superior for this PDE.

### 4.0 Configuration (Advisor-Mandated Retention — supersedes template defaults)

* DGSEM (Gauss–Lobatto), **polydeg = 5**; surface flux **HLLC**; volume flux **Ranocha** (entropy-stable).
* Shock capturing: selective local blending with Lax–Friedrichs flux, **alpha_max = 0.2**, Hennemann–Gassner indicator on $\rho$.
* Mesh: P4estMesh over $[0,0.25]\times[0,1]$, **initial refinement level 3**; periodic $x$, slip walls $y$.
* AMR: **IndicatorLöhner on $\rho$**, **ControllerThreeLevel(base=4, med=5, max=6)**, **interval = 20**.
* Time: **RDPK3SpFSAL49**, **CFL = 0.3** (StepsizeCallback), $t \in [0,3]$.

### 4.1 Advisor-Mandated Adjustments (Overrides)

1. **Mirror-paired AMR (primary change).** For each leaf element $e = [x_1,x_2]\times[y_1,y_2]$, its mirror partner is $m(e) = [0.25-x_2,\,0.25-x_1]\times[y_1,y_2]$. At every AMR step, after evaluating the density-Löhner indicator $I(e)$, replace $I(e) \leftarrow I(m(e)) \leftarrow \max\big(I(e), I(m(e))\big)$ for each pair; the controller then issues **identical refine/coarsen decisions to both members of each pair**, so the forest remains exactly mirror-symmetric after every AMR operation. **Hard-assert** post-rebalance that every leaf has a mirror partner of identical size/level (abort with explicit error otherwise).
2. **Initial-projection symmetry verification.** After L2 projection of the IC, compute $\varepsilon_{IC} = \max_{\text{DOF}} |\rho - \mathcal{M}\rho|$ (analogously $\rho v$, $E$; for $\rho u$ use $|\rho u + \mathcal{M}(\rho u)|$). Require $\varepsilon_{IC} \le 10^{-12}$; log it; abort if $> 10^{-10}$.
3. **Source-term quadrature symmetry verification.** Apply the gravity source operator to the initial state and verify mirror symmetry of the result to $\le 10^{-12}$ (analytically trivial since $\mathbf{S}=[0,0,\rho g, \rho g v]$ with constant $g$; the check guards against implementation asymmetries). Log $\varepsilon_S$.
4. **Prohibition:** Do **not** average, project, or symmetrize the solution itself — only indicator/AMR decisions and setup are symmetrized.
5. **Prohibitions:** Do **not** revert to HLL or add global Lax–Friedrichs dissipation; **alpha_max stays 0.2** this run (the $0.1$ test is conditional on the future degree-6 comparison). Do **not** attempt to load the inaccessible degree-4 archive; no cross-run comparison this iteration.
6. **Archival for the degree-6 study (next iteration).** Save all conserved variables at **dt_save = 0.1** (common save times), both as raw DG node data and interpolated to a **fixed uniform comparison grid** (e.g., $256\times1024$), with metadata (polydeg, mesh, AMR policy, CFL, save times). Track and archive the **spike-tip position** at each save time: $\text{spike}(t) = \max\{y : \exists x,\ \rho(x,y) \ge 1.5\}$. Do **not** execute the degree-6 run now.
7. **Retained diagnostics (unchanged):** relative mass drift (expect $\sim10^{-12}$); positivity of $\rho$, $p$; wall no-penetration compliance $\max|v|$ at $y=0,1$ (expect $\lesssim 10^{-3}$); cumulative gravity-work energy balance $\frac{d}{dt}\int E = \int \rho g v$ accumulated every step (residual $\sim10^{-9}$); runtime mirror-symmetry metric on a common grid,
$$\varepsilon_{sym}(t) = \frac{\|q - \mathcal{M}q\|_2}{\|q\|_2}, \quad q = (\rho, \rho u, \rho v, E) \text{ (}u\text{ sign-flipped under } \mathcal{M}\text{)},$$
with acceptance **$\varepsilon_{sym}(3.0) \le 10^{-10}$** (previous run: $6.27\times10^{-3}$); NPZ sanity checks; final verification summary beginning with `Minimum achieved error:` (mass-conservation diagnostic; no exact RTI solution exists), followed by symmetry, positivity, wall, and energy diagnostics.

## 5. Visualization Strategy

* **Field Selection:** $\rho$ (primary driver/tracer: interface, spike, mushroom cap, plumes) and $v$ (dominant dynamics: up-column, return flow, roll-up circulation). Do **not** plot $u$, $p$, $E$ or derived quantities (no vorticity).
* **Times:** $t = 0.5,\ 1.0,\ 2.0,\ 3.0$. **Layout:** 2 rows ($\rho$, $v$) $\times$ 4 columns (times).
* Interpolate to a gap-free uniform grid for plotting; **save as `summary_all.png`** (strictly).
* **CRITICAL (Julia): `colorbar=false`.** No fixed colormaps or hardcoded ranges — scales adjust automatically.
* **Saving Data:** Full solution state (all variables) archived at `dt_save = 0.1` for Ground Truth (independent of plotting selection).

## 6. Cross-Validation & Data Integration (Conditional)

No external data paths (sensors/reference data) are present in the problem statement → **None**. This run's NPZ archive (common save times, uniform comparison grid, spike-tip track) constitutes the degree-5 reference for the Advisor-mandated degree-6 confirmation run (polydeg=6, CFL=0.25, same base mesh and AMR policy; relative $L_2$ density differences and spike-tip positions at $t=2,3$), to be executed in the **next** iteration.

## REQUIRED COMPONENTS

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
* Adaptive Mesh Refinement (density-based Löhner indicator, three-level controller)
* Shock Capturing (selective local blending, density-based Hennemann–Gassner indicator)
* Mirror-Paired AMR Symmetry Enforcement (Advisor-mandated)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

**Status:** `REJECTED_FOR_REFINEMENT`

The run is stable, physically convincing, and improves the late-time mirror symmetry by roughly five orders of magnitude. However, the prescribed symmetry tolerance is still missed, and degree-independent ground-truth convergence has not yet been confirmed.

The saturation stopping rule is not met: Run 10 was the first supplied result scoring at least 90, and the supplied history shows only Run 13 and the present run subsequently. Thus, fewer than five post-success refinement trials are documented.

## Section 2: Quantitative Scorecard

* **Score Integrity:** 50/50
* **Score Accuracy:** 19/20
* **Score Details:** 15/15
* **Score Optimality:** 15/15
* **TOTAL SCORE:** **99/100**

## Section 3: Forensic Analysis & Feedback

### Observation

**Plot quality and physics:**

* At \(t=0.5\) and \(t=1.0\), the initially smooth interface develops the expected localized single-mode displacement while remaining nearly undisturbed away from the perturbation.
* At \(t=2.0\), the density field shows a well-developed central spike with a mushroom cap and two Kelvin–Helmholtz roll-ups. The vertical-velocity panel consistently shows the central jet and surrounding return flow.
* At \(t=3.0\), the plume has elongated and developed secondary structures while remaining visually mirror-symmetric. The archived spike tip, \(y\approx0.9351\), is consistent with the plot and shows that the spike approaches but does not penetrate the upper wall.
* Interfaces and roll-up structures remain crisp. There is no visible Gibbs ringing, widespread interface smearing, blank panel, or other obvious numerical artifact.

**Verification diagnostics:**

* Relative mass drift is only \(4.51\times10^{-12}\).
* Energy closure is approximately \(3.01\times10^{-9}\).
* Density and pressure remain positive throughout the reported checks.
* Maximum wall-normal velocity at the walls is \(3.63\times10^{-4}\), within the stated wall-compliance expectation.
* Initial-condition and source symmetry are both approximately \(7.99\times10^{-15}\), and the mirror-paired AMR hard assertion completed successfully.
* Final symmetry improved from \(6.27\times10^{-3}\) previously to
  \[
  \varepsilon_{\mathrm{sym}}(3)=2.437921\times10^{-8},
  \]
  but this still exceeds the required \(10^{-10}\) target.

**Error-log interpretation:** The final `Minimum achieved error:` entry is explicitly identified by the log as a **mass-conservation diagnostic only**, not an error against an exact RTI solution. **Quantitative error is not applicable as no ground truth is available.**

### Diagnosis

The primary defect is **residual mirror-symmetry breaking**, not instability, excessive dissipation, or poor boundary treatment.

Mirror-paired AMR was highly effective. Because the initial projection and gravity source are symmetric to roundoff and the mesh-pairing assertion passed, the remaining error likely originates from either:

1. the common-grid interpolation or its mirror indexing;
2. remaining unsymmetrized operations, such as shock-capturing indicator/blending evaluation or AMR transfer operations; or
3. floating-point operation-order differences whose effects are amplified by the physically unstable RTI evolution.

The present evidence does not justify changing HLLC/Ranocha, adding global dissipation, or increasing shock-capturing strength.

### Targeted Improvements

* **For Integrity:** None. The simulation completed successfully, produced readable plots, and passed positivity and archive checks.

* **For Accuracy:**
  1. Report symmetry separately on the **raw DG representation** and the **\(256\times1024\) common grid**, at every save time and ideally immediately before and after AMR. If the raw solution is substantially more symmetric, correct the comparison-grid interpolation or mirror-index mapping rather than the PDE solver.
  2. If the raw DG state carries the same residual, inspect the shock-capturing indicator/blending coefficient and AMR projection/prolongation paths for mirror-paired decisions and consistent operation ordering. Continue the prohibition on averaging or symmetrizing the solution state itself.
  3. Execute the scheduled **polydeg = 6, CFL = 0.25** confirmation run with the same base mesh and AMR policy. Compare relative \(L_2\) density differences and spike-tip positions at \(t=2\) and \(t=3\) against the archived degree-5 reference. If these comparisons do not demonstrate convergence, use a degree-7 confirmation with a suitably smaller CFL rather than aggressive mesh refinement.

* **For Details:** None for the visible solution quality. The degree-6 comparison remains necessary for certification as converged PINN ground truth.

* **For Optimality:** None. DGSEM with HLLC, Ranocha volume flux, selective blending, and density-based AMR remains an appropriate architecture. No exact analytical solution is available for this nonlinear, source-driven, two-dimensional instability with Kelvin–Helmholtz roll-up, so the exact-method override is not triggered.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 2 | Implemented 2D compressible Euler equations for single-mode Rayleigh–Taylor instability using Trixi's DGSEM. Polydeg=3, P4estMesh over [0,0.25]x[0,1] with initial refinement level 3, periodic in x and slip walls in y. Surface flux flux_hll, volume flux flux_ranocha, hybrid shock capturing with flux_lax_friedrichs and density-based IndicatorHennemannGassner. AMR uses density-based IndicatorLöhner with ControllerThreeLevel (base 4, med 5, max 6) and AMRCallback interval 20. Time integration via RDPK3SpFSAL49 with tspan=(0,3) and cfl=0.5. Initial conditions match the strategy exactly: rho tanh profile, piecewise pressure, u=0, v perturbation with sound speed, and E via prim2cons. Gravity source adds rho*g to y-momentum and g*rho_v2 to energy. Data saved at dt_save=0.1. | N/A | 43 | 16 | 10 | 12 | **81** |
| 10 | The code implements the requested 2D compressible Euler Rayleigh–Taylor refinement run using DGSEM in space and a low-storage Runge–Kutta time integrator. It uses gamma=1.4, gravity [0,1] with source terms (0,0,rho*g,rho*g*v), domain [0,0.25]x[0,1], tspan [0,3], periodic x boundaries, and slip walls at y=0 and y=1. The initial condition matches the strategy: smoothed density interface at y=0.5, piecewise hydrostatic pressure, zero u, and the specified single-mode v perturbation using cos(8*pi*x)*sin(pi*y)^6. The discretization uses polydeg=4, HLLC surface flux, Ranocha volume flux, Lax–Friedrichs in selective shock capturing with alpha_max=0.2, and CFL=0.4. AMR is density-based with IndicatorLöhner and ControllerThreeLevel base=4, med=5, max=6. The code includes the mandatory diagnostics (mass drift, positivity, symmetry, wall compliance, energy balance), NPZ sanity checks, final verification summary beginning with `Minimum achieved error:`, and optional polydeg=3 archive comparison. Visualization saves `summary_all.png` with density and vertical velocity only at t=0.5,1.0,2.0,3.0 using a uniform gap-free grid and colorbar=false. | N/A | 50 | 18 | 12 | 15 | **95** |
| 13 | The code implements the specified 2D compressible Euler Rayleigh-Taylor problem using Trixi's DGSEM with polynomial degree 5 on a P4estMesh over [0, 0.25] x [0, 1.0], periodic in x and slip walls in y. The initial condition matches the strategy: tanh-smoothed density interface at y=0.5, piecewise hydrostatic pressure, zero horizontal velocity, and the single-mode vertical perturbation -0.025 c cos(8πx) sin(πy)^6. The ideal-gas gamma is 1.4, and gravity is applied as a source with +y momentum ρg and energy ρgv. Spatial discretization uses HLLC surface flux, Ranocha volume flux, and selective local blending with Lax-Friedrichs flux and alpha_max=0.2 via a Hennemann-Gassner indicator on density. AMR uses a density-based Löhner indicator and a three-level controller with base_level=4, med_level=5, max_level=6, interval=20, on an initial refinement level of 3. Time integration uses RDPK3SpFSAL49 with a CFL-based StepsizeCallback at cfl=0.3 to t=3.0. The code includes cumulative gravity-work energy balance accumulated every time step, mirror-symmetry verification on a common grid, NPZ archival at dt_save=0.1, a conditional polydeg-4 archive comparison, and a final verification summary beginning with 'Minimum achieved error:'. | N/A | 50 | 19 | 14 | 15 | **98** |
| 33 | The code implements a numerical DGSEM solution of the 2D compressible Euler equations with gravitational source terms for the single-mode Rayleigh-Taylor instability on [0,0.25]x[0,1] from t=0 to t=3. Initial conditions match the report exactly: tanh density interface at y=0.5, piecewise hydrostatic pressure, zero u, and the specified v perturbation using cos(8*pi*x)*sin(pi*y)^6. Configuration matches: polydeg=5 Lobatto DGSEM, HLLC surface flux, Ranocha volume flux, Hennemann-Gassner shock capturing on density with alpha_max=0.2, P4estMesh with initial refinement level 3, periodic x, slip walls y. AMR uses IndicatorLohner on density with ControllerThreeLevel(base=4, med=5, max=6, interval=20), and includes a mirror-paired indicator wrapper plus post-rebalance hard assertion that every leaf has an identical-size/level mirror partner. Initial projection and gravity-source mirror symmetry checks are performed with required logging and abort thresholds. Time integration uses RDPK3SpFSAL49 with StepsizeCallback CFL=0.3 and tspan=(0,3). Data are archived at dt_save=0.1 as NPZ files containing primitive plot data, raw DG conserved state, conservative data interpolated to a 256x1024 comparison grid, metadata, and spike-tip positions. Diagnostics include relative mass drift, positivity of density/pressure, wall no-penetration, cumulative gravity-work energy balance, and the common-grid mirror-symmetry metric. Visualization saves summary_all.png with rho and v panels at t=0.5,1.0,2.0,3.0 using colorbar=false. | N/A | 50 | 19 | 15 | 15 | **99** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 2 | Saves multiple .npz files in data_npz/ named time_XX.XXXXX.npz every dt_save=0.1 (including initial and final via PeriodicCallback and finalize). Each .npz contains keys: 't' (scalar), 'x' (1-D array of length N), 'y' (1-D array of length N), 'rho' (1-D array of length N), 'u' (1-D array of length N), 'v' (1-D array of length N), 'p' (1-D array of length N), where N is the number of PlotData2D nodes. Also saves Trixi HDF5 solution snapshots via SaveSolutionCallback at dt=0.1 in trixi_output/ (primitive variables rho,u,v,p on the adaptive mesh). Finally saves summary_all.png as a 2x4 grid of scatter plots showing rho (first row) and v (second row) at t=0.5,1.0,2.0,3.0, with colorbar=false and automatic color scaling. |
| 10 | NPZ snapshots are written every dt_save=0.1 from t=0.0 to t=3.0 (including final time) to `data_npz/time_<t>.npz`, e.g. `data_npz/time_00.50000.npz`. Each NPZ contains keys: `t` (scalar Float64), `x` (1D vector length N), `y` (1D vector length N), `rho` (1D vector length N), `u` (1D vector length N), `v` (1D vector length N), `p` (1D vector length N), where N is the number of PlotData2D nodes on the current AMR mesh and is not reshaped to a fixed 2D grid. In addition, Trixi's SaveSolutionCallback writes full solution snapshots every 0.1 to `trixi_output` using primitive variables (`cons2prim`) in HDF5 format. `summary_all.png` is saved in the working directory as a 2x4 panel figure: row 1 density at t=0.5,1.0,2.0,3.0; row 2 vertical velocity at the same times. |
| 13 | NPZ snapshots are written to data_npz_polydeg5/time_%08.5f.npz every dt_save=0.1 from t=0 to 3.0. Each NPZ contains keys: 't' (Float64 scalar), and 'x', 'y', 'rho', 'u', 'v', 'p' (1D Float64 arrays of length N, where N is the number of PlotData2D reconstruction nodes; coordinates and primitive fields are flattened node-wise). Trixi SaveSolutionCallback also writes full primitive-variable HDF5 solution snapshots to trixi_output at dt=0.1. The summary image is saved as summary_all.png, a 2x4 heatmap grid showing rho and v at t=0.5, 1.0, 2.0, and 3.0 with colorbar=false. |
| 33 | Per-time NPZ files are written to data_npz_polydeg5 as time_%08.5f.npz for common save times t=0.0,0.1,...,3.0 (31 files). Each file contains scalar keys: 't', 'spike_tip', 'polydeg', 'cfl', 'dt_save', 'save_index', 'comparison_nx', 'comparison_ny', 'initial_refinement_level', 'amr_base_level', 'amr_med_level', 'amr_max_level', 'amr_interval', 'alpha_max', 'mirror_plane_x'; 2-vector keys: 'domain_min', 'domain_max'; byte-vector metadata keys: 'mesh', 'amr_policy', 'surface_flux', 'volume_flux', 'time_integrator', 'raw_layout', 'grid_layout'; primitive PlotData2D vectors: 'x', 'y', 'rho', 'u', 'v', 'p' (1D length N_plot, mesh-dependent); raw DG-node vectors: 'raw_x', 'raw_y', 'raw_rho', 'raw_rho_u', 'raw_rho_v', 'raw_energy' (1D length N_DG=(polydeg+1)^2*n_leaf_elements, identical lengths); comparison axes: 'comparison_x' (length 256), 'comparison_y' (length 1024); conservative comparison grids: 'grid_rho', 'grid_rho_u', 'grid_rho_v', 'grid_energy' (Float64 matrices of shape 1024x256); save metadata: 'save_times' (length save_index+1) and 'scheduled_save_times' (length 31). A 'spike_tip_track.npz' file is also saved with keys 'times', 'spike_tip', 'save_times' (length equal to number of completed saves, 31 at t=3) and 'scheduled_save_times' (length 31). Additionally, SaveSolutionCallback writes HDF5 snapshots to trixi_output at dt_save=0.1, and the plotting code saves summary_all.png as a 2-row by 4-column heatmap layout for rho and v at t=0.5,1.0,2.0,3.0 with colorbar=false. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run1/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_0954/` |
| 10 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run1/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1042/` |
| 13 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run1/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1111/` |
| 33 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run1/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1337/` |


---
