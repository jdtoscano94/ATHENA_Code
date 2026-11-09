
# Experiment Report
*Last Updated: 2026-11-06 11:22:52*

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

Single-mode compressible Rayleigh–Taylor instability: heavy fluid (ρ=2) beneath light fluid (ρ=1) in a gravitational field acting in +y, seeded by a single-mode vertical velocity perturbation. The interface grows through linear amplification into a nonlinear rising heavy spike with Kelvin–Helmholtz roll-up, to be resolved on $t\in[0,3]$.

**1. Mathematical Specification (Code-Ready Extraction)**

* **Governing Equations:** 2D Compressible Euler with gravity source:
$$\partial_t \begin{pmatrix} \rho \\ \rho u \\ \rho v \\ E \end{pmatrix} + \partial_x \begin{pmatrix} \rho u \\ \rho u^2 + p \\ \rho u v \\ u(E+p) \end{pmatrix} + \partial_y \begin{pmatrix} \rho v \\ \rho u v \\ \rho v^2 + p \\ v(E+p) \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ \rho g \\ \rho g v \end{pmatrix}$$
with $p = (\gamma-1)\left(E - \tfrac{1}{2}\rho(u^2+v^2)\right)$, $\gamma = 1.4$, $g = 1.0$ (momentum source in $+y$; energy source $\rho g v$).

* **Domain:** $(x,y) \in [0, 0.25] \times [0, 1.0]$, $t \in [0, 3.0]$.

* **Boundary Conditions:** $x$: Periodic. $y=0$ and $y=1$: Reflective/free-slip walls (no-penetration, $v=0$).

* **Initial Conditions:**
$$\rho(x,y,0) = 2.0 + (1.0 - 2.0)\cdot\frac{1}{2}\left[1 + \tanh\!\left(50(y-0.5)\right)\right]$$
$$p(x,y,0) = \begin{cases} 2y + 1, & y < 0.5 \\ y + 1.5, & y \geq 0.5 \end{cases}$$
$$u(x,y,0) = 0$$
$$v(x,y,0) = -0.025\, c(x,y)\, \cos(8\pi x)\, \left[\sin(\pi y)\right]^6, \qquad c(x,y) = \sqrt{\frac{\gamma\, p(x,y,0)}{\rho(x,y,0)}}$$
$$E(x,y,0) = \frac{p}{\gamma - 1} + \frac{1}{2}\rho\left(u^2 + v^2\right)$$

* **Constants & Parameters:** $\gamma = 1.4$; $\mathbf{g} = [0, 1.0]^T$; $\rho_{heavy}=2.0$, $\rho_{light}=1.0$; interface at $y=0.5$ (tanh smoothing 50); perturbation amplitude $0.025$; mode $8\pi$ (exactly one period across $x$-width); $t_{end}=3.0$.

* **Expected Physics:** Exponential linear growth of the single mode ($\lambda_x = 0.25$); nonlinear saturation near $t\approx 1$; heavy spike rising with mushroom cap and KH roll-up on the flanks by $t\approx 2$; multiscale secondary plumes with the spike approaching the top wall by $t=3$. Dominant features: contact discontinuity (density interface) and shear layers; weak compressibility.

**2. Strategic Methodology**

* **Exact solution feasible?** **No.** MoC applies only to 1D simple-wave hyperbolic systems; here we have a fully coupled 2D nonlinear system, a gravity source, reflecting walls, and an instability that amplifies the seed through a nonlinear cascade into vortex roll-up — characteristics cannot describe the post-saturation interface. Linearized RTI theory yields only the early-time growth rate, not the $t\in[0,3]$ evolution. No separation of variables or Green's function exists for the nonlinear convective flux.
* **Selection:** Fourier Spectral is precluded (walls in $y$; developing contact/shear layers → Gibbs). Steady CG is inapplicable (transient). **Select: DGSEM in 2D plus time** — high-order, entropy-stable, mixed-BC capable, AMR-ready for the multiscale roll-up. This is the appropriate high-fidelity architecture; the Advisor confirms no exact-method override is warranted.

**3. Implementation Requirements (CRITICAL)**

* **Configuration:** Use template defaults for all unspecified parameters. The Advisor-mandated overrides below **supersede** all defaults.

* **3.0.1 Adaptive Mesh Refinement & Shock Capturing — Variable Selection Analysis:**
    * *Signal-to-noise:* **Density ($\rho$)** is the material-interface tracer — it remains sharp and well-defined exactly at the features of interest (interface, spike, bubble, mushroom flanks). High SNR, feature-localized. **SELECT density for AMR (Löhner indicator).**
    * *Rejected:* Velocity becomes globally chaotic in the KH roll-up (refining on it wastes resolution domain-wide). Pressure is dominated by the smooth hydrostatic background (poor feature contrast, spurious triggers). Derived quantities (Mach, entropy) are noisier than $\rho$.
    * *Shock capturing:* retain **selective local blending** (LF flux, $\alpha_{max}=0.2$). Do NOT increase global dissipation.

* **3.1 Advisor-Mandated Adjustments (Overrides):**
    1. **polydeg = 5** (upgrade from 4), on the **same base mesh and AMR levels** (initial refinement 3; controller base=4, med=5, max=6, interval 20).
    2. **CFL = 0.3** (reduced from 0.4) for the higher-degree run.
    3. **Retain HLLC surface flux + Ranocha entropy-stable volume flux.** Do NOT revert to HLL; do NOT globally increase LF dissipation.
    4. If polydeg-5 introduces local ringing, adjust **only** the local blending activation threshold / positivity safeguard.
    5. **Cumulative energy balance:** replace the finite-difference $dE/dt$ check with $E(t) - E(0) - \int_0^t\!\!\int_\Omega \rho g v \, dV \, d\tau$, accumulating gravity work at **every time step** with consistent quadrature and AMR cell weights.
    6. **Symmetry verification** on a common, mirror-paired quadrature grid; mirror-symmetric base mesh and symmetric AMR treatment; do NOT average the solution to hide asymmetry.
    7. **Archive polydeg-5 NPZ states**; load archived polydeg-4 states (previous run) if accessible for the controlled p4→p5 comparison (see §5).
    8. Print final verification summary beginning with `Minimum achieved error:`.

**4. Visualization Strategy**

* **Field Selection:** $\rho$ (primary driver — buoyancy from the density contrast; tracks interface/spike/bubble) and $v$ (vertical velocity — rising heavy column and return flow). These two capture the dominant dynamics; omit $u, p, E$ to keep panels readable.
* **Task:** Plot $\rho$ and $v$ at $t = 0.5,\ 1.0,\ 2.0,\ 3.0$.
* **Layout:** Grid — 2 rows (fields) × 4 columns (times).
* **Saving Data:** Save the **full solution state** (all variables) at `dt_save = 0.1` for Ground Truth; plotting selection affects only the summary image.
* **Constraints:** Save as `summary_all.png` (strictly). **DISABLE COLORBARS** (`colorbar=false`) — Julia. Automatic color scales; no fixed colormaps. Plot only primary variables; no derived quantities.

**5. Cross-Validation & Data Integration (Conditional)**

* No external data paths in [USER_QUESTION] → no mandatory external reference.
* **Conditional (Advisor-mandated):** If archived polydeg-4 NPZ states from the previous run are accessible, load them, interpolate both solutions onto an **identical uniform grid** at common save times, and print the **relative $L_2$ density error** and **spike-tip position difference**, with particular attention to $t=2.0$ and $t=3.0$.

**REQUIRED COMPONENTS**

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
* Shock Capturing (selective local blending, $\alpha_{max}=0.2$)
* Adaptive Mesh Refinement (density-based Löhner indicator, three-level controller)
* Cumulative Energy Balance Diagnostic
* Mirror-Symmetric Symmetry Verification
* Polydeg-4 → Polydeg-5 Convergence Comparison (archived-state cross-check)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

**Status:** `REJECTED_FOR_REFINEMENT`

The run is stable and physically convincing, but it does not merit a perfect score because late-time symmetry is not exact and discretization convergence remains unverified. The history contains a previous **95-point** run, but only one subsequent refinement—the present run—is documented. Therefore, the five-post-success refinement requirement for saturation approval has not been met.

## Section 2: Quantitative Scorecard

- **Score Integrity:** 50/50
- **Score Accuracy:** 19/20
- **Score Details:** 14/15
- **Score Optimality:** 15/15
- **TOTAL SCORE:** **98/100**

## Section 3: Forensic Analysis & Feedback

### Observation

The plots show the expected Rayleigh–Taylor development:

- At **\(t=0.5\)** and **\(t=1.0\)**, the initially smooth interface develops a growing, single-mode central deformation.
- At **\(t=2.0\)**, the heavy fluid forms a rising central spike with a mushroom cap and two visible Kelvin–Helmholtz roll-ups. The vertical-velocity panel shows the corresponding upward column and flanking return flow.
- At **\(t=3.0\)**, the spike approaches the top wall and develops thinner secondary plumes and rolled shear layers. There is no obvious catastrophic ringing, blank field, or loss of positivity.
- The solution remains approximately mirror-symmetric, although the late-time fine structures are not perfectly symmetric.

The logs confirm successful completion, NPZ verification, and plot generation. At \(t=3\):

- Relative mass drift is approximately \(4.51\times10^{-12}\).
- Density and pressure remain finite and positive.
- The cumulative energy balance closes to approximately \(3.01\times10^{-9}\), with an energy residual of approximately \(-1.54\times10^{-10}\).
- Maximum wall-normal velocity is approximately \(3.63\times10^{-4}\), indicating good no-penetration compliance.
- The final symmetry metric is \(6.27\times10^{-3}\), small but sufficiently nonzero to prevent a perfect ground-truth score.
- The planned polynomial-degree comparison was skipped because the archived degree-4 states were inaccessible.

**Log error check:** The final verification line contains  
`Minimum achieved error: 0.000000e+00 (mass)`  
This is a **mass-conservation diagnostic**, not an error against an exact Rayleigh–Taylor solution. **Quantitative error is not applicable as no ground truth is available.**

### Diagnosis

This is a **stable, high-order, physically consistent RTI simulation**, not a failed run. The remaining deficiencies are:

1. **Small late-time symmetry error**, potentially associated with non-paired AMR decisions or other symmetry-breaking numerical details; the supplied evidence does not conclusively identify the source.
2. **Incomplete convergence verification**, because the degree-4/degree-5 comparison could not be performed.
3. Any residual interface smoothing appears mild. There is no visual justification for replacing HLLC with a more dissipative flux or for increasing global stabilization.

The exact-method override is not triggered: the nonlinear, coupled two-dimensional Euler system with gravity, mixed boundary conditions, and nonlinear Kelvin–Helmholtz roll-up does not admit a usable closed-form solution through characteristics, Fourier series, or separation of variables.

### Targeted Improvements

- **For Integrity:** None. Retain the current time integrator, CFL control, positivity monitoring, and NPZ verification.

- **For Accuracy:**  
  - Keep `polydeg = 5`, HLLC, Ranocha volume flux, and the current base resolution for the next run.
  - Enforce **mirror-paired AMR**: associate elements across the symmetry plane, use a common refinement decision for each pair, and refine/coarsen them together. Also verify symmetry of the initial projection and source-term quadrature. **Do not average or symmetrize the solution itself.**
  - Because the degree-4 archive is unavailable, subsequently perform a controlled **degree-6 confirmation run** on the same base mesh and AMR policy, using a slightly reduced CFL such as `0.25`. Compare degree 5 and degree 6 on an identical grid using relative \(L_2\) density differences and spike-tip position, especially at \(t=2\) and \(t=3\). Archive all states with common save times.

- **For Details:**  
  - Do **not** revert to HLL or add global Lax–Friedrichs dissipation.
  - First determine whether the thin late-time interfaces change under the degree-6 comparison. If differences localize to strongly blended cells, test reducing only the local blending cap from `alpha_max = 0.2` to `0.1`, while retaining positivity safeguards and monitoring for ringing. Otherwise, leave the shock capturing unchanged.

- **For Optimality:** None. DGSEM with an entropy-stable volume flux, contact-resolving HLLC surface flux, selective local blending, and density-based AMR remains an appropriate architecture for this problem.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 2 | Implemented 2D compressible Euler equations for single-mode Rayleigh–Taylor instability using Trixi's DGSEM. Polydeg=3, P4estMesh over [0,0.25]x[0,1] with initial refinement level 3, periodic in x and slip walls in y. Surface flux flux_hll, volume flux flux_ranocha, hybrid shock capturing with flux_lax_friedrichs and density-based IndicatorHennemannGassner. AMR uses density-based IndicatorLöhner with ControllerThreeLevel (base 4, med 5, max 6) and AMRCallback interval 20. Time integration via RDPK3SpFSAL49 with tspan=(0,3) and cfl=0.5. Initial conditions match the strategy exactly: rho tanh profile, piecewise pressure, u=0, v perturbation with sound speed, and E via prim2cons. Gravity source adds rho*g to y-momentum and g*rho_v2 to energy. Data saved at dt_save=0.1. | N/A | 43 | 16 | 10 | 12 | **81** |
| 10 | The code implements the requested 2D compressible Euler Rayleigh–Taylor refinement run using DGSEM in space and a low-storage Runge–Kutta time integrator. It uses gamma=1.4, gravity [0,1] with source terms (0,0,rho*g,rho*g*v), domain [0,0.25]x[0,1], tspan [0,3], periodic x boundaries, and slip walls at y=0 and y=1. The initial condition matches the strategy: smoothed density interface at y=0.5, piecewise hydrostatic pressure, zero u, and the specified single-mode v perturbation using cos(8*pi*x)*sin(pi*y)^6. The discretization uses polydeg=4, HLLC surface flux, Ranocha volume flux, Lax–Friedrichs in selective shock capturing with alpha_max=0.2, and CFL=0.4. AMR is density-based with IndicatorLöhner and ControllerThreeLevel base=4, med=5, max=6. The code includes the mandatory diagnostics (mass drift, positivity, symmetry, wall compliance, energy balance), NPZ sanity checks, final verification summary beginning with `Minimum achieved error:`, and optional polydeg=3 archive comparison. Visualization saves `summary_all.png` with density and vertical velocity only at t=0.5,1.0,2.0,3.0 using a uniform gap-free grid and colorbar=false. | N/A | 50 | 18 | 12 | 15 | **95** |
| 13 | The code implements the specified 2D compressible Euler Rayleigh-Taylor problem using Trixi's DGSEM with polynomial degree 5 on a P4estMesh over [0, 0.25] x [0, 1.0], periodic in x and slip walls in y. The initial condition matches the strategy: tanh-smoothed density interface at y=0.5, piecewise hydrostatic pressure, zero horizontal velocity, and the single-mode vertical perturbation -0.025 c cos(8πx) sin(πy)^6. The ideal-gas gamma is 1.4, and gravity is applied as a source with +y momentum ρg and energy ρgv. Spatial discretization uses HLLC surface flux, Ranocha volume flux, and selective local blending with Lax-Friedrichs flux and alpha_max=0.2 via a Hennemann-Gassner indicator on density. AMR uses a density-based Löhner indicator and a three-level controller with base_level=4, med_level=5, max_level=6, interval=20, on an initial refinement level of 3. Time integration uses RDPK3SpFSAL49 with a CFL-based StepsizeCallback at cfl=0.3 to t=3.0. The code includes cumulative gravity-work energy balance accumulated every time step, mirror-symmetry verification on a common grid, NPZ archival at dt_save=0.1, a conditional polydeg-4 archive comparison, and a final verification summary beginning with 'Minimum achieved error:'. | N/A | 50 | 19 | 14 | 15 | **98** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 2 | Saves multiple .npz files in data_npz/ named time_XX.XXXXX.npz every dt_save=0.1 (including initial and final via PeriodicCallback and finalize). Each .npz contains keys: 't' (scalar), 'x' (1-D array of length N), 'y' (1-D array of length N), 'rho' (1-D array of length N), 'u' (1-D array of length N), 'v' (1-D array of length N), 'p' (1-D array of length N), where N is the number of PlotData2D nodes. Also saves Trixi HDF5 solution snapshots via SaveSolutionCallback at dt=0.1 in trixi_output/ (primitive variables rho,u,v,p on the adaptive mesh). Finally saves summary_all.png as a 2x4 grid of scatter plots showing rho (first row) and v (second row) at t=0.5,1.0,2.0,3.0, with colorbar=false and automatic color scaling. |
| 10 | NPZ snapshots are written every dt_save=0.1 from t=0.0 to t=3.0 (including final time) to `data_npz/time_<t>.npz`, e.g. `data_npz/time_00.50000.npz`. Each NPZ contains keys: `t` (scalar Float64), `x` (1D vector length N), `y` (1D vector length N), `rho` (1D vector length N), `u` (1D vector length N), `v` (1D vector length N), `p` (1D vector length N), where N is the number of PlotData2D nodes on the current AMR mesh and is not reshaped to a fixed 2D grid. In addition, Trixi's SaveSolutionCallback writes full solution snapshots every 0.1 to `trixi_output` using primitive variables (`cons2prim`) in HDF5 format. `summary_all.png` is saved in the working directory as a 2x4 panel figure: row 1 density at t=0.5,1.0,2.0,3.0; row 2 vertical velocity at the same times. |
| 13 | NPZ snapshots are written to data_npz_polydeg5/time_%08.5f.npz every dt_save=0.1 from t=0 to 3.0. Each NPZ contains keys: 't' (Float64 scalar), and 'x', 'y', 'rho', 'u', 'v', 'p' (1D Float64 arrays of length N, where N is the number of PlotData2D reconstruction nodes; coordinates and primitive fields are flattened node-wise). Trixi SaveSolutionCallback also writes full primitive-variable HDF5 solution snapshots to trixi_output at dt=0.1. The summary image is saved as summary_all.png, a 2x4 heatmap grid showing rho and v at t=0.5, 1.0, 2.0, and 3.0 with colorbar=false. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run1/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_0954/` |
| 10 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run1/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1042/` |
| 13 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run1/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1111/` |


---
