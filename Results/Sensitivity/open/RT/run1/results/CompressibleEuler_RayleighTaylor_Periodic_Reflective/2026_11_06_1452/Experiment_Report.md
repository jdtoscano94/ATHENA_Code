
# Experiment Report
*Last Updated: 2026-11-06 15:17:16*

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
# Scientific Strategy Report — ATHENA Lab

**Classification:** Refinement/Certification Iteration (Run 34+) — continuing from the 99/100 DGSEM baseline.

---

**1. Problem Statement**

We solve the single-mode, inviscid, compressible Rayleigh–Taylor instability (RTI): a heavy fluid ($\rho=2$) lies beneath a light fluid ($\rho=1$) with gravity acting in $+y$, an unstable stratification ($\nabla p \cdot \nabla \rho < 0$ across the interface). The prior DGSEM run is stable, positive, mass-conserving to $10^{-12}$, and physically correct, but **certification is incomplete**: (i) the mirror-symmetry residual is $2.44\times10^{-8}$ versus the required $10^{-10}$, and (ii) degree-independent ground-truth convergence is unconfirmed. This iteration executes the Advisor-mandated **polydeg=6, CFL=0.25 confirmation run** with targeted symmetry-forensic diagnostics, escalating to polydeg=7 only if convergence is not demonstrated.

**1. Mathematical Specification (Code-Ready Extraction)**

* **Governing Equations:** 2D Compressible Euler with gravitational source:
$$\partial_t \begin{pmatrix} \rho \\ \rho u \\ \rho v \\ E \end{pmatrix} + \partial_x \begin{pmatrix} \rho u \\ \rho u^2 + p \\ \rho u v \\ u(E+p) \end{pmatrix} + \partial_y \begin{pmatrix} \rho v \\ \rho u v \\ \rho v^2 + p \\ v(E+p) \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ \rho g \\ \rho g v \end{pmatrix}$$
with equation of state $p = (\gamma - 1)\left(E - \tfrac{1}{2}\rho(u^2 + v^2)\right)$.

* **Domain:** $(x,y) \in [0, 0.25] \times [0, 1.0]$, $\quad t \in [0, 3.0]$.

* **Boundary Conditions:**
  * $x = 0$ and $x = 0.25$: **Periodic**.
  * $y = 0$ and $y = 1$: **Reflective / free-slip walls** (no-penetration: $v = 0$).

* **Initial Conditions** (exact formulas, every field):
$$\rho(x,y) = 2.0 + (1.0 - 2.0)\cdot\tfrac{1}{2}\Big(1 + \tanh\big(50(y - 0.5)\big)\Big)$$
$$p(y) = \begin{cases} 2y + 1, & y < 0.5 \\ y + 1.5, & y \geq 0.5 \end{cases}$$
$$u(x,y) = 0$$
$$v(x,y) = -0.025\; c(y)\, \cos(8\pi x)\, \sin^6(\pi y), \qquad c(y) = \sqrt{\frac{\gamma\, p(y)}{\rho(y)}}$$
$$E(x,y) = \frac{p(y)}{\gamma - 1} + \frac{1}{2}\,\rho(x,y)\, v(x,y)^2$$

* **Constants & Parameters:** $\gamma = 1.4$; gravity vector $\mathbf{g} = [0, 1.0]^T$ (source adds $+\rho g$ to $y$-momentum and $\rho g v$ to energy); $\mu = 0$; $k = 0$.

* **Expected Physics:** Single-mode growth localized at the interface for $t \lesssim 1$; by $t \approx 2$ a central heavy-fluid spike with mushroom cap and two Kelvin–Helmholtz roll-ups; by $t = 3$ an elongated plume with secondary structures, spike tip at $y \approx 0.93$–$0.94$ (approaching, not penetrating, the upper wall). The solution must remain **mirror-symmetric about $x = 0.125$** to $\leq 10^{-10}$.

**2. Strategic Methodology**

* **Can we obtain an exact solution? No — infeasible.** The Method of Characteristics is tractable only for 1D homogeneous hyperbolic systems (simple waves / Riemann invariants). This problem is a **2D nonlinear system with a gravitational source term, wall boundaries, and an unstable interface** whose evolution involves nonlinear mode coupling, KH roll-up, and secondary instabilities. Linear stability theory yields only the early-time growth rate, not the evolution to $t=3$. The Advisor has explicitly confirmed the exact-method override is **not triggered**. Therefore a certified high-order numerical ground truth is the objective.
* **Selection: Discontinuous Galerkin Spectral Element Method in 2D plus time.** Justification: (i) wall boundaries in $y$ exclude Fourier spectral methods; (ii) sharp material interfaces, KH roll-ups, and compressive structures demand local shock capturing and $h$-adaptivity; (iii) the entropy-stable DGSEM architecture has already demonstrated stability, positivity, $10^{-12}$-level mass conservation, and correct RTI morphology in prior runs. The remaining gaps are **verification defects, not architecture defects** — they are addressed by the Advisor's diagnostic and degree-escalation protocol, not by changing fluxes or adding dissipation.

**3. Implementation Requirements (CRITICAL)**

* **Configuration:** Use template defaults **EXCEPT** for the Advisor-mandated overrides below. Retain the same base mesh and AMR policy as the archived degree-5 run.

* **3.0.1 Adaptive Mesh Refinement & Shock Capturing — Variable Selection Analysis:**
  * **AMR indicator variable: Density ($\rho$).** The structural features of RTI — heavy/light interface, spike, mushroom cap, roll-up boundaries, secondary structures — are material-interface phenomena uniquely and sharply marked by density, which remains localized at the features of interest. **Velocity must be avoided:** after roll-up onset ($t \gtrsim 2$), $v$ becomes globally chaotic across the entire mixing region, so velocity-based refinement would waste resources refining everywhere. **Pressure must be avoided:** it is dominated by the smooth hydrostatic stratification ($dp/dy = \rho g$) and would smear refinement vertically over the whole domain. Density is also the archived ground-truth comparison field and feeds the indicator with a bounded, high-contrast signal ($1 \leq \rho \leq 2$).
  * **Shock capturing:** density-based indicator with selective blending, strength unchanged (Advisor: evidence does not justify increasing it, changing HLLC/Ranocha fluxes, or adding global dissipation).
  * **Mirror-paired AMR (retain):** all AMR decisions made symmetrically about $x = 0.125$, with the post-rebalance hard assertion that every leaf has an identical-size/level mirror partner.

* **3.1 Advisor-Mandated Adjustments (Overrides — supersede template defaults):**
  1. **Degree-6 confirmation run:** polydeg = **6** (from 5), CFL = **0.25** (from 0.3), same base mesh and AMR policy.
  2. **Dual symmetry diagnostics:** report the mirror-symmetry residual $\varepsilon_{\mathrm{sym}}$ **separately on (a) the raw DG representation and (b) the $256\times1024$ common comparison grid**, at **every save time** and **immediately before and after each AMR operation**. Target: $\varepsilon_{\mathrm{sym}}(3) \leq 10^{-10}$.
  3. **Conditional repair directives:** if the raw DG state is substantially more symmetric than the common-grid value, fix the comparison-grid interpolation or mirror-index mapping — **not** the PDE solver. If the raw DG state carries the same residual, inspect the shock-capturing indicator/blending evaluation and AMR projection/prolongation paths for mirror-paired decisions and consistent floating-point operation ordering.
  4. **Prohibition (retain):** no averaging or symmetrization of the solution state itself.
  5. **Degree-5 archive comparison:** load the archived degree-5 NPZ reference; compute and print relative $L_2$ density differences and spike-tip position deltas at $t = 2$ and $t = 3$.
  6. **Escalation rule:** if the degree-6 vs. degree-5 comparison does not demonstrate convergence, execute a **polydeg = 7** run with a suitably smaller CFL — **not** aggressive mesh refinement.
  7. **Retain all integrity diagnostics:** relative mass drift, positivity of $\rho$ and $p$, wall no-penetration compliance, cumulative gravity-work energy balance, initial-condition and source symmetry checks (roundoff, $\sim 10^{-15}$, with logging/abort thresholds), and the final verification summary beginning with `Minimum achieved error:` (explicitly labeled as a mass-conservation diagnostic).

**4. Visualization Strategy**

* **Field Selection (Analysis):** Plot **Density ($\rho$)** and **vertical velocity ($v$)** only. $\rho$ is the primary driver of the instability (Atwood-number buoyancy) and marks the interface, spike, and mushroom structures; $v$ captures the dominant dynamics (central jet ascent and surrounding return flow). Do not plot $u$, $p$, or $E$ — additional panels would be too small to read.
* **Task:** Plot $\rho$ and $v$ at $t = 0.5,\ 1.0,\ 2.0,\ 3.0$.
* **Layout:** Grid — Rows = [$\rho$, $v$], Columns = [4 time steps].
* **Saving Data:** Archive the **full solution state** (all variables: primitive plot data, raw DG conserved state, common-grid interpolated conservative data, metadata, spike-tip positions) at `dt_save = 0.1` for Ground Truth.
* **Constraints:** Save as `summary_all.png` (strictly). **CRITICAL: disable colorbars (`colorbar=false`) — Julia.** No fixed colormaps; let scales adjust automatically. Plot only primary variables; no derived quantities (no vorticity).

**5. Cross-Validation & Data Integration (Conditional)**

* No external sensor/reference paths are present in the problem statement. However, per the Advisor mandate, cross-validation against the **archived degree-5 reference** (prior run's NPZ archive, `dt_save=0.1`) is required:
  * Load the archived degree-5 NPZ files.
  * Compute and print the **Relative $L_2$ density difference** at $t = 2.0$ and $t = 3.0$ between the new degree-6 solution and the degree-5 archive.
  * Compute and print the **spike-tip position deltas** at $t = 2.0$ and $t = 3.0$.
  * **Certification criteria:** (i) $\varepsilon_{\mathrm{sym}}(3) \leq 10^{-10}$ on both raw DG and common grid; (ii) degree-independent agreement in the above metrics; (iii) all integrity diagnostics pass. If (ii) fails, escalate per §3.1 item 6.

---

**REQUIRED COMPONENTS**

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
* Shock Capturing Indicator
* Adaptive Mesh Refinement (Mirror-Paired)
* Dual-Grid Symmetry Diagnostics (Raw DG + Common Grid)
* Degree-Escalation Convergence Protocol (polydeg 6 → 7, with Archived Degree-5 Cross-Validation)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

**Status:** **APPROVED — History Saturation Reached**

Run 10 first exceeded the high-fidelity threshold with **95/100**, followed by scores of 98 and 99. The current Run 34+ is well beyond five subsequent refinement trials. Exploration should therefore stop to prevent an infinite refinement loop.

This approval is a **saturation decision**, not a claim that the present run achieved perfect certification. The symmetry and degree-independence targets remain incomplete.

## Section 2: Quantitative Scorecard

* **Score Integrity:** **50/50**
* **Score Accuracy:** **19/20**
* **Score Details:** **14/15**
* **Score Optimality:** **15/15**
* **TOTAL SCORE:** **98/100**

## Section 3: Forensic Analysis & Feedback

### Log and Ground-Truth Check

The requested log entry is:

> `Minimum achieved error: 0.000000e+00 (mass)`

The log explicitly identifies this as a **mass-conservation diagnostic only**. It must not be interpreted as zero solution error.

**Quantitative error is not applicable as no ground truth is available.** The full nonlinear, two-dimensional RTI evolution—with gravitational forcing, wall boundaries, mushroom formation, and Kelvin–Helmholtz roll-up—does not admit a practical closed-form exact solution. Consequently, the exact-method override is not triggered, and DGSEM is an appropriate ground-truth approximation strategy.

### Observation

* **Execution and conservation:** The solver completed through \(t=3\), saved and verified the NPZ archives, and generated `summary_all.png`. Density and pressure remained positive. Relative mass drift was only \(6.66\times10^{-12}\), while the energy-balance residual and closure were approximately \(9.51\times10^{-11}\) and \(1.82\times10^{-9}\), respectively.
* **Physical development:** The plots show the expected progression:
  * At \(t=0.5\) and \(t=1\), the initially smooth interface develops a localized, symmetric perturbation.
  * At \(t=2\), a central heavy-fluid spike with a mushroom cap and two rolled-up regions is visible.
  * At \(t=3\), the plume elongates and develops secondary structures, while the vertical-velocity field shows a strong central jet and surrounding return flow.
* **Spike position:** The reported spike tip at \(t=3\) is \(y\approx0.93115\), consistent with the expected \(0.93\)–\(0.94\) range and below the upper wall. There is no visible wall penetration.
* **Fine-scale quality:** Interfaces and rolled-up filaments appear comparatively crisp, without obvious global smearing, blank panels, or conspicuous grid-scale Gibbs ringing. The very finest \(t=3\) structures cannot be convergence-certified from the image alone.
* **Boundary and symmetry diagnostics:** The maximum reported wall-normal velocity is small, \(2.03\times10^{-4}\), but not exactly zero. More importantly, the final symmetry residuals are
  \[
  \varepsilon_{\rm raw}=2.276682\times10^{-7},\qquad
  \varepsilon_{\rm grid}=1.943475\times10^{-7},
  \]
  so both miss the \(10^{-10}\) target by more than three orders of magnitude. The current raw residual is also larger than the prior baseline value stated in the strategy report.
* **Cross-validation:** The degree-5 reference archives were missing. Therefore, relative density differences and spike-tip agreement between degrees 5 and 6 were not measured, and degree-independent convergence remains uncertified.

### Diagnosis

The principal issue is **incomplete verification, not a visible failure of RTI physics**.

The initial condition and gravity source are symmetric to approximately \(5.33\times10^{-15}\). Because the raw-DG and common-grid residuals are similar at final time, common-grid interpolation or mirror-index mapping is unlikely to be the dominant cause. The remaining candidates are non-mirror-invariant floating-point operation ordering in AMR projection/prolongation, shock-indicator/blending evaluation, flux reductions, or time integration, potentially amplified during the nonlinear instability. The supplied log does not isolate which operation is responsible.

The missing degree-5 archive is a **data-management problem**, not evidence of numerical non-convergence. It correctly did not trigger an automatic degree-7 run.

### Targeted Improvements / Recommendation

* **For Integrity:** **None.** No crash, NaN, positivity failure, or plotting failure occurred.
* **For Accuracy:** No change to gravity, boundary conditions, fluxes, or mesh is justified by the plots. Restore or relocate the degree-5 NPZ archive and perform the required \(t=2\) and \(t=3\) comparisons as post-processing. For symmetry certification, inspect deterministic mirror-paired operation ordering in the raw-DG AMR and shock-capturing paths; do **not** symmetrize or average the solution state.
* **For Details:** Do not repeat another degree-6 run or add dissipation merely to pursue the symmetry metric. The additional degree-6 cost has not yet been shown to improve degree-independent accuracy. Escalate to `polydeg=7`, CFL \(=0.20\), only if an actual degree-5/degree-6 comparison demonstrates material degree dependence.
* **For Optimality:** **None.** HLLC surface flux, Ranocha volume flux, selective Lax–Friedrichs blending, and density-based AMR remain well matched to this problem.

Under the saturation rule, no further solver-exploration iteration is recommended. Run 33 remains the highest-scoring archived result; the present degree-6 dataset is high quality but should not supersede it as certified ground truth until the missing cross-validation and symmetry investigations are completed.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 2 | Implemented 2D compressible Euler equations for single-mode Rayleigh–Taylor instability using Trixi's DGSEM. Polydeg=3, P4estMesh over [0,0.25]x[0,1] with initial refinement level 3, periodic in x and slip walls in y. Surface flux flux_hll, volume flux flux_ranocha, hybrid shock capturing with flux_lax_friedrichs and density-based IndicatorHennemannGassner. AMR uses density-based IndicatorLöhner with ControllerThreeLevel (base 4, med 5, max 6) and AMRCallback interval 20. Time integration via RDPK3SpFSAL49 with tspan=(0,3) and cfl=0.5. Initial conditions match the strategy exactly: rho tanh profile, piecewise pressure, u=0, v perturbation with sound speed, and E via prim2cons. Gravity source adds rho*g to y-momentum and g*rho_v2 to energy. Data saved at dt_save=0.1. | N/A | 43 | 16 | 10 | 12 | **81** |
| 10 | The code implements the requested 2D compressible Euler Rayleigh–Taylor refinement run using DGSEM in space and a low-storage Runge–Kutta time integrator. It uses gamma=1.4, gravity [0,1] with source terms (0,0,rho*g,rho*g*v), domain [0,0.25]x[0,1], tspan [0,3], periodic x boundaries, and slip walls at y=0 and y=1. The initial condition matches the strategy: smoothed density interface at y=0.5, piecewise hydrostatic pressure, zero u, and the specified single-mode v perturbation using cos(8*pi*x)*sin(pi*y)^6. The discretization uses polydeg=4, HLLC surface flux, Ranocha volume flux, Lax–Friedrichs in selective shock capturing with alpha_max=0.2, and CFL=0.4. AMR is density-based with IndicatorLöhner and ControllerThreeLevel base=4, med=5, max=6. The code includes the mandatory diagnostics (mass drift, positivity, symmetry, wall compliance, energy balance), NPZ sanity checks, final verification summary beginning with `Minimum achieved error:`, and optional polydeg=3 archive comparison. Visualization saves `summary_all.png` with density and vertical velocity only at t=0.5,1.0,2.0,3.0 using a uniform gap-free grid and colorbar=false. | N/A | 50 | 18 | 12 | 15 | **95** |
| 13 | The code implements the specified 2D compressible Euler Rayleigh-Taylor problem using Trixi's DGSEM with polynomial degree 5 on a P4estMesh over [0, 0.25] x [0, 1.0], periodic in x and slip walls in y. The initial condition matches the strategy: tanh-smoothed density interface at y=0.5, piecewise hydrostatic pressure, zero horizontal velocity, and the single-mode vertical perturbation -0.025 c cos(8πx) sin(πy)^6. The ideal-gas gamma is 1.4, and gravity is applied as a source with +y momentum ρg and energy ρgv. Spatial discretization uses HLLC surface flux, Ranocha volume flux, and selective local blending with Lax-Friedrichs flux and alpha_max=0.2 via a Hennemann-Gassner indicator on density. AMR uses a density-based Löhner indicator and a three-level controller with base_level=4, med_level=5, max_level=6, interval=20, on an initial refinement level of 3. Time integration uses RDPK3SpFSAL49 with a CFL-based StepsizeCallback at cfl=0.3 to t=3.0. The code includes cumulative gravity-work energy balance accumulated every time step, mirror-symmetry verification on a common grid, NPZ archival at dt_save=0.1, a conditional polydeg-4 archive comparison, and a final verification summary beginning with 'Minimum achieved error:'. | N/A | 50 | 19 | 14 | 15 | **98** |
| 33 | The code implements a numerical DGSEM solution of the 2D compressible Euler equations with gravitational source terms for the single-mode Rayleigh-Taylor instability on [0,0.25]x[0,1] from t=0 to t=3. Initial conditions match the report exactly: tanh density interface at y=0.5, piecewise hydrostatic pressure, zero u, and the specified v perturbation using cos(8*pi*x)*sin(pi*y)^6. Configuration matches: polydeg=5 Lobatto DGSEM, HLLC surface flux, Ranocha volume flux, Hennemann-Gassner shock capturing on density with alpha_max=0.2, P4estMesh with initial refinement level 3, periodic x, slip walls y. AMR uses IndicatorLohner on density with ControllerThreeLevel(base=4, med=5, max=6, interval=20), and includes a mirror-paired indicator wrapper plus post-rebalance hard assertion that every leaf has an identical-size/level mirror partner. Initial projection and gravity-source mirror symmetry checks are performed with required logging and abort thresholds. Time integration uses RDPK3SpFSAL49 with StepsizeCallback CFL=0.3 and tspan=(0,3). Data are archived at dt_save=0.1 as NPZ files containing primitive plot data, raw DG conserved state, conservative data interpolated to a 256x1024 comparison grid, metadata, and spike-tip positions. Diagnostics include relative mass drift, positivity of density/pressure, wall no-penetration, cumulative gravity-work energy balance, and the common-grid mirror-symmetry metric. Visualization saves summary_all.png with rho and v panels at t=0.5,1.0,2.0,3.0 using colorbar=false. | N/A | 50 | 19 | 15 | 15 | **99** |
| 39 | The code implements a 2D compressible Euler solver for the single-mode Rayleigh-Taylor instability using Trixi DGSEM. The physical domain is [0,0.25]x[0,1] with periodic x boundaries and slip walls in y, gamma=1.4, and gravity g=1.0 acting in +y. Initial conditions exactly match the strategy: rho = 2 + (1-2)*0.5*(1+tanh(50*(y-0.5))), piecewise hydrostatic pressure p=2y+1 for y<0.5 and p=y+1.5 for y>=0.5, u=0, v=-0.025*c(y)*cos(8*pi*x)*sin(pi*y)^6 with c(y)=sqrt(gamma*p/rho), and E via prim2cons. The numerical configuration uses polydeg=6, CFL=0.25, P4estMesh with 1x1 trees and initial_refinement_level=3, density-based Loehner AMR (base_level=4, med_level=5, max_level=6, med_threshold=0.1, max_threshold=0.3, interval=20, adapt_initial_condition=true) with mirror-paired indicator decisions and hard mirror-partner assertions. Shock capturing uses IndicatorHennemannGassner with density variable, alpha_max=0.2, alpha_min=1e-3, alpha_smooth=true, volume_flux_dg=flux_ranocha, volume_flux_fv=flux_lax_friedrichs, and surface_flux=flux_hllc. Time integration is RDPK3SpFSAL49 with adaptive=false and StepsizeCallback(cfl=0.25) over t in [0,3]. The code implements dual-grid mirror-symmetry diagnostics separately on the raw DG representation and on a 256x1024 common comparison grid at every save time and before/after each AMR operation, with target 1e-10. Conditional repair branches handle comparison-grid interpolation/mirror-index issues and raw-DG shock/AMR path inspection without symmetrizing the solution state. It includes degree-5 archive cross-validation for relative L2 density differences and spike-tip deltas at t=2 and t=3, and conditional degree-7 escalation with CFL=0.20 if convergence is not demonstrated. Integrity diagnostics include mass drift, positivity of rho and p, wall no-penetration, cumulative gravity-work energy balance, initial-condition and source symmetry checks with roundoff logging/abort thresholds, and a final verification summary beginning with Minimum achieved error: labeled as a mass-conservation diagnostic. Visualization generates summary_all.png with density and vertical velocity at t=0.5,1.0,2.0,3.0 in a 2x4 layout with colorbar=false. | N/A | 50 | 19 | 14 | 15 | **98** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 2 | Saves multiple .npz files in data_npz/ named time_XX.XXXXX.npz every dt_save=0.1 (including initial and final via PeriodicCallback and finalize). Each .npz contains keys: 't' (scalar), 'x' (1-D array of length N), 'y' (1-D array of length N), 'rho' (1-D array of length N), 'u' (1-D array of length N), 'v' (1-D array of length N), 'p' (1-D array of length N), where N is the number of PlotData2D nodes. Also saves Trixi HDF5 solution snapshots via SaveSolutionCallback at dt=0.1 in trixi_output/ (primitive variables rho,u,v,p on the adaptive mesh). Finally saves summary_all.png as a 2x4 grid of scatter plots showing rho (first row) and v (second row) at t=0.5,1.0,2.0,3.0, with colorbar=false and automatic color scaling. |
| 10 | NPZ snapshots are written every dt_save=0.1 from t=0.0 to t=3.0 (including final time) to `data_npz/time_<t>.npz`, e.g. `data_npz/time_00.50000.npz`. Each NPZ contains keys: `t` (scalar Float64), `x` (1D vector length N), `y` (1D vector length N), `rho` (1D vector length N), `u` (1D vector length N), `v` (1D vector length N), `p` (1D vector length N), where N is the number of PlotData2D nodes on the current AMR mesh and is not reshaped to a fixed 2D grid. In addition, Trixi's SaveSolutionCallback writes full solution snapshots every 0.1 to `trixi_output` using primitive variables (`cons2prim`) in HDF5 format. `summary_all.png` is saved in the working directory as a 2x4 panel figure: row 1 density at t=0.5,1.0,2.0,3.0; row 2 vertical velocity at the same times. |
| 13 | NPZ snapshots are written to data_npz_polydeg5/time_%08.5f.npz every dt_save=0.1 from t=0 to 3.0. Each NPZ contains keys: 't' (Float64 scalar), and 'x', 'y', 'rho', 'u', 'v', 'p' (1D Float64 arrays of length N, where N is the number of PlotData2D reconstruction nodes; coordinates and primitive fields are flattened node-wise). Trixi SaveSolutionCallback also writes full primitive-variable HDF5 solution snapshots to trixi_output at dt=0.1. The summary image is saved as summary_all.png, a 2x4 heatmap grid showing rho and v at t=0.5, 1.0, 2.0, and 3.0 with colorbar=false. |
| 33 | Per-time NPZ files are written to data_npz_polydeg5 as time_%08.5f.npz for common save times t=0.0,0.1,...,3.0 (31 files). Each file contains scalar keys: 't', 'spike_tip', 'polydeg', 'cfl', 'dt_save', 'save_index', 'comparison_nx', 'comparison_ny', 'initial_refinement_level', 'amr_base_level', 'amr_med_level', 'amr_max_level', 'amr_interval', 'alpha_max', 'mirror_plane_x'; 2-vector keys: 'domain_min', 'domain_max'; byte-vector metadata keys: 'mesh', 'amr_policy', 'surface_flux', 'volume_flux', 'time_integrator', 'raw_layout', 'grid_layout'; primitive PlotData2D vectors: 'x', 'y', 'rho', 'u', 'v', 'p' (1D length N_plot, mesh-dependent); raw DG-node vectors: 'raw_x', 'raw_y', 'raw_rho', 'raw_rho_u', 'raw_rho_v', 'raw_energy' (1D length N_DG=(polydeg+1)^2*n_leaf_elements, identical lengths); comparison axes: 'comparison_x' (length 256), 'comparison_y' (length 1024); conservative comparison grids: 'grid_rho', 'grid_rho_u', 'grid_rho_v', 'grid_energy' (Float64 matrices of shape 1024x256); save metadata: 'save_times' (length save_index+1) and 'scheduled_save_times' (length 31). A 'spike_tip_track.npz' file is also saved with keys 'times', 'spike_tip', 'save_times' (length equal to number of completed saves, 31 at t=3) and 'scheduled_save_times' (length 31). Additionally, SaveSolutionCallback writes HDF5 snapshots to trixi_output at dt_save=0.1, and the plotting code saves summary_all.png as a 2-row by 4-column heatmap layout for rho and v at t=0.5,1.0,2.0,3.0 with colorbar=false. |
| 39 | The code saves time-series NPZ archives under data_npz_polydeg6 (or data_npz_polydeg7 after escalation), with filenames time_XX.XXXXX.npz at dt_save=0.1 from t=0.0 to t=3.0. Each NPZ contains: t (scalar Float64); x, y, rho, u, v, p (1D arrays of length n_plot_nodes, flattened PlotData2D primitive data); raw_x, raw_y, raw_rho, raw_rho_u, raw_rho_v, raw_energy (1D arrays of length n_raw_nodes, flattened raw DG node coordinates and conserved variables); comparison_x (1D length 256), comparison_y (1D length 1024), grid_rho, grid_rho_u, grid_rho_v, grid_energy (2D arrays of shape 1024x256, common-grid interpolated conservative data); spike_tip (scalar), symmetry_raw (scalar), symmetry_grid (scalar). Metadata keys include polydeg, cfl, dt_save, comparison_nx, comparison_ny, save_index, initial_refinement_level, amr_base_level, amr_med_level, amr_max_level, amr_interval, alpha_max, mirror_plane_x as scalars; save_times as a 1D array of length save_index+1; scheduled_save_times as a 1D array of length 31; domain_min and domain_max as 1D length-2 arrays; and mesh, amr_policy, surface_flux, volume_flux, time_integrator, raw_layout, grid_layout as 1D byte arrays. A separate spike_tip_track.npz is saved with times, spike_tip, save_times, and scheduled_save_times. Trixi SaveSolutionCallback also writes HDF5 full-state snapshots at dt_save to trixi_output (or trixi_output_polydeg7). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run1/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_0954/` |
| 10 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run1/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1042/` |
| 13 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run1/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1111/` |
| 33 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run1/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1337/` |
| 39 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run1/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1452/` |


---
