
# Experiment Report
*Last Updated: 2026-11-06 11:52:59*

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
# Scientific Strategy Report — 2D Compressible Euler Rayleigh–Taylor Instability (Refinement Trial)

**1. Problem Statement**

Single-mode Rayleigh–Taylor instability (RTI) of the inviscid compressible Euler equations: a heavy fluid ($\rho=2$) lies below a light fluid ($\rho=1$) while gravity acts in $+y$ — an unstable stratification ($\mathbf{g}\cdot\nabla\rho < 0$). A single-mode vertical-velocity perturbation seeds rising heavy-fluid spikes, descending light-fluid bubbles, Kelvin–Helmholtz roll-up, and a filament-rich mixing layer over $t\in[0,3]$. This is a **one-factor refinement trial** on top of the 97-point Run 5 architecture: the sole physics change is a controlled reduction of shock-capturing dissipation, plus a hardening of the certification output.

**1. Mathematical Specification (Code-Ready Extraction)**

* **Governing Equations:** 2D Compressible Euler with gravitational source:
$$\partial_t \begin{pmatrix}\rho\\ \rho u\\ \rho v\\ E\end{pmatrix} + \partial_x \begin{pmatrix}\rho u\\ \rho u^2 + p\\ \rho u v\\ u(E+p)\end{pmatrix} + \partial_y \begin{pmatrix}\rho v\\ \rho u v\\ \rho v^2 + p\\ v(E+p)\end{pmatrix} = \begin{pmatrix}0\\ 0\\ \rho g\\ \rho g\, v\end{pmatrix}, \qquad E = \frac{p}{\gamma-1} + \frac{1}{2}\rho\left(u^2+v^2\right)$$

* **Domain:** $(x,y) \in [0, 0.25] \times [0, 1.0]$; $t \in [0, 3.0]$.

* **Boundary Conditions:** $x = 0$ and $x = 0.25$: **Periodic**. $y = 0$ and $y = 1$: **Reflective / free-slip walls** (no-penetration $v = 0$; tangential velocity and all thermodynamic quantities mirrored).

* **Initial Conditions** (interface at $y = 0.5$):
$$\rho(x,y,0) = 2.0 + (1.0 - 2.0)\cdot\frac{1}{2}\left[1 + \tanh\!\left(50\,(y - 0.5)\right)\right]$$
$$p(x,y,0) = \begin{cases} 2y + 1, & y < 0.5 \\ y + 1.5, & y \geq 0.5 \end{cases}$$
$$u(x,y,0) = 0$$
$$v(x,y,0) = -0.025\; c(x,y)\,\cos(8\pi x)\,\left[\sin(\pi y)\right]^6, \qquad c(x,y) = \sqrt{\frac{\gamma\, p(x,y,0)}{\rho(x,y,0)}}$$
$$E(x,y,0) = \frac{p(x,y,0)}{\gamma - 1} + \frac{1}{2}\,\rho(x,y,0)\,v(x,y,0)^2$$

* **Constants & Parameters:** $\gamma = 1.4$; gravitational acceleration $g = 1.0$ (gravity vector $\mathbf{g} = [0,\, 1.0]^T$; source $\mathbf{S} = [0,\,0,\,\rho g,\,\rho g v]^T$); $\mu = 0$, thermal conductivity $k = 0$.

* **Expected Physics:** $t\approx 0.75$: early nonlinear growth of the single $\cos(8\pi x)$ mode (wavelength $= 0.25 =$ domain width). $t\approx 1.5$: rising heavy-fluid spike with bulbous cap; descending light-fluid bubble. $t\approx 2.25$: KH-driven mushroom roll-up at the spike cap with thin side filaments. $t = 3.0$: mixing layer extends toward the upper wall with multiple narrow filaments; approximate left–right symmetry preserved. Dominant features are a **contact discontinuity** (material interface) and **shear layers** — contact resolution, not shock resolution, is the accuracy bottleneck.

**2. Strategic Methodology**

* **Can we obtain an exact solution? No.** (i) The system is a nonlinear multi-dimensional hyperbolic conservation law: characteristic speeds depend on the solution itself, and characteristics from the smooth initial data intersect — no closed-form or implicit-characteristic inversion exists for the full 2D IVP. (ii) The solution develops a contact interface, shear layers, and multi-scale vortical roll-up; no Fourier-series representation is possible (non-periodic walls in $y$, discontinuities in $x$-$y$-$t$). (iii) The only exact object is the unperturbed hydrostatic equilibrium ($v \equiv 0$), which is **not** the solution of this IVP since the perturbation is part of the initial data. The Advisor has explicitly confirmed no exact-method override is warranted.
* **Selection: DGSEM in 2D plus time.** It is the only toolbox method that simultaneously provides (i) high order for the smooth vortex cores, (ii) Riemann-solver element interfaces for contacts/shear, (iii) local AMR for thin filaments, and (iv) native periodic + slip-wall boundaries with entropy-stable flux combinations. The Run-5 architecture (DGSEM + HLLC surface flux + Ranocha volume flux + HG shock capturing + density AMR) is retained; only the dissipation level is tuned.

**3. Implementation Requirements (CRITICAL)**

* **Configuration:** Use template defaults, **except** for the Advisor-mandated overrides in 3.1.

* **3.0.1 Adaptive Mesh Refinement & Shock Capturing — Variable Selection Analysis:**
  * **Selected variable: Density $\rho$** (for both the AMR indicator and the shock-capturing indicator).
  * **Justification:** The structural features of RTI — material interface, spike caps, bubble fronts, filaments — are *defined* by the density field. $\rho$ remains sharp and localized at these features for all $t$, so the indicator triggers refinement exactly in the interface band. **Velocity is rejected:** by $t\approx 2$ the mixing layer is filled with small-scale KH vortices, making $u, v$ globally chaotic there — refining on velocity wastes DOFs on the entire turbulent band instead of the interface. **Pressure is rejected:** it is essentially continuous across the contact (hydrostatic stratification), so it localizes the interface poorly. Density is strictly superior for this PDE and keeps the AMR and shock-capturing mechanisms aligned on the same variable.

* **3.1 Advisor-Mandated Adjustments (Overrides):**
  1. **Retain exactly (do not change):** polydeg $= 6$; CFL $= 0.4$; HLLC surface flux; Ranocha entropy-conservative volume flux; FV stabilization flux `flux_lax_friedrichs` (do **not** swap it this trial); AMR: Löhner indicator on $\rho$, `ControllerThreeLevel` levels 4/5/6, interval $= 10$; explicit RK time integration (RDPK3SpFSAL49). **Do NOT increase polydeg to 7.**
  2. **CHANGE (the one-factor dissipation trial):** Reduce the Hennemann–Gassner blending strength from `alpha_max = 0.5` to **`alpha_max = 0.25`**, with all other parameters fixed. Monitor positivity ($\min\rho > 0$, $\min p > 0$) and check for new interface ringing/Gibbs artifacts.
  3. **ADD (certification block):** Print to stdout, **at every output time** ($t = 0.75, 1.5, 2.25, 3.0$): $\min(\rho)$, $\min(p)$, wall $\max|v|$ at $y\in\{0,1\}$, relative mass drift, energy-balance residual (vs. gravitational source work), interface apex position, and mixing-layer width. This block must be retained in the supplied stdout tail, and the log must end with an explicit statement that the quantitative $L_2$ error is N/A (no analytical ground truth exists for the nonlinear RTI IVP).
  4. **ADD (p-convergence certification):** Explicitly print the polydeg-5 vs polydeg-6 differences at $t = 3$ (apex, mixing width, filament metrics), or state clearly why the stored polydeg-5 baseline was unavailable.
  5. **Contingency (do not stack changes):** If filaments remain soft under `alpha_max = 0.25`, the *next* isolated trial will tighten the AMR controller (interface pinned at level 6, or a narrow level-7 interface band). No simultaneous AMR + dissipation changes in this run.

**4. Visualization Strategy:**

* **Field Selection (Analysis):** (1) **Density $\rho$** — the primary driver and diagnostic of RTI (interface, spikes, bubbles, filaments). (2) **Vertical velocity $v$** — captures the spike/bubble circulation and return flow. Do not plot $u, p, E$; two fields keep all panels readable.
* **Task:** Plot $\rho$ (row 1) and $v$ (row 2) at $t = 0.75,\ 1.5,\ 2.25,\ 3.0$ (columns).
* **Layout:** Grid — Rows = Selected Fields ($\rho$, $v$), Columns = Time (4 snapshots).
* **Saving Data:** Save the **full solution state** (all conservative variables $\rho, \rho u, \rho v, E$) at frequent intervals (e.g., `dt_save = 0.1`, with exact snapshots guaranteed at $t = 0.75, 1.5, 2.25, 3.0$) for Ground Truth. The $\rho$/$v$ plotting selection affects only `summary_all.png`.
* **Constraints:** Save as `summary_all.png` (strictly). **CRITICAL: DISABLE COLORBARS (`colorbar = false`)** — the implementation is Julia. Do not use fixed colormaps; let scales adjust automatically per panel. Plot only primary variables; do not compute derived quantities (no vorticity, schlieren, etc.).

**5. Cross-Validation & Data Integration (Conditional)**

* No external data paths (sensors / reference solutions) are provided in [USER_QUESTION] → **None**. The polydeg-5 baseline comparison in 3.1 uses the internally stored baseline if available; otherwise the mandated unavailability statement is printed.

**REQUIRED COMPONENTS**

**Main Method:**
* Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
* Shock Capturing Indicator (density-based, Hennemann–Gassner) — Advisor-mandated blending-strength reduction `alpha_max: 0.5 → 0.25`
* Adaptive Mesh Refinement (density-based, Löhner indicator, ControllerThreeLevel 4/5/6)
* Certification Diagnostics (stdout block: positivity, wall $\max|v|$, mass/energy drift, apex, mixing width; polydeg-5 vs polydeg-6 comparison at $t=3$)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

**Status: APPROVED**

**Reason:** The history-saturation criterion has been reached. Run 1 first achieved a score above 90, and the present trial—following Run 5—is the fifth subsequent refinement trial. Further iterations are therefore not required. The current run is also the best-scoring result, although it does not achieve a perfect score.

## Section 2: Quantitative Scorecard

| Category | Score |
|---|---:|
| **Score Integrity** | **50 / 50** |
| **Score Accuracy** | **19 / 20** |
| **Score Details** | **14 / 15** |
| **Score Optimality** | **15 / 15** |
| **TOTAL SCORE** | **98 / 100** |

## Section 3: Forensic Analysis & Feedback

### Observation

- The simulation completed successfully, generated a readable `summary_all.png`, and reported positive density and pressure at every output time. There is no evidence of `NaN`, `Inf`, solver failure, or blank plotting panels.
- The density evolution is physically coherent:
  - At \(t=0.75\), the initially smooth interface has developed a localized upward perturbation.
  - At \(t=1.5\), a heavy-fluid spike with a bulbous cap is visible.
  - At \(t=2.25\), the cap has rolled into a mushroom structure with side filaments, consistent with Kelvin–Helmholtz instability.
  - At \(t=3.0\), the spike approaches the upper wall and produces multiple narrow, vertically elongated filaments. Approximate left–right symmetry is retained.
- The vertical-velocity panels show the corresponding strengthening and vertical extension of the spike circulation and adjacent return flow. No obvious nonphysical velocity artifact appears at the horizontal walls.
- The certification diagnostics support the visual assessment:
  - Relative mass drift remains below approximately \(4.13\times10^{-12}\).
  - The largest reported wall \(\max|v|\) is approximately \(2.40\times10^{-4}\).
  - The final energy-balance residual has magnitude \(2.61\times10^{-4}\).
  - The interface apex advances from approximately \(0.5398\) to \(0.9290\), while the mixing-layer width grows from \(0.0767\) to \(0.7500\).
- Compared with the preceding architecture, reducing `alpha_max` to `0.25` appears to preserve the thin filament structure without introducing visible catastrophic ringing or loss of positivity. Some softness remains at the thinnest density interfaces, and the unavailable polynomial-degree-5 baseline means that formal p-convergence was not demonstrated.

### Quantitative Error

The requested **“Minimum achieved error:”** entry does not occur in the supplied log. The log explicitly states:

> “Quantitative \(L_2\) error: N/A — no analytical ground truth exists for the nonlinear RTI IVP.”

**Quantitative error is not applicable as no ground truth is available.** The conservation and energy-balance diagnostics above are numerical-health checks, not solution-error measurements.

### Diagnosis

This is a stable, high-fidelity numerical RTI solution with **minor residual contact-interface dissipation**, rather than a stability failure. HLLC is appropriate for contact resolution, the Ranocha volume flux provides a suitable entropy-conservative DG flux, and density-based AMR plus limited Hennemann–Gassner stabilization targets the material interface and shear-driven filaments effectively.

No exact-method override is warranted: the full nonlinear, two-dimensional, gravitationally forced Euler IVP with vortical roll-up does not admit a practical closed-form, Fourier-series, or invertible characteristic solution. The exact unperturbed hydrostatic equilibrium is not the solution of the perturbed problem.

### Targeted Improvements

- **For Integrity:** None.
- **For Accuracy:** None required for this decision. If certification is revisited independently, restore or regenerate the polynomial-degree-5 baseline to complete the missing convergence comparison.
- **For Details:** No further refinement trial is required under the saturation rule. The remaining point was withheld for slight thinning/filament softness and the absence of formal resolution convergence; further reducing `alpha_max` is not recommended without such verification because it could reintroduce ringing or positivity problems.
- **For Optimality:** None. The DGSEM/HLLC/Ranocha/AMR configuration is well matched to this contact- and shear-dominated problem.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The code implements a 2D inviscid compressible Euler simulation of single-mode Rayleigh-Taylor instability using a numerical discontinuous Galerkin spectral element method (DGSEM) with Trixi.jl. The governing equations use gamma=1.4, gravitational source terms in +y with g=1.0, domain [0,0.25] x [0,1], t in [0,3.0], periodic x-boundaries, and free-slip/reflective y-walls. Initial conditions match the strategy: smoothed density interface with tanh(50(y-0.5)), piecewise hydrostatic pressure, u=0, and v=-0.025*c*cos(8*pi*x)*sin(pi*y)^6. The solver uses polydeg=3, P4estMesh with initial_refinement_level=3, density-based AMR via IndicatorLöhner with ControllerThreeLevel levels 4-6 and interval 20, density-based shock capturing via IndicatorHennemannGassner, HLL surface flux, entropy-conserving volume flux flux_ranocha, FV stabilization flux flux_lax_friedrichs, RDPK3SpFSAL49 time integration, and StepsizeCallback with CFL=0.5. The visualization code plots density and vertical velocity at t=0.75, 1.5, 2.25, and 3.0 in a 2x4 layout and saves summary_all.png with colorbar=false. | N/A | 45 | 17 | 10 | 12 | **84** |
| 1 | The code implements a 2D compressible Euler Rayleigh-Taylor simulation using Trixi's DGSEM on a P4estMesh over [0,0.25]x[0,1] with t in [0,3]. It uses CompressibleEulerEquations2D(gamma=1.4), polydeg=5, LobattoLegendreBasis, surface flux flux_hllc, entropy-conserving volume flux flux_ranocha, FV stabilization flux_lax_friedrichs, density-based IndicatorHennemannGassner shock capturing, density-based IndicatorLöhner AMR with ControllerThreeLevel levels 4/5/6 and interval=10, and RDPK3SpFSAL49 with StepsizeCallback(cfl=0.5). Initial conditions match the strategy: smoothed heavy/light density profile, piecewise hydrostatic pressure, zero u, v perturbation -0.025*c(y)*cos(8*pi*x)*sin(pi*y)^6, and E via prim2cons. Gravity source [0,0,+rho,+rho*v] is implemented. Boundaries are periodic in x and slip walls at y=0,1. Runtime diagnostics check finiteness, positivity, wall v extrema, total mass, energy tendency vs source work, interface apex/mixing width, and polydeg=3 baseline comparison when baseline data exist. Visualization plots rho and v at t=0.75,1.5,2.25,3.0 in a 2x4 heatmap layout with colorbar=false and gap-free rasterization. | N/A | 50 | 19 | 13 | 15 | **97** |
| 5 | The code implements the required Trixi DGSEM solver for the 2D inviscid compressible Euler equations with gravity source. It uses polydeg=6, CFL=0.4, P4estMesh on [0,0.25]x[0,1] with initial_refinement_level=3, periodic x and slip-wall y boundaries. Surface flux is flux_hllc; volume integral is VolumeIntegralShockCapturingHG with flux_ranocha as DG flux, flux_lax_friedrichs as FV flux, and density-based IndicatorHennemannGassner. AMR uses density-based IndicatorLöhner and ControllerThreeLevel with levels 4/5/6 and interval=10. The initial condition matches the strategy exactly: rho tanh profile, piecewise hydrostatic pressure, u=0, v=-0.025*c(y)*cos(8*pi*x)*sin(pi*y)^6 with c=sqrt(gamma*p/rho), and prim2cons for E. The source term is [0,0,rho*g,rho*g*v]. Time integration is explicit RDPK3SpFSAL49 with adaptive=false and StepsizeCallback(cfl=0.4) over t in [0,3]. The code computes and prints min(rho), min(p), wall max|v| at y=0/1, relative mass drift, energy-balance residual, interface apex, and mixing-layer width; it also attempts optional polydeg=5 baseline comparison and the mandated contingency message. Visualization plots rho and v at t=0.75,1.5,2.25,3.0 in a 2x4 layout, saves summary_all.png, and disables colorbars. No strategy mismatch was found. | N/A | 50 | 19 | 13 | 15 | **97** |
| 6 | The code implements the 2D compressible Euler Rayleigh–Taylor instability using Trixi.jl. The spatial discretization is DGSEM with polydeg=6, HLLC surface flux, Ranocha entropy-conservative volume flux, and Lax–Friedrichs FV stabilization via Hennemann–Gassner shock capturing with alpha_max=0.25. Density-based Löhner AMR uses ControllerThreeLevel with levels 4/5/6 and interval=10. Time integration is explicit RDPK3SpFSAL49 with CFL=0.4 over t∈[0,3]. The domain is [0,0.25]×[0,1], periodic in x and slip walls at y=0,1. Initial conditions match the strategy: tanh density profile, piecewise hydrostatic pressure, zero u, prescribed single-mode v, and total energy via prim2cons. Gravity source is [0,0,ρg,ρg v] with g=1. The code includes the mandated certification diagnostics at output times, p-convergence comparison against an optional polydeg-5 baseline, and prints the required L2 N/A statement at the end. Visualization saves summary_all.png with density and vertical velocity in a 2×4 grid with colorbars disabled. | N/A | 50 | 19 | 14 | 15 | **98** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves NPZ snapshots in data_npz/ named time_%08.5f.npz (e.g., time_00.00000.npz through time_03.00000.npz) at t=0.0,0.1,0.2,...,3.0 plus the visualization times. Each NPZ contains keys: 't' (scalar Float64), 'x' and 'y' (1D Vector{Float64} of scattered PlotData2D node coordinates, length N_points), primitive variables 'rho','u','v','p' (1D Vector{Float64}, length N_points), conservative variables 'conservative_rho','rho_u','rho_v','rho_e' (1D Vector{Float64}, length N_points), and 'state_vector' (1D Vector{Float64} raw DG state, length N_DOFs). N_points and N_DOFs vary with AMR. Also saves Trixi HDF5 solution files in trixi_output/ via SaveSolutionCallback at dt=0.1, including initial and final states, with solution_variables=cons2cons (full conservative DG state; exact HDF5 structure is Trixi's native format and mesh/DOF-dependent). Saves summary_all.png as a PNG 2x4 panel plot: row 1 density (rho), row 2 vertical velocity (v), columns t=0.75,1.5,2.25,3.0, with colorbars disabled. |
| 1 | Saves NPZ snapshots under data_npz with filename pattern time_%08.5f.npz (e.g., time_00.00000.npz) using NPZ.jl, at t=0.0,0.1,...,3.0 plus the visualization times 0.75,1.5,2.25,3.0. Each .npz contains: t scalar Float64; x, y, rho, u, v, p, conservative_rho, rho_u, rho_v, rho_e as 2D arrays on the current PlotData2D visualization grid (dimensions vary with AMR and are oriented to match plotting); state_vector as a 1D vector of raw DG conservative degrees of freedom; and scalar diagnostics min_rho, min_p, total_mass, mass_drift, total_energy, energy_tendency, source_work, energy_balance_residual, interface_apex, mixing_layer_width, wall_v_y0_min, wall_v_y0_max, wall_v_y1_min, wall_v_y1_max. It also invokes Trixi SaveSolutionCallback into trixi_output with dt=0.1, saving full conservative variables (cons2cons) at initial, final, and intermediate times. Finally it saves summary_all.png as a 2x4 continuous heatmap grid of rho and v at the four requested times. It optionally reads data_npz_polydeg3 for baseline convergence comparison; no baseline files are written by this code. |
| 5 | The custom DiscreteCallback writes NPZ snapshots to data_npz/time_XXXXXXXX.npz at t=0 and when crossing npz_save_times (0.0:0.1:3.0 plus 0.75,1.5,2.25,3.0). Each NPZ contains keys: scalar t; arrays x, y, rho, u, v, p, conservative_rho, rho_u, rho_v, rho_e (PlotData2D sample arrays with identical 2D grid shape; exact n_x-by-n_y dimensions are AMR/plot-sampling dependent); state_vector (1D Vector of the full DG conservative state, length ndofs); and scalars min_rho, min_p, total_mass, mass_drift, relative_mass_drift, total_energy, energy_tendency, source_work, source_work_interval_mean, energy_balance_residual, integrated_energy_balance_residual, interface_apex, mixing_layer_width, filament_thickness, wall_v_y0_min, wall_v_y0_max, wall_v_y1_min, wall_v_y1_max, wall_v_y0_maxabs, wall_v_y1_maxabs. Additionally, Trixi SaveSolutionCallback writes the full conservative DG state to trixi_output at dt=0.1, including initial/final states, with solution_variables=cons2cons. The summary image summary_all.png is saved as a 2x4 rho/v panel grid with colorbar=false. |
| 6 | Saves NPZ snapshots in data_npz/ named time_XXXXX.npz at dt_save=0.1 plus the exact visualization times 0.75,1.5,2.25,3.0. Each NPZ contains keys: 't' (scalar), 'x' (1D array length N), 'y' (1D array length N), 'rho' (N,), 'u' (N,), 'v' (N,), 'p' (N,), 'conservative_rho' (N,), 'rho_u' (N,), 'rho_v' (N,), 'rho_e' (N,), 'state_vector' (1D array length M, full DG state), scalar metrics 'min_rho', 'min_p', 'total_mass', 'mass_drift', 'relative_mass_drift', 'total_energy', 'energy_tendency', 'source_work', 'source_work_interval_mean', 'energy_balance_residual', 'integrated_energy_balance_residual', 'interface_apex', 'mixing_layer_width', 'filament_thickness', and wall diagnostics 'wall_v_y0_min', 'wall_v_y0_max', 'wall_v_y1_min', 'wall_v_y1_max', 'wall_v_y0_maxabs', 'wall_v_y1_maxabs'. N is the number of PlotData2D visualization nodes (varies with AMR), M is the number of DG degrees of freedom. Additionally saves summary_all.png as a 2×4 heatmap grid: row 1 density ρ, row 2 vertical velocity v, columns t=0.75,1.5,2.25,3.0, with colorbar=false. A Trixi SaveSolutionCallback also writes full conservative-state HDF5 files under trixi_output/ every 0.1 time units. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run3/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective_TanhInterface/2026_11_06_1051/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run3/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective_TanhInterface/2026_11_06_1107/` |
| 5 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run3/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective_TanhInterface/2026_11_06_1130/` |
| 6 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run3/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective_TanhInterface/2026_11_06_1144/` |


---
