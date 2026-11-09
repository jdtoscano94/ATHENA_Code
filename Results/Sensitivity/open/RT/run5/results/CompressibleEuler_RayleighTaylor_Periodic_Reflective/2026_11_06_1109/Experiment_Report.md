
# Experiment Report
*Last Updated: 2026-11-06 11:15:26*

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
2D inviscid compressible Euler equations with gravitational source, modeling the single-mode Rayleigh–Taylor Instability (RTI). Heavy fluid ($\rho=2$) lies below light fluid ($\rho=1$) with gravity acting in $+y$, an unstable stratification. This is Trial 2: Run 1 (DGSEM + density AMR + HLL) scored 89/100 with correct physics but excessive contact-interface dissipation and visualization artifacts. The Advisor mandates a targeted refinement: contact-resolving flux, decoupled shock capturing, clean rendering, and quantitative diagnostics.

**1. Mathematical Specification (Code-Ready Extraction)**
* **Governing Equations:**
$$\frac{\partial}{\partial t}\begin{pmatrix}\rho\\ \rho u\\ \rho v\\ E\end{pmatrix} + \frac{\partial}{\partial x}\begin{pmatrix}\rho u\\ \rho u^2 + p\\ \rho u v\\ u(E+p)\end{pmatrix} + \frac{\partial}{\partial y}\begin{pmatrix}\rho v\\ \rho u v\\ \rho v^2 + p\\ v(E+p)\end{pmatrix} = \begin{pmatrix}0\\ 0\\ \rho g\\ \rho g v\end{pmatrix}$$
with EOS $p = (\gamma-1)\left(E - \tfrac{1}{2}\rho(u^2+v^2)\right)$, $\gamma = 1.4$.
* **Domain:** $(x,y) \in [0,\,0.25]\times[0,\,1.0]$, $t \in [0,\,3.0]$.
* **Boundary Conditions:** $x$: Periodic. $y=0$ and $y=1$: Reflective / free-slip walls (no-penetration $v=0$).
* **Initial Conditions** (verbatim, no modification):
$$\rho(x,y,0) = 2.0 + (1.0-2.0)\cdot\frac{1}{2}\left[1+\tanh\!\left(50(y-0.5)\right)\right]$$
$$p(x,y,0) = \begin{cases} 2y + 1, & y < 0.5\\ y + 1.5, & y \geq 0.5\end{cases}$$
$$u(x,y,0) = 0$$
$$v(x,y,0) = -0.025\; c(x,y)\,\cos(8\pi x)\,\sin^6(\pi y), \qquad c(x,y)=\sqrt{\frac{\gamma\, p(x,y,0)}{\rho(x,y,0)}}$$
$$E(x,y,0) = \frac{p(x,y,0)}{\gamma-1} + \frac{1}{2}\rho(x,y,0)\,v(x,y,0)^2$$
* **Constants & Parameters:** $\gamma = 1.4$; gravity vector $\mathbf{g}=[0,\,1.0]^T$ (momentum source in $+y$); $\rho_{heavy}=2.0$, $\rho_{light}=1.0$; $\mu = 0$, $k = 0$ (inviscid).
* **Expected Physics:** Interface deformation from $t\approx 0.5$; heavy-fluid spike rises at $x=0.125$ with light-fluid bubbles descending at $x=0,\,0.25$; mushroom roll-up by $t\approx 2$; mixing region interacts with the top wall by $t=3.0$. Weak acoustic shocks and slip-wall reflections. Solution remains symmetric about $x=0.125$.

**2. Strategic Methodology**
* **Exact solution feasible?** **No.** This is a nonlinear, multi-dimensional hyperbolic system with a gravitational source, contact discontinuities, interface roll-up, shocks, and wall reflections. The Method of Characteristics yields no closed-form solution in 2D (characteristics cross; multi-d coupling precludes implicit inversion). The Advisor explicitly confirms no exact-solution override is warranted. No external reference data is provided.
* **Selection:** **DGSEM in 2D plus time.** Justification: hyperbolic conservation law with discontinuities demands conservative discretization with numerical fluxes; high-order DG resolves the smooth roll-up vortices; AMR tracks the evolving interface; slip walls and periodicity are natural. Fourier spectral is rejected (non-periodic in $y$; Gibbs contamination at contacts/shocks). Run 1 validated this architecture (89/100) — the architecture is retained; only the Advisor's targeted corrections are applied.

**3. Implementation Requirements (CRITICAL)**
* **Configuration:** Use template defaults. Per the Advisor, **hold the Run-1 configuration fixed** (DGSEM, P4estMesh, density-based Löhner AMR, Ranocha DG volume flux, Lax-Friedrichs FV flux, current time integrator, CFL). **Do NOT change polynomial degree, CFL, AMR levels, or the time integrator this run** — the Advisor's conditional `polydeg=5` / CFL 0.3–0.4 test is deferred to a future run *only if* smearing persists.

* **3.0.1 Adaptive Mesh Refinement & Shock Capturing — Variable Selection Analysis:**
    * **AMR variable: Density ($\rho$).** Signal-to-noise reasoning: $\rho$ is the material-interface tracer — it remains sharp, bimodal ($1 \leftrightarrow 2$), and well-defined exactly at the bubble/spike fronts of interest. Velocity $(u,v)$ becomes globally turbulent/chaotic in the mixing layer (wasteful near-uniform refinement). Pressure carries a smooth hydrostatic background gradient ($dp/dy = \rho g$, a factor-2 change across the domain) that would dominate any indicator and trigger spurious global refinement, plus acoustic noise. **Density is strictly superior for AMR.**
    * **Shock-capturing variable: Pressure ($p$) — Advisor-mandated decoupling from AMR.** The RTI interface is a *contact discontinuity*, across which pressure is continuous. The Run-1 density-based indicator misclassified the steep contact as nonsmooth, activating the diffusive FV hybrid precisely at the interface and smearing it. A pressure-based indicator activates only at genuine compressive/acoustic shocks (pressure-discontinuous), leaving the contact to the sharp pure-DG scheme. This decouples shock detection from AMR.

* **3.1 Advisor-Mandated Adjustments (Overrides):**
    1. **Surface flux: `flux_hllc`** (replaces HLL) — restores the contact wave for sharp material interfaces. All else held fixed to isolate its effect.
    2. **Shock-capturing indicator: pressure-based `IndicatorHennemannGassner`** (replaces density-based); retain the hybrid DG-FV volume integral (`flux_ranocha` DG / `flux_lax_friedrichs` FV).
    3. **Visualization cleanup:** remove the horizontal mesh-line/stripe overlay entirely; render `summary_all.png` at higher resolution.
    4. **Diagnostics (print to stdout at every save):** total mass $\int\rho\,dA$ and drift vs. $t=0$; $\min\rho$, $\min p$ (positivity); min/max timestep (CFL behavior); max $|v|$ at $y=0$ and $y=1$ (no-penetration check); symmetry residual $\|q(x,y)-q(0.25-x,y)\|/\|q\|$ about $x=0.125$.
    5. **Initial condition verbatim** — no re-tuning of the prescribed profiles.

**4. Visualization Strategy:**
* **Field Selection:** **Density $\rho$** (primary driver — material interface, bubbles/spikes, mixing) and **Pressure $p$** (hydrostatic trend + acoustic/shock content; Advisor retained these panels). Do not plot $u, v, E$.
* **Layout:** Grid, 2 rows (fields) × 4 columns (times $t = 0.75,\, 1.5,\, 2.25,\, 3.0$).
* **Saving Data:** Save the **full solution state** (all variables) with `dt_save = 0.1` for Ground Truth. Print min/max of $\rho$ and $p$ at each save.
* **Constraints:** Save as `summary_all.png` (strict). **Disable colorbars** (`colorbar=false`, Julia). **No mesh/element-boundary lines** in the plots. Automatic per-panel color scaling; no fixed colormaps. No derived quantities.

**5. Cross-Validation & Data Integration (Conditional)**
* No external data paths (sensors / reference solutions) are provided in [USER_QUESTION]. **None.** Ground truth consists of the saved full solution states; the diagnostics in §3.1 serve as the quantitative fidelity check.

**REQUIRED COMPONENTS**

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
* Adaptive Mesh Refinement (Density-based Löhner indicator)
* Shock Capturing (Pressure-based indicator, hybrid DG-FV)
* HLLC Surface Flux (Advisor-mandated contact-resolving upgrade)
* Physical Diagnostics Suite (mass conservation, positivity minima, wall no-penetration, symmetry residual)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

* **Status:** `REJECTED_FOR_REFINEMENT`
* **Reason:** The simulation completed and shows credible Rayleigh–Taylor physics, but the score is below 100 and visualization artifacts remain. Run 1 scored **89**, so there is no previous run with score ≥90; the five-run saturation rule therefore does not apply.

## Section 2: Quantitative Scorecard

* **Score Integrity:** 47/50
* **Score Accuracy:** 19/20
* **Score Details:** 12/15
* **Score Optimality:** 15/15
* **TOTAL SCORE:** **93/100**

**Error-norm check:** The supplied stdout tail contains no `Minimum achieved error:` line; it ends with the successful generation of `summary_all.png` and `Done.` **Quantitative error is not applicable as no ground truth is available.** No error value should be inferred from the timing output.

## Section 3: Forensic Analysis & Feedback

### Observation

* **Execution:** The solver ran through \(t=3.0\), performed AMR and solution output, and successfully saved the summary figure. The tail contains no crash, NaN, or non-finite-value indication.
* **Density evolution:** The top row shows the expected instability sequence:
  * At \(t=0.75\), a centered heavy-fluid protrusion has formed.
  * At \(t=1.5\), it develops into a narrow rising spike.
  * At \(t=2.25\), the spike rolls up into a mushroom structure while light fluid descends on either side.
  * At \(t=3.0\), the roll-up interacts with the upper part of the domain and produces increasingly complex secondary vortices.
  
  This agrees with the prescribed perturbation: upward motion near \(x=0.125\), downward motion near the periodic side boundaries, and approximate left–right symmetry.
* **Pressure:** The bottom row retains the expected predominantly hydrostatic vertical variation, with increased compression near the upper wall at late time. There is no obvious large-scale pressure oscillation or sign of an incorrect gravity direction.
* **Interfaces:** The principal density fronts and mushroom rims are relatively crisp, consistent with the intended HLLC/contact-resolving upgrade. Some late-time fine-scale roll-up remains soft, but there is no clear large-amplitude Gibbs ringing.
* **Remaining artifact:** Narrow, nearly vertical stripes remain visible in both density and pressure, including regions that should still be smooth or nearly one-dimensional. Their wavelength and persistence are inconsistent with the imposed single-mode \(\cos(8\pi x)\) perturbation. Thus, they are unlikely to represent genuine RTI structures.
* **Diagnostics:** Although the implementation reports a diagnostic suite, none of its numerical values appear in the supplied tail. Consequently, mass drift, positivity, timestep range, wall-normal velocity, and symmetry cannot be independently verified from the provided evidence.

### Diagnosis

The principal remaining defect is **visualization striping/aliasing**, most likely caused by element-wise rendering, inter-element gaps, or nonuniform solution sampling. The claimed removal of explicit mesh lines did not eliminate the artifact. A secondary limitation is mild numerical dissipation of the smallest late-time vortical structures.

The numerical architecture itself is appropriate: DGSEM with HLLC, pressure-based shock detection, and separate density-based AMR correctly distinguishes acoustic/compressive shocks from the material contact. There is no evidence justifying a return to HLL or density-based shock capturing. A global analytical solution is not feasible for this nonlinear, multidimensional RTI problem with gravity, roll-up, and wall interactions, so no exact-method override is warranted.

### Targeted Improvements

* **For Integrity:**
  * Solver execution requires no stability fix.
  * Replace the current element-wise plotting path with **uniformly sampled solution arrays rendered as one heatmap/image per panel**, with no element edge strokes or gaps.
  * As a decisive rendering test, sample the initial condition: density and pressure must be independent of \(x\). Any remaining vertical stripes at \(t=0\) are conclusively visualization artifacts.
  * Include a compact final diagnostic summary in stdout so the physical checks are auditable.

* **For Accuracy:**
  * Report the actual values of relative mass drift, \(\min\rho\), \(\min p\), \(dt_{\min}/dt_{\max}\), wall \(|v|\), and the symmetry residual.
  * Verify that mass drift is near solver/AMR conservation tolerance, density and pressure remain positive, and wall-normal velocities are consistent with weak slip-wall enforcement.
  * Keep the initial conditions, gravity source, HLLC flux, and boundary conditions unchanged.

* **For Details:**
  * Fix the renderer before attributing the stripes to numerical oscillations.
  * Since the same polynomial-degree/CFL configuration has now been used twice, do not repeat it unchanged. For the final resolution refinement, use **`polydeg=5` with CFL approximately `0.35`**, while retaining HLLC, the pressure-based shock indicator, density-based AMR, and the current AMR limits.
  * Do **not** add further mesh refinement; the higher polynomial degree should improve smooth vortex resolution more efficiently. Inspect the resulting contacts for ringing before considering any additional stabilization.

* **For Optimality:**
  * **None.** Retain the present DGSEM/HLLC/decoupled-indicator architecture.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | The code implements the 2D compressible Euler equations with gravitational source using Trixi.jl. The spatial discretization is DGSEM with polynomial degree 3 on a P4estMesh (domain [0,0.25]×[0,1], periodic in x, reflective slip walls in y via boundary_condition_slip_wall). Shock capturing is density-based using IndicatorHennemannGassner with a hybrid DG-FV volume integral (flux_ranocha for DG, flux_lax_friedrichs for FV) and HLL surface flux. AMR is density-based using IndicatorLöhner with ControllerThreeLevel (base_level=4, med_level=5, max_level=6) and AMRCallback interval=20. The initial condition exactly matches the strategy: tanh density profile, piecewise hydrostatic pressure, zero u, and the specified single-mode v perturbation. Gravity source is (0,0,ρg,ρgv) with g=1.0. Time integration uses RDPK3SpFSAL49 with adaptive=false, StepsizeCallback cfl=0.5, and tspan=(0.0,3.0). Visualization produces summary_all.png with density (top row) and pressure (bottom row) at t=0.75, 1.5, 2.25, 3.0, with colorbars disabled and automatic color scaling. | N/A | 47 | 18 | 11 | 13 | **89** |
| 2 | The implementation faithfully reproduces the DGSEM solution of the 2D inviscid compressible Euler equations with a gravitational source for the single-mode Rayleigh–Taylor instability. Governing equations use CompressibleEulerEquations2D(γ=1.4); the initial condition is the verbatim strategy profile (smoothed density 2→1 via tanh(50(y-0.5))·0.5, piecewise pressure 2y+1 / y+1.5 at y=0.5, u=0, v=-0.025·c·cos(8πx)·sin^6(πy) with c=sqrt(γp/ρ), and energy via prim2cons giving E=p/(γ-1)+0.5ρv²). The gravity source term returns SVector(0,0,g·ρ, g·ρv), i.e. (0,0,ρg,ρgv). Discretization: P4estMesh on [0,0.25]×[1.0] with x-periodic (periodicity=(true,false)) and slip-wall BCs on :y_neg/:y_pos; DGSEM with polydeg=3, LobattoLegendreBasis. Advisor-mandated upgrades are present: surface_flux=flux_hllc, pressure-based IndicatorHennemannGassner shock capture (pressure_indicator derived from conservative state) feeding VolumeIntegralShockCapturingHG with volume_flux_dg=flux_ranocha and volume_flux_fv=flux_lax_friedrichs, while AMR uses a separate density-based IndicatorLöhner (variable=density_indicator=u[1]) with ControllerThreeLevel (base 4 / med 5 @0.1 / max 6 @0.3), interval=20, adapt_initial_condition=true. Held-fixed config is respected: RDPK3SpFSAL49 time integrator, adaptive=false, StepsizeCallback(cfl=0.5), tspan=(0.0,3.0). The full diagnostic suite is implemented and printed at every save: total mass (Trixi.integrate) with absolute/relative drift vs t=0, min/max ρ and p (positivity), dt_min/dt_max/current dt (CFL), max|v| at y=0 and y=1 (wall no-penetration) and an approximate symmetry residual ‖q-q(0.25-x)‖/‖q‖ about x=0.125. Visualization renders a clean 2×4 grid (density top row, pressure bottom row) at t=0.75,1.5,2.25,3.0 with colorbar=false, no mesh lines (framestyle=:none, axis=nothing), per-panel automatic scaling, and higher resolution size=(3200,2400) saved as summary_all.png. | N/A | 47 | 19 | 12 | 15 | **93** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | NPZ files are saved in directory `data_npz` with filenames `time_%08.5f.npz` at dt_save=0.1 and additionally at the visualization times 0.75, 1.5, 2.25, 3.0. Each NPZ file contains keys: `t` (scalar Float64), `x` (1D array of node coordinates, length N_nodes(t)), `y` (1D array, length N_nodes(t)), `rho` (1D array, length N_nodes(t)), `rho_u` (1D array), `rho_v` (1D array), `rho_e` (1D array), and `p` (1D array, length N_nodes(t)). N_nodes varies with AMR. Additionally, `SaveSolutionCallback` writes HDF5 solution files to `trixi_output` at dt=0.1 with full conservative state. The plot `summary_all.png` is saved in the working directory, containing a 2×4 grid: top row density, bottom row pressure at the four specified times, with colorbars disabled. |
| 2 | Per-snapshot .npz files (e.g. time_00.75000.npz, time_01.50000.npz, time_02.25000.npz, time_03.00000.npz) written to <script_dir>/data_npz/ via npzwrite. Each file contains a Dict with keys: 't' (scalar Float64), 'x' (Float64 array of DG solution-node x-coordinates from PlotData2D), 'y' (Float64 array of node y-coordinates), 'rho' (Float64 array of DG-node conserved density), 'rho_u' (Float64 array, x-momentum ρu), 'rho_v' (Float64 array, y-momentum ρv), 'rho_e' (Float64 array, total energy ρE), 'p' (Float64 array, pressure from cons2prim). All field arrays (rho, rho_u, rho_v, rho_e, p, x, y) are structured on the same DG-node grid (per-element nodes × elements; flattened with vec() by the rasterizer during plotting). Save times = union of {0.0,0.1,...,3.0} plus the four visualization times {0.75,1.5,2.25,3.0}, ensuring dt_save=0.1 cadence and exact plot snapshots. Additionally, SaveSolutionCallback writes full conservative state (cons2cons) HDF5 snapshots at dt=0.1 (plus initial) into <script_dir>/trixi_output/. Final figure summary_all.png (2 rows × 4 columns, 3200×2400) is saved to <script_dir>/. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1051/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1109/` |


---
