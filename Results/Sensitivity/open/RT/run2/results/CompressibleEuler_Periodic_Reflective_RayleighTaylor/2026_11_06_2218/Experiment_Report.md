
# Experiment Report
*Last Updated: 2026-11-06 22:24:02*

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

We solve the 2D inviscid, compressible Euler equations with a gravitational source for the single-mode Rayleigh–Taylor Instability (RTI). Heavy fluid ($\rho=2$) lies below light fluid ($\rho=1$) with gravity acting in $+y$ (from heavy into light) — a buoyantly unstable configuration. A single-wavelength vertical-velocity perturbation seeds the instability. Objective: a high-fidelity, standalone ground-truth dataset on $[0,0.25]\times[0,1]$ for $t\in[0,3]$, resolving linear growth, nonlinear spike/bubble formation, and Kelvin–Helmholtz roll-up — with the visualization defects of the previous trial eliminated.

---

**2. Mathematical Specification (Code-Ready Extraction)**

* **Governing Equations:** 2D Compressible Euler with gravity source ($\mathbf{g} = [0, g]^T$, $g=1.0$):
$$\partial_t \begin{pmatrix} \rho \\ \rho u \\ \rho v \\ \rho e \end{pmatrix} + \partial_x \begin{pmatrix} \rho u \\ \rho u^2 + p \\ \rho u v \\ u(\rho e + p) \end{pmatrix} + \partial_y \begin{pmatrix} \rho v \\ \rho u v \\ \rho v^2 + p \\ v(\rho e + p) \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ \rho g \\ \rho g v \end{pmatrix}$$
with equation of state $p = (\gamma - 1)\left(\rho e - \tfrac{1}{2}\rho(u^2 + v^2)\right)$.

* **Domain:** $(x,y) \in [0, 0.25] \times [0, 1.0]$, $t \in [0, 3.0]$.

* **Boundary Conditions:**
  * $x = 0$ and $x = 0.25$: **Periodic**.
  * $y = 0$ and $y = 1$: **Reflective / free-slip walls** (no-penetration $v = 0$; slip in $u$).

* **Initial Conditions** (interface at $y = 0.5$):
$$\rho(x,y,0) = 2.0 + (1.0 - 2.0)\cdot\frac{1}{2}\left(1 + \tanh\left(50(y - 0.5)\right)\right)$$
$$p(x,y,0) = \begin{cases} 2y + 1, & y < 0.5 \\ y + 1.5, & y \geq 0.5 \end{cases}$$
$$u(x,y,0) = 0$$
$$v(x,y,0) = -0.025 \cdot c(x,y) \cdot \cos(8\pi x) \cdot \sin^6(\pi y), \qquad c(x,y) = \sqrt{\gamma\, p / \rho}$$
$$\rho e(x,y,0) = \frac{p}{\gamma - 1} + \frac{1}{2}\rho\, v^2$$

* **Constants & Parameters:** $\gamma = 1.4$; gravity vector $\mathbf{g} = [0, 1.0]^T$; $\rho_{heavy}=2.0$, $\rho_{light}=1.0$; interface $y_0 = 0.5$; tanh smoothing width $50$; perturbation amplitude $0.025$; perturbation wavenumber $8\pi$ (exactly one period across the domain width); $\mu = 0$, $k = 0$.

* **Expected Physics:** $t \lesssim 0.75$: exponential single-mode growth, smooth interface deformation. $t \approx 1.5$–$2.25$: nonlinear phase — heavy-fluid spike ascending in $+y$ near $x = 0.125$, light-fluid bubbles descending near the periodic sides; KH shear layers on the spike flanks. $t \approx 3.0$: mushroom/roll-up structures, elongated jets approaching the upper wall, possible weak compressions from spike–wall interaction. **Contacts dominate; shocks are secondary.**

---

**3. Strategic Methodology**

* **Can we obtain an exact solution?** **No.** The system is nonlinear and multi-dimensional; the Method of Characteristics is tractable only for 1D simple waves/Riemann problems and cannot produce closed-form characteristic relations for 2D Euler with a gravity source, contact roll-up, and KH shear. Fourier spectral methods are inapplicable (non-periodic walls in $y$; steep contacts destroy global spectral convergence). No closed-form solution exists through $t=3$; the Advisor confirms no exact-method override is warranted.
* **Selection:** **Discontinuous Galerkin Spectral Element Method in 2D plus time.** It natively supports the mixed periodic/wall boundary setup, delivers high-order accuracy for the smooth roll-up, and provides AMR + shock capturing for the contact-dominated interface. This is a refinement of the previous trial's architecture (which scored 75/100 with correct qualitative physics), not a methodological change.

---

**4. Implementation Requirements (CRITICAL)**

* **Configuration:** Use template defaults, **except** for the Advisor-mandated overrides in §4.1.

* **4.0.1 Adaptive Mesh Refinement & Shock Capturing — Variable Selection Analysis:**
  * **Selected variable: Density $\rho$** — for both the AMR (Löhner) indicator and the shock-capturing (Hennemann–Gassner) indicator.
  * **Justification:** The material interface (contact discontinuity) is the dominant structural feature of RTI, and density is the *only* field that labels it — pressure and velocity are continuous across a contact and cannot localize it. Density remains sharp and localized at the interface/spike/bubble boundaries for the entire run (high signal-to-noise). In contrast, velocity becomes globally chaotic once KH roll-up and mixing begin (refining on $v$ would refine everywhere and waste elements), and pressure is smooth except for weak late-time shocks. Density is strictly superior for this PDE.

* **4.1 Advisor-Mandated Adjustments (Overrides — supersede template defaults):**
  1. **`polydeg`: 3 → 5**, retaining the existing density-driven AMR (`max_level = 6`, existing controller thresholds, `adapt_initial_condition = true`).
  2. **CFL: 0.5 → 0.3** (compensates the stricter high-order timestep restriction; within the Advisor's mandated 0.25–0.30 window).
  3. **Surface flux: `flux_hll` → `flux_hllc`** (contact-resolving; reduces smearing of density interfaces).
  4. **Shock-capturing `alpha_max`: 1.0 → 0.5**, retaining the Ranocha/Lax–Friedrichs blending with `alpha_min = 0.001`, `alpha_smooth = true`. If genuine oscillations appear after these changes, increase stabilization moderately — do **not** revert to the most dissipative configuration.
  5. **Visualization repair (critical):** Render the AMR solution by interpolating the DG polynomial onto a fine uniform Cartesian grid (or the library's native AMR-aware plot-data structure) with **seamless element assembly — no mesh-edge lines, no row gaps, no horizontal banding**. Verify orientation ($x$ horizontal, $y$ vertical) and aspect ratio (domain is $0.25$ wide $\times\,1.0$ tall → tall panels). Regenerate `summary_all.png` from this new run.
  6. **Runtime diagnostics (printed at save intervals):** finiteness of all fields; $\min(\rho)$ and $\min(p)$ (positivity — do not rely on the figure); mass $\int \rho\, dA$ (must be conserved); extrema of $\rho$ and $p$; $\max|v|$ on both walls $y=0$ and $y=1$ (no-penetration check); energy balance including gravity-source work — the modified total energy $\int (E - \rho g y)\, dA$ (internal + kinetic + gravitational potential) must remain constant; report its drift.
  7. **Standalone trial:** fresh folder, run from scratch; no loading, re-rendering, or comparison against previous run outputs; all figures and data regenerated from this run.

---

**5. Visualization Strategy**

* **Field Selection (Analysis):** Plot **$\rho$ (density)** — the primary driver of RTI; the density contrast *is* the instability and the interface morphology is the solution — and **$v$ (vertical velocity)** — captures the spike/bubble jets and shear layers. Do **not** plot $u$, $p$, or $E$.
* **Times:** $t = 0.75,\ 1.5,\ 2.25,\ 3.0$.
* **Layout:** Grid — 2 rows (fields $\rho$, $v$) $\times$ 4 columns (times).
* **Rendering:** AMR-consistent interpolated rendering per §4.1 item 5 (no striping/mesh artifacts). **`colorbar = false` (Julia — CRITICAL)**. No fixed colormaps; let scales adjust automatically. No derived quantities.
* **Saving Data:** Save the **full solution state** (all conserved variables) at `dt_save = 0.1` for ground truth.
* **Save as:** `summary_all.png` (strictly).

---

**6. Cross-Validation & Data Integration (Conditional)**

No external reference data or sensor paths are provided in the problem statement → **None**. Ground-truth certification is established via the runtime positivity/conservation diagnostics of §4.1 item 6.

---

**REQUIRED COMPONENTS**

**Main Method:**
* Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
* Adaptive Mesh Refinement (density-based Löhner indicator, `max_level = 6`)
* Shock Capturing (Hennemann–Gassner indicator on density, `alpha_max = 0.5`)
* Contact-Resolving Surface Flux (`flux_hllc`) — Advisor-mandated
* AMR-Consistent Interpolated Visualization — Advisor-mandated
* Runtime Positivity & Conservation Diagnostics — Advisor-mandated

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

**Status:** `REJECTED_FOR_REFINEMENT`

The run is high quality, but it has not reached a perfect score. The history contains only one previous score, **75/100**, so there is no previous run scoring at least 90 and the five-run saturation rule does not apply.

## Section 2: Quantitative Scorecard

- **Score Integrity:** 50/50
- **Score Accuracy:** 18/20
- **Score Details:** 13/15
- **Score Optimality:** 15/15
- **TOTAL SCORE:** **96/100**

## Section 3: Forensic Analysis & Feedback

### Observation

**Execution and visualization**

- The log reaches `Saving summary_all.png... Done.`, and solution files are written at all four requested times. No crash, `NaN`, or `Inf` is reported in the supplied tail.
- All eight panels are readable, correctly oriented, and have the expected tall aspect ratio.
- The previous visualization defects appear resolved: there are no obvious element outlines, row gaps, horizontal bands, or blank panels.

**Physical development**

- At \(t=0.75\), the interface shows the expected smooth single-mode deformation.
- At \(t=1.5\), a central heavy-fluid spike rises in \(+y\), complemented by descending light-fluid regions near the periodic sides. This is consistent with the imposed perturbation and gravity direction.
- At \(t=2.25\), the spike develops a mushroom-like cap with rolled shear layers on its flanks.
- At \(t=3.0\), the spike approaches the upper wall and elongated Kelvin–Helmholtz structures are visible. The vertical-velocity panels consistently localize the strongest motion around the central jet and its shear layers.
- The solution remains broadly left–right symmetric. There is no obvious large-scale Gibbs ringing, checkerboarding, or catastrophic interface breakup.

The main residual visual limitation is that the thinnest late-time rolled filaments appear somewhat soft. In addition, automatic per-panel scaling and the absence of quantitative axis/color information prevent density magnitudes and interface thickness from being audited quantitatively.

**Missing verification evidence**

The supplied tail contains timing information and output paths, but not the mandated numerical values for positivity, mass conservation, modified-energy conservation, or wall normal velocity. Consequently:

- Positivity of \(\rho\) and \(p\) cannot be certified from the supplied log.
- Mass and modified-energy drift cannot be evaluated.
- The images show no obvious through-wall flow, but they cannot certify the no-penetration condition.

### Error and Exact-Solution Assessment

The supplied log tail contains **no** `Minimum achieved error:` entry.

**Quantitative error is not applicable as no ground truth is available.** This is a forward nonlinear RTI simulation, not a manufactured-solution or analytical-reference calculation; no numerical error value should be inferred from the plotting output.

An exact solution through \(t=3\) is not feasible: nonlinear multidimensional Euler with gravity, contact roll-up, and Kelvin–Helmholtz shear does not admit a closed-form characteristic or Fourier solution. Therefore, the selected numerical architecture is appropriate and no exact-method override is warranted.

### Diagnosis

The result shows **correct qualitative RTI physics with substantially improved contact resolution**. The remaining issues are:

1. **Incomplete conservation/positivity verification**, rather than evidence of a failed simulation.
2. **Mild late-time numerical dissipation or AMR-limited filament resolution.** Density-based shock sensing can activate Lax–Friedrichs blending at steep but physically smooth contacts, while `max_level=6` limits the smallest rolled structures that can be represented.

There is no evidence requiring a change away from DGSEM, HLLC, or the current boundary-condition implementation.

### Targeted Improvements

All changes should be made in a **fresh standalone run in a new folder**, regenerating every solution file and `summary_all.png` from scratch.

- **For Integrity:** None. Execution and figure generation were successful.

- **For Accuracy:**  
  Retain the current ICs, BCs, `polydeg=5`, `flux_hllc`, and CFL \(=0.3\). Ensure the next log includes a compact final verification table containing:
  - finiteness of all conserved variables;
  - \(\min(\rho)\) and \(\min(p)\);
  - relative mass drift;
  - relative drift of  
    \[
    \mathcal E^*=\int_\Omega (E-\rho g y)\,dA;
    \]
  - maximum wall \(|v|\), and preferably the wall normal mass flux, at \(y=0\) and \(y=1\).
  
  Also print density and pressure extrema at the four plotted times so the autoscale visualization can be interpreted quantitatively.

- **For Details:**  
  Prefer **local AMR improvement over global refinement or another polynomial-degree increase**. As a concrete next configuration, retain `max_level=6` but test:
  - `med_threshold: 0.1 → 0.05`
  - `max_threshold: 0.3 → 0.15`
  
  This should retain more elements around the late-time roll-up and spike–wall interaction without refining the entire domain.
  
  Because no significant ringing is visible, a controlled dissipation reduction is also justified: test `alpha_max: 0.5 → 0.25`, keeping `alpha_min=0.001`, `alpha_smooth=true`, and HLLC unchanged. If this produces ringing or nonpositive states, restore `alpha_max=0.5` in the subsequent fresh run while retaining the improved AMR; do **not** revert to the more dissipative HLL configuration.

- **For Optimality:** None. DGSEM with contact-resolving HLLC, density-driven AMR, high-order time integration, and moderate shock capturing is well matched to this contact-dominated, shock-secondary problem.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 2 | Numerical implementation of 2D inviscid compressible Euler RTI using DGSEM in Trixi. Uses CompressibleEulerEquations2D with gamma=1.4, P4estMesh on [0,0.25]x[0,1] with periodicity=(true,false), initial_refinement_level=3, polydeg=3. DGSEM has surface_flux=flux_hll and VolumeIntegralShockCapturingHG with IndicatorHennemannGassner on density (alpha_max=1.0, alpha_min=0.001, alpha_smooth=true) blending flux_ranocha and flux_lax_friedrichs. Slip-wall boundary conditions are set on y_neg/y_pos; x is periodic. Gravity source term implements S=(0,0,rho*g,g*rho*v) with g=1.0 in +y. Initial conditions match the strategy: smoothed tanh density interface at y=0.5 (rho=2 below, rho=1 above), piecewise hydrostatic pressure (2y+1 for y<0.5, y+1.5 for y>=0.5), u=0, v=-0.025*c*cos(8*pi*x)*sin(pi*y)^6 with c=sqrt(gamma*p/rho), and energy via prim2cons. AMR uses IndicatorLöhner on density with ControllerThreeLevel base_level=4, med_level=5, med_threshold=0.1, max_level=6, max_threshold=0.3, interval=20, adapt_initial_condition=true. Time integration is RDPK3SpFSAL49, adaptive=false, tspan=(0.0,3.0), CFL=0.5 via StepsizeCallback, saveat=0.1. Visualization plots density rho and vertical velocity v at t=0.75,1.5,2.25,3.0 in a 2x4 layout, colorbar=false, no fixed colormap, saving summary_all.png. | N/A | 40 | 15 | 8 | 12 | **75** |
| 4 | The generated code implements the requested 2D inviscid compressible Euler RTI with gravity using Trixi.jl DGSEM. Domain [0,0.25]x[0,1], t in [0,3]. Equations use CompressibleEulerEquations2D(1.4); gravity source S=(0,0,rho*g,rho*g*v) with g=1. IC matches: smoothed density interface at y=0.5 via tanh(50(y-0.5)), piecewise hydrostatic pressure, u=0, v=-0.025*sqrt(gamma p/rho)*cos(8πx)*sin^6(πy). BCs: periodic x, slip walls y. Solver: LobattoLegendreBasis polydeg=5, DGSEM with surface_flux=flux_hllc; shock capturing via IndicatorHennemannGassner on density with alpha_max=0.5, alpha_min=0.001, alpha_smooth=true, volume flux blend flux_ranocha/flux_lax_friedrichs. AMR is density-based Löhner with ControllerThreeLevel max_level=6, adapt_initial_condition=true. CFL=0.3. Time integration RDPK3SpFSAL49, adaptive=false, tspan=(0,3). Runtime diagnostics print finiteness, positivity, rho/p extrema, mass drift, modified energy drift, wall max|v|. Visualization plots only rho and v at t=0.75,1.5,2.25,3.0 in a 2x4 layout with colorbar=false, AMR-consistent rasterization to 256x1024, and saves summary_all.png. | N/A | 50 | 18 | 13 | 15 | **96** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 2 | Creates a fresh rti_run/ directory and rti_run/data_npz/. NPZ snapshots are written with npzwrite to files named time_%08.5f.npz (e.g., time_00.75000.npz) at t=0.0,0.1,0.2,...,3.0 plus 0.75,1.5,2.25. Each NPZ file contains keys: 't' (scalar Float64), 'x' (1D vector length N_points), 'y' (1D vector length N_points), 'rho' (1D vector length N_points), 'rho_u' (1D vector length N_points), 'rho_v' (1D vector length N_points), 'rho_e' (1D vector length N_points), 'u' (1D vector length N_points), 'v' (1D vector length N_points), 'p' (1D vector length N_points). N_points is the number of PlotData2D visualization points from the adaptive DG mesh and varies with AMR. Additionally, SaveSolutionCallback writes Trixi HDF5 solution files to rti_run/trixi_output/ at dt=0.1 with initial and final solutions saved, using solution_variables=cons2cons. The final 2x4 visualization is saved as summary_all.png in the script directory. |
| 4 | NPZ snapshots are written to rti_run/data_npz/time_%08.5f.npz at times 0.0:0.1:3.0 plus 0.75 and 2.25. Each NPZ contains keys: t (scalar Float64); x, y, rho, rho_u, rho_v, rho_e, u, v, p (arrays with the raw Trixi PlotData2D element-patch shape; the code does not reshape before writing). With default PlotData2D sampling this is the AMR element-patch layout, typically nvis=2*polydeg+1=11 per element and n_elements AMR-dependent. The code also uses SaveSolutionCallback(dt=0.1, solution_variables=cons2cons) writing full DG conservative-state snapshots to rti_run/trixi_output. The final figure summary_all.png is saved at @__DIR__/summary_all.png with size 1600x2200, layout 2 rows (rho, v) by 4 columns (t=0.75,1.5,2.25,3.0), and colorbar=false. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run2/results//CompressibleEuler_Periodic_Reflective_RayleighTaylor/2026_11_06_2147/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run2/results//CompressibleEuler_Periodic_Reflective_RayleighTaylor/2026_11_06_2218/` |


---
