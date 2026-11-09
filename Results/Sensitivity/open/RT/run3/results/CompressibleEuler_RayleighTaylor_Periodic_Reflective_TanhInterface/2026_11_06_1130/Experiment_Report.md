
# Experiment Report
*Last Updated: 2026-11-06 11:38:44*

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

2D inviscid compressible Euler equations with a gravitational source term, evolving the single-mode Rayleigh–Taylor (RT) instability. Unstable stratification: heavy fluid ($\rho=2$) rests below light fluid ($\rho=1$) while gravity pushes in $+y$ — the mirror-equivalent of the classic heavy-over-light configuration. This is a **refinement run**: the Advisor has accepted the polydeg=5 architecture (Run 1, score 97) and mandated a focused p-convergence trial (polydeg=6, CFL=0.4) with explicit diagnostic certification. Do not re-architect; verify and certify.

**1. Mathematical Specification (Code-Ready Extraction)**

* **Governing Equations:** 2D Compressible Euler with gravity (source adds momentum in $+y$):
$$\partial_t \rho + \partial_x(\rho u) + \partial_y(\rho v) = 0$$
$$\partial_t(\rho u) + \partial_x(\rho u^2 + p) + \partial_y(\rho u v) = 0$$
$$\partial_t(\rho v) + \partial_x(\rho u v) + \partial_y(\rho v^2 + p) = \rho g$$
$$\partial_t E + \partial_x\big((E+p)u\big) + \partial_y\big((E+p)v\big) = \rho g v$$
$$E = \frac{p}{\gamma - 1} + \frac{1}{2}\rho(u^2 + v^2), \qquad \gamma = 1.4, \qquad g = 1.0$$

* **Domain:** $(x,y) \in [0, 0.25] \times [0, 1.0]$, $t \in [0, 3.0]$

* **Boundary Conditions:** $x$: Periodic. $y = 0$ and $y = 1$: Reflective / free-slip walls (no-penetration, $v = 0$).

* **Initial Conditions:**
$$\rho(x,y,0) = 2.0 + (1.0 - 2.0)\cdot\frac{1}{2}\left[1 + \tanh\!\left(50(y - 0.5)\right)\right]$$
$$p(x,y,0) = \begin{cases} 2y + 1, & y < 0.5 \\ y + 1.5, & y \geq 0.5 \end{cases}$$
$$u(x,y,0) = 0$$
$$v(x,y,0) = -0.025\, c(y)\,\cos(8\pi x)\,\sin^6(\pi y), \qquad c(y) = \sqrt{\gamma\, p(y)/\rho(y)}$$
$$E(x,y,0) = \frac{p(y)}{\gamma - 1} + \frac{1}{2}\rho(y)\, v^2(x,y,0)$$

* **Constants & Parameters:** $\gamma = 1.4$; $g = 1.0$ (source vector $[0,\,0,\,\rho g,\,\rho g v]^T$); $\rho_{heavy}=2.0$, $\rho_{light}=1.0$; interface at $y_0 = 0.5$; tanh smoothing parameter $50$; perturbation amplitude $0.025$; mode $\cos(8\pi x)$ → wavelength $\lambda = 0.25$ (exactly one period across the domain); Atwood number $A_t = 1/3$.

* **Expected Physics:** Linear RT growth ($t \lesssim 1$) of the single mode; nonlinear phase with heavy-fluid spikes rising in $+y$ and light-fluid bubbles sinking; Kelvin–Helmholtz roll-up forming mushroom caps ($t \approx 2$–$2.5$); late-time mixing layer with thin density filaments, spike approaching the $y=1$ wall by $t = 3.0$. Flow is essentially smooth except steep gradients at the material interface; wall interaction may generate weak acoustic waves late-time.

**2. Strategic Methodology**

* **Exact solution feasibility: NOT FEASIBLE.**
    * **MoC:** Applicable only to 1D hyperbolic systems/simple waves. Here the 2D characteristic field intersects itself to generate vorticity (KH roll-up); no closed-form characteristic network exists. The *only* exact solution of this system is the unperturbed hydrostatic equilibrium ($u=v=0$, $dp/dy = \rho g$), which the imposed velocity perturbation destroys — it is not the solution of the posed IVP.
    * **Fourier Spectral:** Requires periodicity in all directions (the $y$-direction has walls) and global smoothness (steep interface + filaments). Inapplicable.
    * The Advisor explicitly concurs: no exact-solution override is warranted.
* **Selection: DGSEM in 2D plus time.** High-order accuracy for the smooth vortex-dominated bulk, entropy-stable flux splitting, natural handling of mixed periodic/wall boundaries, and AMR + localized FV blending for the steep interface. This is the accepted Run-1 architecture; only the Advisor's targeted refinements change.

**3. Implementation Requirements (CRITICAL)**

* **Configuration:** Use template defaults **EXCEPT** for the following Advisor-mandated overrides (these supersede all defaults):
    * `polydeg = 6` (focused p-convergence trial vs. the accepted polydeg=5 baseline).
    * `CFL = 0.4` (StepsizeCallback).
    * **Retain** (do not change): surface flux `flux_hllc`, entropy-conserving volume flux `flux_ranocha`, FV stabilization flux `flux_lax_friedrichs`, density-based `IndicatorHennemannGassner` shock capturing, density-based `IndicatorLöhner` AMR with `ControllerThreeLevel` (levels 4/5/6, interval = 10), `P4estMesh` with `initial_refinement_level = 3`.
    * **Do NOT** revert to HLL or polydeg=3.

* **3.0.1 Adaptive Mesh Refinement & Shock Capturing — Variable Selection Analysis:**
    * **Selected variable: Density $\rho$** (for both the Löhner AMR indicator and the Hennemann–Gassner shock indicator).
    * **Justification:** The structural features of RT — spikes, bubbles, and filaments — are material interfaces, i.e., density discontinuities. Density remains sharp and coherent *only* at these features. Vertical velocity $v$ becomes globally chaotic inside the mixing layer (KH vortices, shear layers); refining on it would scatter refinement across smooth vortical regions and waste resources. Pressure is dominated by the smooth hydrostatic background ($p \approx 1$–$3$) with weak perturbations, so its indicator is poorly localized to the interface. Density is strictly superior for this PDE.

* **3.1 Advisor-Mandated Diagnostics (MUST be computed and PRINTED to stdout at every output time $t = 0.75, 1.5, 2.25, 3.0$):**
    * $\min(\rho)$ and $\min(p)$ over the domain (positivity certification).
    * $\max|v|$ on $y=0$ and $y=1$ (no-penetration certification).
    * Relative total-mass drift: $|M(t) - M(0)|/M(0)$.
    * Energy-balance residual: compare discrete $\frac{d}{dt}\int E\, dA$ against the source work $\int \rho g v\, dA$.
    * Interface-apex position (e.g., $y_{\max} = \max\{y : \rho > 1.5\}$) and mixing-layer width (e.g., $y_{\max} - y_{\min}$ from the $\rho = 1.5$ threshold).
    * **p-convergence comparison:** if the stored polydeg=5 baseline state is available on disk, load it and report $\Delta$apex, $\Delta$mixing-width, and late-time filament thickness at $t=3.0$. **Contingency (Advisor):** if the p=6 metrics are unchanged, explicitly report that residual smearing is controlled by FV blending / the AMR level cap — not polynomial resolution.

**4. Visualization Strategy**

* **Field Selection (Analysis):** Plot **$\rho$** (primary driver: interface, spikes, bubbles, filaments — the RT dynamics *is* the density field) and **$v$** (vertical velocity: spike/bubble motion and the gravity source-work coupling). Do not plot $u$, $p$, $E$ — subplots become unreadable.
* **Task:** Plot $\rho$ and $v$ at $t = 0.75, 1.5, 2.25, 3.0$.
* **Layout:** Grid — 2 rows (fields) × 4 columns (times).
* **Saving Data:** Save the **full solution state** (all conserved variables) at frequent intervals (`dt_save = 0.1`) for Ground Truth archiving; plotting selection affects only the summary image.
* **Constraints:** Save as `summary_all.png` (strict). **DISABLE COLORBARS** (`colorbar=false`) — Julia. No fixed colormaps; let scales adjust automatically. Gap-free rasterization. Plot only primary variables; no derived quantities.

**5. Cross-Validation & Data Integration (Conditional)**

* No external data paths (sensors / reference solutions) are present in [USER_QUESTION] → **None**.
* Verification is internal: the Advisor-mandated p-convergence comparison against the stored polydeg=5 baseline (Section 3.1) plus the full diagnostics suite constitutes the verification layer.

**REQUIRED COMPONENTS**

**Main Method:**
* Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
* Adaptive Mesh Refinement (Density-based IndicatorLöhner, ControllerThreeLevel)
* Shock Capturing (Density-based IndicatorHennemannGassner, FV blending)
* Advisor-Mandated p-Convergence Trial (polydeg=6, CFL=0.4)
* Runtime Diagnostics & Baseline Comparison Suite (positivity, wall no-penetration, mass drift, energy balance, apex/mixing width)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

**Status:** `REJECTED_FOR_REFINEMENT`

Run 1 previously achieved **97 points**, so the history-saturation rule is active. The present submission is only the **first refinement trial since that high-score run**: count = 1 < 5. Consequently, exploration has not saturated. Four additional post-success refinement trials are required unless a perfect score is achieved earlier.

## Section 2: Quantitative Scorecard

- **Score Integrity:** 50/50
- **Score Accuracy:** 19/20
- **Score Details:** 13/15
- **Score Optimality:** 15/15
- **TOTAL SCORE:** **97/100**

**Error-log check:** The supplied stdout tail does **not** contain a `Minimum achieved error:` line. It ends with timing information followed by successful generation and saving of `summary_all.png`.

**Quantitative error is not applicable as no ground truth is available.** This is a forward simulation of the nonlinear Rayleigh–Taylor problem, not a manufactured-solution or analytical-reference calculation.

## Section 3: Forensic Analysis & Feedback

### Observation

- The simulation completed successfully, and all eight requested panels are present and readable. There are no blank fields, obvious NaN artifacts, or catastrophic instabilities.
- The density evolution is physically coherent:
  - At \(t=0.75\), the initial single-mode interface deformation is visible.
  - At \(t=1.5\), the heavy-fluid spike has developed a bulbous cap.
  - At \(t=2.25\), the cap has rolled up into a mushroom structure with thin side filaments, consistent with Kelvin–Helmholtz instability.
  - At \(t=3.0\), the spike and mixing layer extend toward the upper wall, with multiple narrow filaments and no obvious large-scale topological artifact.
- The vertical-velocity field is concentrated around the rising spike and surrounding return flow, which is consistent with the density evolution. Approximate left–right symmetry is retained, and there is no conspicuous Gibbs ringing in the plotted velocity.
- Some late-time filaments appear soft or only a few pixels wide. The image alone cannot distinguish modest numerical diffusion from rasterization limits, so this is not classified as severe interface smearing.
- The implementation report states that positivity, wall velocity, mass, energy, and interface diagnostics were computed. However, their numerical values and the requested polydeg-5/polydeg-6 comparison are absent from the supplied log tail. Thus, positivity, exact no-penetration, conservation, and p-convergence cannot be independently certified here.

### Diagnosis

The remaining limitation is **possible residual contact-interface dissipation and unverified p-convergence**, not a stability failure. The likely dissipative components are the local Lax–Friedrichs FV blending and the AMR level cap, rather than the already contact-resolving HLLC surface flux.

Increasing the polynomial degree further is unlikely to cure filament softness if the polydeg-5 and polydeg-6 apex, mixing-width, and filament metrics are effectively unchanged. The cost of polydeg 6 is justified only if that comparison demonstrates a measurable improvement.

The architecture remains methodologically excellent: DGSEM with Ranocha volume flux, HLLC surface flux, localized FV stabilization, and density-based AMR is well matched to this problem. No exact-solution override is warranted: the perturbed nonlinear 2D Euler Rayleigh–Taylor problem with wall boundaries and vortex roll-up has no feasible closed-form, Fourier-series, or method-of-characteristics solution. The unperturbed hydrostatic equilibrium is exact, but it is not the solution of this IVP.

### Targeted Improvements

- **For Integrity:** None. Preserve the successful CFL \(=0.4\) configuration and output workflow.

- **For Accuracy:**  
  Add a compact final certification block to stdout, or retain it in the supplied tail, containing at every output time:
  \(\min(\rho)\), \(\min(p)\), wall \(\max|v|\), relative mass drift, energy-balance residual, apex position, and mixing-layer width. Explicitly print the polydeg-5/polydeg-6 differences at \(t=3\), or state clearly why the stored baseline was unavailable. Successful completion alone does not certify these properties.

- **For Details:**  
  Do **not** increase to `polydeg=7`. Instead, perform a one-factor dissipation trial: keep polydeg 6, CFL \(=0.4\), HLLC surface flux, Ranocha volume flux, and AMR fixed, while explicitly reducing the Hennemann–Gassner blending strength—for example, `alpha_max: 0.5 -> 0.25` if 0.5 was the previous value. Monitor positivity and check for new interface ringing.  
  If this does not sharpen the filaments, the next isolated trial should tighten the AMR controller so the material interface remains at level 6, or permit level 7 only in a narrow interface band subject to the DOF/runtime budget. A contact-resolving FV stabilization flux can also be considered in place of Lax–Friedrichs, but should be tested separately rather than simultaneously with AMR changes.

- **For Optimality:** None. Retain the DGSEM/HLLC/Ranocha architecture; no re-architecture or exact-method substitution is indicated.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The code implements a 2D inviscid compressible Euler simulation of single-mode Rayleigh-Taylor instability using a numerical discontinuous Galerkin spectral element method (DGSEM) with Trixi.jl. The governing equations use gamma=1.4, gravitational source terms in +y with g=1.0, domain [0,0.25] x [0,1], t in [0,3.0], periodic x-boundaries, and free-slip/reflective y-walls. Initial conditions match the strategy: smoothed density interface with tanh(50(y-0.5)), piecewise hydrostatic pressure, u=0, and v=-0.025*c*cos(8*pi*x)*sin(pi*y)^6. The solver uses polydeg=3, P4estMesh with initial_refinement_level=3, density-based AMR via IndicatorLöhner with ControllerThreeLevel levels 4-6 and interval 20, density-based shock capturing via IndicatorHennemannGassner, HLL surface flux, entropy-conserving volume flux flux_ranocha, FV stabilization flux flux_lax_friedrichs, RDPK3SpFSAL49 time integration, and StepsizeCallback with CFL=0.5. The visualization code plots density and vertical velocity at t=0.75, 1.5, 2.25, and 3.0 in a 2x4 layout and saves summary_all.png with colorbar=false. | N/A | 45 | 17 | 10 | 12 | **84** |
| 1 | The code implements a 2D compressible Euler Rayleigh-Taylor simulation using Trixi's DGSEM on a P4estMesh over [0,0.25]x[0,1] with t in [0,3]. It uses CompressibleEulerEquations2D(gamma=1.4), polydeg=5, LobattoLegendreBasis, surface flux flux_hllc, entropy-conserving volume flux flux_ranocha, FV stabilization flux_lax_friedrichs, density-based IndicatorHennemannGassner shock capturing, density-based IndicatorLöhner AMR with ControllerThreeLevel levels 4/5/6 and interval=10, and RDPK3SpFSAL49 with StepsizeCallback(cfl=0.5). Initial conditions match the strategy: smoothed heavy/light density profile, piecewise hydrostatic pressure, zero u, v perturbation -0.025*c(y)*cos(8*pi*x)*sin(pi*y)^6, and E via prim2cons. Gravity source [0,0,+rho,+rho*v] is implemented. Boundaries are periodic in x and slip walls at y=0,1. Runtime diagnostics check finiteness, positivity, wall v extrema, total mass, energy tendency vs source work, interface apex/mixing width, and polydeg=3 baseline comparison when baseline data exist. Visualization plots rho and v at t=0.75,1.5,2.25,3.0 in a 2x4 heatmap layout with colorbar=false and gap-free rasterization. | N/A | 50 | 19 | 13 | 15 | **97** |
| 5 | The code implements the required Trixi DGSEM solver for the 2D inviscid compressible Euler equations with gravity source. It uses polydeg=6, CFL=0.4, P4estMesh on [0,0.25]x[0,1] with initial_refinement_level=3, periodic x and slip-wall y boundaries. Surface flux is flux_hllc; volume integral is VolumeIntegralShockCapturingHG with flux_ranocha as DG flux, flux_lax_friedrichs as FV flux, and density-based IndicatorHennemannGassner. AMR uses density-based IndicatorLöhner and ControllerThreeLevel with levels 4/5/6 and interval=10. The initial condition matches the strategy exactly: rho tanh profile, piecewise hydrostatic pressure, u=0, v=-0.025*c(y)*cos(8*pi*x)*sin(pi*y)^6 with c=sqrt(gamma*p/rho), and prim2cons for E. The source term is [0,0,rho*g,rho*g*v]. Time integration is explicit RDPK3SpFSAL49 with adaptive=false and StepsizeCallback(cfl=0.4) over t in [0,3]. The code computes and prints min(rho), min(p), wall max|v| at y=0/1, relative mass drift, energy-balance residual, interface apex, and mixing-layer width; it also attempts optional polydeg=5 baseline comparison and the mandated contingency message. Visualization plots rho and v at t=0.75,1.5,2.25,3.0 in a 2x4 layout, saves summary_all.png, and disables colorbars. No strategy mismatch was found. | N/A | 50 | 19 | 13 | 15 | **97** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves NPZ snapshots in data_npz/ named time_%08.5f.npz (e.g., time_00.00000.npz through time_03.00000.npz) at t=0.0,0.1,0.2,...,3.0 plus the visualization times. Each NPZ contains keys: 't' (scalar Float64), 'x' and 'y' (1D Vector{Float64} of scattered PlotData2D node coordinates, length N_points), primitive variables 'rho','u','v','p' (1D Vector{Float64}, length N_points), conservative variables 'conservative_rho','rho_u','rho_v','rho_e' (1D Vector{Float64}, length N_points), and 'state_vector' (1D Vector{Float64} raw DG state, length N_DOFs). N_points and N_DOFs vary with AMR. Also saves Trixi HDF5 solution files in trixi_output/ via SaveSolutionCallback at dt=0.1, including initial and final states, with solution_variables=cons2cons (full conservative DG state; exact HDF5 structure is Trixi's native format and mesh/DOF-dependent). Saves summary_all.png as a PNG 2x4 panel plot: row 1 density (rho), row 2 vertical velocity (v), columns t=0.75,1.5,2.25,3.0, with colorbars disabled. |
| 1 | Saves NPZ snapshots under data_npz with filename pattern time_%08.5f.npz (e.g., time_00.00000.npz) using NPZ.jl, at t=0.0,0.1,...,3.0 plus the visualization times 0.75,1.5,2.25,3.0. Each .npz contains: t scalar Float64; x, y, rho, u, v, p, conservative_rho, rho_u, rho_v, rho_e as 2D arrays on the current PlotData2D visualization grid (dimensions vary with AMR and are oriented to match plotting); state_vector as a 1D vector of raw DG conservative degrees of freedom; and scalar diagnostics min_rho, min_p, total_mass, mass_drift, total_energy, energy_tendency, source_work, energy_balance_residual, interface_apex, mixing_layer_width, wall_v_y0_min, wall_v_y0_max, wall_v_y1_min, wall_v_y1_max. It also invokes Trixi SaveSolutionCallback into trixi_output with dt=0.1, saving full conservative variables (cons2cons) at initial, final, and intermediate times. Finally it saves summary_all.png as a 2x4 continuous heatmap grid of rho and v at the four requested times. It optionally reads data_npz_polydeg3 for baseline convergence comparison; no baseline files are written by this code. |
| 5 | The custom DiscreteCallback writes NPZ snapshots to data_npz/time_XXXXXXXX.npz at t=0 and when crossing npz_save_times (0.0:0.1:3.0 plus 0.75,1.5,2.25,3.0). Each NPZ contains keys: scalar t; arrays x, y, rho, u, v, p, conservative_rho, rho_u, rho_v, rho_e (PlotData2D sample arrays with identical 2D grid shape; exact n_x-by-n_y dimensions are AMR/plot-sampling dependent); state_vector (1D Vector of the full DG conservative state, length ndofs); and scalars min_rho, min_p, total_mass, mass_drift, relative_mass_drift, total_energy, energy_tendency, source_work, source_work_interval_mean, energy_balance_residual, integrated_energy_balance_residual, interface_apex, mixing_layer_width, filament_thickness, wall_v_y0_min, wall_v_y0_max, wall_v_y1_min, wall_v_y1_max, wall_v_y0_maxabs, wall_v_y1_maxabs. Additionally, Trixi SaveSolutionCallback writes the full conservative DG state to trixi_output at dt=0.1, including initial/final states, with solution_variables=cons2cons. The summary image summary_all.png is saved as a 2x4 rho/v panel grid with colorbar=false. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run3/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective_TanhInterface/2026_11_06_1051/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run3/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective_TanhInterface/2026_11_06_1107/` |
| 5 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run3/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective_TanhInterface/2026_11_06_1130/` |


---
