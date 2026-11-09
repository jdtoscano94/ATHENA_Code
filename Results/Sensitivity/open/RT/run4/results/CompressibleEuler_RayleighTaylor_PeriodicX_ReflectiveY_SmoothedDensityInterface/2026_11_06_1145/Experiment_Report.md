
# Experiment Report
*Last Updated: 2026-11-06 11:54:17*

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
# Scientific Strategy Report — Trial #8 (Refinement)
**Problem:** 2D Compressible Euler Rayleigh–Taylor Instability | **Prior:** Run #7 (85/100, REJECTED_FOR_REFINEMENT)

---

**1. Problem Statement**

Refinement of the inviscid compressible Euler RTI simulation. The Run #7 DGSEM/AMR architecture is physically sound (correct morphology, no NaN/blowup). The Advisor's forensics identify three fixable deficiencies, none of which justify an architecture change: (i) **contact-wave dissipation** from `flux_hll` softening the RT interface (the feature of interest), (ii) **striped AMR-stitching visualization artifacts** masquerading as oscillations, and (iii) **missing quantitative verification and data archival**. Strategy: retain the DGSEM+AMR skeleton, upgrade to a contact-resolving HLLC surface flux, raise `polydeg` 3→5, repair the visualization pipeline, and certify the ground truth via a p=3/p=5 convergence comparison.

---

**1. Mathematical Specification (Code-Ready Extraction)**

* **Governing Equations:** 2D compressible Euler with gravitational source ($g = 1.0$, acting in $+y$):

$$\partial_t \begin{pmatrix} \rho \\ \rho u \\ \rho v \\ E \end{pmatrix} + \partial_x \begin{pmatrix} \rho u \\ \rho u^2 + p \\ \rho u v \\ u(E+p) \end{pmatrix} + \partial_y \begin{pmatrix} \rho v \\ \rho u v \\ \rho v^2 + p \\ v(E+p) \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ \rho g \\ \rho v g \end{pmatrix}$$

Equation of state: $p = (\gamma - 1)\left(E - \tfrac{1}{2}\rho(u^2 + v^2)\right)$, $\gamma = 1.4$. Inviscid: $\mu = 0$, $k = 0$.

* **Domain:** $(x,y) \in [0, 0.25] \times [0, 1.0]$, $t \in [0, 3.0]$.
* **Boundary Conditions:**
  * $x = 0$ and $x = 0.25$: **Periodic**.
  * $y = 0$ and $y = 1$: **Reflective / free-slip walls** — no-penetration $v = 0$; mirror state $(\rho, \rho u, \rho v, E) \mapsto (\rho, \rho u, -\rho v, E)$.

* **Initial Conditions** (interface at $y_0 = 0.5$):
$$\rho(y) = 2.0 + (1.0 - 2.0)\cdot\frac{1}{2}\left[1 + \tanh\left(50\,(y - 0.5)\right)\right]$$
$$p(y) = \begin{cases} 2y + 1, & y < 0.5 \\ y + 1.5, & y \geq 0.5 \end{cases}$$
$$u(x,y) = 0$$
$$v(x,y) = -0.025\, c(y)\, \cos(8\pi x)\, \sin^6(\pi y), \qquad c(y) = \sqrt{\frac{\gamma\, p(y)}{\rho(y)}}$$
$$E(x,y) = \frac{p(y)}{\gamma - 1} + \frac{1}{2}\,\rho(y)\, v(x,y)^2$$

* **Constants & Parameters:** $\gamma = 1.4$; gravity vector $\mathbf{g} = [0, +1.0]^T$; $\rho_{heavy} = 2.0$, $\rho_{light} = 1.0$; interface $y_0 = 0.5$; tanh steepness $50$; perturbation amplitude $0.025$; perturbation wavenumber $8\pi$ (single mode over the $x$-period $0.25$); $\mu = 0$, $k = 0$.
* **Expected Physics:** Piecewise hydrostatic balance $\partial_y p = \rho g$ holds initially; dynamics driven by the single-mode perturbation. Linear RT growth ($t \lesssim 1$) → nonlinear plumes with mushroom caps ($t \approx 1.5$–$2$) → KH roll-up on plume flanks and secondary instabilities ($t \approx 2$–$3$). Heavy-fluid spikes penetrate in $+y$; light-fluid bubbles penetrate in $-y$. The interface is a **contact discontinuity** (the certification target); only weak acoustic transients are expected. Positivity of $\rho, p$ must hold through $t = 3$.

---

**2. Strategic Methodology**

* **Exact solution feasible? NO.** (i) MoC requires invertible 1D characteristic relations; this problem is 2D, gravitationally forced, and its late-time dynamics are vortical (RT/KH roll-up) — characteristics neither decouple nor close, and no shock-root selection problem exists to exploit. (ii) Separation of variables / Green's functions fail: nonlinear convective fluxes, non-periodic $y$, inhomogeneous source. (iii) Linear RT stability theory covers only the early growth regime, not $t \le 3$. The Advisor concurs: no exact-method override is warranted.
* **Selection: Discontinuous Galerkin Spectral Element Method in 2D plus time** (retained). It is the only toolbox method delivering simultaneously: high-order accuracy for the smooth vortical roll-up, subcell FV blending for positivity at the steep contact, and AMR to track the moving interface. Run #7's failures were flux- and post-processing-level, not method-level.
* **AMR variable (Signal-to-Noise Analysis):** **Density $\rho$.** The interface, spikes, and bubbles are *density structures*: $\rho$ is sharp exactly at the features of interest and quiescent (constant 2.0 / 1.0) elsewhere — maximal signal-to-noise. Velocity becomes globally chaotic across the entire mixing layer (refining on it wastes resources on turbulence, not structure); pressure is dominated by the smooth hydrostatic profile $p(y)$ and does not localize the interface. Density-based indication is superior for this PDE and retained.

---

**3. Implementation Requirements (CRITICAL)**

* **Configuration:** Use template defaults **except** the Advisor-mandated overrides in §3.1. Do NOT alter mesh, AMR limits, CFL, or time stepper beyond what is listed.

* **3.0.1 Adaptive Mesh Refinement & Shock Capturing:**
  * *Indicator variable:* $\rho$ for both the Löhner AMR indicator and the Hennemann–Gassner shock-capturing indicator (justified in §2).
  * *AMR:* retain `IndicatorLohner`($\rho$) with `ControllerThreeLevel` (base=4, med=5, max=6). Do **not** raise the maximum level.
  * *Shock capturing:* retain `VolumeIntegralShockCapturingHG` with `flux_lax_friedrichs` FV subcells; **modestly reduce `alpha_max` (e.g., 1.0 → 0.5)** so the steep-but-non-shocking RT contact is not perpetually blended to FV. If ringing appears, restore blending locally (raise `alpha_max`) — do **not** revert to globally dissipative HLL.

* **3.1 Advisor-Mandated Adjustments (Overrides):**
  1. **Surface flux:** `flux_hllc` (replaces `flux_hll`) for contact resolution. Volume flux `flux_ranocha` retained.
  2. **Primary run:** `polydeg=5`, retaining AMR limits and adaptive stepping (`RDPK3SpFSAL49`, initial dt = 1e-5). A smaller stable step is expected — the adaptive controller handles it; **no CFL reduction**.
  3. **Companion verification run:** `polydeg=3` with otherwise identical settings (HLLC, same AMR) to isolate the flux change and enable p-convergence certification.
  4. **Data archival (mandatory):** save the full conservative state $(\rho, \rho u, \rho v, E)$, times, and mesh/AMR metadata every `dt_save = 0.1`. (Run #7 saved only the PNG.)
  5. **Log checks (mandatory prints):** successful arrival at $t = 3.0$; all fields finite (no NaN/Inf); $\min \rho > 0$; $\min p > 0$.
  6. **Visualization repair:** gapless, AMR-aware rendering — at each plot time, interpolate the DG/AMR solution onto a common **uniform visualization grid** (square cells, ≥ 4 points per finest AMR element edge, e.g., $512 \times 2048$), sampling each point from the leaf element containing it (hanging-node correct). Render one gapless heatmap per field; disable element-edge strokes/gaps. Do **not** smooth the physical data to hide stripes.

---

**4. Visualization Strategy**

* **Field Selection (2):** $\rho$ — primary driver (interface, spikes, bubbles, mixing layer); $v = \rho v / \rho$ — vertical velocity (plume dynamics). Omit $u, p, E$ from the figure.
* **Times:** $t = 0.5,\ 1.0,\ 2.0,\ 3.0$.
* **Layout:** 2 rows (fields) × 4 columns (times), rendered from the uniform gapless grid of §3.1(6). Figure generated from the **p=5 primary run**.
* **Julia:** `colorbar=false` (mandatory). No fixed colormaps — auto-scaled per field/time.
* **Save as:** `summary_all.png` (strict).
* **Data:** full-state saving at `dt_save = 0.1` proceeds independently of the figure.

---

**5. Cross-Validation & Data Integration (Conditional)**

* No external reference data paths in [USER_QUESTION] → no external $L_2$ comparison.
* **Internal cross-validation (Advisor-mandated):** interpolate the p=3 and p=5 runs onto a common uniform output grid at the saved times ($t = 1.0, 2.0, 3.0$) and **print**:
  * Spike/bubble tip positions ($y$-extremes of the $\rho = 1.5$ contour);
  * Mixing-layer width (tip-to-tip separation, or $y$-extent of $1.1 < \rho < 1.9$);
  * Kinetic energy $\int_\Omega \tfrac{1}{2}\rho(u^2+v^2)\,dV$ vs. time for both runs;
  * Relative $L_2$ differences of $\rho$ and $v$ between p=3 and p=5 at each saved time.
* The **p=5 run is the certified ground truth**; these metrics constitute the certification. A single visually plausible run is insufficient.

---

**REQUIRED COMPONENTS**

**Main Method:**
* Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
* Shock Capturing Indicator (density-based, reduced `alpha_max`)
* Adaptive Mesh Refinement (density-based Löhner, three-level controller)
* Contact-Resolving Surface Flux (HLLC)
* Polynomial-Degree Convergence Verification (p=3 vs. p=5)
* AMR-Aware Gapless Visualization
* Full-State Solution Archival (`dt_save=0.1`)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

**Status:** `REJECTED_FOR_REFINEMENT`

Trial #8 is a substantial improvement over Run #7: it is stable, visually clean, and physically credible. However, the p=3/p=5 comparison does **not** yet certify the p=5 result as high-fidelity ground truth, particularly for the late-time velocity and fine-scale roll-up.

The saturation rule does not apply: the only previous trial, Run #7, scored **85**, so there is no earlier score ≥90 from which to count five refinement trials.

## Section 2: Quantitative Scorecard

- **Score Integrity:** 50/50
- **Score Accuracy:** 18/20
- **Score Details:** 12/15
- **Score Optimality:** 15/15
- **TOTAL SCORE:** **95/100**

## Section 3: Forensic Analysis & Feedback

### Observation

**Execution and integrity**

- The log reaches \(t=3.00\), reports a final kinetic energy of approximately \(1.63945\times10^{-2}\), and ends with `Done.`
- The kinetic-energy history increases smoothly through the late-time interval, with no evidence of a numerical blow-up.
- All eight panels are populated and readable. The previous AMR-stitching/striping problem is no longer apparent.
- There are no visible NaN/Inf artifacts, blank panels, or catastrophic positivity failures. The numerical values of the minimum density and pressure checks are not included in the supplied tail, so they should remain explicitly visible in future final log summaries.

**Error-log check**

No `Minimum achieved error:` line appears in the supplied stdout tail.

**Quantitative error is not applicable as no ground truth is available.** This is a nonlinear forward RTI simulation without an analytical reference. The reported p=3/p=5 relative \(L_2\) values are **resolution-difference diagnostics**, not errors against an exact solution.

**Physical behavior**

The density sequence shows the expected progression:

1. **\(t=0.5\):** a perturbed, still-thin heavy/light interface.
2. **\(t=1.0\):** a coherent central heavy-fluid protrusion in \(+y\).
3. **\(t=2.0\):** a mushroom-shaped cap with rolled flanks.
4. **\(t=3.0\):** an elongated spike and complicated secondary Kelvin–Helmholtz-like filaments.

This is consistent with the prescribed upward body force and inverted configuration: heavy-fluid spikes penetrate in \(+y\), while light-fluid bubbles penetrate in \(-y\). The vertical-velocity panels consistently localize the strongest motion around the plume and mixing layer. At \(t=3\), the p=5 spike and bubble tips remain inside the domain, so there is no visible wall-penetration problem.

The large-scale morphology is approximately symmetric during the early coherent growth stage, and there is no obvious Gibbs ringing. Increasing kinetic energy is physically compatible with gravitational potential energy being converted into fluid motion; total energy conservation is not expected in the presence of the gravitational source.

**Resolution certification**

The p=3/p=5 differences grow substantially with time:

| Time | Relative \(\rho\) difference | Relative \(v\) difference |
|---|---:|---:|
| \(t=1\) | 0.0162 | 0.0757 |
| \(t=2\) | 0.0599 | 0.1379 |
| \(t=3\) | 0.1589 | 0.2585 |

The integral and large-scale measures are less sensitive:

- At \(t=3\), mixing-layer widths are 0.72372 and 0.74257, a difference of about **2.5%**.
- The corresponding kinetic energies differ by about **4.9%**.
- The spike tips are close, but the bubble tips differ by approximately **0.0226** in \(y\), which is significant relative to the plume displacement.

Thus, the large-scale growth is reasonably consistent, but the pointwise late-time fields—especially velocity and the rolled-up filaments—remain resolution-sensitive. Two polynomial degrees are also insufficient to establish a convergence trend. The p=5 run can serve as the better current reference, but it should not yet be described as certified or exact ground truth.

### Diagnosis

**Primary diagnosis:** resolution-sensitive nonlinear RT/KH mixing and remaining discretization uncertainty, **not** a stability failure.

The HLLC flux, higher polynomial degree, and repaired visualization have addressed the major Run #7 deficiencies. The remaining late-time discrepancies are plausibly caused by a combination of:

- different numerical dissipation between p=3 and p=5;
- sensitivity of thin vortical filaments to their phase and location;
- density-based AMR preferentially resolving the contact while potentially under-resolving some shear-dominated structures.

No exact-method override is warranted. The full problem is two-dimensional, nonlinear, gravitationally forced, wall-bounded, and develops vortical roll-up. Method of characteristics, separation of variables, and linear RT theory cannot provide the complete solution through \(t=3\).

### Targeted Improvements

- **For Integrity:** None. Preserve the successful completion, archival, and positivity checks. Keep the final finite/minimum-\(\rho\)/minimum-\(p\) summaries near the end of the log for auditability.

- **For Accuracy:**  
  Perform a **new p=7 primary run with p=5 as the companion**, changing only polynomial degree. Retain HLLC, Ranocha volume flux, `alpha_max=0.5`, the present AMR limits, and adaptive time stepping. This respects the recommended `polydeg ≤ 7` limit and is preferable to costly uniform refinement. Compare p=3/p=5/p=7 trends in tips, mixing width, kinetic energy, and common-grid field differences. Do not repeat the p=3/p=5-only certification.

- **For Details:**  
  If p=5/p=7 differences remain concentrated in the late-time rolled-up filaments, use **targeted rather than uniform refinement**: augment the density AMR signal with a shear/vorticity measure and, if necessary, permit one additional local AMR level only around the mixing layer. Do not increase shock-capturing dissipation or revert to HLL; there is no visible ringing justifying that trade-off, and it would soften the contact.

- **For Optimality:** None. DGSEM with HLLC, entropy-conservative volume flux, subcell shock capturing, AMR, and adaptive high-order time integration is an appropriate architecture. The remaining requirement is convergence evidence, not another method change.

**Recommendation:** Refine once at p=7 and reassess convergence. Until then, treat Trial #8 as an excellent numerical solution, but not yet as fully certified PINN ground truth.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 7 | Implemented as a 2D DGSEM (polydeg=3) for CompressibleEulerEquations2D with gamma=1.4 on domain x in [0,0.25], y in [0,1], t in [0,3]. Uses periodic x boundaries, slip-wall y boundaries, gravitational source [0,0,rho,rho*v] with g=1.0, and exact initial rho/p/u/v/E profiles including tanh interface and single-mode v perturbation. Spatial operator uses flux_hll at interfaces, flux_ranocha for DG volume, flux_lax_friedrichs for FV subcells via VolumeIntegralShockCapturingHG with Hennemann-Gassner indicator on density. AMR uses IndicatorLohner on density with ControllerThreeLevel base=4, med=5, max=6 and AMRCallback interval=20. Time integration uses RDPK3SpFSAL49 with StepsizeCallback and dt=1e-5 initial; tspan 0-3. Post-processing plots density and vertical velocity v=rho_v2/rho at t=0.5,1.0,2.0,3.0 in 2x4 layout, colorbar=false, saved as summary_all.png. | N/A | 45 | 17 | 10 | 13 | **85** |
| 9 | The code implements the 2D compressible Euler Rayleigh–Taylor problem with Trixi using DGSEM plus AMR. Physics match the strategy: CompressibleEulerEquations2D(1.4), domain [0,0.25]×[0,1], tspan [0,3], periodic x, slip walls at y, gravity source [0,0,ρg,ρvg], and initial conditions with tanh density transition, piecewise hydrostatic pressure, single-mode vertical-velocity perturbation, and prim2cons total energy. The primary run uses polydeg=5 and the companion verification run uses polydeg=3 with otherwise identical settings. The surface flux is flux_hllc; the volume integral is VolumeIntegralShockCapturingHG with density-based IndicatorHennemannGassner, alpha_max=0.5, volume_flux_dg=flux_ranocha, and volume_flux_fv=flux_lax_friedrichs. AMR uses density-based IndicatorLöhner and ControllerThreeLevel base=4, med=5, max=6, interval=20. Time integration is RDPK3SpFSAL49 with dt=1e-5 and StepsizeCallback. The code archives full conservative state and AMR metadata every 0.1, performs mandatory t=3/finite/positivity log checks, builds a 512×2048 uniform AMR-aware gapless interpolation for density and vertical velocity at t=0.5,1,2,3, saves summary_all.png from the p=5 run, and prints p3/p5 cross-validation metrics including interface tips, mixing width, kinetic energy history, and relative L2 differences of rho and v. | N/A | 50 | 18 | 12 | 15 | **95** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 7 | NPZ files are written to data_npz/time_%08.5f.npz at dt_save=0.1 (including t=0 and final t=3.0). Each NPZ contains keys: 't' (scalar Float64); 'x','y','rho','rho_v1','rho_v2','rho_e' (arrays from PlotData2D/cons2cons). The arrays have shape (16 x n_elements), where 16=(polydeg+1)^2 and n_elements is the current AMR-adapted number of elements; PlotData2D returns x,y and each conservative variable in this matrix layout, and plotting flattens them with vec(). Additionally, SaveSolutionCallback writes Trixi HDF5 solution files in trixi_output at dt=0.1 with solution_variables=cons2cons, i.e., full conservative state. |
| 9 | NPZ archives are written every dt_save=0.1 from t=0 to t=3.0 (plus a final t=3.0 callback) to data_npz/p5/time_%08.5f.npz for the primary run and data_npz/p3/time_%08.5f.npz for the companion run. Each NPZ file contains: 't' (Float64 scalar); 'polydeg' (Int scalar); 'x','y','rho','rho_v1','rho_v2','rho_e' (collected Trixi PlotData2D arrays of shape n_nodes_1d^2 × n_elements, with n_nodes_1d=polydeg+1, i.e. 6 for p5 and 4 for p3); 'u_cons' (4 × n_nodes_1d × n_nodes_1d × n_elements full conservative DG state); 'leaf_node_coordinates' (2 × n_nodes_1d × n_nodes_1d × n_elements); 'leaf_element_bounds' (4 × n_elements, rows xmin,xmax,ymin,ymax); 'element_levels' (n_elements Int vector); 'element_ids' (n_elements Int vector); 'n_elements' (Int scalar); 'n_nodes_1d' (Int scalar); 'domain_min' and 'domain_max' (length-2 Float64 vectors); 'periodicity' (length-2 Int [1,0]); 'amr_controller_levels' (length-3 Int [4,5,6]); 'amr_interval' (Int 20); 'amr_indicator_component' (Int 1); 'shock_indicator_component' (Int 1). Additionally, Trixi SaveSolutionCallback writes HDF5 snapshots to trixi_output/p5 and trixi_output/p3 at dt_save=0.1. The figure summary_all.png is saved at the script directory as a 2×4 gapless heatmap layout with density in the first row and vertical velocity in the second row. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 7 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run4/results//CompressibleEuler_RayleighTaylor_PeriodicX_ReflectiveY_SmoothedDensityInterface/2026_11_06_1114/` |
| 9 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run4/results//CompressibleEuler_RayleighTaylor_PeriodicX_ReflectiveY_SmoothedDensityInterface/2026_11_06_1145/` |


---
