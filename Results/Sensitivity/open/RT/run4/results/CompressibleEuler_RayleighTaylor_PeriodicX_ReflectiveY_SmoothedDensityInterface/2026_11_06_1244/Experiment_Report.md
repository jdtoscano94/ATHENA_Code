
# Experiment Report
*Last Updated: 2026-11-06 13:02:57*

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
## 2D Compressible Euler Rayleigh–Taylor Instability (Refinement Trial #10: p=7 Certification Run)

---

**1. Problem Statement**

We solve the inviscid 2D compressible Euler equations with a gravitational source for the single-mode Rayleigh–Taylor instability in the inverted configuration (heavy fluid below light, body force in $+y$). Per the Advisor's directive, this trial is a **pure polynomial-degree refinement**: a **p=7 primary run** with a **p=5 companion**, all other numerics frozen at the Run #9 configuration, to certify convergence of the late-time rolled-up filaments.

---

**1. Mathematical Specification (Code-Ready Extraction)**

* **Governing Equations:** 2D Compressible Euler with gravity source ($\gamma = 1.4$):
$$\partial_t \mathbf{U} + \partial_x \mathbf{F}(\mathbf{U}) + \partial_y \mathbf{G}(\mathbf{U}) = \mathbf{S}(\mathbf{U})$$
$$\mathbf{U} = \begin{pmatrix}\rho\\ \rho u\\ \rho v\\ \rho E\end{pmatrix},\quad \mathbf{F} = \begin{pmatrix}\rho u\\ \rho u^2 + p\\ \rho u v\\ u(\rho E + p)\end{pmatrix},\quad \mathbf{G} = \begin{pmatrix}\rho v\\ \rho u v\\ \rho v^2 + p\\ v(\rho E + p)\end{pmatrix},\quad \mathbf{S} = \begin{pmatrix}0\\ 0\\ \rho g\\ \rho g\, v\end{pmatrix}$$
with $g = 1.0$ (positive $y$), equation of state $p = (\gamma-1)\rho e$, and total energy $E = e + \tfrac{1}{2}(u^2 + v^2)$, i.e. $\rho E = \frac{p}{\gamma-1} + \frac{1}{2}\rho(u^2+v^2)$.

* **Domain:** $(x,y) \in [0, 0.25] \times [0, 1.0]$, $t \in [0, 3.0]$.

* **Boundary Conditions:**
  * $x$-direction: Periodic.
  * $y = 0$ and $y = 1$: Reflective / free-slip walls (no-penetration: $v = 0$ at the walls).

* **Initial Conditions** (interface at $y = 0.5$; every field exact):
$$\rho(x,y) = 2.0 + (1.0 - 2.0)\cdot\frac{1}{2}\left[1 + \tanh\!\left(50\,(y - 0.5)\right)\right]$$
$$p(x,y) = \begin{cases} 2y + 1, & y < 0.5\\[2pt] y + 1.5, & y \geq 0.5\end{cases}$$
$$u(x,y) = 0$$
$$v(x,y) = -0.025\; c(x,y)\,\cos(8\pi x)\,\sin^6(\pi y), \qquad c(x,y) = \sqrt{\frac{\gamma\, p(x,y)}{\rho(x,y)}}$$
$$\rho E(x,y) = \frac{p(x,y)}{\gamma - 1} + \frac{1}{2}\,\rho(x,y)\left(u^2 + v^2\right)$$

* **Constants & Parameters:** $\gamma = 1.4$; gravity vector $\mathbf{g} = [0, 1.0]^T$ (source adds momentum in $+y$); $\mu = 0$, $k = 0$; $\rho_{heavy} = 2.0$, $\rho_{light} = 1.0$; interface $y = 0.5$; tanh smoothing parameter $50$; perturbation amplitude $0.025$; perturbation wavenumber $8\pi$ (single mode across the domain).

* **Expected Physics:** Linear RT growth, then nonlinear mushroom-cap formation by $t \approx 2$; heavy-fluid spikes penetrate $+y$, light-fluid bubbles penetrate $-y$; secondary Kelvin–Helmholtz roll-up filaments on the spike flanks by $t = 3$; mixing-layer width $\approx 0.72$–$0.74$ at $t=3$; monotonically increasing kinetic energy (potential → kinetic conversion; total energy not conserved due to the gravitational source).

---

**2. Strategic Methodology**

* **Can we obtain an exact solution?** **No.** The system is 2D, nonlinear, gravitationally forced, wall-bounded, and develops multi-scale vortical mixing (RT spikes/bubbles + KH roll-up). The Method of Characteristics applies only to 1D simple-wave/Riemann structures; here multi-dimensional wave interactions preclude any implicit characteristic representation of the full field through $t=3$. Separation of variables and linear RT growth-rate theory cover only the early-time regime and cannot deliver the fields. The Advisor explicitly confirms: **no exact-method override is warranted.**
* **Selection:** The one best method is the **Discontinuous Galerkin Spectral Element Method in 2D plus time**, with subcell shock capturing and AMR, certified by **polynomial-degree convergence (p=7 primary vs. p=5 companion, with archived p=3 trend)**. Since MoC/exact solutions are infeasible, no entropy-condition root selection is required.

---

**3. Implementation Requirements (CRITICAL)**

* **Configuration:** Use "template defaults" **EXCEPT** for the Advisor-mandated adjustments in §3.1 below.

* **3.0.1 Adaptive Mesh Refinement & Shock Capturing — Variable Selection Analysis:**
  * **AMR variable: $\rho$ (density).** The material interface/contact is the defining structural feature of RTI; density remains sharp and banded *only* at the interface and mixing layer for all $t \in [0,3]$, making it the highest signal-to-noise refinement key. It is superior to velocity, which becomes globally chaotic in the turbulent mixing layer (refining on it wastes elements domain-wide), and to pressure, which is a smooth hydrostatic background weakly informative of the contact.
  * **Shock-capturing indicator: $\rho$** (Hennemann–Gassner), since the dominant discontinuity is the material interface.
  * **Advisor-mandated conditional augmentation:** if p=5/p=7 differences remain concentrated in the late-time rolled-up filaments, augment the density AMR signal with a **shear/vorticity measure restricted to the mixing layer** (density alone may under-resolve shear-dominated KH structures). Do **not** refine globally on velocity.

* **3.1 Advisor-Mandated Adjustments (Overrides — supersede template defaults):**
  1. **Primary run: `polydeg = 7`. Companion run: `polydeg = 5`.** Change **only** the polynomial degree relative to Run #9 (p=5/p=3). Respect the `polydeg ≤ 7` limit. Do **not** switch to uniform refinement.
  2. **Retain** surface flux `flux_hllc`; volume integral `VolumeIntegralShockCapturingHG` with `IndicatorHennemannGassner` on density, `alpha_max = 0.5`, `volume_flux_dg = flux_ranocha`, `volume_flux_fv = flux_lax_friedrichs`.
  3. **Retain** AMR: `IndicatorLöhner` on density, `ControllerThreeLevel` (base=4, med=5, max=6), AMR interval = 20.
  4. **Retain** time integration: `RDPK3SpFSAL49` with `StepsizeCallback` (adaptive), initial dt = 1e-5, $t \in [0,3]$.
  5. **Retain** archival of the full conservative state + AMR metadata every `dt_save = 0.1`, and the final integrity log (t=3.0 reached, all-finite check, minimum $\rho$, minimum $p$) printed near the end of stdout for auditability.
  6. **Conditional targeted refinement:** if p=5/p=7 differences persist in the late-time filaments, add the shear/vorticity AMR augmentation and permit **one additional local AMR level (max=7) only around the mixing layer**. Do **not** increase shock-capturing dissipation and do **not** revert to HLL — there is no ringing justifying that trade-off.
  7. **Convergence certification:** report **p=3/p=5/p=7 trends** in spike/bubble tip positions, mixing-layer width, kinetic-energy history, and common-grid (512×2048 AMR-aware, gapless) relative $L_2$ differences of $\rho$ and $v$ at $t = 1, 2, 3$. Reuse the archived Run #9 p=3 (and p=5) snapshots if present; do **not** repeat the p=3/p=5-only certification.

---

**4. Visualization Strategy**

* **Field Selection (Analysis):** Plot **$\rho$ (density)** and **$v$ (vertical velocity)** only. Density is the primary driver and interface marker of RTI (spike/bubble/filament morphology); vertical velocity localizes the plume and mixing-layer dynamics. $u$ and $p$ are secondary and would shrink the panels without adding diagnostic value.
* **Task:** Plot these two fields at **4 distinct times: $t = 0.5,\ 1.0,\ 2.0,\ 3.0$**, taken from the **p=7 primary run**.
* **Layout:** Grid — 2 rows (fields: $\rho$, $v$) × 4 columns (times).
* **Saving Data:** Archive the **full solution state** (all conservative variables + AMR metadata) every `dt_save = 0.1` for Ground Truth. Plotting selection affects only `summary_all.png`.
* **Constraints:**
  * Save as: `summary_all.png` (strictly).
  * **CRITICAL: DISABLE COLORBARS (`colorbar = false`)** — the code is in Julia.
  * Do not use fixed colormaps; let scales adjust automatically per panel.
  * Plot only primary variables ($\rho$, $v = \rho v / \rho$); do not compute derived quantities (e.g., vorticity) in the figure.

---

**5. Cross-Validation & Data Integration (Conditional)**

* **Trigger check:** The [USER_QUESTION] contains **no external data paths** (no sensors, no reference `.npz`). No external loading is mandated.
* **Self-convergence metrics (mandatory, printed to stdout):** relative $L_2$ differences of $\rho$ and $v$ between p=7 and p=5 (and archived p=3) on the common 512×2048 grid at $t = 1, 2, 3$; spike/bubble tip $y$-positions; mixing-layer widths; kinetic-energy histories. These are resolution-difference diagnostics, not errors against an exact solution, and must be labeled as such.

---

**REQUIRED COMPONENTS**

**Main Method:**
* Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
* Shock Capturing (subcell, density-keyed)
* Adaptive Mesh Refinement (density-keyed, three-level controller)
* Polynomial-Degree Convergence Study (p=7 primary / p=5 companion)
* Conditional Shear-Aware Targeted AMR (mixing layer only)
* Final Positivity & Finiteness Audit (min-$\rho$, min-$p$, finite-state log)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

**Status:** `REJECTED_FOR_REFINEMENT`

The simulation is stable and physically convincing, but the late-time solution is not yet certified as converged ground truth. Run 9 previously scored **95**, and the present Run 10 is only the **first subsequent refinement visible in the history**. Therefore, the five-post-success refinement saturation condition has not been reached.

## Section 2: Quantitative Scorecard

| Category | Score |
|---|---:|
| **Score Integrity** | **50 / 50** |
| **Score Accuracy** | **18 / 20** |
| **Score Details** | **12 / 15** |
| **Score Optimality** | **15 / 15** |
| **TOTAL SCORE** | **95 / 100** |

**Ground-truth error check:** No `Minimum achieved error:` line appears in the supplied log. **Quantitative error is not applicable as no ground truth is available.** The printed relative \(L_2\) values are differences between numerical resolutions, not errors against an exact solution.

## Section 3: Forensic Analysis & Feedback

### Observation

- **Integrity is excellent.** All three polynomial-degree runs reach \(t=3.0\), all archived fields are finite, and density and pressure remain strictly positive. For the primary \(p=7\) run, the minima are approximately \(\rho_{\min}=0.7706\) and \(p_{\min}=1.2136\). There is no evidence of NaN/Inf failure, positivity failure, or a blank/corrupted plot.
- **The qualitative physics is correct:**
  - At \(t=0.5\), the initially smoothed interface develops the expected single-mode deformation.
  - At \(t=1.0\), the central heavy-fluid spike rises while light fluid penetrates downward on either side.
  - At \(t=2.0\), a clear mushroom cap and symmetric roll-up structures have formed.
  - At \(t=3.0\), the spike approaches the upper wall and the density field contains elongated, thin Kelvin–Helmholtz-like filaments. The vertical-velocity panels localize the same plume and mixing-layer dynamics.
- The fields remain approximately symmetric about the domain centerline. There is no obvious periodic-seam artifact, wall penetration, or visible large-scale Gibbs ringing.
- The macroscopic \(p=7\) diagnostics are plausible: at \(t=3\), the bubble and spike tips are approximately \(y=0.1846\) and \(y=0.9286\), respectively, with mixing width \(0.7440\). Kinetic energy increases between the reported \(t=2\) and \(t=3\) states, consistent with gravitational potential-to-kinetic energy conversion.
- **The remaining concern is late-time convergence.** The \(p=5/p=7\) common-grid resolution differences are modest at \(t=2\)—about \(2.4\%\) for density and \(5.2\%\) for vertical velocity—but grow to approximately \(18.5\%\) and \(25.6\%\), respectively, at \(t=3\). Meanwhile, tip positions, mixing width, and kinetic energy are much closer. This combination indicates that the large-scale plume is converging substantially better than the small-scale filament locations and topology.

### Diagnosis

**Late-time shear/filament under-resolution and chaotic mixing sensitivity**, rather than a stability failure.

At late times, small differences in the rolled-up filaments can produce large pointwise \(L_2\) resolution differences even when the macroscopic plume diagnostics agree. Polynomial refinement alone cannot certify these structures: \(p=7\) is already at the recommended limit, and density-keyed refinement can under-prioritize shear-dominated regions once the interface becomes highly convoluted.

There is no visual justification for increasing shock-capturing dissipation or replacing HLLC with a more dissipative flux. Doing so would risk further smearing the contact and rolled-up filaments.

No exact-method override is warranted: the full nonlinear, multidimensional, gravitationally forced Euler solution with wall interactions and vortical mixing does not admit a usable closed-form or characteristic-based exact field solution through \(t=3\).

### Targeted Improvements

- **For Integrity:** **None.** Retain the adaptive time integration, current CFL control, HLLC surface flux, and positivity-preserving subcell stabilization.
- **For Accuracy:** Preserve the exact initial conditions, source term, and boundary conditions, but do not certify the archived fields as ground truth until the late-time filament sensitivity is reduced. Continue using tip positions, mixing width, kinetic energy, and common-grid field differences as **resolution diagnostics**, not exact-solution errors.
- **For Details:** Do **not** simply repeat the same \(p=7/p=5\) configuration.
  1. Keep `polydeg = 7`; do not increase it further.
  2. Strengthen the already-specified shear-aware AMR rather than merely leaving it conditional: activate it before or during mushroom roll-up, approximately \(t=1.5\)–\(2.0\), using vorticity or a velocity-gradient measure **restricted to the mixing layer**.
  3. Reduce the AMR interval from **20 to 10** and tighten the relevant refinement thresholds so thin filaments are refined before they roll up. Permit `max_level = 7` only in that local region.
  4. Use the archived \(p=7\) result as a baseline and perform a **\(p=7\) local-\(h\)-refinement comparison**. This isolates mesh sensitivity now that polynomial degree is capped. A further \(p=5\) companion is secondary and can be omitted if cost is limiting.
  5. Retain `flux_hllc`, `flux_ranocha`, and `alpha_max = 0.5`; there is no observed ringing warranting additional dissipation. Avoid global refinement or global velocity-based AMR.
- **For Optimality:** **None.** DGSEM with contact-resolving HLLC flux, entropy-stable DG volume flux, subcell shock capturing, and targeted AMR remains an appropriate high-fidelity method for this problem.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 7 | Implemented as a 2D DGSEM (polydeg=3) for CompressibleEulerEquations2D with gamma=1.4 on domain x in [0,0.25], y in [0,1], t in [0,3]. Uses periodic x boundaries, slip-wall y boundaries, gravitational source [0,0,rho,rho*v] with g=1.0, and exact initial rho/p/u/v/E profiles including tanh interface and single-mode v perturbation. Spatial operator uses flux_hll at interfaces, flux_ranocha for DG volume, flux_lax_friedrichs for FV subcells via VolumeIntegralShockCapturingHG with Hennemann-Gassner indicator on density. AMR uses IndicatorLohner on density with ControllerThreeLevel base=4, med=5, max=6 and AMRCallback interval=20. Time integration uses RDPK3SpFSAL49 with StepsizeCallback and dt=1e-5 initial; tspan 0-3. Post-processing plots density and vertical velocity v=rho_v2/rho at t=0.5,1.0,2.0,3.0 in 2x4 layout, colorbar=false, saved as summary_all.png. | N/A | 45 | 17 | 10 | 13 | **85** |
| 9 | The code implements the 2D compressible Euler Rayleigh–Taylor problem with Trixi using DGSEM plus AMR. Physics match the strategy: CompressibleEulerEquations2D(1.4), domain [0,0.25]×[0,1], tspan [0,3], periodic x, slip walls at y, gravity source [0,0,ρg,ρvg], and initial conditions with tanh density transition, piecewise hydrostatic pressure, single-mode vertical-velocity perturbation, and prim2cons total energy. The primary run uses polydeg=5 and the companion verification run uses polydeg=3 with otherwise identical settings. The surface flux is flux_hllc; the volume integral is VolumeIntegralShockCapturingHG with density-based IndicatorHennemannGassner, alpha_max=0.5, volume_flux_dg=flux_ranocha, and volume_flux_fv=flux_lax_friedrichs. AMR uses density-based IndicatorLöhner and ControllerThreeLevel base=4, med=5, max=6, interval=20. Time integration is RDPK3SpFSAL49 with dt=1e-5 and StepsizeCallback. The code archives full conservative state and AMR metadata every 0.1, performs mandatory t=3/finite/positivity log checks, builds a 512×2048 uniform AMR-aware gapless interpolation for density and vertical velocity at t=0.5,1,2,3, saves summary_all.png from the p=5 run, and prints p3/p5 cross-validation metrics including interface tips, mixing width, kinetic energy history, and relative L2 differences of rho and v. | N/A | 50 | 18 | 12 | 15 | **95** |
| 16 | The code implements the specified 2D compressible Euler Rayleigh–Taylor instability using Trixi's DGSEM with p=7 primary and p=5 companion runs, plus an archived p=3 trend. It uses gamma=1.4, domain [0,0.25] x [0,1.0], t in [0,3.0], HLLC surface flux, Hennemann–Gassner shock capturing on density with alpha_max=0.5, Ranocha DG volume flux, Lax–Friedrichs FV volume flux, density-keyed Löhner AMR with ControllerThreeLevel base=4/med=5/max=6 and interval=20, and conditional shear-aware vorticity-masked AMR with max_level=7 for targeted mixing-layer refinement. Time integration uses RDPK3SpFSAL49 with StepsizeCallback cfl=0.5 and initial dt=1e-5. The initial density, piecewise pressure, zero horizontal velocity, vertical perturbation, sound speed, and conservative energy are implemented exactly as specified. The code saves NPZ snapshots every dt_save=0.1, saves summary_all.png for rho and v at t=0.5,1.0,2.0,3.0, and prints the required convergence, spike/bubble, mixing-width, kinetic-energy, and final integrity/positivity diagnostics. | N/A | 50 | 18 | 12 | 15 | **95** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 7 | NPZ files are written to data_npz/time_%08.5f.npz at dt_save=0.1 (including t=0 and final t=3.0). Each NPZ contains keys: 't' (scalar Float64); 'x','y','rho','rho_v1','rho_v2','rho_e' (arrays from PlotData2D/cons2cons). The arrays have shape (16 x n_elements), where 16=(polydeg+1)^2 and n_elements is the current AMR-adapted number of elements; PlotData2D returns x,y and each conservative variable in this matrix layout, and plotting flattens them with vec(). Additionally, SaveSolutionCallback writes Trixi HDF5 solution files in trixi_output at dt=0.1 with solution_variables=cons2cons, i.e., full conservative state. |
| 9 | NPZ archives are written every dt_save=0.1 from t=0 to t=3.0 (plus a final t=3.0 callback) to data_npz/p5/time_%08.5f.npz for the primary run and data_npz/p3/time_%08.5f.npz for the companion run. Each NPZ file contains: 't' (Float64 scalar); 'polydeg' (Int scalar); 'x','y','rho','rho_v1','rho_v2','rho_e' (collected Trixi PlotData2D arrays of shape n_nodes_1d^2 × n_elements, with n_nodes_1d=polydeg+1, i.e. 6 for p5 and 4 for p3); 'u_cons' (4 × n_nodes_1d × n_nodes_1d × n_elements full conservative DG state); 'leaf_node_coordinates' (2 × n_nodes_1d × n_nodes_1d × n_elements); 'leaf_element_bounds' (4 × n_elements, rows xmin,xmax,ymin,ymax); 'element_levels' (n_elements Int vector); 'element_ids' (n_elements Int vector); 'n_elements' (Int scalar); 'n_nodes_1d' (Int scalar); 'domain_min' and 'domain_max' (length-2 Float64 vectors); 'periodicity' (length-2 Int [1,0]); 'amr_controller_levels' (length-3 Int [4,5,6]); 'amr_interval' (Int 20); 'amr_indicator_component' (Int 1); 'shock_indicator_component' (Int 1). Additionally, Trixi SaveSolutionCallback writes HDF5 snapshots to trixi_output/p5 and trixi_output/p3 at dt_save=0.1. The figure summary_all.png is saved at the script directory as a 2×4 gapless heatmap layout with density in the first row and vertical velocity in the second row. |
| 16 | NPZ snapshots are written to data_npz/<run_id>/time_%08.5f.npz every dt_save=0.1, including t=0.0 and final t=3.0. Each NPZ contains: 't' (scalar Float64), 'polydeg' (Int), 'x' and 'y' (1-D PlotData2D coordinate vectors), 'rho', 'rho_v1', 'rho_v2', 'rho_e' (reconstructed PlotData2D fields; 1-D nodal arrays), 'u_cons' (4 x n_nodes_1d x n_nodes_1d x n_elements, where n_nodes_1d = polydeg+1), 'leaf_node_coordinates' (2 x n_nodes_1d x n_nodes_1d x n_elements), 'leaf_element_bounds' (4 x n_elements), 'element_levels' (n_elements Int vector), 'element_ids' (n_elements Int vector, 1:n_elements), 'n_elements' (scalar Int), 'n_nodes_1d' (scalar Int), 'domain_min' (2 Float64), 'domain_max' (2 Float64), 'periodicity' (2 Int, [1,0]), 'amr_controller_levels' (3 Int, [4,5,6] baseline or [4,5,7] targeted), 'amr_interval' (scalar Int, 20), 'amr_max_level' (scalar Int, 6 baseline or 7 targeted), 'amr_shear_aware' (scalar Int, 0/1), 'amr_targeted_to_mixing_layer' (scalar Int, 0/1), 'amr_indicator_component' (scalar Int, 1), and 'shock_indicator_component' (scalar Int, 1). Trixi SaveSolutionCallback also archives restart/mesh output under trixi_output/<run_id>. The figure summary_all.png is saved with a 2x4 layout of heatmaps for rho and v at t=0.5,1.0,2.0,3.0 from the p=7 primary run, with colorbars disabled. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 7 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run4/results//CompressibleEuler_RayleighTaylor_PeriodicX_ReflectiveY_SmoothedDensityInterface/2026_11_06_1114/` |
| 9 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run4/results//CompressibleEuler_RayleighTaylor_PeriodicX_ReflectiveY_SmoothedDensityInterface/2026_11_06_1145/` |
| 16 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run4/results//CompressibleEuler_RayleighTaylor_PeriodicX_ReflectiveY_SmoothedDensityInterface/2026_11_06_1244/` |


---
