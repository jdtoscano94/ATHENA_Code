
# Experiment Report
*Last Updated: 2026-11-06 21:51:11*

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
2D inviscid compressible Euler Rayleigh–Taylor Instability (RTI). A heavy fluid ($\rho=2$) lies below $y=0.5$ and a light fluid ($\rho=1$) above; with gravity acting in $+y$, the heavy fluid is gravitationally "overlying" the light fluid — an unstable configuration. A single-mode velocity perturbation seeds the instability, which must be evolved to $t=3.0$ through the linear, nonlinear bubble/spike, and mixing phases.

---

**1. Mathematical Specification (Code-Ready Extraction)**

* **Governing Equations:** 2D Compressible Euler with gravity source:
$$
\partial_t \mathbf{U} + \partial_x \mathbf{F}(\mathbf{U}) + \partial_y \mathbf{G}(\mathbf{U}) = \mathbf{S}(\mathbf{U}), \qquad \mathbf{U} = \left(\rho,\ \rho u,\ \rho v,\ E\right)^T
$$
$$
\mathbf{F} = \begin{pmatrix}\rho u\\ \rho u^2 + p\\ \rho u v\\ u(E+p)\end{pmatrix}, \qquad
\mathbf{G} = \begin{pmatrix}\rho v\\ \rho u v\\ \rho v^2 + p\\ v(E+p)\end{pmatrix}, \qquad
\mathbf{S} = \begin{pmatrix}0\\ 0\\ \rho g\\ \rho g v\end{pmatrix}
$$
Equation of state: $p = (\gamma-1)\left(E - \tfrac{1}{2}\rho(u^2+v^2)\right)$.

* **Domain:** $(x,y) \in [0, 0.25] \times [0, 1.0]$, $\quad t \in [0, 3.0]$.

* **Boundary Conditions:** $x$: Periodic. $y=0$ and $y=1$: Reflective / free-slip walls (no-penetration $v=0$; mirror reflection of $\rho, u, p$).

* **Initial Conditions:**
$$
\rho(x,y,0) = 2.0 + (1.0 - 2.0)\cdot\frac{1}{2}\left[1 + \tanh\left(50\,(y - 0.5)\right)\right]
$$
$$
p(x,y,0) = \begin{cases} 2y + 1, & y < 0.5 \\ y + 1.5, & y \geq 0.5 \end{cases}
$$
$$
u(x,y,0) = 0
$$
$$
v(x,y,0) = -0.025\, c(x,y)\, \cos(8\pi x)\, \left[\sin(\pi y)\right]^6, \qquad c(x,y) = \sqrt{\frac{\gamma\, p(x,y,0)}{\rho(x,y,0)}}
$$
$$
E(x,y,0) = \frac{p(x,y,0)}{\gamma - 1} + \frac{1}{2}\,\rho(x,y,0)\, v(x,y,0)^2
$$

* **Constants & Parameters:** $\gamma = 1.4$; gravity magnitude $g = 1.0$ with $\mathbf{g} = [0, +1.0]^T$; $\rho_{heavy}=2.0$, $\rho_{light}=1.0$; interface at $y_0 = 0.5$; interface smoothing factor $50$; perturbation amplitude $0.025$; perturbation wavenumber $k = 8\pi$ (exactly one wavelength across the periodic $x$-extent).

* **Expected Physics:** Exponential linear growth of the single mode ($\sigma \approx \sqrt{Akg} \approx 2.9$ with $A = 1/3$) until $t \approx 0.7$–$1.0$; transition to nonlinear regime with bubbles of light fluid rising toward $y=0$ and spikes of heavy fluid sinking toward $y=1$; secondary Kelvin–Helmholtz roll-up along the shear layers; mixing zone spanning roughly $y \in [0.2, 0.8]$ by $t = 3.0$; weak shocks/acoustic transients from the small pressure–density imbalance inside the smoothed interface layer and from late-time jet impingement on the $y$-walls.

---

**2. Strategic Methodology**

* **Can we obtain an exact solution?** **No.** The Method of Characteristics yields closed-form solutions only for 1D simple-wave/Riemann-type problems. RTI is a genuinely multi-D instability: perturbed hydrostatic equilibrium with a contact discontinuity, exponential growth, secondary KH instabilities, and shock formation — no characteristic invariant can be integrated in closed form, and the dynamics are hydrodynamically unstable (no analytical ground truth exists).
* **Fourier Spectral:** Rejected — the $y$-direction has reflective walls (non-periodic), and the solution develops non-smooth features (contacts, shear layers, weak shocks) that destroy global spectral convergence.
* **Steady Continuous Galerkin:** Rejected — the problem is unsteady by design.
* **Selection:** **DGSEM in 2D plus time** — natively supports periodic $x$ / reflective $y$ boundaries, provides high-order accuracy for the smooth early phase, and offers local conservation with shock capturing and AMR for the late mixing phase.

---

**3. Implementation Requirements (CRITICAL)**

* **Configuration:** Use template defaults.
* **3.0.1 Adaptive Mesh Refinement & Shock Capturing — Variable Selection Analysis:**
    * *Structural features:* The material interface, bubble/spike fronts, and mixing zone are all **density** discontinuities. $\rho$ remains bimodal (values $\approx 1$ vs. $\approx 2$) and sharp exactly at the features of interest → high signal-to-noise.
    * *Noisy variables:* The velocity field becomes globally chaotic after the secondary KH instabilities (turbulent billows fill the whole mixing zone) → refining on $u$ or $v$ wastes resolution everywhere. Pressure is dominated by the smooth hydrostatic gradient plus weak acoustics → poor feature indicator.
    * **Decision: AMR and shock-capturing indicator on density $\rho$.** It is superior to velocity (globally turbulent) and pressure (smooth background) for this PDE.
* **3.1 Advisor-Mandated Adjustments:** None.
* **Standalone Trials:** Single fresh run in a new folder; no dependence on prior outputs.

---

**4. Visualization Strategy**

* **Field Selection:** Plot **density $\rho$** (primary — the density contrast drives the instability and traces the mixing zone) and **vertical velocity $v$** (secondary — reveals the bubble/spike jets and shear structures). Do not plot $u$, $p$, or $E$.
* **Task:** Plot these two fields at **4 time steps: $t = 0.75,\ 1.5,\ 2.25,\ 3.0$**.
* **Layout:** Grid — 2 rows ($\rho$, $v$) × 4 columns (time).
* **Saving Data:** Save the **full solution state** (all variables) at `dt_save = 0.1` for Ground Truth.
* **Constraints:**
    * Save as: `summary_all.png` (strictly).
    * **DISABLE COLORBARS** (`colorbar=false`) if the code is in Julia.
    * Do not use fixed colormaps; let scales adjust automatically.
    * Plot only primary variables; do not compute derived quantities (e.g., vorticity, schlieren).

---

**5. Cross-Validation & Data Integration (Conditional)**
* **Trigger check:** No external data paths (sensors / reference solutions) are provided in [USER_QUESTION] → **not triggered**. Metrics: None.

---

**REQUIRED COMPONENTS**

**Main Method:**
* Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
* Shock Capturing (density-based indicator)
* Adaptive Mesh Refinement (density-driven)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

* **Status:** **REJECTED_FOR_REFINEMENT**

The simulation completed and exhibits qualitatively plausible Rayleigh–Taylor dynamics, but the visualization is substantially degraded and the density interfaces cannot yet be certified as ground-truth quality. There is no trial history or saturation basis for approval.

## Section 2: Quantitative Scorecard

* **Score Integrity:** 40/50
* **Score Accuracy:** 15/20
* **Score Details:** 8/15
* **Score Optimality:** 12/15
* **TOTAL SCORE:** **75/100**

**Error-log check:** The exact string `Minimum achieved error:` is **not present** in the supplied STDOUT tail, so no error value can be extracted. **Quantitative error is not applicable as no ground truth is available.** This is a nonlinear forward RTI simulation; an error against zero or another placeholder would not be meaningful.

## Section 3: Forensic Analysis & Feedback

### Observation

* The log ends with successful generation of the requested time files, `Saving summary_all.png...`, and `Done.` There is no crash indication in the supplied tail.
* The density sequence shows a physically plausible progression:
  * At \(t=0.75\), the interface has developed a smooth single-mode deformation.
  * At \(t=1.5\) and \(t=2.25\), a central heavy-fluid spike propagates in \(+y\), accompanied by light-fluid structures near the periodic sides.
  * At \(t=3.0\), the structures are strongly nonlinear, with elongated jets and mushroom/roll-up-like features approaching the upper wall.
* The vertical-velocity jets approximately colocate with the density structures, and the evolution retains approximate left–right symmetry. These are positive qualitative checks.
* However, **strong, regular horizontal striping appears in every panel**, including otherwise quiescent regions. Because it occurs in both variables and follows regular mesh-like rows, it is more likely a plotting artifact—such as cell-edge rendering, row gaps, or incorrect AMR visualization—than physical Gibbs oscillation. Nevertheless, it severely obscures the solution.
* The visible density-interface edges appear soft, but their true thickness cannot be measured reliably through the striping. The combination of `flux_hll` and Lax–Friedrichs shock-capturing blending is potentially quite dissipative for the contact-dominated RTI structures.
* With the supplied auto-scaled plots, positivity of density/pressure, exact periodic matching, and \(v=0\) at the walls cannot be verified quantitatively.

### Diagnosis

The primary diagnosis is **insufficiently verified ground truth due to defective visualization and possible excessive contact dissipation**, not a fatal solver instability.

DGSEM with density-driven AMR is an appropriate architecture. The full nonlinear, multi-dimensional RTI—with gravity, wall interactions, contact roll-up, and possible weak shocks—does not admit a closed-form analytical solution through \(t=3\). Method of Characteristics or a Fourier solution cannot supply the requested nonlinear ground truth, so no exact-method override is warranted.

### Targeted Improvements

* **For Integrity:**
  * In the **next standalone full run**, fix the plotting routine so that mesh edges, row gaps, and horizontal rendering bands are not drawn. Correctly interpolate or assemble the AMR solution for visualization, verify the aspect ratio and orientation, and regenerate `summary_all.png` from that new run.
  * Add runtime checks for finite values, \(\min(\rho)\), and \(\min(p)\); do not rely on the figure to establish positivity.

* **For Accuracy:**
  * Increase **`polydeg` from 3 to 5** while retaining the existing density-driven AMR and `max_level=6`. This should improve resolution of smooth roll-up without further mesh refinement.
  * Compensate for the stricter high-order time-step restriction by reducing **CFL from 0.5 to approximately 0.25–0.30**.
  * Record mass conservation, extrema of density and pressure, wall-normal velocity at both walls, and the energy balance including gravity-source work.

* **For Details:**
  * Replace **`flux_hll` with the contact-resolving `flux_hllc`** to reduce smearing of density interfaces.
  * Since strong shocks are not the dominant feature, cautiously reduce **`alpha_max` from 1.0 to about 0.5**, while retaining the Ranocha/Lax–Friedrichs blending for robustness. The density indicator may otherwise apply substantial stabilization directly at material interfaces.
  * If genuine oscillations appear after these changes, increase stabilization moderately rather than returning immediately to the most dissipative configuration.

* **For Optimality:**
  * Retain **DGSEM + density-driven AMR**; it is well matched to this problem. The main methodological improvement is the contact-resolving surface flux and better discrimination between shocks and material interfaces.

**Decision rationale:** No previous trial has scored at least 90, and the five-run saturation condition is therefore inactive. A refined, fully standalone rerun is required before this dataset can be accepted as high-fidelity PINN ground truth.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 2 | Numerical implementation of 2D inviscid compressible Euler RTI using DGSEM in Trixi. Uses CompressibleEulerEquations2D with gamma=1.4, P4estMesh on [0,0.25]x[0,1] with periodicity=(true,false), initial_refinement_level=3, polydeg=3. DGSEM has surface_flux=flux_hll and VolumeIntegralShockCapturingHG with IndicatorHennemannGassner on density (alpha_max=1.0, alpha_min=0.001, alpha_smooth=true) blending flux_ranocha and flux_lax_friedrichs. Slip-wall boundary conditions are set on y_neg/y_pos; x is periodic. Gravity source term implements S=(0,0,rho*g,g*rho*v) with g=1.0 in +y. Initial conditions match the strategy: smoothed tanh density interface at y=0.5 (rho=2 below, rho=1 above), piecewise hydrostatic pressure (2y+1 for y<0.5, y+1.5 for y>=0.5), u=0, v=-0.025*c*cos(8*pi*x)*sin(pi*y)^6 with c=sqrt(gamma*p/rho), and energy via prim2cons. AMR uses IndicatorLöhner on density with ControllerThreeLevel base_level=4, med_level=5, med_threshold=0.1, max_level=6, max_threshold=0.3, interval=20, adapt_initial_condition=true. Time integration is RDPK3SpFSAL49, adaptive=false, tspan=(0.0,3.0), CFL=0.5 via StepsizeCallback, saveat=0.1. Visualization plots density rho and vertical velocity v at t=0.75,1.5,2.25,3.0 in a 2x4 layout, colorbar=false, no fixed colormap, saving summary_all.png. | N/A | 40 | 15 | 8 | 12 | **75** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 2 | Creates a fresh rti_run/ directory and rti_run/data_npz/. NPZ snapshots are written with npzwrite to files named time_%08.5f.npz (e.g., time_00.75000.npz) at t=0.0,0.1,0.2,...,3.0 plus 0.75,1.5,2.25. Each NPZ file contains keys: 't' (scalar Float64), 'x' (1D vector length N_points), 'y' (1D vector length N_points), 'rho' (1D vector length N_points), 'rho_u' (1D vector length N_points), 'rho_v' (1D vector length N_points), 'rho_e' (1D vector length N_points), 'u' (1D vector length N_points), 'v' (1D vector length N_points), 'p' (1D vector length N_points). N_points is the number of PlotData2D visualization points from the adaptive DG mesh and varies with AMR. Additionally, SaveSolutionCallback writes Trixi HDF5 solution files to rti_run/trixi_output/ at dt=0.1 with initial and final solutions saved, using solution_variables=cons2cons. The final 2x4 visualization is saved as summary_all.png in the script directory. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run2/results//CompressibleEuler_Periodic_Reflective_RayleighTaylor/2026_11_06_2147/` |


---
