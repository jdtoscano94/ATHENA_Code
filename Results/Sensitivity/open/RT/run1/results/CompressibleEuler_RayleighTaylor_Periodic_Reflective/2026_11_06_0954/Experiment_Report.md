
# Experiment Report
*Last Updated: 2026-11-06 09:58:19*

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
# Scientific Strategy Report — 2D Rayleigh–Taylor Instability (Compressible Euler + Gravity)

---

## 1. Problem Statement

Solve the inviscid compressible Euler equations with a gravitational source term for the single-mode Rayleigh–Taylor instability. A heavy fluid ($\rho = 2.0$) rests below a light fluid ($\rho = 1.0$) across a smoothed interface at $y = 0.5$; gravity acts in $+y$, i.e., the acceleration is directed **from the heavy into the light fluid** — the RT-unstable configuration (Atwood number $A = 1/3$). A single-mode perturbation (one wavelength per domain width) seeds the instability. Objective: capture the full evolution — exponential linear growth → nonlinear spike/bubble formation → Kelvin–Helmholtz roll-up → late-time mixing — on $t \in [0, 3.0]$.

---

## 2. Mathematical Specification (Code-Ready Extraction)

* **Governing Equations:** 2D Compressible Euler with gravity source:
$$\partial_t \begin{pmatrix}\rho\\ \rho u\\ \rho v\\ E\end{pmatrix} + \partial_x \begin{pmatrix}\rho u\\ \rho u^2 + p\\ \rho u v\\ (E+p)u\end{pmatrix} + \partial_y \begin{pmatrix}\rho v\\ \rho u v\\ \rho v^2 + p\\ (E+p)v\end{pmatrix} = \begin{pmatrix}0\\ 0\\ \rho g\\ \rho g\, v\end{pmatrix}$$
with equation of state:
$$p = (\gamma - 1)\left(E - \tfrac{1}{2}\rho\left(u^2 + v^2\right)\right)$$

* **Domain:** $(x,y) \in [0, 0.25] \times [0, 1.0]$, $\quad t \in [0, 3.0]$

* **Boundary Conditions:**
  * $x$-direction: Periodic.
  * $y$-direction: Reflective / free-slip walls at $y=0$ and $y=1$ (no-penetration: $v = 0$; mirror tangential velocity $u$).

* **Initial Conditions** (exact, for every field):
$$\rho(x,y,0) = 2.0 + (1.0 - 2.0)\cdot \tfrac{1}{2}\left[1 + \tanh\left(50\,(y - 0.5)\right)\right]$$
$$p(x,y,0) = \begin{cases} 2y + 1, & y < 0.5\\[2pt] y + 1.5, & y \geq 0.5\end{cases}$$
$$u(x,y,0) = 0$$
$$v(x,y,0) = -0.025\; c(x,y)\,\cos(8\pi x)\,\sin^6(\pi y), \qquad c(x,y) = \sqrt{\frac{\gamma\, p(x,y,0)}{\rho(x,y,0)}}$$
$$E(x,y,0) = \frac{p(x,y,0)}{\gamma - 1} + \frac{1}{2}\,\rho(x,y,0)\,\Big[u(x,y,0)^2 + v(x,y,0)^2\Big]$$

* **Constants & Parameters:** $\gamma = 1.4$; $g = 1.0$ (source adds momentum in $+y$); $\rho_{heavy} = 2.0$, $\rho_{light} = 1.0$; interface $y_i = 0.5$; smoothing steepness $= 50$; perturbation amplitude $= 0.025$; mode $k_x = 8\pi$ (wavelength $= 0.25$, exactly one period per domain); $\mu = 0$, $k_{\text{thermal}} = 0$.

* **Expected Physics:**
  * **Early ($t \lesssim 1$):** Exponential growth of the single mode (incompressible estimate $\sigma \approx \sqrt{A k g} \approx 2.9$); a small localized adjustment transient at the interface due to the tanh-smoothed $\rho$ vs. piecewise-linear $p$ mismatch — subsumed into the instability growth.
  * **Intermediate ($t \approx 1$–$2$):** Nonlinear saturation; heavy-fluid **spikes rise ($+y$)** near $x = 0.125$, light-fluid **bubbles sink ($-y$)** near $x = 0,\, 0.25$; shear layers on spike flanks seed Kelvin–Helmholtz roll-up.
  * **Late ($t \approx 2$–$3$):** Mushroom structures, vortex-dominated mixing layer broadening toward both walls; steep interface gradients persist throughout; flow remains essentially subsonic.

---

## 3. Strategic Methodology

**Exact solution: NOT feasible.** RTI is a genuinely 2D, nonlinear, multi-scale instability: the flow evolves from a perturbed equilibrium through transverse mode growth, shear-driven vortex roll-up, and mixing. The Method of Characteristics applies only to 1D hyperbolic systems / simple waves — it cannot represent transverse instability growth or vortex formation, and no separation-of-variables / Green's-function ansatz exists for the nonlinear Euler system. The Fourier Spectral Method is inapplicable: the $y$-direction has solid (non-periodic) walls, and the solution develops steep interface gradients (Gibbs risk).

**Selection: Discontinuous Galerkin Spectral Element Method in 2D plus time.** It handles wall boundaries naturally, provides local element-wise high order for the smooth vortex cores, and supports AMR + shock capturing for the sharp material interface.

---

## 4. Implementation Requirements (CRITICAL)

* **Configuration:** Use template defaults.

* **3.0.1 Adaptive Mesh Refinement & Shock Capturing — Variable Selection:**
  * **Selected sensor variable: Density $\rho$.**
  * *Signal-to-Noise Analysis:* The structural feature of RTI is the **material interface** (later: spike/bubble boundaries). Density defines this feature sharply and *locally* at all times — a clean factor-2 binary-fluid tracer. In contrast, **velocity** becomes globally chaotic/turbulent in the mixing layer by late time (refining on it wastes elements on noise), and **pressure** is dominated by the smooth hydrostatic background, so its indicator smears over the entire domain instead of localizing at the interface. Density is therefore strictly superior for both the AMR indicator and the shock-capturing indicator for this PDE.

* **3.1 Advisor-Mandated Adjustments (Overrides):** None (Advisor report is empty).

---

## 5. Visualization Strategy

* **Field Selection (Analysis):**
  * **Primary: $\rho$** — the interface/mixing tracer; the driver of the RTI dynamics.
  * **Secondary: $v$** — vertical velocity, showing spike rise, bubble descent, and vortex roll-up.
* **Task:** Plot $\rho$ and $v$ at **4 distinct time steps**: $t = 0.5,\; 1.0,\; 2.0,\; 3.0$ (linear growth → nonlinear onset → roll-up → late mixing).
* **Layout:** Grid — 2 rows (fields: $\rho$, $v$) × 4 columns (times).
* **Saving Data:** Save the **full solution state** (all variables) at `dt_save = 0.1` for Ground Truth. Plotting selection affects only the `summary_all.png` image.
* **Constraints:**
  * Save as: `summary_all.png` (strictly).
  * **CRITICAL: DISABLE COLORBARS (`colorbar=false`)** if the code is in Julia.
  * Do not use fixed colormaps; let scales adjust automatically.
  * Plot only the primitive fields $\rho$ and $v$; do not compute derived quantities.

---

## 6. Cross-Validation & Data Integration (Conditional)

No external reference data or sensor paths are provided in the problem statement → **not applicable**.

---

## REQUIRED COMPONENTS

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
* Shock Capturing Indicator (density-based)
* Adaptive Mesh Refinement (density-based)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

**Status:** `REJECTED_FOR_REFINEMENT`

The simulation completed successfully and shows a qualitatively plausible Rayleigh–Taylor evolution. However, the figure has a significant rendering artifact, the material interfaces are not demonstrably sharp enough for PINN ground truth, and no convergence or physical-conservation verification is supplied.

The saturation rule is not applicable: there is no trial history and no previous score ≥90.

## Section 2: Quantitative Scorecard

- **Score Integrity:** 43/50
- **Score Accuracy:** 16/20
- **Score Details:** 10/15
- **Score Optimality:** 12/15
- **TOTAL SCORE:** **81/100**

## Section 3: Forensic Analysis & Feedback

### Observation

**Execution and error reporting**

- The log reaches `t=3.0`, saves `time_03.00000.npz`, generates all four requested plot times, and ends with `Done.` There is no reported crash, solver failure, or NaN/Inf diagnostic.
- The requested string **`Minimum achieved error:` is not present in the supplied STDOUT tail**.
- **Quantitative error is not applicable as no ground truth is available.** This is a forward nonlinear RTI simulation, so an error against zero or another artificial reference would not be meaningful.

**Physical evolution**

- At \(t=0.5\), the density interface remains close to \(y=0.5\), with a small perturbation and a localized vertical-velocity response.
- At \(t=1.0\), the central heavy-fluid protrusion is rising in \(+y\), consistent with the initial perturbation and the stated gravity direction.
- At \(t=2.0\), the spike has become strongly nonlinear, with side lobes suggestive of shear-driven roll-up and adjacent light-fluid penetration downward.
- At \(t=3.0\), a long heavy spike and strong vertical-velocity column extend toward the upper wall, with increasingly complex side structures. The evolution is broadly consistent with spike growth, bubble descent, and late-time RTI mixing.
- No gross left–right symmetry violation is apparent. Nevertheless, the spike approaches the upper wall at late time, making verification of the reflective condition \(v=0\) and possible wall interaction particularly important.

**Plot and interface quality**

- Every panel contains dense, regular horizontal white stripes, including otherwise quiescent regions. Because they occur systematically in both fields and at all times, they are more consistent with **mesh-edge/plot-rendering artifacts or missing plot rows** than with physical Gibbs oscillations. They substantially obscure interface sharpness and boundary behavior.
- The visible density fronts are coherent, and there is no clear evidence of severe ringing. However, the spike flanks and roll-up structures appear somewhat soft/pixelated. The figure does not establish that the material interface is resolved within one or two cells.

### Diagnosis

1. **Visualization artifact:** The horizontal striping must be corrected before the solution can be reliably assessed visually. Raw NPZ data should also be checked to exclude missing or non-finite rows.
2. **Potential contact-interface dissipation:** `flux_hll` does not resolve stationary/contact discontinuities as effectively as HLLC. Combined with Lax–Friedrichs shock-capturing blending, it can unnecessarily diffuse the density interface in this predominantly subsonic flow.
3. **Insufficient verification:** Qualitative plausibility alone is not enough for ground-truth generation. Mass conservation, gravitational energy-work balance, positivity, wall behavior, and discretization convergence have not been demonstrated.
4. **Exact-method check:** No exact Method-of-Characteristics, separation-of-variables, or Fourier solution is available for the full nonlinear, two-dimensional, wall-bounded RTI evolution through vortex roll-up and mixing. Linear RTI theory would only describe the early regime. Therefore, DGSEM is an admissible numerical fallback; no exact-method architecture override is warranted.

### Targeted Improvements

- **For Integrity:**
  - Remove element-edge strokes from the plots, e.g. use `edgecolors='none'` and zero line width, or interpolate the DG/AMR solution onto a uniform visualization grid with no gaps between cells.
  - Check every saved NPZ state for finite \(\rho,u,v,p\), positive density and pressure, and missing rows.
  - Add physical axis extents/ticks so that interface position and wall distance can be inspected.

- **For Accuracy:**
  - Add quantitative physical diagnostics: total mass drift, minimum/maximum density and pressure, symmetry error about the expected symmetry plane, wall-normal velocity at \(y=0,1\), and the energy balance
    \[
    \frac{d}{dt}\int E\,dV \approx \int \rho g v\,dV
    \]
    rather than expecting total energy conservation in the presence of gravity.
  - Perform a controlled discretization-convergence comparison at the saved output times. Use **`polydeg=4`** for the next run; `polydeg=5` is preferable to further aggressive global mesh refinement if cost and stability remain acceptable. Reduce CFL to approximately 0.3–0.4 only if the higher-degree run requires it.

- **For Details:**
  - Replace **`flux_hll` with `flux_hllc`** while retaining `flux_ranocha` in the volume. This should reduce dissipation at the material/contact interface.
  - Keep density-based AMR, ensuring the spike and bubble boundaries reach the permitted refinement level. Avoid further global refinement unless the convergence test demonstrates that it is necessary.
  - Because strong shocks are not expected, reduce the aggressiveness of the Lax–Friedrichs blending—through `alpha_max` or its activation threshold—so it acts only where genuinely needed. Retain a local stabilization/positivity safeguard if HLLC or higher polynomial degree produces interface oscillations.

- **For Optimality:**
  - Retain DGSEM, density-based AMR, the Ranocha volume flux, and the adaptive explicit time integrator. The principal methodological upgrade is **HLLC plus more selective shock capturing**, not a change of PDE solver architecture.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 2 | Implemented 2D compressible Euler equations for single-mode Rayleigh–Taylor instability using Trixi's DGSEM. Polydeg=3, P4estMesh over [0,0.25]x[0,1] with initial refinement level 3, periodic in x and slip walls in y. Surface flux flux_hll, volume flux flux_ranocha, hybrid shock capturing with flux_lax_friedrichs and density-based IndicatorHennemannGassner. AMR uses density-based IndicatorLöhner with ControllerThreeLevel (base 4, med 5, max 6) and AMRCallback interval 20. Time integration via RDPK3SpFSAL49 with tspan=(0,3) and cfl=0.5. Initial conditions match the strategy exactly: rho tanh profile, piecewise pressure, u=0, v perturbation with sound speed, and E via prim2cons. Gravity source adds rho*g to y-momentum and g*rho_v2 to energy. Data saved at dt_save=0.1. | N/A | 43 | 16 | 10 | 12 | **81** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 2 | Saves multiple .npz files in data_npz/ named time_XX.XXXXX.npz every dt_save=0.1 (including initial and final via PeriodicCallback and finalize). Each .npz contains keys: 't' (scalar), 'x' (1-D array of length N), 'y' (1-D array of length N), 'rho' (1-D array of length N), 'u' (1-D array of length N), 'v' (1-D array of length N), 'p' (1-D array of length N), where N is the number of PlotData2D nodes. Also saves Trixi HDF5 solution snapshots via SaveSolutionCallback at dt=0.1 in trixi_output/ (primitive variables rho,u,v,p on the adaptive mesh). Finally saves summary_all.png as a 2x4 grid of scatter plots showing rho (first row) and v (second row) at t=0.5,1.0,2.0,3.0, with colorbar=false and automatic color scaling. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run1/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_0954/` |


---
