
# Experiment Report
*Last Updated: 2025-11-22 15:01:01*

## User Request

Let's solve the 2D Compressible **Euler equations** for the **Kelvin-Helmholtz Instability**.

**Physics Parameters:**
* $\gamma = 1.4$
* Viscosity $\mu = 0.0$ (Inviscid flow)
* Gravity $g = 0.0$ (Instability is driven by shear, not buoyancy).

**Domain:**
$(x,y) \in [0, 1.0] \times [0, 1.0]$
Time interval: $t \in [0, 2.5]$

**Boundary Conditions:**
* $x$-direction: **Periodic**
* $y$-direction: **Slip walls** (Reflective, normal velocity $v=0$) at $y=0$ and $y=1$.

**Initial Conditions:**
The domain is initialized with a smoothed shear layer at $y=0.5$.
* **Density ($\rho$):** Stratified with a smooth transition.
    $$ \rho(y) = 1.5 - 0.5 \cdot \tanh\left(\frac{y - 0.5}{0.03}\right) $$
    (Density varies from 1.0 at the top to 2.0 at the bottom).
* **Pressure ($p$):** Constant uniform pressure.
    $$ p = 2.5 $$
* **Velocity:**
    * **Horizontal ($u$):** Opposing flows smoothed by a tanh profile (Shear layer).
        $$ u(y) = 0.5 \cdot \tanh\left(\frac{y - 0.5}{0.03}\right) $$
    * **Vertical ($v$):** A single-mode perturbation to trigger the roll-up.
        $$ v(x,y) = 0.01 \cdot \sin(4\pi x) \cdot \exp\left(-\frac{(y - 0.5)^2}{0.01}\right) $$



---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
### 1. Problem Statement

- **Equation:** 2D compressible Euler equations (conservative form)
  $$
  \partial_t \mathbf{U} + \partial_x \mathbf{F}(\mathbf{U}) + \partial_y \mathbf{G}(\mathbf{U}) = 0,
  $$
  with $\mathbf{U} = (\rho, \rho u, \rho v, E)$, ideal gas EOS with $\gamma=1.4$, $p=(\gamma-1)\left(E-\frac12\rho(u^2+v^2)\right)$.

- **Type:** Hyperbolic (nonlinear, inviscid).

- **Domain:** $(x,y)\in[0,1]\times[0,1]$, $t\in[0,2.5]$.
  - $x$-direction: periodic boundaries.
  - $y$-direction: slip walls at $y=0,1$ ($v=0$; zero normal mass/momentum flux).

- **Initial conditions:**
  - Density:
    $$
    \rho(y) = 1.5 - 0.5\,\tanh\!\left(\frac{y-0.5}{0.03}\right)
    $$
  - Pressure:
    $$
    p = 2.5
    $$
  - Velocities:
    $$
    u(y) = 0.5\,\tanh\!\left(\frac{y-0.5}{0.03}\right)
    $$
    $$
    v(x,y) = 0.01\,\sin(4\pi x)\,\exp\!\left(-\frac{(y-0.5)^2}{0.01}\right)
    $$
  - Total energy:
    $$
    E = \frac{p}{\gamma-1} + \frac12 \rho\,(u^2+v^2)
    $$

- **Expected Physics:** Kelvin–Helmholtz instability of a thin shear layer:
  - Linear growth of perturbation in $v$.
  - Roll-up into large coherent vortices around midplane $y\approx 0.5$.
  - Subsequent pairing / secondary instabilities; no shocks expected (subsonic).

---

### 2. Strategic Methodology

- **Selected Main Method:** Discontinuous Galerkin Spectral Element Method in 2D plus time (DGSEM).

- **Rationale:** Strongly hyperbolic, inviscid, shock-free shear-flow instability on a smooth periodic/slip domain is best captured by high-order DGSEM with minimal artificial dissipation and refinement targeted at the shear/density interface.

---

### 3. Implementation Requirements (CRITICAL)

**Configuration:**
- Use template defaults **except** for the following mandatory changes requested by the Advisor.

**Main Method:**
- Discontinuous Galerkin Spectral Element Method in 2D for Compressible Euler with:
  - High-order nodal basis (e.g. Lobatto–Legendre).
  - Explicit strong-stability-preserving Runge–Kutta time integrator (as in template).

**Enhancements (Physics-Driven, per Advisor):**

1. **AMR Targeting of Shear Layer (CRITICAL)**
   - **AMR indicator variable:** switch to **density** (or directly `rho`), *not* velocity magnitude.
   - **Controller:** unchanged template controller (three-level) but driven by density gradients.
   - Purpose: force refinement around the sharp density (and hence shear) transition at $y\approx0.5$, so that the tanh layer with thickness $\sim 0.03$ is resolved by several elements and high-order polynomials.

2. **Disable Artificial Shock-Capturing Dissipation (CRITICAL)**
   - **Volume integral operator:** use **`VolumeIntegralWeakForm`**.
   - **Do not** use `VolumeIntegralShockCapturingHG` for this run.
   - Reason: flow is smooth, subsonic, and dominated by linear/weakly nonlinear shear; shock capturing is unnecessary and excessively diffusive here.

3. **Increase Polynomial Order (CRITICAL)**
   - Set **`polydeg = 5`** (or at least 4; target 5 as per Advisor).
   - Maintain existing element layout / AMR levels, letting higher order reduce numerical dissipation and better resolve the vortex cores.

4. **Flux Choice (Keep Template Default)**
   - Keep the existing approximate Riemann solver (e.g. HLL) at surfaces, as in prior run.
   - Do not introduce extra limiting or slope-flattening beyond the disabled shock capturing.

5. **Boundary Conditions**
   - Maintain:
     - Periodic in $x$.
     - Slip wall (reflective) in $y$ with $v=0$ enforced weakly via boundary fluxes.
   - Ensure primitive variable conversion is consistent with these BCs at each output.

6. **Output / Data Management**
   - Retain existing callbacks for:
     - NPZ snapshot export (conservative or primitive variables).
     - HDF5 DG solution output.
   - Ensure snapshots include at least: $t$, $x$, $y$, $\rho$, $u$, $v$, $p$ on a plotting grid.

7. **Optional (Only if later needed after this run)**
   - If excessive small-scale noise appears at late times (after clear KH roll-up has formed):
     - Re-enable shock capturing but with:
       - Much higher activation threshold.
       - Same AMR on density.
     - This is **not** part of the primary strategy run; only a contingency.

---

### 4. Visualization Strategy

- **Task:** Generate a multi-panel figure `summary_all.png` with all primary fields at 4 times to clearly see KH growth and roll-up.

- **Fields (Rows):**
  1. Density $\rho(x,y)$.
  2. Horizontal velocity $u(x,y)$.
  3. Vertical velocity $v(x,y)$.
  4. Pressure $p(x,y)$.

- **Times (Columns):**
  - $t = 0.0$ (initial).
  - $t \approx 0.8$–$1.0$ (linear growth, early modulation of shear).
  - $t \approx 1.6$–$1.8$ (onset of roll-up; formation of vortical billows).
  - $t = 2.5$ (fully nonlinear stage with deformed vortices / secondary structures).

- **Layout:**
  - 4 (rows) × 4 (columns) grid: rows = fields, columns = times.
  - Axes: use $x$ horizontal, $y$ vertical; identical domain [0,1]×[0,1] for all panels.
  - Titles:
    - Top row: include time in column titles (e.g., “$t=0.0$”, “$t=1.0$”, ...).
    - Row labels: “Density”, “u-velocity”, “v-velocity”, “Pressure”.

- **Style Constraints:**
  - Save exactly as `summary_all.png`.
  - No colorbars (for Julia plotting).
  - Allow colormap scales to auto-adjust per panel (no fixed global colormaps).
  - Use only primary fields listed above; no derived diagnostics are required.

---

**Main Method:** Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
- AMR indicator on **density** (`rho`).
- **Disable shock capturing**: use `VolumeIntegralWeakForm`.
- Increase **polynomial degree to 5**.

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
Observation: The pressure field at t=0 shows minute fluctuations (±0.0002 around 2.5) due to projection error, which is negligible. Between t=1.45 and t=2.5 the shear layer rolls up into two distinct vortices, preserving a vivid density contrast and demonstrating that the solver does not smear the contact discontinuity. At t=2.5, thin density filaments are resolved by AMR level 6, though the finest filaments approach sub‑grid scales, creating slight texture. Diagnosis: The phenomenon is inviscid vortex roll‑up. The lack of explicit viscosity (μ=0) and disabled shock capturing leave only the very low inherent dissipation of the DG upwind flux, allowing the Kelvin‑Helmholtz instability to grow at the physical rate. The use of polydeg=5 provides high spectral resolution per element, mitigating dispersion errors that would otherwise distort the vortices.

Recommendation:
None required for this dataset.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Numerical Discontinuous Galerkin Spectral Element Method (DGSEM) in 2D for compressible Euler equations with gamma=1.4. Polynomial degree polydeg=5 using LobattoLegendreBasis. Surface flux: HLL. Volume integral: VolumeIntegralWeakForm (shock capturing disabled). P4estMesh on [0,1]x[0,1] with initial_refinement_level=3, periodicity in x, slip walls in y. AMR via IndicatorLöhner on density variable with ControllerThreeLevel (base_level=4, med=5 threshold=0.1, max=6 threshold=0.3), interval=20. Time span [0,2.5] with RDPK3SpFSAL49 low-storage RK integrator, CFL=0.5, adaptive=false. Initial conditions match KH shear layer exactly. | 0.00e+00 | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves multiple NPZ files in 'data_npz/' directory named 'time_XXXXXXX.npz' (e.g., time_00000.00000.npz) containing keys: 't' (scalar), 'x' (1D vector of length N from PlotData2D grid), 'y' (1D vector length N), 'rho' (1D vector N), 'u' (N), 'v' (N), 'p' (N). Additionally, HDF5 DG solution files via SaveSolutionCallback (cons2prim variables) in 'trixi_output/' directory at interval=20 including initial. Generates 'summary_all.png' visualization. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//Euler_Periodic_SlipWalls_ShearLayer/2025_11_22_1457/` |


---
### Scoring Legend
* **Integrity (0-50):** Code stability, valid plots, no NaNs.
* **Accuracy (0-20):** Physics correctness, boundary conditions.
* **Details (0-15):** Oscillations, smearing, grid efficiency.
* **Optimality (0-15):** Correct numerical method choice (e.g., DG vs FD).
* **PASSING SCORE:** > 75
