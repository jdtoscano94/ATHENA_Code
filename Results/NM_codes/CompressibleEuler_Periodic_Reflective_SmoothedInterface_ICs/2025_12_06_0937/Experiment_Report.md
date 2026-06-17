
# Experiment Report
*Last Updated: 2025-12-06 10:00:16*

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

### Mathematical Specification
*   **Governing Equations:** Compressible Euler Equations with Gravity Source Term
    $$ \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0 $$
    $$ \frac{\partial (\rho \mathbf{v})}{\partial t} + \nabla \cdot (\rho \mathbf{v} \otimes \mathbf{v} + p\mathbf{I}) = \rho \mathbf{g} $$
    $$ \frac{\partial E}{\partial t} + \nabla \cdot ((E + p)\mathbf{v}) = \rho \mathbf{v} \cdot \mathbf{g} $$
    where $E = \frac{p}{\gamma - 1} + \frac{1}{2}\rho \|\mathbf{v}\|^2$ and $\mathbf{g} = [0, 1.0]^T$.

*   **Domain:** $\Omega = [0, 0.25] \times [0, 1.0]$
*   **Boundary Conditions:**
    *   $x$-boundaries ($x=0, x=0.25$): **Periodic**
    *   $y$-boundaries ($y=0, y=1$): **Reflective (Slip Wall, $\mathbf{v} \cdot \mathbf{n} = 0$)**
*   **Initial Conditions:**
    *   **Density ($\rho$):**
        $$ \rho(x,y) = 2.0 + (1.0 - 2.0) \cdot \frac{1}{2}\left(1 + \tanh\left(50(y - 0.5)\right)\right) $$
    *   **Pressure ($p$):**
        $$ p(y) = \begin{cases} 2y + 1 & \text{if } y < 0.5 \\ y + 1.5 & \text{if } y \geq 0.5 \end{cases} $$
    *   **Velocity ($\mathbf{v} = [u, v]^T$):**
        $$ u = 0 $$
        $$ v(x,y) = -0.025 \sqrt{\frac{\gamma p}{\rho}} \cos(8\pi x) \sin(\pi y)^6 $$
*   **Constants & Parameters:** $\gamma = 1.4$, $\mathbf{g} = [0, 1.0]^T$.
*   **Expected Physics:** Rayleigh-Taylor instability driven by the acceleration of the heavy fluid ($y<0.5$) into the light fluid ($y>0.5$). Formation of symmetric "mushroom" structures and Kelvin-Helmholtz vortices at the interface.

---

**2. Strategic Methodology**

*   **Selected Main Method:** Discontinuous Galerkin Spectral Element Method in 2D plus time (DGSEM).
*   **Rationale:** The problem involves complex vortex generation and sharp density interfaces; DGSEM provides high-order accuracy for the vortices while shock capturing handles the contact discontinuity.

---

**3. Implementation Requirements**

*   **Configuration:** Use template defaults unless specified below.

*   **3.1 Adaptive Mesh Refinement & Shock Capturing:**
    *   **Variable Selection Analysis:** The primary feature is the contact discontinuity (interface) between the heavy and light fluids.
    *   **Selected Variable:** **Density ($\rho$)**.
    *   **Justification:** Density is the only variable that distinctly marks the interface between the two fluids. Velocity becomes turbulent and noisy throughout the mixing zone, making it a poor candidate for targeted refinement. Pressure is relatively smooth across the contact discontinuity.

*   **3.2 Advisor-Mandated Adjustments (Overrides):**
    *   **AMR Levels:** Increase maximum refinement level to **7** (Range: 4 to 7).
    *   **Surface Flux:** Use `flux_hllc` (HLLC Flux) to minimize dissipation at the contact discontinuity.
    *   **Polynomial Degree:** Keep `polydeg=4`.
    *   **Save Interval:** Ensure `dt_save` is set (e.g., 0.1) to guarantee a snapshot at exactly $t=1.0$.

---

**4. Visualization Strategy**

*   **Field Selection:**
    *   **Primary Field:** **Density ($\rho$)**.
    *   *Reasoning:* Density visualizes the mixing of the two fluids and the structure of the instability (bubbles and spikes) most clearly.
*   **Time Steps:** Plot at $t = 0.0, 1.0, 2.0, 3.0$.
*   **Layout:** 1 Row (Density) $\times$ 4 Columns (Time).
*   **Constraints:**
    *   Save as `summary_all.png`.
    *   **DISABLE COLORBARS** (`colorbar=false`).
    *   Ensure the aspect ratio respects the domain $[0, 0.25] \times [0, 1.0]$ (tall and thin).

---

**REQUIRED COMPONENTS**

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
*   Shock Capturing Indicator (Hennemann-Gassner)
*   Adaptive Mesh Refinement (Löhner Indicator on Density)
*   HLLC Surface Flux (Advisor Mandate)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
### Section 1: Status Decision
*   **Status:** APPROVED

### Section 2: Quantitative Scorecard
*   **Score Integrity:** **50** (Code ran to completion, logs clean, plots generated successfully.)
*   **Score Accuracy:** **20** (Physics are flawless. The instability evolves symmetrically and respects the upward gravitational forcing. Boundary conditions are stable.)
*   **Score Details:** **15** (Excellent resolution. The use of `flux_hllc` combined with AMR Level 7 has resulted in razor-sharp contact discontinuities. The Kelvin-Helmholtz vortices on the mushroom caps are distinct and well-resolved without visible smearing.)
*   **Score Optimality:** **15** (The choice of DGSEM with HLLC flux is theoretically optimal for this problem, balancing high-order accuracy for vortices with robust shock capturing for the density interfaces.)
*   **TOTAL SCORE:** **100**

### Section 3: Forensic Analysis & Feedback

**1. Observation:**
The simulation depicts a high-fidelity evolution of the Rayleigh-Taylor Instability.
*   **Physics:** The setup involves a heavy fluid ($\rho=2$, Red) on the bottom and a light fluid ($\rho=1$, Blue) on top, with a gravitational force acting **upwards** ($+y$). As expected, the heavy fluid is driven into the light fluid, forming characteristic "spikes" (rising red structures) and "bubbles" (falling blue structures).
*   **Resolution:** The interface between the fluids is remarkably sharp. At $t=2.0$ and $t=3.0$, secondary Kelvin-Helmholtz instabilities (rolling vortices) are clearly visible along the shear layers of the primary spike. These features are often washed out in more dissipative schemes but are preserved here.
*   **Symmetry:** The solution maintains perfect symmetry about the vertical centerline ($x=0.125$), indicating that the numerical grid and scheme introduced no biasing errors.

**2. Diagnosis:**
**Perfect Execution.** The high quality of the result is directly attributable to the strategic override to use the **HLLC Surface Flux** (`flux_hllc`). Unlike the standard local Lax-Friedrichs flux (which is very dissipative), HLLC is designed to resolve contact discontinuities (density interfaces) exactly. Combined with aggressive AMR (Level 7), this minimized numerical diffusion, allowing the fine-scale turbulence to develop naturally.

**3. Targeted Improvements:**
*   **None.** The solution has achieved the maximum score. The trade-off between cost and accuracy is optimal for this resolution.
*   **Note on Error:** As this is a forward simulation of a chaotic instability without a closed-form analytical solution, no quantitative error norm is applicable. The "Accuracy" is judged by the physical validity of the flow structures.

**Minimum achieved error:** N/A (Forward Simulation / No Analytical Solution)


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Numerical implementation using Discontinuous Galerkin Spectral Element Method (DGSEM) in 2D plus time with polydeg=3, LobattoLegendreBasis, surface_flux=flux_hll, volume_integral=VolumeIntegralShockCapturingHG (DG flux_ranocha blended with FV flux_lax_friedrichs based on IndicatorHennemannGassner on density variable), CompressibleEulerEquations2D(gamma=1.4), gravity source term g=(0,1), domain [0,0.25]x[0,1] via P4estMesh (periodic x, slip walls y), initial_refinement_level=3, AMR via IndicatorLöhner on density with ControllerThreeLevel (levels 4-6), time integration RDPK3SpFSAL49 (LSRK) with CFL=0.5 over tspan=(0.0,3.0), exact matching initial conditions (tanh density, piecewise pressure, velocity perturbation), reflective slip BCs. | N/A | 40 | 18 | 10 | 10 | **78** |
| 1 | Numerical implementation using Discontinuous Galerkin Spectral Element Method (DGSEM) in 2D plus time via Trixi framework. Compressible Euler equations with gravity source term (g=[0,1.0]), gamma=1.4. Domain [0,0.25]x[0,1.0], periodic in x, slip walls in y. Polynomial degree polydeg=4. Surface flux flux_hllc. Shock capturing via IndicatorHennemannGassner on density (volume_flux_dg=flux_ranocha, volume_flux_fv=flux_lax_friedrichs). AMR via IndicatorLöhner on density with ControllerThreeLevel (levels 4-6). Time integration RDPK3SpFSAL49 over tspan=(0.0,3.0) with CFL=0.5. Initial conditions exactly match: tanh density profile, piecewise hydrostatic pressure, velocity perturbation. Plots density fields. | N/A | 50 | 20 | 14 | 15 | **99** |
| 2 | Numerical implementation using Discontinuous Galerkin Spectral Element Method (DGSEM) in 2D plus time. Polynomial degree polydeg=4 with LobattoLegendreBasis. HLLC surface flux, VolumeIntegralShockCapturingHG (Hennemann-Gassner indicator on density variable). Adaptive Mesh Refinement with IndicatorLöhner on density, ControllerThreeLevel (base_level=4, max_level=7). Domain [0,0.25]x[0,1], periodic in x, slip walls in y. Gravity source g=[0,1.0], gamma=1.4. Time span tspan=(0.0,3.0), RDPK3SpFSAL49 integrator with CFL=0.5. Initial conditions, pressure profile, density tanh interface, and velocity perturbation match exactly. | N/A | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves multiple .npz files (e.g., time_00.00000.npz, time_00.10000.npz, etc.) every dt_save=0.1s to data_npz/ containing keys: 't' (scalar), 'x' (1D vector of quadrature points length ~num_dofs), 'y' (1D vector same length), 'rho' (1D vector), 'u' (1D vector), 'v' (1D vector), 'p' (1D vector). Additionally, SaveSolutionCallback saves full conservative solution states (in primitive variables via cons2prim) as .h5 files every 20 steps to trixi_output/. Generates summary_all.png (density plots at t~0,1,2,3). |
| 1 | Saves multiple files in data_npz/time_XXXXX.npz (every dt_save=0.1, including t=0 and t=3.0) in NPZ format containing keys: 't' (scalar), 'x' (N,), 'y' (N,), 'rho' (N,), 'u' (N,), 'v' (N,), 'p' (N,) where N is the number of points from PlotData2D (flattened nodal data). Also saves Trixi solution files in trixi_output/ via SaveSolutionCallback (interval=20, primitive variables). Generates summary_all.png (1x4 density scatter plots at t≈0,1,2,3 with colorbar=false). |
| 2 | Saves multiple files in 'data_npz/time_XXXXX.npz' (NPZ format) at intervals of dt_save=0.1 (with tstops), each containing keys: 't' (scalar), 'x' (1D vector of length ndofs), 'y' (1D vector of length ndofs), 'rho' (1D vector), 'u' (1D vector), 'v' (1D vector), 'p' (1D vector). Also saves Trixi .h5 solutions via SaveSolutionCallback (interval=20) to 'trixi_output/'. Generates 'summary_all.png' (density heatmaps at t≈0.0,1.0,2.0,3.0; 1x4 layout, no colorbar). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//CompressibleEuler_Periodic_Reflective_SmoothedInterface_ICs/2025_12_06_0918/` |
| 1 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//CompressibleEuler_Periodic_Reflective_SmoothedInterface_ICs/2025_12_06_0926/` |
| 2 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//CompressibleEuler_Periodic_Reflective_SmoothedInterface_ICs/2025_12_06_0937/` |


---
