
# Experiment Report
*Last Updated: 2025-12-06 11:05:37*

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

### 1. Mathematical Specification
*   **Governing Equations:** Compressible Euler Equations with Gravity Source Term ($\gamma = 1.4$):
    $$ \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0 $$
    $$ \frac{\partial (\rho \mathbf{v})}{\partial t} + \nabla \cdot (\rho \mathbf{v} \otimes \mathbf{v} + p \mathbf{I}) = \rho \mathbf{g} $$
    $$ \frac{\partial E}{\partial t} + \nabla \cdot ((E + p)\mathbf{v}) = \rho \mathbf{g} \cdot \mathbf{v} $$
    where $E = \frac{p}{\gamma - 1} + \frac{1}{2}\rho \|\mathbf{v}\|^2$ and $\mathbf{g} = [0, 1.0]^T$.

*   **Domain:** $(x,y) \in [0, 0.25] \times [0, 1.0]$
*   **Boundary Conditions:**
    *   $x$-faces ($x=0, x=0.25$): **Periodic**
    *   $y$-faces ($y=0, y=1$): **Slip Wall** (Reflective, $\mathbf{v} \cdot \mathbf{n} = 0$)
*   **Initial Conditions:**
    *   **Density:** $\rho(y) = 2.0 + (1.0 - 2.0) \cdot \frac{1}{2}\left(1 + \tanh\left(50(y - 0.5)\right)\right)$
    *   **Pressure:** $p(y) = \begin{cases} 2y + 1 & \text{if } y < 0.5 \\ y + 1.5 & \text{if } y \geq 0.5 \end{cases}$
    *   **Velocity:** $u=0, \quad v(x,y) = -0.025 \sqrt{\frac{\gamma p}{\rho}} \cos(8\pi x) \sin(\pi y)^6$
*   **Constants & Parameters:** $\gamma = 1.4$, Gravity $\mathbf{g} = [0, 1.0]^T$.
*   **Expected Physics:** Rayleigh-Taylor instability. Growth of mushroom-shaped plumes, Kelvin-Helmholtz vortices on plume stems, and turbulent mixing at late times ($t=3.0$).

**2. Strategic Methodology**

*   **Selected Main Method:** Discontinuous Galerkin Spectral Element Method in 2D plus time.
*   **Rationale:** The problem involves the evolution of complex vortical structures and steep density gradients (interfaces); DGSEM provides the necessary high-order accuracy for vortices while shock capturing handles the stability at the interface.

**3. Implementation Requirements**

*   **Configuration:** Use template defaults, subject to the overrides below.

*   **3.1 Adaptive Mesh Refinement & Shock Capturing:**
    *   **Variable Selection Analysis:** The **Density ($\rho$)** field defines the interface between the heavy and light fluids. This is the primary structural feature. Velocity becomes turbulent and noisy, making it a poor candidate for refinement.
    *   **Selection:** Use **Density** for both the AMR Indicator (Löhner) and Shock Capturing Indicator (Hennemann-Gassner).
    *   **Advisor-Mandated Adjustments (Overrides):**
        *   **Polynomial Degree:** `polydeg = 4`
        *   **AMR Max Level:** `max_level = 7`
        *   **Data Saving:** You must use a **Time-Based** save interval to guarantee snapshots at integer times. Set `save_solution` with `dt = 1.0` (or `save_at_times=[0.0, 1.0, 2.0, 3.0]`). Do **not** use step-based intervals.

**4. Visualization Strategy**

*   **Field Selection:** **Density ($\rho$)**. (This captures the mixing of fluids most clearly).
*   **Time Steps:** Plot at $t = 0.0, 1.0, 2.0, 3.0$.
*   **Layout:** 1 Row (Density) $\times$ 4 Columns (Time).
*   **Saving Data:** Save full solution state at `dt=1.0` (as mandated above).
*   **Constraints:**
    *   Save as `summary_all.png`.
    *   **CRITICAL:** `colorbar=false` (Disable colorbars).
    *   Use automatic scale adjustment.

**5. Cross-Validation & Data Integration**
*   None.

**REQUIRED COMPONENTS**

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
*   Shock Capturing (Hennemann-Gassner)
*   Adaptive Mesh Refinement (Löhner)
*   Source Term (Gravity)
*   **Advisor Override:** Time-Based Save Interval (`dt=1.0`)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
### Section 1: Status Decision
*   **Status:** APPROVED

### Section 2: Quantitative Scorecard
*   **Score Integrity:** **50**
*   **Score Accuracy:** **20**
*   **Score Details:** **15**
*   **Score Optimality:** **15**
*   **TOTAL SCORE:** **100**

### Section 3: Forensic Analysis & Feedback

**1. Observation:**
*   **Visual Quality:** The simulation produces a high-fidelity representation of the Rayleigh-Taylor Instability. The interface between the heavy ($\rho=2$) and light ($\rho=1$) fluids remains sharp and distinct throughout the evolution ($t=0$ to $t=3$).
*   **Physics:**
    *   **Symmetry:** The solution maintains perfect left-right symmetry, which is a strong indicator of numerical stability and grid alignment.
    *   **Instability Growth:** The linear growth phase ($t=1$) transitions correctly into the non-linear mushroom cap formation ($t=2$) and subsequent turbulent mixing ($t=3$).
    *   **Secondary Features:** Kelvin-Helmholtz vortices are clearly resolved on the stems of the plumes at $t=2.0$ and $t=3.0$, indicating that the effective numerical viscosity is low enough to capture shear instabilities.
*   **Grid & Resolution:** The background grid (visible as faint white lines) shows the Adaptive Mesh Refinement (AMR) in action. The mesh is highly refined (Level 7) specifically along the complex interface and mixing zones, while remaining coarse in the unperturbed regions. This demonstrates optimal resource usage.

**2. Diagnosis:**
*   **Success:** The combination of **DGSEM (Order 4)**, **AMR (Level 7)**, and **Shock Capturing** has successfully generated a "Ground Truth" quality solution.
*   **Dissipation Control:** Despite using the robust (but dissipative) `flux_hll` surface flux, the high refinement level ($L=7$) provided sufficient resolution to counteract numerical smearing, resulting in crisp contact discontinuities.
*   **Stability:** The simulation completed without warnings, oscillations, or symmetry breaking.

**3. Quantitative Error Check:**
*   **Metric:** N/A (Forward Simulation).
*   **Note:** As this is a forward simulation of an instability starting from a perturbation, there is no analytical solution to compare against for error norms. The "Error" in the logs is not applicable.

**4. Conclusion:**
*   The simulation meets all criteria for **High-Fidelity Ground Truth Data**.
*   The "5-Run Rule" for saturation is satisfied: A previous run scored 98, and this refinement (increasing AMR to Level 7) has maximized the visual quality and physical resolution without incurring instability.
*   No further refinement is necessary.

**Targeted Improvements:**
*   **None.** The solution is approved.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The code implements a numerical Discontinuous Galerkin Spectral Element Method (DGSEM) in 2D plus time for the Compressible Euler Equations with gravity source term. Key parameters include polydeg=3 (LobattoLegendreBasis), domain [0.0, 0.25] x [0.0, 1.0], tspan=(0.0, 3.0), γ=1.4, g=(0,1.0), HLL surface flux, Ranocha DG volume flux blended with Lax-Friedrichs FV via Hennemann-Gassner indicator on density (alpha_max=1.0, alpha_min=0.001), Löhner AMR indicator on density with three-level controller (levels 4-6), RDPK3SpFSAL49 time integrator with CFL=0.5. Initial conditions, source terms, periodic x / slip y BCs match exactly. Visualization plots density at ~t=0,1,2,3. | N/A | 45 | 20 | 15 | 15 | **95** |
| 1 | Numerical implementation using Discontinuous Galerkin Spectral Element Method (DGSEM) in 2D plus time. Polynomial degree polydeg=4 with LobattoLegendreBasis. Shock capturing via Hennemann-Gassner indicator on density (alpha_max=1.0, alpha_min=0.001) blending flux_ranocha (DG) and flux_lax_friedrichs (FV). Surface flux flux_hll. Adaptive Mesh Refinement using Löhner indicator on density with ThreeLevel controller (levels 4-6, thresholds 0.1/0.3, interval=20). Domain [0.0,0.25]x[0.0,1.0] with P4estMesh (trees_per_dim=(1,1), initial_refinement_level=3, periodic x). Time span (0.0,3.0) integrated with RDPK3SpFSAL49 LSRK scheme, CFL=0.5. Gravity source g=1.0 in +y direction. Slip wall BCs on y-boundaries. | N/A | 50 | 20 | 13 | 15 | **98** |
| 2 | The code implements a Numerical Discontinuous Galerkin Spectral Element Method (DGSEM) in 2D plus time for the 2D Compressible Euler equations with gravity source term. Key parameters: polydeg=4, surface_flux=flux_hll, AMR with IndicatorLöhner and max_level=7 on density, shock capturing with IndicatorHennemannGassner on density using flux_ranocha (DG) and flux_lax_friedrichs (FV), time integrator RDPK3SpFSAL49 (LSRK), CFL=0.5, tspan=(0.0, 3.0), domain [0.0, 0.25] x [0.0, 1.0], periodic in x, slip walls in y. Initial conditions, pressure profile, density tanh interface, and velocity perturbation match exactly. Source term implements gravity g=1.0 in y-momentum and energy. | N/A | 45 | 20 | 15 | 15 | **95** |
| 3 | Numerical implementation using Discontinuous Galerkin Spectral Element Method (DGSEM) in 2D plus time for Compressible Euler Equations with gravity source term (gamma=1.4, g=[0,1.0]). Domain [0,0.25]x[0,1.0], polydeg=4, LobattoLegendreBasis, surface_flux=flux_hll, volume_integral=VolumeIntegralShockCapturingHG (IndicatorHennemannGassner on density, flux_ranocha DG, flux_lax_friedrichs FV), AMR with IndicatorLohner on density and ControllerThreeLevel (max_level=7), tspan=(0.0,3.0), time integrator RDPK3SpFSAL49 with CFL=0.5 StepsizeCallback. | N/A | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves multiple files to data_npz/time_XXXXXXXXX.npz every dt_save=0.1 (condition: rem(t,0.1)<1e-5 or end): keys 't' (scalar), 'x' (vector of node x-coords), 'y' (vector of node y-coords), 'rho' (vector), 'u' (vector), 'v' (vector), 'p' (vector) from PlotData2D cons2prim. Also saves Trixi solution files to trixi_output/ every 20 steps (primitive vars). Generates summary_all.png (1x4 density scatter plots, no colorbar). |
| 1 | Saves multiple .npz files in data_npz/ (e.g., time_XXXXX.npz) with keys: 't' (scalar), 'x' (1D vector of length N_points), 'y' (1D vector of length N_points), 'rho' (1D vector of length N_points), 'u' (1D vector of length N_points), 'v' (1D vector of length N_points), 'p' (1D vector of length N_points). Also saves Trixi solution files (HDF5 format) in trixi_output/ via SaveSolutionCallback (interval=20, primitive variables). Generates summary_all.png (1x4 density plots at t≈0,1,2,3). |
| 2 | Saves Trixi .h5 solution files to 'trixi_output/' via SaveSolutionCallback (interval=10, primitive variables cons2prim, including initial solution). Saves NPZ files to 'data_npz/time_XXXXXXXXX.npz' (every ~0.1 or at end) containing keys: 't' (scalar), 'x' (1D vector of nodal x-coordinates), 'y' (1D vector of nodal y-coordinates), 'rho' (1D vector), 'u' (1D vector), 'v' (1D vector), 'p' (1D vector) all same length from PlotData2D. Saves 'summary_all.png' (1x4 layout of density scatter plots at t~0,1,2,3). |
| 3 | Saves full solution state via SaveSolutionCallback (dt=1.0, save_initial=true) to trixi_output/ (likely .h5 files at t=0.0,1.0,2.0,3.0 with primitive variables cons2prim on adaptive mesh). Additionally saves data_npz/time_XXXXX.npz at integer times t=0.0,1.0,2.0,3.0 containing keys: 't' (scalar), 'x' (N,), 'y' (N,), 'rho' (N,), 'u' (N,), 'v' (N,), 'p' (N,) where N=length of PlotData2D points (varies with AMR/polydeg). Saves summary_all.png (1x4 density plots, colorbar=false). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//CompressibleEuler_Periodic_Reflective_SmoothedInterface_ICs/2025_12_06_1028/` |
| 1 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//CompressibleEuler_Periodic_Reflective_SmoothedInterface_ICs/2025_12_06_1037/` |
| 2 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//CompressibleEuler_Periodic_Reflective_SmoothedInterface_ICs/2025_12_06_1048/` |
| 3 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//CompressibleEuler_Periodic_Reflective_SmoothedInterface_ICs/2025_12_06_1059/` |


---
