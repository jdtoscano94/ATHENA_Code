
# Experiment Report
*Last Updated: 2025-12-08 09:41:50*

## User Request

Let's solve the 2D Compressible Navier-Stokes equations for the Rayleigh-Taylor Instability.

**Physics Parameters:**
* $\gamma = 1.4$
* Viscosity $\mu = 0.0001$ (Prandtl number $Pr = 0.72$)
* Gravity $g = 1.0$ acting in the positive $y$-direction.

**Domain:**
$(x,y) \in [0, 0.25] \times [0, 1.0]$
Time interval: $t \in [0, 3.0]$

**Boundary Conditions:**
* $x$-direction: Periodic
* $y$-direction: **No-slip adiabatic walls** at $y=0$ and $y=1$ (Velocity is zero).

**Initial Conditions:**
The domain is split at $y=0.5$.
* **Density ($\rho$):** A smoothed interface between $\rho_{heavy} = 2.0$ (bottom) and $\rho_{light} = 1.0$ (top).
    $$ \rho(y) = \rho_{heavy} + (\rho_{light} - \rho_{heavy}) \cdot \frac{1}{2}\left(1 + \tanh\left(50(y - 0.5)\right)\right) $$
* **Pressure ($p$):** Hydrostatic balance ($\frac{dp}{dy} = \rho g$).
    $$ p(y) = \begin{cases} 2y + 1 & \text{if } y < 0.5 \\ y + 1.5 & \text{if } y \geq 0.5 \end{cases} $$
* **Velocity:** Initialized with zero horizontal velocity ($u=0$) and a perturbation in vertical velocity ($v$):
    $$ v(x,y) = -0.025 \cdot c \cdot \cos(8\pi x) \cdot \sin(\pi y)^6 $$
    Where $c = \sqrt{\gamma p / \rho}$ is the local speed of sound.

**Output:**
The final plot must be saved as `summary_all.png` and should include snapshots of several timesteps to visualize the mixing.


---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
### 1. Problem Statement

**1. Mathematical Specification**

*   **Governing Equations:** Compressible Navier-Stokes Equations with Gravity Source Term
    $$ \partial_t \rho + \nabla \cdot (\rho \mathbf{v}) = 0 $$
    $$ \partial_t (\rho \mathbf{v}) + \nabla \cdot (\rho \mathbf{v}\mathbf{v}^T + p\mathbf{I}) = \nabla \cdot \tau + \rho \mathbf{g} $$
    $$ \partial_t E + \nabla \cdot ((E+p)\mathbf{v}) = \nabla \cdot (\tau \cdot \mathbf{v} - \mathbf{q}) + \rho \mathbf{g} \cdot \mathbf{v} $$
    Where $\tau$ is the viscous stress tensor and $\mathbf{q}$ is the heat flux.

*   **Domain:** $(x,y) \in [0, 0.25] \times [0, 1.0]$
*   **Boundary Conditions:**
    *   $x$-faces ($x=0, x=0.25$): Periodic
    *   $y$-faces ($y=0, y=1$): No-slip adiabatic walls ($\mathbf{v}=\mathbf{0}, \mathbf{q}\cdot\mathbf{n}=0$)
*   **Initial Conditions:**
    *   **Density:**
        $$ \rho(x,y) = 2.0 + (1.0 - 2.0) \cdot \frac{1}{2}\left(1 + \tanh\left(50(y - 0.5)\right)\right) $$
    *   **Velocity:**
        $$ u(x,y) = 0, \quad v(x,y) = -0.025 \sqrt{\frac{\gamma p}{\rho}} \cos(8\pi x) \sin(\pi y)^6 $$
    *   **Pressure:**
        $$ p(x,y) = \begin{cases} 2y + 1 & \text{if } y < 0.5 \\ y + 1.5 & \text{if } y \geq 0.5 \end{cases} $$
*   **Constants & Parameters:**
    *   $\gamma = 1.4$
    *   $\mu = 0.0001$
    *   $Pr = 0.72$
    *   Gravity Vector $\mathbf{g} = [0, 1.0]^T$
*   **Expected Physics:** Rayleigh-Taylor Instability driven by gravity acting in the positive y-direction on the heavy fluid at the bottom. Formation of mushroom-shaped plumes and Kelvin-Helmholtz vortices at the interface.

### 2. Strategic Methodology

*   **Selected Main Method:** Discontinuous Galerkin Spectral Element Method in 2D plus time.
*   **Rationale:** The problem involves complex boundary conditions (walls) and requires high-order accuracy to resolve the fine-scale Kelvin-Helmholtz vortices that develop on the fluid interface, which DGSEM handles efficiently.

### 3. Implementation Requirements

*   **Configuration:** Use template defaults, with the following **Advisor-Mandated Adjustments**:
    *   **Polynomial Degree:** Set `polydeg = 5` (Increased from 3).
    *   **Mesh Refinement:** Set `initial_refinement = 4` (Increased from 3).
    *   **AMR Controller:** Set `levels = 4` (min) to `7` (max).

*   **3.1 Adaptive Mesh Refinement & Shock Capturing:**
    *   **Variable Selection:** **Density ($\rho$)**
    *   **Justification:** The physics is dominated by the mixing of two immiscible-like fluids. The density gradient defines the interface structure (the "signal"). Velocity becomes globally turbulent and noisy, which would trigger wasteful refinement in smooth regions. Pressure is hydrostatically dominated and relatively smooth. Therefore, refining on Density maximizes the resolution of the instability features.

### 4. Visualization Strategy

*   **Field Selection:** **Density ($\rho$)**
    *   *Reasoning:* Density is the primary variable distinguishing the two fluids and clearly visualizes the mixing process and the instability morphology.
*   **Task:** Plot Density at 4 distinct time steps (e.g., $t=0.0, 1.0, 2.0, 3.0$).
*   **Layout:** Grid (1 Row, 4 Columns).
*   **Saving Data:** Save full solution state at `dt_save=0.1`.
*   **Constraints:**
    *   Save as: `summary_all.png`
    *   **CRITICAL:** `colorbar=false` (Disable colorbars).
    *   Style: Automatic scaling.

### 5. Cross-Validation & Data Integration

*   **Trigger:** None (No external data provided).

### REQUIRED COMPONENTS

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
*   Shock Capturing Indicator (on Density)
*   Adaptive Mesh Refinement (on Density)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
### Section 1: Status Decision
* **Status:** APPROVED

### Section 2: Quantitative Scorecard
* **Score Integrity:** **50** / 50
* **Score Accuracy:** **20** / 20
* **Score Details:** **15** / 15
* **Score Optimality:** **15** / 15
* **TOTAL SCORE:** **100** / 100

### Section 3: Forensic Analysis & Feedback

**1. Observation:**
The simulation successfully captures the full evolution of the Rayleigh-Taylor Instability (RTI) from the linear growth phase ($t=1.0$) to the non-linear mixing phase ($t=3.0$).
*   **Symmetry:** The solution maintains perfect symmetry about the vertical centerline ($x=0.125$), indicating that the numerical scheme preserves the symmetry of the governing equations and initial conditions without introducing biased numerical noise.
*   **Interface Resolution:** The density interface remains distinct throughout the simulation. The transition from heavy (red) to light (blue) fluid is sharp, consistent with the initial `tanh` smoothing and the physical viscosity.
*   **Fine-Scale Features:** Crucially, the simulation resolves the **Kelvin-Helmholtz (KH) vortices** rolling up along the shear layers of the mushroom cap stem (visible at $t=2.0$ and $t=3.0$). This is a hallmark of high-fidelity resolution; lower-order or highly dissipative schemes often smear these features out entirely.

**2. Diagnosis:**
**High-Fidelity Physical Resolution.**
The combination of High-Order DGSEM (`polydeg=5`) and Adaptive Mesh Refinement (AMR Levels 4-7) has successfully resolved the multiscale nature of the RTI.
*   The **Polynomial Degree (N=5)** minimizes dispersion errors, allowing the secondary KH instabilities to form naturally rather than being damped out.
*   The **Physical Viscosity ($\mu=0.0001$)** provides necessary regularization, preventing the non-physical grid-scale noise often seen in inviscid Euler simulations of RTI, resulting in a clean, physically consistent flow field.
*   **Shock Capturing:** While primarily for shocks, the blending method here successfully stabilized the steep density gradients at the interface without excessive smearing, as evidenced by the survival of the vortices.

**3. Targeted Improvements:**
*   **None Required.** The solution has achieved a perfect score. The balance between computational cost (AMR) and accuracy (Polydeg=5) is optimal for generating Ground Truth data for PINN training.
*   **Next Steps:** Proceed immediately to data extraction and PINN training using this dataset.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Numerical implementation using Discontinuous Galerkin Spectral Element Method (DGSEM) in 2D plus time for Compressible Navier-Stokes with gravity source term. Polydeg=3, LobattoLegendreBasis, HLL surface flux, shock capturing via IndicatorHennemannGassner on density with blending (flux_ranocha DG + flux_lax_friedrichs FV), AMR via IndicatorLöhner on density with ControllerThreeLevel (levels 4-6). Domain [0.0,0.25]x[0.0,1.0] with P4estMesh (trees_per_dim=(1,4), initial_refinement=3, periodicity=(true,false)). BCs: periodic x, no-slip adiabatic walls y=0/1.0. Initial conditions match tanh density profile, piecewise pressure, velocity perturbation. Source: gravity [0,1]. Time span (0.0,3.0), RDPK3SpFSAL49 integrator, CFL=0.5. | N/A | 50 | 20 | 12 | 13 | **95** |
| 1 | Numerical implementation using Discontinuous Galerkin Spectral Element Method (DGSEM) in 2D plus time for Compressible Navier-Stokes with gravity. Polynomial degree polydeg=5, initial_refinement_level=4, AMR with IndicatorLöhner and ControllerThreeLevel (base_level=4, med=5 threshold=0.1, max_level=7 threshold=0.3) on density, shock capturing with IndicatorHennemannGassner (alpha_max=1.0, alpha_min=0.001) on density using flux_ranocha (DG) and flux_lax_friedrichs (FV), surface_flux=flux_hll. Domain [0.0,0.25]x[0.0,1.0] with x-periodic and y no-slip adiabatic walls, tspan=(0.0,3.0), RDPK3SpFSAL49 integrator with CFL=0.5. | 0.00e+00 | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves Trixi solution files via SaveSolutionCallback every dt=0.1 (including initial) to 'trixi_output/' directory in native format with primitive variables (rho,u,v,p). Additionally saves NPZ files via DiscreteCallback (every 40 steps or end, plus initial) to 'data_npz/time_XXXXXXXX.npz' containing keys: 't' (scalar), 'x' (1D vector of all nodal points), 'y' (1D vector same length), 'rho' (1D vector), 'u' (1D vector), 'v' (1D vector), 'p' (1D vector). Saves 'summary_all.png' with 1x4 density plots at approximate times t=0,1,2,3. |
| 1 | Saves to trixi_output/ via SaveSolutionCallback (Trixi format, primitive variables cons2prim, every dt=0.1 including initial). Saves to data_npz/time_XXXXXXXX.npz (every 40 steps or end) containing keys: 't' (scalar), 'x' (N,), 'y' (N,), 'rho' (N,), 'u' (N,), 'v' (N,), 'p' (N,) where N is number of PlotData2D points. Saves summary_all.png (density scatter plots at t~0.0,1.0,2.0,3.0, 1x4 layout, colorbar=false). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//CompressibleNavierStokes_Periodic_NoSlipAdiabatic_SmoothedInterface/2025_12_08_0859/` |
| 1 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//CompressibleNavierStokes_Periodic_NoSlipAdiabatic_SmoothedInterface/2025_12_08_0914/` |


---
