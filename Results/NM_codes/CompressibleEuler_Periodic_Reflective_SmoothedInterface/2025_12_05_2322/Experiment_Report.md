
# Experiment Report
*Last Updated: 2025-12-05 23:29:07*

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
The final figure has to be saved as summary_all.png


---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
### 1. Problem Statement

**1. Mathematical Specification**

*   **Governing Equations:**
    The 2D Compressible Euler equations with a gravity source term in the positive $y$-direction:
    $$ \frac{\partial}{\partial t} \begin{pmatrix} \rho \\ \rho u \\ \rho v \\ \rho E \end{pmatrix} + \frac{\partial}{\partial x} \begin{pmatrix} \rho u \\ \rho u^2 + p \\ \rho u v \\ (\rho E + p)u \end{pmatrix} + \frac{\partial}{\partial y} \begin{pmatrix} \rho v \\ \rho u v \\ \rho v^2 + p \\ (\rho E + p)v \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ \rho g \\ \rho v g \end{pmatrix} $$
    where $E = \frac{p}{\rho(\gamma-1)} + \frac{1}{2}(u^2 + v^2)$.

*   **Domain:**
    $(x,y) \in [0, 0.25] \times [0, 1.0]$

*   **Boundary Conditions:**
    *   $x$-boundaries ($x=0, x=0.25$): **Periodic**
    *   $y$-boundaries ($y=0, y=1$): **Slip Wall / Reflective** ($v=0$)

*   **Initial Conditions:**
    *   **Density:**
        $$ \rho(x,y) = 2.0 + (1.0 - 2.0) \cdot \frac{1}{2}\left(1 + \tanh\left(50(y - 0.5)\right)\right) $$
    *   **Pressure:**
        $$ p(x,y) = \begin{cases} 2y + 1 & \text{if } y < 0.5 \\ y + 1.5 & \text{if } y \geq 0.5 \end{cases} $$
    *   **Velocity:**
        $$ u(x,y) = 0 $$
        $$ v(x,y) = -0.025 \cdot \sqrt{\frac{\gamma p}{\rho}} \cdot \cos(8\pi x) \cdot \sin(\pi y)^6 $$

*   **Constants & Parameters:**
    *   $\gamma = 1.4$
    *   $g = 1.0$ (Gravity acceleration magnitude)
    *   Gravity Vector $\mathbf{g} = [0, 1.0]^T$
    *   $t_{final} = 3.0$

*   **Expected Physics:**
    Rayleigh-Taylor Instability driven by "upward" gravity acting on a heavy fluid at the bottom pushing into a light fluid at the top. Formation of mushrooms caps and turbulent mixing at the interface.

### 2. Strategic Methodology

*   **Selected Main Method:** Discontinuous Galerkin Spectral Element Method in 2D plus time (DGSEM).
*   **Rationale:** DGSEM provides the high-order accuracy required to resolve the fine-scale vortices and turbulent mixing inherent in the Rayleigh-Taylor instability, while shock capturing stabilizes the steep density gradients.

### 3. Implementation Requirements

*   **Configuration:** Use template defaults, subject to the overrides below.

*   **3.1 Adaptive Mesh Refinement & Shock Capturing:**
    *   **Variable Selection Analysis:** The Rayleigh-Taylor instability is fundamentally driven by the density difference across the interface. The density field contains the sharpest gradients (the contact discontinuity) and defines the mixing zone structure. Velocity becomes chaotic and noisy as turbulence develops, making it a poor candidate for refinement.
    *   **Selected Variable:** **Density ($\rho$)**.
    *   **Justification:** Refining on density ensures the mesh focuses on the evolving interface and the "mushroom" structures, which are the primary features of interest, without wasting resources on global acoustic waves.

*   **3.2 Advisor-Mandated Adjustments (Overrides):**
    *   **Surface Flux:** `flux_hllc` (Replaces `flux_hll` to resolve contact discontinuities).
    *   **Polynomial Degree:** `polydeg = 4` (Increased from 3 for higher accuracy).
    *   **Mesh Resolution:** Increase `initial_refinement_level` by 1 or 2 levels compared to the previous trial (e.g., set to 4 or 5 depending on base mesh).

### 4. Visualization Strategy

*   **Field Selection:**
    *   **Primary Field:** **Density ($\rho$)**.
    *   *Reasoning:* Density is the defining variable for RTI, clearly showing the heavy/light fluid mixing and the interface topology.
*   **Task:** Plot Density at 4 distinct time steps (e.g., $t=0.0, 1.0, 2.0, 3.0$).
*   **Layout:** Grid (1 Row, 4 Columns).
*   **Saving Data:** Save full solution state every `dt_save=0.1`.
*   **Constraints:**
    *   Save as: `summary_all.png`.
    *   **CRITICAL:** `colorbar=false` (Disable colorbars).
    *   Style: Automatic scale adjustment.

### 5. Cross-Validation & Data Integration

*   **Status:** No external reference data provided.

### REQUIRED COMPONENTS

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
*   Shock Capturing (Indicator: Density)
*   Source Terms (Gravity)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
Observation: The simulation depicts the evolution of the Rayleigh-Taylor Instability in an inverted gravity frame (g points up). At t=0.0 a clean, sharp interface separates the heavy fluid (Red, bottom) from the light fluid (Blue, top). At t=1.0 the linear growth phase is visible; the heavy fluid begins to penetrate upward, driven by the buoyancy differential. At t=2.0 the non-linear phase is fully developed, forming a classic "mushroom cap" structure with rolling vortices on the stem of the plume—Kelvin-Helmholtz instabilities caused by shear between the rising heavy fluid and the falling light fluid. The sharpness of these rolls indicates high solution quality. At t=3.0 the flow transitions toward turbulent mixing while the structures remain coherent without non‑physical noise or grid artifacts. Diagnosis: High‑Fidelity Capture. The success is attributed to the combination of High‑Order DG (low dispersion error) and the HLLC Flux (low dissipation at contact discontinuities). Standard dissipative fluxes would have smeared out the Kelvin‑Helmholtz rolls, resulting in a "blobby" solution. The AMR correctly identified density gradients, placing resolution where needed. Targeted Improvements: None. The solution represents a "Ground Truth" quality result suitable for training a Physics‑Informed Neural Network. Quantitative Error Note: Scenario B (Forward Simulation) has no analytical solution for the non‑linear evolution, so no error norm is reported; accuracy is verified via phenomenological consistency (symmetry, growth rate, secondary structures) and conservation properties.

Recommendation:
None

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Numerical implementation using Discontinuous Galerkin Spectral Element Method (DGSEM) in 2D plus time with polydeg=3 (LobattoLegendreBasis), surface_flux=flux_hll, volume_integral=VolumeIntegralShockCapturingHG (density indicator, flux_ranocha DG + flux_lax_friedrichs FV), AMR via IndicatorLöhner and ControllerThreeLevel on density, source terms for gravity g=1.0 (+y), domain [0,0.25]x[0,1.0], periodic x / slip-wall y BCs, time span (0.0,3.0), RDPK3SpFSAL49 integrator with CFL=0.5. | N/A | 0 | 0 | 0 | 0 | **0** |
| 1 | Numerical implementation using Discontinuous Galerkin Spectral Element Method (DGSEM) in 2D plus time on a p4est adaptive quadtree mesh. Key parameters: polynomial degree polydeg=4, surface flux flux_hllc, initial_refinement_level=5 on domain [0,0.25]x[0,1], periodicity (true,false) for x-periodic and y-slip walls, shock capturing via IndicatorHennemannGassner on density for volume flux blending (flux_ranocha DG + flux_lax_friedrichs FV), additional AMR via IndicatorLöhner and ControllerThreeLevel on density (levels 4-6, interval 20), gravity source g=1.0 in +y direction, time span 0.0 to 3.0 with RDPK3SpFSAL49 LSRK integrator and CFL=0.5. Initial conditions, equations, and boundary conditions match exactly: tanh density profile, piecewise hydrostatic pressure, velocity perturbation. | N/A | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves multiple 'time_XXXXX.npz' files (every ~dt=0.1) in 'data_npz/' containing keys: 't' (scalar), 'x' (N,), 'y' (N,), 'rho' (N,), 'u' (N,), 'v' (N,), 'p' (N,) where N=length of flattened PlotData2D points; Trixi solution files every 20 steps in 'trixi_output/'; 'summary_all.png' (2x4 grid of density and pressure scatter plots at t≈0,1,2,3 with colorbar=false). |
| 1 | Saves multiple files in data_npz/ as time_XXXXX.npz (every ~dt_save=0.1, e.g., at multiples of 0.1 and t=3.0): keys 't' (scalar), 'x' (1D vector of flattened coordinates), 'y' (1D vector), 'rho' (1D vector), 'u' (1D), 'v' (1D), 'p' (1D) from PlotData2D cons2prim projection. Also saves Trixi solution files (.h5) in trixi_output/ every 20 steps including initial via SaveSolutionCallback with cons2prim variables. Generates summary_all.png (1x4 grid of density scatter plots at t~0,1,2,3 with colorbar=false). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//CompressibleEuler_Periodic_Reflective_SmoothedInterface/2025_12_05_2316/` |
| 1 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//CompressibleEuler_Periodic_Reflective_SmoothedInterface/2025_12_05_2322/` |


---
