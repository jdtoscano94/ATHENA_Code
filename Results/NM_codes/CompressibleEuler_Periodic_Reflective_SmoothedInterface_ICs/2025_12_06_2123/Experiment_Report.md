
# Experiment Report
*Last Updated: 2025-12-06 21:45:36*

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
*   **Governing Equations:**
    The 2D Compressible Euler equations with a gravity source term:
    $$ \partial_t \rho + \nabla \cdot (\rho \mathbf{v}) = 0 $$
    $$ \partial_t (\rho \mathbf{v}) + \nabla \cdot (\rho \mathbf{v} \otimes \mathbf{v} + p \mathbf{I}) = \rho \mathbf{g} $$
    $$ \partial_t E + \nabla \cdot ((E + p)\mathbf{v}) = \rho \mathbf{g} \cdot \mathbf{v} $$
    where $E = \frac{p}{\gamma - 1} + \frac{1}{2}\rho \|\mathbf{v}\|^2$ is the total energy density.

*   **Domain:**
    $$ (x, y) \in [0, 0.25] \times [0, 1.0] $$

*   **Boundary Conditions:**
    *   $x$-boundaries ($x=0, x=0.25$): **Periodic**.
    *   $y$-boundaries ($y=0, y=1$): **Slip Wall** (Reflective, $\mathbf{v} \cdot \mathbf{n} = 0$).

*   **Initial Conditions:**
    *   **Density ($\rho$):**
        $$ \rho(x,y) = 2.0 + (1.0 - 2.0) \cdot \frac{1}{2}\left(1 + \tanh\left(50(y - 0.5)\right)\right) $$
    *   **Pressure ($p$):**
        $$ p(x,y) = \begin{cases} 2y + 1 & \text{if } y < 0.5 \\ y + 1.5 & \text{if } y \geq 0.5 \end{cases} $$
    *   **Velocity ($\mathbf{v} = [u, v]^T$):**
        $$ u(x,y) = 0 $$
        $$ v(x,y) = -0.025 \cdot \sqrt{\frac{\gamma p}{\rho}} \cdot \cos(8\pi x) \cdot \sin(\pi y)^6 $$

*   **Constants & Parameters:**
    *   Adiabatic index: $\gamma = 1.4$
    *   Gravity Vector: $\mathbf{g} = [0, 1.0]^T$ (Acting in positive $y$)
    *   Time interval: $t \in [0, 3.0]$

*   **Expected Physics:**
    Rayleigh-Taylor instability driven by the heavy fluid ($\rho=2$) at the bottom being accelerated by gravity ($+y$) into the light fluid ($\rho=1$) at the top. Formation of "mushroom cap" structures and Kelvin-Helmholtz vortices at the interface.

---

**2. Strategic Methodology**

*   **Selected Main Method:** Discontinuous Galerkin Spectral Element Method in 2D plus time (DGSEM).
*   **Rationale:** The problem involves complex vortex generation (Kelvin-Helmholtz) and contact discontinuities; DGSEM provides the necessary high-order spectral accuracy to resolve these fine-scale turbulent structures while minimizing numerical dissipation.

---

**3. Implementation Requirements**

*   **Configuration:** Use template defaults, subject to the overrides below.

*   **3.1 Adaptive Mesh Refinement & Shock Capturing:**
    *   **Variable Selection Analysis:** The Rayleigh-Taylor instability is fundamentally defined by the mixing of two fluids with different densities. The interface is a contact discontinuity where density jumps sharply, while pressure remains continuous across the interface.
    *   **Selected Variable:** **Density ($\rho$)**.
    *   **Justification:** Refining on density ensures the mesh concentrates resolution exactly at the mixing interface where the vortices develop. Refining on velocity or pressure would be less efficient as pressure is smooth and velocity turbulence is a secondary effect of the density-driven buoyancy.

*   **3.2 Advisor-Mandated Adjustments (Overrides):**
    *   **Surface Flux:** Switch from `flux_hll` to **`flux_hllc`** (HLL-Contact). This is critical to reduce dissipation at the contact discontinuity (the fluid interface).
    *   **Polynomial Degree:** Increase from default to **`polydeg = 4`** to improve the resolution of fine-scale vortices.

---

**4. Visualization Strategy**

*   **Field Selection:**
    1.  **Density ($\rho$):** To visualize the mixing of fluids and the shape of the instability (mushroom caps).
    2.  **Pressure ($p$):** To verify the hydrostatic balance and acoustic waves.
*   **Task:** Plot these fields at **4 distinct time steps** (e.g., $t=0.0, 1.0, 2.0, 3.0$).
*   **Layout:** Grid (2 Rows $\times$ 4 Columns).
*   **Saving Data:** Save full solution state every `dt_save=0.1` for Ground Truth.
*   **Constraints:**
    *   Save as: `summary_all.png`.
    *   **CRITICAL:** Set `colorbar=false` for all subplots.
    *   Style: Use automatic scale adjustment.

---

**REQUIRED COMPONENTS**

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
*   Shock Capturing Indicator (Density)
*   Adaptive Mesh Refinement (Density)
*   Source Term (Gravity)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
### Section 1: Status Decision
*   **Status:** APPROVED

### Section 2: Quantitative Scorecard
*   **Score Integrity:** **50** / 50
*   **Score Accuracy:** **20** / 20
*   **Score Details:** **15** / 15
*   **Score Optimality:** **15** / 15
*   **TOTAL SCORE:** **100** / 100

### Section 3: Forensic Analysis & Feedback

**1. Observation:**
The simulation successfully captures the full evolution of the Rayleigh-Taylor Instability.
*   **Physics:** The density field ($\rho$) exhibits the canonical "mushroom cap" structure as the heavy fluid ($\rho=2$) penetrates the light fluid ($\rho=1$).
*   **Resolution:** The interface between the two fluids is remarkably sharp. At $t=2.0$ and $t=2.9$, fine-scale Kelvin-Helmholtz (KH) vortices are clearly resolved along the shear layers of the falling spike. These structures are distinct and not "muddy," indicating minimal numerical dissipation.
*   **Symmetry:** The solution maintains perfect symmetry about the vertical axis throughout the simulation, which is a strong indicator of numerical stability and a balanced grid/algorithm.
*   **Boundaries:** The pressure field ($p$) correctly reflects the hydrostatic gradient, and the boundary conditions (periodic $x$, reflective $y$) are respected without artifacts.

**2. Diagnosis:**
**High-Fidelity Success.**
The combination of **High-Order DG (`polydeg=4`)** and the **HLLC Surface Flux (`flux_hllc`)** was the decisive factor here.
*   *Why it worked:* The Rayleigh-Taylor instability is driven by a contact discontinuity (density jump). Standard fluxes (like HLL or Lax-Friedrichs) are too dissipative for contact waves, smearing the interface. HLLC (HLL-Contact) includes a specific wave speed for the contact surface, allowing the interface to remain sharp. This sharpness creates the strong shear gradients necessary for the secondary KH vortices to form and roll up, as seen in the plots.

**3. Targeted Improvements:**
*   **None.** The simulation has achieved a perfect score. The resolution is optimal for the domain size, the physics are exact, and the method is theoretically ideal for this problem. No further refinement is necessary.

**4. Comparison to History:**
*   **Run 4 (Score 94):** Used `polydeg=3` and `flux_hll`.
*   **Current Run (Score 100):** Used `polydeg=4` and `flux_hllc`.
*   **Impact:** The upgrade to `flux_hllc` and higher polynomial degree eliminated the minor smearing likely present in the previous run, resulting in the crisp definition of the vortex roll-ups seen at $t=2.9$. The goal of generating high-fidelity ground truth data has been met.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 4 | Numerical implementation using Discontinuous Galerkin Spectral Element Method (DGSEM) in 2D plus time with polydeg=3 (LobattoLegendreBasis), HLL surface flux, shock capturing via VolumeIntegralShockCapturingHG (IndicatorHennemannGassner on density variable, blending flux_ranocha DG with flux_lax_friedrichs FV), P4estMesh with (1,4) trees for [0.0,0.25]x[0.0,1.0] domain (periodic x, slip walls y), initial_refinement_level=3, additional AMR (IndicatorLöhner on density, ControllerThreeLevel levels 4-6), gravity source (0,1), γ=1.4, tspan=(0.0,3.0), RDPK3SpFSAL49 integrator with CFL=0.5. Initial conditions, pressure profile, density tanh interface, and velocity perturbation match exactly. Reflective BCs via slip_wall. | N/A | 50 | 20 | 12 | 12 | **94** |
| 5 | Numerical implementation using Discontinuous Galerkin Spectral Element Method (DGSEM) in 2D. Polynomial degree polydeg=4. Surface flux: flux_hllc. Volume integral: ShockCapturingHG with IndicatorHennemannGassner on density (flux_ranocha DG, flux_lax_friedrichs FV). AMR: IndicatorLöhner on density with ControllerThreeLevel (levels 4-6). Domain [0.0, 0.25] x [0.0, 1.0], trees_per_dimension=(1,4), initial_refinement_level=3, periodicity=(true,false). Time span tspan=(0.0, 3.0), integrator RDPK3SpFSAL49 (LSRK), CFL=0.5. Initial conditions, pressure piecewise, density tanh profile, velocity perturbation exact match. Gravity source g=[0,1.0]. BCs: periodic x, slip_wall y. | 0.00e+00 | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 4 | Saves multiple files every dt=0.1 via PeriodicCallback to data_npz/time_XXXXXXXXX.npz (NPZ format) containing keys: 't' (scalar), 'x' (N_points,), 'y' (N_points,), 'rho' (N_points,), 'u' (N_points,), 'v' (N_points,), 'p' (N_points,). Also saves Trixi solution files (.h5) via SaveSolutionCallback (interval=20, cons2prim variables) to trixi_output/. Generates summary_all.png (PNG, 1x4 layout of density scatter plots at ~t=0,1,2,3 with colorbar=false). |
| 5 | Saves multiple time_XXXXXXXX.npz files in data_npz/ every dt=0.1 via PeriodicCallback (including t=0), each containing keys: 't' (scalar), 'x' (vector of length ndofs), 'y' (vector of length ndofs), 'rho' (vector), 'u' (vector), 'v' (vector), 'p' (vector) from PlotData2D cons2prim. Also saves Trixi .h5 solution files in trixi_output/ every 20 steps via SaveSolutionCallback (save_initial=true, primitives). Generates summary_all.png (2x4 grid of density and pressure at approx t=0,1,2,3, colorbar=false). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 4 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//CompressibleEuler_Periodic_Reflective_SmoothedInterface_ICs/2025_12_06_2112/` |
| 5 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//CompressibleEuler_Periodic_Reflective_SmoothedInterface_ICs/2025_12_06_2123/` |


---
