
# Experiment Report
*Last Updated: 2025-12-06 21:20:39*

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
    $$ \partial_t \rho + \nabla \cdot (\rho \mathbf{v}) = 0 $$
    $$ \partial_t (\rho \mathbf{v}) + \nabla \cdot (\rho \mathbf{v} \otimes \mathbf{v} + p \mathbf{I}) = \rho \mathbf{g} $$
    $$ \partial_t E + \nabla \cdot ((E + p)\mathbf{v}) = \rho \mathbf{g} \cdot \mathbf{v} $$
    where $E = \frac{p}{\gamma - 1} + \frac{1}{2}\rho \|\mathbf{v}\|^2$ and $\mathbf{g} = (0, 1.0)^T$.

*   **Domain:** $\Omega = [0, 0.25] \times [0, 1.0]$
*   **Boundary Conditions:**
    *   $x = 0, 0.25$: Periodic
    *   $y = 0, 1.0$: Reflective (Slip wall, $\mathbf{v} \cdot \mathbf{n} = 0$)
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
    *   Gravity $\mathbf{g} = [0.0, 1.0]^T$
*   **Expected Physics:** Rayleigh-Taylor Instability driven by "upward" gravity acting on a heavy fluid at the bottom pushing into a light fluid at the top. Formation of mushrooms/plumes at the interface.

**2. Strategic Methodology**

*   **Selected Main Method:** Discontinuous Galerkin Spectral Element Method in 2D plus time
*   **Rationale:** The problem involves the evolution of a complex interface (contact discontinuity) and turbulent mixing structures which require the high-order accuracy and low dissipation of DGSEM, stabilized by shock capturing.

**3. Implementation Requirements**

*   **Configuration:** Use template defaults.
*   **3.1 Adaptive Mesh Refinement & Shock Capturing:**
    *   **Variable Selection Analysis:** The Rayleigh-Taylor instability is defined by the mixing of two fluids with different densities. The interface is a contact discontinuity where density jumps sharply, while pressure remains relatively smooth (hydrostatic balance). Velocity becomes turbulent and noisy later.
    *   **Selected Variable:** **Density** ($\rho$).
    *   **Justification:** Density provides the clearest signal of the interface location and the evolving mixing layer structures, ensuring stabilization is applied exactly where the gradients are steepest without damping the turbulent velocity field unnecessarily.
*   **3.2 Advisor-Mandated Adjustments:** None.

**4. Visualization Strategy**

*   **Field Selection:** **Density** ($\rho$).
    *   *Reasoning:* Density is the primary variable distinguishing the two fluids; it visualizes the "mushroom" structures of the instability most clearly.
*   **Task:** Plot Density at 4 distinct time steps (e.g., $t=0.0, 1.0, 2.0, 3.0$).
*   **Layout:** 1 Row (Density) $\times$ 4 Columns (Time).
*   **Saving Data:** Save full solution state every `dt_save=0.1`.
*   **Constraints:**
    *   Save as `summary_all.png`.
    *   **DISABLE COLORBARS** (`colorbar=false`).
    *   Do not use fixed scales; allow auto-scaling.

**REQUIRED COMPONENTS**

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
Shock Capturing Indicator (Density)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
### Section 1: Status Decision
*   **Status:** REJECTED_FOR_REFINEMENT

### Section 2: Quantitative Scorecard
*   **Score Integrity:** **50** (Perfect execution. Code ran to completion, plots generated successfully, no stability warnings.)
*   **Score Accuracy:** **20** (Physics are excellent. The Rayleigh-Taylor instability develops correctly with the expected symmetry: a central rising plume of light fluid and falling spikes of heavy fluid at the periodic boundaries. The growth rate and topology match theoretical expectations for the compressible regime.)
*   **Score Details:** **12** (High quality. The Kelvin-Helmholtz (KH) shear instabilities on the side of the mushroom cap are visible. However, the interface shows signs of numerical diffusion—the contact discontinuity is slightly "soft" rather than razor-sharp, likely due to the choice of the HLL flux.)
*   **Score Optimality:** **12** (DGSEM is appropriate. However, the use of `flux_hll` is sub-optimal for this specific problem. RTI is driven by a contact discontinuity (density interface), and HLL is known to be overly dissipative for contact waves compared to HLLC.)
*   **TOTAL SCORE:** **94**

### Section 3: Forensic Analysis & Feedback

**1. Observation**
*   **Physics:** The simulation correctly captures the non-linear evolution of the instability. At $t=2.0$ and $t=2.9$, the characteristic "mushroom cap" is well-formed.
*   **Artifacts:** The secondary Kelvin-Helmholtz vortices (the "rolls" on the side of the plume) are present but appear slightly blurred or "chunky." This indicates that while the AMR is capturing the feature, the numerical dissipation is smoothing out the finest scales of turbulence.
*   **Log Analysis:** The simulation completed successfully without time-step crashes. The AMR statistics show active refinement (`~AMR~` taking significant time), confirming the grid is adapting to the interface.

**2. Diagnosis**
*   **Primary Issue:** **Dissipative Surface Flux.** The simulation uses the **HLL (Harten-Lax-van Leer)** flux. While HLL is extremely robust for strong shocks, it ignores the contact wave (the middle wave in the Riemann problem). In the Rayleigh-Taylor instability, the interface between the two fluids is a contact discontinuity. HLL smears this interface, suppressing the fine-scale mixing and small vortices.
*   **Secondary Issue:** **Polynomial Resolution.** `polydeg=3` is a standard starting point, but for turbulence and mixing, a slightly higher order (e.g., 4 or 5) often yields significantly better resolution per degree of freedom (spectral accuracy) for the vortices.

**3. Targeted Improvements**
*   **For Optimality (CRITICAL):** Switch the surface flux from `flux_hll` to **`flux_hllc`** (HLL-Contact). HLLC explicitly resolves the contact wave, which will significantly sharpen the density interface and allow finer KH vortices to form without artificial damping.
*   **For Details:** Increase the polynomial degree from `polydeg=3` to **`polydeg=4`**. This provides a boost in spatial accuracy (5th order) to better resolve the turbulent structures, with a manageable increase in computational cost.

**Quantitative Error Statement:**
Quantitative error is not applicable as no ground truth is available for this forward simulation of an instability. Accuracy is assessed via physical plausibility and feature resolution.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 4 | Numerical implementation using Discontinuous Galerkin Spectral Element Method (DGSEM) in 2D plus time with polydeg=3 (LobattoLegendreBasis), HLL surface flux, shock capturing via VolumeIntegralShockCapturingHG (IndicatorHennemannGassner on density variable, blending flux_ranocha DG with flux_lax_friedrichs FV), P4estMesh with (1,4) trees for [0.0,0.25]x[0.0,1.0] domain (periodic x, slip walls y), initial_refinement_level=3, additional AMR (IndicatorLöhner on density, ControllerThreeLevel levels 4-6), gravity source (0,1), γ=1.4, tspan=(0.0,3.0), RDPK3SpFSAL49 integrator with CFL=0.5. Initial conditions, pressure profile, density tanh interface, and velocity perturbation match exactly. Reflective BCs via slip_wall. | N/A | 50 | 20 | 12 | 12 | **94** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 4 | Saves multiple files every dt=0.1 via PeriodicCallback to data_npz/time_XXXXXXXXX.npz (NPZ format) containing keys: 't' (scalar), 'x' (N_points,), 'y' (N_points,), 'rho' (N_points,), 'u' (N_points,), 'v' (N_points,), 'p' (N_points,). Also saves Trixi solution files (.h5) via SaveSolutionCallback (interval=20, cons2prim variables) to trixi_output/. Generates summary_all.png (PNG, 1x4 layout of density scatter plots at ~t=0,1,2,3 with colorbar=false). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 4 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//CompressibleEuler_Periodic_Reflective_SmoothedInterface_ICs/2025_12_06_2112/` |


---
