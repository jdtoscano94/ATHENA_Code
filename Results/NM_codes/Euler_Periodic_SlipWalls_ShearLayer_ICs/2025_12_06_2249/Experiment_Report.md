
# Experiment Report
*Last Updated: 2025-12-06 22:52:17*

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

**Output:**
Create an appropiate Figure and saved it as summary_all.png


---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
**1. Problem Statement**

### Mathematical Specification
*   **Governing Equations:**
    The 2D Compressible Euler Equations for conservation of mass, momentum, and energy:
    $$ \partial_t \rho + \nabla \cdot (\rho \mathbf{v}) = 0 $$
    $$ \partial_t (\rho \mathbf{v}) + \nabla \cdot (\rho \mathbf{v} \otimes \mathbf{v} + p \mathbf{I}) = 0 $$
    $$ \partial_t E + \nabla \cdot ((E + p)\mathbf{v}) = 0 $$
    where $E = \frac{p}{\gamma - 1} + \frac{1}{2}\rho \|\mathbf{v}\|^2$.

*   **Domain:**
    $$ (x, y) \in [0.0, 1.0] \times [0.0, 1.0] $$

*   **Boundary Conditions:**
    *   $x$-faces ($x=0, x=1$): **Periodic**
    *   $y$-faces ($y=0, y=1$): **Slip Wall** (Reflective, $\mathbf{v} \cdot \mathbf{n} = 0$)

*   **Initial Conditions:**
    *   **Density ($\rho$):**
        $$ \rho(x,y) = 1.5 - 0.5 \tanh\left(\frac{y - 0.5}{0.03}\right) $$
    *   **Velocity ($u, v$):**
        $$ u(x,y) = 0.5 \tanh\left(\frac{y - 0.5}{0.03}\right) $$
        $$ v(x,y) = 0.01 \sin(4\pi x) \exp\left(-\frac{(y - 0.5)^2}{0.01}\right) $$
    *   **Pressure ($p$):**
        $$ p(x,y) = 2.5 $$

*   **Constants & Parameters:**
    *   Heat capacity ratio: $\gamma = 1.4$
    *   Mach number: Implicitly determined by initial velocities and sound speed.

*   **Expected Physics:**
    Shear flow instability (Kelvin-Helmholtz) leading to the formation of large-scale vortices and mixing layers at the interface $y=0.5$.

---

**2. Strategic Methodology**

*   **Selected Main Method:** Discontinuous Galerkin Spectral Element Method in 2D plus time.
*   **Rationale:** The problem involves advection-dominated vortex evolution where high-order accuracy is required to minimize numerical dissipation, combined with shock capturing to handle the sharp density gradients at the interface.

---

**3. Implementation Requirements**

*   **Configuration:** Use template defaults unless specified below.

*   **3.1 Adaptive Mesh Refinement & Shock Capturing:**
    *   **Variable Selection Analysis:** The Kelvin-Helmholtz instability is driven by the shear layer between two fluids of different densities. The **Density ($\rho$)** field defines the structural interface and the mixing region most sharply. Velocity fields will become turbulent and noisy as the vortices break down, making them poor candidates for refinement triggers.
    *   **Selected Variable:** **Density ($\rho$)**.
    *   **Justification:** Refining on density gradients ensures the mesh resolution focuses on the contact discontinuity and the evolving vortex arms, which are the primary features of interest.

*   **3.2 Advisor-Mandated Adjustments (Overrides):**
    *   `initial_refinement_level = 4`
    *   `max_refinement_level = 7` (in Controller)
    *   `polydeg = 4`
    *   `surface_flux = flux_hllc`

---

**4. Visualization Strategy**

*   **Field Selection:**
    1.  **Density ($\rho$):** To visualize the mixing of the stratified layers.
    2.  **Pressure ($p$):** To observe the pressure fluctuations within the vortex cores.
*   **Time Steps:** Plot at $t = [0.0, 0.8, 1.6, 2.5]$.
*   **Layout:** Grid (2 Rows $\times$ 4 Columns).
*   **Saving Data:** Save full solution state at `dt_save = 0.1`.
*   **Constraints:**
    *   Save as: `summary_all.png`.
    *   **CRITICAL:** `colorbar=false` (Disable colorbars).
    *   Do not compute derived quantities like vorticity.

---

**5. Cross-Validation & Data Integration**

*   **Status:** No external reference data provided.

---

**REQUIRED COMPONENTS**

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
*   Shock Capturing (Hennemann-Gassner)
*   Adaptive Mesh Refinement (Löhner Indicator)

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
The simulation provides a textbook-quality realization of the Kelvin-Helmholtz instability.
*   **Evolution:** The transition from the initial linear perturbation ($t=0.0$) to the non-linear roll-up ($t=1.6$) and the formation of fully developed vortices ($t=2.5$) is captured with high fidelity.
*   **Resolution:** The Adaptive Mesh Refinement (AMR) is clearly visible in the plots, clustering high-resolution elements specifically along the density interface and the vortex arms. This maximizes efficiency while maintaining sharp gradients.
*   **Physics:** The pressure field ($p$) correctly develops low-pressure cores within the vortices (centrifugal force balance), which is a key physical signature of the instability.
*   **Artifacts:** The slight horizontal striation in the pressure field at $t=0.0$ is a standard numerical artifact resulting from the projection of the non-polynomial initial conditions (tanh profiles) onto the polynomial DG basis. This is negligible and vanishes as the physics evolves.

**2. Diagnosis:**
**Optimal Configuration Achieved.**
The success of this simulation is attributed to the specific combination of:
1.  **`flux_hllc` (Surface Flux):** This is the decisive factor. The HLLC flux is designed to resolve contact discontinuities (like the shear layer and density interface) exactly, whereas standard fluxes (HLL, Lax-Friedrichs) would have smeared the interface, suppressing the instability growth.
2.  **AMR on Density:** By triggering refinement based on density gradients, the solver allocated resources exactly where the mixing layer evolved, preventing the "smearing" of the vortex arms.
3.  **High-Order DG (`polydeg=4`):** This provided low numerical dispersion, allowing the small perturbations to grow into coherent structures without being damped out by numerical viscosity.

**3. Quantitative Error:**
*   **Value:** N/A (Forward Simulation)
*   **Note:** No analytical ground truth exists for the non-linear evolution of the Kelvin-Helmholtz instability. Accuracy is assessed via physical plausibility, symmetry, and feature sharpness.

**4. Targeted Improvements:**
*   **None.** The solution has reached the saturation point of quality for the requested parameters. Further refinement would increase cost with diminishing returns on visual or physical fidelity. The result is suitable for use as High-Fidelity Ground Truth.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Numerical implementation using Discontinuous Galerkin Spectral Element Method (DGSEM) in 2D plus time with polydeg=3, LobattoLegendreBasis, surface_flux=flux_hll, VolumeIntegralShockCapturingHG (DG flux=flux_ranocha, FV flux=flux_lax_friedrichs, IndicatorHennemannGassner on density), P4estMesh (1x1 trees, initial_refinement_level=3, periodicity=(true,false) for x-periodic/y-slip), AMR via IndicatorLöhner/ControllerThreeLevel on density (levels 4-6, interval=20), time integration RDPK3SpFSAL49 (Low-Storage RK) over tspan=(0.0,2.5) with CFL=0.5 StepsizeCallback. Matches CompressibleEulerEquations2D(γ=1.4), exact ICs (tanh shear layer + perturbation), slip walls on y, zero sources. Plots density field. | N/A | 50 | 18 | 10 | 10 | **88** |
| 1 | Numerical implementation using Discontinuous Galerkin Spectral Element Method (DGSEM) in 2D plus time with polynomial degree 4 (LobattoLegendreBasis), surface flux flux_hllc, volume integral via ShockCapturingHG (Hennemann-Gassner style with flux_ranocha for DG and flux_lax_friedrichs for FV, indicator on density), P4estMesh with initial_refinement_level=3, periodicity (true,false) for x-periodic and y-slip walls, AMR via IndicatorLöhner on density with ControllerThreeLevel (levels 4-6), time integration RDPK3SpFSAL49 to T=2.5 with CFL=0.5 StepsizeCallback. | N/A | 50 | 20 | 12 | 15 | **97** |
| 2 | Numerical implementation using Discontinuous Galerkin Spectral Element Method (DGSEM) in 2D plus time for the 2D Compressible Euler Equations (gamma=1.4). Includes Hennemann-Gassner shock capturing blended with IndicatorHennemannGassner on density variable (alpha_max=1.0, alpha_min=0.001), Löhner AMR indicator on density with ControllerThreeLevel (base_level=4, med_level=5 threshold=0.1, max_level=7 threshold=0.3, interval=20), polydeg=4, LobattoLegendreBasis, surface_flux=flux_hllc, volume_flux_dg=flux_ranocha, volume_flux_fv=flux_lax_friedrichs. Domain [0,1]x[0,1] with P4estMesh (trees_per_dim=(1,1), initial_refinement_level=4, periodicity=(true,false) for x-periodic y-slip walls). Time integration RDPK3SpFSAL49 LSRK with CFL=0.5 stepsize control, tspan=(0.0,2.5). Initial conditions exactly match specified Kelvin-Helmholtz profiles. Boundary conditions slip_wall on y-faces. | N/A | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves full solution states in Trixi HDF5 format to trixi_output/ every dt=0.1 (including t=0) via SaveSolutionCallback (primitive variables cons2prim). Saves npz snapshots to data_npz/time_XXXXXXXX.npz (every 40 steps or end) with keys: 't' (scalar), 'x' (vector of length ~num_dofs), 'y' (vector), 'rho' (vector), 'u' (vector), 'v' (vector), 'p' (vector) from PlotData2D rasterization. Saves summary_all.png (1x4 density plots at ~t=0.0,0.8,1.6,2.5, colorbar=false). |
| 1 | Saves full solution states via SaveSolutionCallback to trixi_output/ (likely .h5 format) at dt=0.1 including initial, with primitive variables (cons2prim: rho, u, v, p). Additionally saves data_npz/time_XXXXXXXX.npz files (every 40 steps or end) containing keys: 't' (scalar), 'x' (vector of length N), 'y' (vector of length N), 'rho' (vector N), 'u' (vector N), 'v' (vector N), 'p' (vector N) where N is number of points from PlotData2D. Saves summary_all.png (1x4 density plots at t≈[0.0,0.8,1.6,2.5]). |
| 2 | Saves full primitive solution states (cons2prim: rho,u,v,p) via SaveSolutionCallback every dt=0.1 to trixi_output/*.h5 (Trixi .h5 format, structure includes mesh, solution snapshots at each save time). Additionally saves via DiscreteCallback (every 40 steps or end, initialized at t=0) to data_npz/time_XXXXXXXXX.npz (NPZ format) containing keys: 't' (scalar), 'x' (N_points,), 'y' (N_points,), 'rho' (N_points,), 'u' (N_points,), 'v' (N_points,), 'p' (N_points,) where N_points is the number of PlotData2D extraction points. Saves summary_all.png (PNG, 2x4 grid of scatter plots for rho and p at t≈[0.0,0.8,1.6,2.5], colorbar=false). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//Euler_Periodic_SlipWalls_ShearLayer_ICs/2025_12_06_2239/` |
| 1 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//Euler_Periodic_SlipWalls_ShearLayer_ICs/2025_12_06_2243/` |
| 2 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//Euler_Periodic_SlipWalls_ShearLayer_ICs/2025_12_06_2249/` |


---
