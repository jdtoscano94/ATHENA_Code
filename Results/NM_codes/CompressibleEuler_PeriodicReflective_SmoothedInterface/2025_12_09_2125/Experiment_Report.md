
# Experiment Report
*Last Updated: 2025-12-09 21:47:44*

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
*   **Governing Equations:** 2D Compressible Euler Equations with Gravity Source Term
    $$ \frac{\partial}{\partial t} \begin{pmatrix} \rho \\ \rho u \\ \rho v \\ E \end{pmatrix} + \frac{\partial}{\partial x} \begin{pmatrix} \rho u \\ \rho u^2 + p \\ \rho u v \\ (E+p)u \end{pmatrix} + \frac{\partial}{\partial y} \begin{pmatrix} \rho v \\ \rho u v \\ \rho v^2 + p \\ (E+p)v \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ \rho g \\ \rho g v \end{pmatrix} $$
    where $E = \frac{p}{\gamma-1} + \frac{1}{2}\rho(u^2 + v^2)$ and $p = (\gamma-1)(E - \frac{1}{2}\rho(u^2+v^2))$.

*   **Domain:** $(x,y) \in [0, 0.25] \times [0, 1.0]$
*   **Boundary Conditions:**
    *   $x$-faces ($x=0, x=0.25$): **Periodic**
    *   $y$-faces ($y=0, y=1$): **Slip Wall** (Reflective, $\mathbf{v} \cdot \mathbf{n} = 0$)
*   **Initial Conditions:**
    *   **Density ($\rho$):**
        $$ \rho(x,y) = 2.0 + (1.0 - 2.0) \cdot \frac{1}{2}\left(1 + \tanh\left(50(y - 0.5)\right)\right) $$
    *   **Pressure ($p$):**
        $$ p(x,y) = \begin{cases} 2y + 1 & \text{if } y < 0.5 \\ y + 1.5 & \text{if } y \geq 0.5 \end{cases} $$
    *   **Velocity ($u, v$):**
        $$ u(x,y) = 0 $$
        $$ v(x,y) = -0.025 \sqrt{\frac{\gamma p}{\rho}} \cos(8\pi x) \sin(\pi y)^6 $$
*   **Constants & Parameters:**
    *   $\gamma = 1.4$
    *   Gravity $g = 1.0$ (acting in $+y$ direction).
*   **Expected Physics:** Rayleigh-Taylor Instability. The heavy fluid ($\rho \approx 2$) at the bottom is accelerated upward by gravity ($g=+1$) faster than the light fluid ($\rho \approx 1$) at the top. The interface will destabilize, forming a rising "mushroom" plume with Kelvin-Helmholtz vortices on the shear layers.

---

**2. Strategic Methodology**

*   **Selected Main Method:** **Discontinuous Galerkin Spectral Element Method in 2D plus time (DGSEM)**
*   **Rationale:** The problem involves the evolution of complex vortical structures (Kelvin-Helmholtz rolls) and sharp density interfaces; DGSEM provides the necessary high-order spectral accuracy to resolve vortices while minimizing dissipation, and AMR allows efficient resolution of the moving interface.

---

**3. Implementation Requirements**

*   **Configuration:** Use template defaults for Time Stepper and CFL unless specified below.

*   **3.1 Adaptive Mesh Refinement & Shock Capturing:**
    *   **Variable Selection Analysis:** The primary feature of interest is the contact discontinuity (interface) between the heavy and light fluids. Velocity fields will become turbulent and globally noisy as the instability progresses. Refining on velocity would waste resources on small-scale acoustic noise. Refining on density focuses computational power strictly on the structural evolution of the plume.
    *   **Selected Variable:** **Density ($\rho$)**
    *   **Justification:** Density gradients define the interface driving the instability.

*   **3.2 Advisor-Mandated Adjustments (Overrides):**
    *   **Surface Flux:** **`flux_hllc`** (Mandatory replacement of `flux_hll` to reduce dissipation at contact discontinuities).
    *   **Polynomial Degree:** **`polydeg = 4`** (Increased from 3 to improve spectral resolution of vortices).

---

**4. Visualization Strategy**

*   **Field Selection:**
    1.  **Density ($\rho$):** To visualize the mixing and the shape of the mushroom plume.
    2.  **Pressure ($p$):** To visualize the hydrostatic balance and acoustic waves.
*   **Time Steps:** Plot at $t \approx [0.0, 1.0, 2.0, 3.0]$.
*   **Layout:** 2 Rows (Density, Pressure) $\times$ 4 Columns (Time).
*   **Constraints:**
    *   Save file as: `summary_all.png`
    *   **CRITICAL:** Set `colorbar=false` for all plots.
    *   Do not use fixed color scales; allow auto-scaling.

---

**5. Cross-Validation & Data Integration**

*   **Status:** No external reference data provided.
*   **Action:** None.

---

**REQUIRED COMPONENTS**

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
*   Shock Capturing Indicator (Hennemann-Gassner)
*   Adaptive Mesh Refinement (AMR)
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
*   **Physics & Dynamics:** The simulation correctly captures the Rayleigh-Taylor Instability under the specified "inverted gravity" conditions ($g=+1.0$). The heavy fluid (Red, $\rho=2$) at the bottom is accelerated upward into the light fluid (Blue, $\rho=1$), forming a classic mushroom plume.
*   **Interface Resolution:** The contact discontinuity between the two fluids is remarkably sharp, particularly at $t=2.0$. The Kelvin-Helmholtz (KH) vortices forming on the shear layers of the mushroom stem are distinct and well-resolved, not smeared out.
*   **Symmetry:** The solution maintains perfect symmetry about the vertical centerline ($x=0.125$) up to the final time $t=2.9$. This is a strong indicator of low numerical dispersion and a balanced mesh, as asymmetric numerical noise often triggers chaotic breakdown earlier in RTI simulations.
*   **Pressure Field:** The pressure gradient correctly increases with height (Blue at bottom $\to$ Red at top), consistent with the hydrostatic balance $\frac{\partial p}{\partial y} = \rho g$ where $g$ is positive.

**2. Diagnosis:**
*   **Methodological Success:** The switch to `flux_hllc` (Harten-Lax-van Leer-Contact) was decisive. Unlike `flux_hll` or `flux_lax_friedrichs`, HLLC includes a wave specifically for the contact discontinuity, preventing the artificial diffusion that typically blurs density interfaces in Euler simulations.
*   **Spectral Resolution:** The use of `polydeg=4` provided sufficient spectral resolving power to capture the fine-scale vorticity in the late-stage turbulence ($t=2.9$) without introducing Gibbs oscillations, likely aided by the Hennemann-Gassner shock capturing which applied dissipation only where necessary.

**3. Conclusion:**
*   **Perfect Score Achieved.** The simulation meets all criteria for "High-Fidelity Ground Truth."
*   The physics are consistent, the numerical artifacts are negligible, and the resolution of turbulent features is excellent for the computational cost.
*   No further refinement is necessary. The data generated is suitable for training the PINN.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Numerical implementation using Discontinuous Galerkin Spectral Element Method (DGSEM) in 2D plus time for Compressible Euler Equations with gravity source term. Polydeg=3, LobattoLegendreBasis, flux_hll surface flux, shock capturing via IndicatorHennemannGassner on density variable blending flux_ranocha (DG) and flux_lax_friedrichs (FV). Domain [0.0, 0.25] x [0.0, 1.0], periodic in x, slip walls in y. Initial refinement level=3 with AMR (IndicatorLöhner on density, ControllerThreeLevel levels 4-6). Time span (0.0, 3.0), RDPK3SpFSAL49 integrator, CFL=0.5. Matches initial conditions, source terms, and boundary conditions exactly. | N/A | 50 | 20 | 10 | 15 | **95** |
| 2 | Numerical implementation using Discontinuous Galerkin Spectral Element Method (DGSEM) in 2D plus time for Compressible Euler Equations with gravity source term. Polynomial degree polydeg=4, surface flux flux_hllc, volume integral with Hennemann-Gassner shock capturing on density indicator, AMR using IndicatorLöhner on density with ControllerThreeLevel (levels 4-6). Domain [0.0, 0.25] x [0.0, 1.0] with P4estMesh trees_per_dimension=(1,4), initial_refinement_level=3, periodic in x, slip wall in y. Time span tspan=(0.0, 3.0), RDPK3SpFSAL49 integrator with CFL=0.5. Initial conditions, pressure, density, velocity perturbation, and gravity source g=1.0 in +y direction match exactly. Plots density and pressure at t≈[0,1,2,3] saved as summary_all.png with 2x4 layout and colorbar=false. | 0.00e+00 | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves multiple files to 'data_npz/time_XXXXX.npz' (every dt=0.1) containing keys: 't' (scalar), 'x' (N_points,), 'y' (N_points,), 'rho' (N_points,), 'u' (N_points,), 'v' (N_points,), 'p' (N_points,). Also saves Trixi solution files to 'trixi_output/' every 20 steps (HDF5 format with primitive variables). Generates 'summary_all.png' (1x4 layout of density scatter plots at t≈0,1,2,3). |
| 2 | Saves multiple .npz files in data_npz/ (e.g., time_XXXXX.npz) with keys: 't' (scalar), 'x' (1D vector of length N_points), 'y' (1D vector of length N_points), 'rho' (1D vector of length N_points), 'u' (1D vector of length N_points), 'v' (1D vector of length N_points), 'p' (1D vector of length N_points) from PlotData2D cons2prim projection every ~0.1 time units. Trixi .h5 solution files in trixi_output/ via SaveSolutionCallback (interval=20, cons2prim variables, including initial). Final visualization summary_all.png. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//CompressibleEuler_PeriodicReflective_SmoothedInterface/2025_12_09_2114/` |
| 2 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//CompressibleEuler_PeriodicReflective_SmoothedInterface/2025_12_09_2125/` |


---
