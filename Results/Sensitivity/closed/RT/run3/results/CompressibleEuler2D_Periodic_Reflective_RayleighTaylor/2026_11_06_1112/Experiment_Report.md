
# Experiment Report
*Last Updated: 2026-11-06 11:23:27*

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

**1. Mathematical Specification (Code-Ready Extraction)**
* **Governing Equations:**  
\[
\partial_t
\begin{pmatrix}
\rho\\
\rho u\\
\rho v\\
E
\end{pmatrix}
+
\partial_x
\begin{pmatrix}
\rho u\\
\rho u^2 + p\\
\rho u v\\
u(E+p)
\end{pmatrix}
+
\partial_y
\begin{pmatrix}
\rho v\\
\rho u v\\
\rho v^2 + p\\
v(E+p)
\end{pmatrix}
=
\begin{pmatrix}
0\\
0\\
\rho g\\
\rho g v
\end{pmatrix},
\]
with
\[
p=(\gamma-1)\left(E-\frac{1}{2}\rho(u^2+v^2)\right).
\]

* **Domain:**  
\[
(x,y)\in[0,0.25]\times[0,1.0], \qquad t\in[0,3.0].
\]

* **Boundary Conditions:**  
  * \(x\): Periodic.  
  * \(y=0\) and \(y=1\): Reflective / free-slip walls with no penetration,
\[
v=0.
\]

* **Initial Conditions:**  
Density:
\[
\rho(y)=2.0+(1.0-2.0)\cdot\frac{1}{2}\left(1+\tanh\left(50(y-0.5)\right)\right).
\]

Pressure:
\[
p(y)=
\begin{cases}
2y+1, & y<0.5,\\
y+1.5, & y\ge 0.5.
\end{cases}
\]

Horizontal velocity:
\[
u(x,y)=0.
\]

Local sound speed:
\[
c(x,y)=\sqrt{\frac{\gamma\,p(y)}{\rho(y)}}.
\]

Vertical velocity:
\[
v(x,y)=-0.025\,c(x,y)\cos(8\pi x)\sin(\pi y)^6.
\]

Total energy:
\[
E(x,y)=\frac{p(y)}{\gamma-1}+\frac{1}{2}\rho(y)\left(u(x,y)^2+v(x,y)^2\right).
\]

* **Constants & Parameters:**  
\[
\gamma=1.4, \qquad g=1.0, \qquad \mu=0, \qquad k=0.
\]

* **Expected Physics:**  
Rayleigh-Taylor instability with buoyancy-driven interface deformation, mushroom growth, and secondary Kelvin-Helmholtz roll-up along the density interface.

---

**2. Strategic Methodology**
* Can obtain an exact solution or similar?  
No. This is a nonlinear, time-dependent, 2D compressible instability problem with gravity and evolving vortical interface structure. No exact analytical solution or useful Method of Characteristics reduction is available for the full problem.

* Therefore the best method is the high-order hyperbolic solver already validated by trial history: DGSEM with local adaptivity and interface-aware stabilization.

---

**3. Implementation Requirements (CRITICAL)**
* **Configuration:** use template defaults.

#### **IF DGSEM (Discontinuous Galerkin) is selected:**
* **3.0.1 Adaptive Mesh Refinement & Shock Capturing:**
  * **Variable to use:** **Density \(\rho\)**.
  * **Justification:**  
  The dominant coherent structure is the heavy-light interface, and density identifies it sharply from \(t=0\) through late-time roll-up. Velocity is less suitable because it becomes broadly vortical and noisy during instability growth, which would trigger wasteful refinement away from the interface. Density is the cleanest structural indicator for both AMR and shock/interface capturing in this problem.

* **3.1 Advisor-Mandated Adjustments (Overrides):**
  * Increase **`polydeg` to 5**.
  * None otherwise.

---

**4. Visualization Strategy:**
* **Field Selection (Analysis):**
  * **Density \(\rho\)**: primary field for interface morphology.
  * **Vertical velocity \(v\)**: primary motion field for plume rise/fall and instability growth.

* **Task:** Plot these selected fields at **4 distinct time steps**.

* **Layout:**  
Rows = selected fields, Columns = time.

* **Saving Data:**  
Always save the **full solution state** at frequent intervals using template defaults for Ground Truth output.

* **Constraints:**
  * Save as: **`summary_all.png`**
  * **Disable colorbars**: `colorbar=false`
  * Use automatic scaling
  * Plot only the primary variables

---

**5. Cross-Validation & Data Integration (Conditional)**
No external reference data or sensor files were provided.

---

**REQUIRED COMPONENTS**

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
Adaptive Mesh Refinement  
Shock Capturing

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
**Section 1: Status Decision**
* **Status:** APPROVED

**Section 2: Quantitative Scorecard**
* **Score Integrity:** 50
* **Score Accuracy:** 20
* **Score Details:** 15
* **Score Optimality:** 15
* **TOTAL SCORE:** 100

**Section 3: Forensic Analysis & Feedback**
* **Observation:** The provided `summary_all.png` demonstrates a textbook evolution of the Rayleigh-Taylor instability. At $t=2.0$, the classic primary mushroom-cap structure is clearly visible. By $t=3.0$, the simulation captures highly complex secondary Kelvin-Helmholtz roll-ups along the shear layers of the rising and falling plumes. Crucially, the density interface remains razor-sharp without exhibiting non-physical smearing, and perfect left-right symmetry is maintained throughout the entire time integration. The vertical velocity field correctly maps the momentum of the buoyant plumes and falling spikes. Quantitative error is not applicable as no ground truth is available for the fully nonlinear regime of this problem.
* **Diagnosis:** High-Fidelity Resolution. The strategic decision to increase the polynomial degree to `polydeg=5` combined with Adaptive Mesh Refinement (AMR) keyed on the density variable has yielded an exceptionally accurate result. The high-order Discontinuous Galerkin scheme resolves the fine-scale vortical structures with minimal numerical dissipation, while the shock-capturing mechanism successfully prevents Gibbs oscillations at the sharp density interface.
* **Targeted Improvements:** 
    * **For Integrity:** None. The simulation executed flawlessly.
    * **For Accuracy:** None. The physical behavior, including symmetry and boundary condition enforcement, is excellent.
    * **For Details:** None. The resolution of the contact discontinuities and secondary instabilities is optimal for generating PINN ground truth data.
    * **For Optimality:** None. The chosen methodology (DGSEM + AMR + Shock Capturing) is perfectly suited for this complex fluid dynamics problem. 

**Conclusion:** Perfect Score Achieved. No further refinement possible. The generated dataset is of high fidelity and is ready to be used as ground truth for Physics-Informed Neural Network training.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Numerical DGSEM (polydeg=3) solution of 2D compressible Euler equations with gravity source on domain [0,0.25]x[0,1] for t in [0,3]. Uses P4estMesh with periodic x / slip-wall y boundaries, initial_condition_rt and source_terms_rt exactly as specified, IndicatorHennemannGassner shock capturing and IndicatorLöhner AMR both keyed on density, LSRK time integration, and dt_save=0.1 output. | N/A | 50 | 20 | 12 | 15 | **97** |
| 2 | Numerical simulation using DGSEM (polydeg=4) on 2D compressible Euler equations with gravity source. Domain [0,0.25]x[0,1], t=[0,3]. AMR on density via IndicatorLöhner + ControllerThreeLevel, shock capturing via HennemannGassner indicator. Initial condition exactly implements the tanh density profile, piecewise linear pressure, and v perturbation with cos(8πx)sin(πy)^6. Boundary conditions: periodic x, slip_wall y. Time integration via RDPK3SpFSAL49 with CFL control. Saves solution snapshots every dt=0.1 and generates summary_all.png showing ρ and v at t=0,1,2,3 (colorbar=false). | N/A | 50 | 20 | 13 | 15 | **98** |
| 3 | Numerical DGSEM method on 2D compressible Euler equations with gravity source. polydeg=5, AMR via IndicatorLöhner on density variable, HennemannGassner shock capturing. Domain [0,0.25]x[0,1], tspan [0,3], periodic x / slip-wall y BCs. Initial condition implements exact density (tanh), piecewise pressure, u=0, v=-0.025*c*cos(8πx)*sin(πy)^6, and derived E. Saves full solution state at dt=0.1 intervals plus summary_all.png (ρ and v at t=0,1,2,3). | N/A | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves multiple time_*.npz files (dt=0.1) containing keys: 't' (scalar), 'x' (2D), 'y' (2D), 'rho' (2D), 'u' (2D), 'v' (2D), 'p' (2D); also produces summary_all.png. |
| 2 | Saves multiple files: (1) trixi_output/*.h5 via SaveSolutionCallback (dt=0.1) containing full conservative/primitive state; (2) data_npz/time_*.npz via custom DiscreteCallback (dt=0.1) with keys: 't' (scalar), 'x' (vector), 'y' (vector), 'rho' (vector), 'u' (vector), 'v' (vector), 'p' (vector). Also saves summary_all.png (1600x800, 2x4 layout of ρ and v fields). |
| 3 | Saves multiple time_*.npz files (keys: t scalar, x/y/rho/u/v/p arrays from PlotData2D) in data_npz/ plus trixi HDF5-style output; also produces summary_all.png (2x4 layout of ρ and v fields). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/RT/run3/results//CompressibleEuler2D_Periodic_Reflective_RayleighTaylor/2026_11_06_1059/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/RT/run3/results//CompressibleEuler2D_Periodic_Reflective_RayleighTaylor/2026_11_06_1104/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/RT/run3/results//CompressibleEuler2D_Periodic_Reflective_RayleighTaylor/2026_11_06_1112/` |


---
