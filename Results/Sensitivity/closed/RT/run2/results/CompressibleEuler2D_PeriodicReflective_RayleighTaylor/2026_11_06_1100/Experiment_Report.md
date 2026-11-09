
# Experiment Report
*Last Updated: 2026-11-06 11:07:56*

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
  The 2D compressible Euler equations with gravity source in the positive \(y\)-direction are
  \[
  \partial_t
  \begin{bmatrix}
  \rho\\
  \rho u\\
  \rho v\\
  E
  \end{bmatrix}
  +
  \partial_x
  \begin{bmatrix}
  \rho u\\
  \rho u^2 + p\\
  \rho u v\\
  u(E+p)
  \end{bmatrix}
  +
  \partial_y
  \begin{bmatrix}
  \rho v\\
  \rho u v\\
  \rho v^2 + p\\
  v(E+p)
  \end{bmatrix}
  =
  \begin{bmatrix}
  0\\
  0\\
  \rho g\\
  \rho g v
  \end{bmatrix}.
  \]
  Equation of state:
  \[
  p = (\gamma - 1)\left(E - \frac{1}{2}\rho(u^2+v^2)\right).
  \]

* **Domain:**
  \[
  (x,y)\in[0,0.25]\times[0,1.0],\qquad t\in[0,3.0].
  \]

* **Boundary Conditions:**
  * \(x=0\) and \(x=0.25\): periodic.
  * \(y=0\) and \(y=1\): reflective / free-slip wall, with no penetration
    \[
    v=0.
    \]

* **Initial Conditions:**
  Density:
  \[
  \rho(x,y,0)=\rho(y)=2.0+(1.0-2.0)\cdot \frac{1}{2}\left(1+\tanh\left(50(y-0.5)\right)\right).
  \]
  Equivalently,
  \[
  \rho(x,y,0)=2.0-\frac{1}{2}\left(1+\tanh\left(50(y-0.5)\right)\right).
  \]

  Pressure:
  \[
  p(x,y,0)=p(y)=
  \begin{cases}
  2y+1, & y<0.5,\\
  y+1.5, & y\ge 0.5.
  \end{cases}
  \]

  Horizontal velocity:
  \[
  u(x,y,0)=0.
  \]

  Local sound speed:
  \[
  c(x,y,0)=\sqrt{\frac{\gamma\,p(y)}{\rho(y)}}.
  \]

  Vertical velocity:
  \[
  v(x,y,0)=-0.025\,c(x,y,0)\,\cos(8\pi x)\,\sin(\pi y)^6.
  \]

  Total energy:
  \[
  E(x,y,0)=\frac{p(x,y,0)}{\gamma-1}+\frac{1}{2}\rho(x,y,0)\left(u(x,y,0)^2+v(x,y,0)^2\right).
  \]

* **Constants & Parameters:**
  \[
  \gamma=1.4,\qquad g=1.0,\qquad \mu=0,\qquad k=0.
  \]

* **Expected Physics:**
  Rayleigh-Taylor instability with unstable density stratification under upward gravity, growth of the seeded mode, plume formation, rolling interfaces, and secondary Kelvin-Helmholtz structures. Primary feature: evolving density interface near \(y=0.5\).

---

**2. Strategic Methodology**

* Can obtain an exact solution or similar?  
  No. This is a fully nonlinear, time-dependent, 2D compressible instability with variable density interface, source term, and nonlinear mode coupling. No practical exact analytical solution is available beyond the initial hydrostatic base state. Method of characteristics is not feasible for the full 2D Euler instability evolution.

* **Main method selection:**  
  **Discontinuous Galerkin Spectral Element Method in 2D plus time**

  Reason: compressible Euler, evolving sharp interface, inviscid instability, mixed smooth and under-resolved small scales, and non-periodic walls in \(y\). DGSEM is the correct high-order shock/interface-resolving method.

---

**3. Implementation Requirements (CRITICAL)**

* **Configuration:** use template defaults.

#### **IF DGSEM (Discontinuous Galerkin) is selected:**

* **3.0.1 Adaptive Mesh Refinement & Shock Capturing:**
  * **Variable for AMR/Shock Capturing:** **Density \(\rho\)**.
  * **Justification:** The dominant coherent structure is the heavy/light interface. Density remains the cleanest marker of the contact layer and plume morphology. Velocity develops widespread vortical fine scales and becomes globally oscillatory during Kelvin-Helmholtz roll-up, so refining on velocity would waste resolution away from the interface. Pressure is comparatively smoother due to near-hydrostatic balance and is less sensitive to contact sharpness. Therefore density is superior for both AMR and shock/interface sensing in this problem.

* **3.1 Advisor-Mandated Adjustments (Overrides):**
  * Increase `polydeg` from **3** to **4**.
  * Increase AMR controller levels from **4-6** to **4-7**.
  * None else.

---

**4. Visualization Strategy:**

* **Field Selection (Analysis):**
  * **Density \(\rho\)**: primary field; directly captures RTI interface, plume, and mixing layer.
  * **Vertical momentum \(\rho v\)**: best secondary field; shows upward/downward motion and symmetry of plume dynamics without overplotting all primitive variables.

* **Task:** Plot these selected fields at **4 distinct time steps**:
  \[
  t=\{0,\ 1,\ 2,\ 3\}.
  \]

* **Layout:**  
  Grid with rows = selected fields, columns = time.

* **Saving Data:**  
  Save the **full solution state** at frequent intervals, e.g.
  \[
  dt_{\text{save}}=0.1
  \]
  for ground-truth archive.

* **Constraints:**
  * Save figure as **`summary_all.png`**
  * **Disable colorbars** (`colorbar=false`) if the code is in Julia.
  * Use automatic scaling; no fixed colormap specification.
  * Plot only primary variables \(\rho\) and \(\rho v\).

---

**5. Cross-Validation & Data Integration (Conditional)**

No external reference data or sensor paths were provided. No action required.

---

**REQUIRED COMPONENTS**

**Main Method:**
- Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
- Adaptive Mesh Refinement
- Shock Capturing
- Increase polynomial degree to 4
- Increase AMR levels to 4-7

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
* **Observation:** The simulation successfully captures the complex, nonlinear evolution of the Rayleigh-Taylor instability. The density field ($\rho$) shows a remarkably sharp contact discontinuity between the heavy and light fluids. By $t=2.0$ and $t=3.0$, the primary mushroom plume and the secondary Kelvin-Helmholtz roll-ups along the shear layers are beautifully resolved with intricate fine-scale structures. Crucially, perfect left-right symmetry is maintained throughout the entire time integration, which is a strong indicator of high numerical fidelity and the absence of biased truncation errors. Quantitative error is not applicable as no analytical ground truth is available for the fully developed nonlinear regime of this instability.
* **Diagnosis:** High-Fidelity Resolution (Optimal State). The strategic adjustments from the previous trial—specifically increasing the polynomial degree (`polydeg=4`) and expanding the Adaptive Mesh Refinement bounds (levels 4-7)—have successfully eliminated the slight interface smearing observed previously. The Discontinuous Galerkin Spectral Element Method (DGSEM) combined with density-based AMR is perfectly tracking and resolving the chaotic mixing layer without introducing non-physical Gibbs oscillations.
* **Targeted Improvements:** 
    * **For Integrity:** None. The simulation ran flawlessly to completion.
    * **For Accuracy:** None. The physical behavior, including hydrostatic balance away from the interface and symmetric perturbation growth, is perfectly captured.
    * **For Details:** None. The contact discontinuities are razor-sharp, and the fine-scale vortical structures are exceptionally well-resolved.
    * **For Optimality:** None. The chosen architecture (DGSEM + AMR + HLL Flux + Hennemann-Gassner shock capturing) is theoretically and practically optimal for this specific inviscid, interface-driven problem. 

**Verdict Justification:** Perfect Score Achieved. The current configuration yields exceptionally high-quality, artifact-free data that is ideal for training a Physics-Informed Neural Network (PINN). No further refinement is necessary.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Numerical simulation using 2D DGSEM (polydeg=3, LobattoLegendreBasis) on P4estMesh with AMR (IndicatorLöhner on density, ControllerThreeLevel levels 4-6) and hybrid shock capturing (IndicatorHennemannGassner on density). Solves compressible Euler equations with gravity source term (g=1.0) on domain [0,0.25]x[0,1] with periodic x / slip-wall y boundaries. Initial condition implements exact tanh density profile, piecewise linear pressure, and perturbation velocity. Time integration via RDPK3SpFSAL49 to t=3.0 with CFL stepping. Saves full solution snapshots (rho, rho_v, ...) every dt=0.1 via npz and produces summary_all.png showing rho and rho_v at t={0,1,2,3}. | N/A | 50 | 20 | 12 | 15 | **97** |
| 2 | Numerical DGSEM method on 2D compressible Euler equations with gravity source. Uses polydeg=4, AMR on density (levels 4-7), shock capturing via HennemannGassner indicator, HLL surface flux, and source_terms_rt. Domain [0,0.25]x[0,1], tspan [0,3], periodic x / slip y BCs, exact ICs for rho(y), piecewise p(y), v seeding. Saves npz snapshots every dt=0.1 and summary_all.png (rho, rho_v at t=0,1,2,3). | N/A | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves multiple time_*.npz files (dt_save=0.1) each containing keys: 't' (scalar), 'x' (vector), 'y' (vector), 'rho' (vector), 'rho_v' (vector), 'u' (vector), 'v' (vector), 'p' (vector). Also saves summary_all.png (PNG image, 2x4 layout of rho/rho_v fields). |
| 2 | Saves multiple files time_*.npz (dt_save=0.1) containing keys: 't' (scalar), 'x' (vector), 'y' (vector), 'rho' (vector), 'rho_v' (vector), 'u' (vector), 'v' (vector), 'p' (vector). Also saves summary_all.png (2x4 layout of rho and rho_v scatter plots at t={0,1,2,3}). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/RT/run2/results//CompressibleEuler2D_PeriodicReflective_RayleighTaylor/2026_11_06_1055/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/RT/run2/results//CompressibleEuler2D_PeriodicReflective_RayleighTaylor/2026_11_06_1100/` |


---
