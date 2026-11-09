
# Experiment Report
*Last Updated: 2026-11-06 11:11:18*

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
\end{pmatrix}.
\]

Equation of state:
\[
p=(\gamma-1)\left(E-\frac{1}{2}\rho(u^2+v^2)\right).
\]

* **Domain:**
\[
(x,y)\in [0,0.25]\times[0,1.0], \qquad t\in[0,3.0].
\]

* **Boundary Conditions:**
  * \(x=0\) and \(x=0.25\): periodic.
  * \(y=0\) and \(y=1\): reflective / free-slip wall, i.e.
\[
v=0
\]
at the wall, with tangential slip allowed.

* **Initial Conditions:**

Density:
\[
\rho(y)=\rho_{\text{heavy}}+(\rho_{\text{light}}-\rho_{\text{heavy}})\cdot \frac{1}{2}\left(1+\tanh\left(50(y-0.5)\right)\right),
\]
with
\[
\rho_{\text{heavy}}=2.0,\qquad \rho_{\text{light}}=1.0.
\]

Thus explicitly,
\[
\rho(y)=2.0+(1.0-2.0)\cdot \frac{1}{2}\left(1+\tanh\left(50(y-0.5)\right)\right).
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
c(x,y)=\sqrt{\frac{\gamma p(y)}{\rho(y)}}.
\]

Vertical velocity perturbation:
\[
v(x,y)=-0.025\, c(x,y)\cos(8\pi x)\sin(\pi y)^6.
\]

Total energy:
\[
E(x,y)=\frac{p(y)}{\gamma-1}+\frac{1}{2}\rho(y)\left(u(x,y)^2+v(x,y)^2\right).
\]

* **Constants & Parameters:**
\[
\gamma=1.4,\qquad g=1.0,\qquad \mu=0,\qquad k=0.
\]

* **Expected Physics:**
Rayleigh-Taylor instability driven by gravitationally unfavorable stratification, with interface growth, mushroom structures, and secondary Kelvin-Helmholtz roll-up along the density interface.

---

**2. Strategic Methodology**

* Can obtain an exact solution or similar?  
No. This is a nonlinear 2D instability problem for the compressible Euler system with nontrivial gravity source, reflective walls, and perturbation-driven interface evolution. No useful exact solution or Method of Characteristics reduction is feasible for the full problem.

* **Main numerical choice:**  
**Discontinuous Galerkin Spectral Element Method in 2D plus time**

Reason: hyperbolic compressible flow, sharp density interface, evolving contact structures, and localized small-scale roll-up. The advisor already confirmed the architecture is correct. This is a refinement pass, not a redesign.

---

**3. Implementation Requirements (CRITICAL)**

* **Configuration:** use template defaults.

#### **3.0.1 Adaptive Mesh Refinement & Shock Capturing**
* **AMR/Shock variable:** **Density \(\rho\)**.
* **Justification:** The dominant coherent structure is the material interface between heavy and light fluid. Density remains the cleanest marker for that interface and for the mushroom cap/contact layers. Velocity will develop broad small-scale vortical content and become noisy as Kelvin-Helmholtz structures grow, so refining on velocity would waste resolution over too much of the domain. Pressure is comparatively smoother due to near-hydrostatic initialization and is less selective for interface tracking. Therefore density is superior for both AMR and shock/interface capturing.

* **Advisor-Mandated Adjustments (Overrides):**
  * Increase `polydeg` from **3** to **4**.
  * No other overrides.

This directly targets the diagnosed projection/Gibbs artifact at \(t=0\) and improves interface sharpness at late time.

---

**4. Visualization Strategy:**

* **Field Selection (Analysis):**
  * **Density \(\rho\)**
  * **Vertical velocity \(v\)**

Reason: density is the primary instability marker and best reveals interface morphology; vertical velocity shows the buoyancy-driven motion and roll-up dynamics without overcrowding the summary.

* **Task:** Plot these fields at **4 distinct time steps**:
\[
t = 0.0,\; 1.0,\; 2.0,\; 3.0.
\]

* **Layout:** Grid with rows = fields, columns = time.

* **Saving Data:** Save the full solution state for all variables at frequent intervals with
\[
dt_{\text{save}}=0.1.
\]

* **Constraints:**
  * Save figure as `summary_all.png`
  * **Disable colorbars** (`colorbar=false`) if the code is in Julia.
  * Use automatic scaling; no fixed colormap requirements.
  * Plot only the selected primary variables.

---

**5. Cross-Validation & Data Integration (Conditional)**

No external reference data or sensor files were provided.  
No relative \(L_2\) comparison required.

---

**REQUIRED COMPONENTS**

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
Shock Capturing Indicator  
Adaptive Mesh Refinement  
Increase `polydeg` to 4

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
**Section 1: Status Decision**
* **Status:** REJECTED_FOR_REFINEMENT

**Section 2: Quantitative Scorecard**
* **Score Integrity:** 50
* **Score Accuracy:** 20
* **Score Details:** 13
* **Score Optimality:** 15
* **TOTAL SCORE:** 98

**Section 3: Forensic Analysis & Feedback**

* **Observation:** The simulation successfully executes and captures the complex physics of the Rayleigh-Taylor instability. The late-time dynamics ($t=2.0$ and $t=3.0$) show excellent resolution of the primary mushroom structures and the secondary Kelvin-Helmholtz roll-ups along the interface. However, a close inspection of the density field at $t=0.0$ reveals a distinct, high-frequency "comb-like" or jagged oscillation along the horizontal interface. Quantitative error is not applicable as no ground truth is available for this nonlinear instability problem.
* **Diagnosis:** Projection Error / Gibbs Phenomenon. The initial condition uses a steep `tanh` profile to define the density interface. Despite increasing the polynomial degree to 4, the basis is still slightly insufficient to perfectly project this sharp gradient without introducing grid-aligned spectral ringing (overshoots/undershoots) at the element boundaries. 
* **Targeted Improvements:**
    * **For Integrity:** None. The simulation is perfectly stable.
    * **For Accuracy:** None. The macroscopic physics and boundary conditions are correct.
    * **For Details:** To achieve a perfect score (100) and eliminate the $t=0$ projection artifact, **increase `polydeg` to 5**. This will provide the necessary spectral resolution to smoothly capture the steep initial density gradient and will further enhance the sharpness of the intricate, small-scale vortical structures at $t=3.0$.
    * **For Optimality:** None. DGSEM with Adaptive Mesh Refinement is the optimal choice for resolving the evolving, sharp interfaces in this problem.

*(Note: We are in the Verification Phase. A high score of 97 was achieved in Run 1. This is the first refinement attempt. We will continue refining up to 5 times to achieve a perfect 100 before locking in the best model.)*


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Numerical DGSEM (polydeg=3) solution of 2D compressible Euler equations with gravity source on domain [0,0.25]x[0,1] for t in [0,3]. Uses P4estMesh with periodic x / slip-wall y boundaries, initial_condition_rt and source_terms_rt exactly as specified, IndicatorHennemannGassner shock capturing and IndicatorLöhner AMR both keyed on density, LSRK time integration, and dt_save=0.1 output. | N/A | 50 | 20 | 12 | 15 | **97** |
| 2 | Numerical simulation using DGSEM (polydeg=4) on 2D compressible Euler equations with gravity source. Domain [0,0.25]x[0,1], t=[0,3]. AMR on density via IndicatorLöhner + ControllerThreeLevel, shock capturing via HennemannGassner indicator. Initial condition exactly implements the tanh density profile, piecewise linear pressure, and v perturbation with cos(8πx)sin(πy)^6. Boundary conditions: periodic x, slip_wall y. Time integration via RDPK3SpFSAL49 with CFL control. Saves solution snapshots every dt=0.1 and generates summary_all.png showing ρ and v at t=0,1,2,3 (colorbar=false). | N/A | 50 | 20 | 13 | 15 | **98** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves multiple time_*.npz files (dt=0.1) containing keys: 't' (scalar), 'x' (2D), 'y' (2D), 'rho' (2D), 'u' (2D), 'v' (2D), 'p' (2D); also produces summary_all.png. |
| 2 | Saves multiple files: (1) trixi_output/*.h5 via SaveSolutionCallback (dt=0.1) containing full conservative/primitive state; (2) data_npz/time_*.npz via custom DiscreteCallback (dt=0.1) with keys: 't' (scalar), 'x' (vector), 'y' (vector), 'rho' (vector), 'u' (vector), 'v' (vector), 'p' (vector). Also saves summary_all.png (1600x800, 2x4 layout of ρ and v fields). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/RT/run3/results//CompressibleEuler2D_Periodic_Reflective_RayleighTaylor/2026_11_06_1059/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/RT/run3/results//CompressibleEuler2D_Periodic_Reflective_RayleighTaylor/2026_11_06_1104/` |


---
