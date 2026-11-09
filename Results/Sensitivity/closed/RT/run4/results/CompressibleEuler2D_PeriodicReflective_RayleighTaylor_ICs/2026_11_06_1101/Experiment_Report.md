
# Experiment Report
*Last Updated: 2026-11-06 11:08:15*

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
For the 2D compressible Euler equations with gravity source in the positive \(y\)-direction,
\[
\frac{\partial}{\partial t}
\begin{pmatrix}
\rho\\
\rho u\\
\rho v\\
E
\end{pmatrix}
+
\frac{\partial}{\partial x}
\begin{pmatrix}
\rho u\\
\rho u^2 + p\\
\rho u v\\
u(E+p)
\end{pmatrix}
+
\frac{\partial}{\partial y}
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
with equation of state
\[
p = (\gamma - 1)\left(E - \frac{1}{2}\rho(u^2+v^2)\right).
\]

* **Domain:**
\[
(x,y)\in[0,0.25]\times[0,1.0], \qquad t\in[0,3.0].
\]

* **Boundary Conditions:**
  * \(x=0\) and \(x=0.25\): periodic.
  * \(y=0\) and \(y=1\): reflective / free-slip wall, i.e.
\[
v=0 \quad \text{at } y=0,1,
\]
with tangential velocity unconstrained by no-slip.

* **Initial Conditions:**
Density:
\[
\rho(x,y,0)=\rho(y)=2.0+(1.0-2.0)\cdot \frac{1}{2}\left(1+\tanh\left(50(y-0.5)\right)\right).
\]

Pressure:
\[
p(x,y,0)=p(y)=
\begin{cases}
2y+1, & y<0.5,\\[4pt]
y+1.5, & y\ge 0.5.
\end{cases}
\]

Horizontal velocity:
\[
u(x,y,0)=0.
\]

Vertical velocity:
\[
v(x,y,0)=-0.025\, c(x,y)\,\cos(8\pi x)\,\sin(\pi y)^6,
\]
where
\[
c(x,y)=\sqrt{\frac{\gamma\,p(y)}{\rho(y)}}.
\]

Total energy:
\[
E(x,y,0)=\frac{p(x,y,0)}{\gamma-1}+\frac{1}{2}\rho(x,y,0)\left(u(x,y,0)^2+v(x,y,0)^2\right).
\]

* **Constants & Parameters:**
\[
\gamma=1.4,\qquad g=1.0,\qquad \mu=0,\qquad k=0.
\]

Heavy/light densities:
\[
\rho_{\text{heavy}}=2.0,\qquad \rho_{\text{light}}=1.0.
\]

* **Expected Physics:**
Rayleigh-Taylor instability driven by unstable stratification under upward gravity. Interface roll-up, mushroom plumes, secondary instabilities, and progressive nonlinear mixing are expected by \(t=3.0\).

---

**2. Strategic Methodology**

* Can obtain an exact solution or similar?  
No. This is a fully nonlinear, time-dependent, 2D compressible instability with interface deformation and secondary roll-up. No useful exact analytical solution or Method of Characteristics reduction exists for the full problem.

* **Main method selection:**  
Use **Discontinuous Galerkin Spectral Element Method in 2D plus time**.

Reason: compressible Euler, mixed smooth/sharp structure, reflective walls, periodic direction, evolving contact interface, and nonlinear instability. This is precisely a high-order DGSEM problem.

---

**3. Implementation Requirements (CRITICAL)**

* **Configuration:** use template defaults.

#### **IF DGSEM (Discontinuous Galerkin) is selected:**

* **3.0.1 Adaptive Mesh Refinement & Shock Capturing:**
  * **AMR/Shock variable:** **Density \(\rho\)**.
  * **Justification:** The key structure is the heavy-light material interface. Density remains the cleanest marker of that interface and stays localized at the contact layer. Velocity becomes vortical and noisy during nonlinear roll-up, so refining on velocity would waste resolution over broad regions. Pressure is comparatively smoother due to near-hydrostatic initialization and is less selective for interface tracking. Therefore, density is superior for both AMR and shock/interface capturing.

* **Advisor-Mandated Adjustments (Overrides):**
  * Increase maximum AMR level from 6 to **7 or 8**; select **8**.
  * Increase polynomial degree from 3 to **4**.
  * None otherwise.

This directly addresses interface smearing by concentrating more resolution at the density gradient and reducing truncation/dispersion error in the mushroom-cap structures.

---

**4. Visualization Strategy:**

* **Field Selection (Analysis):**
  * **Density \(\rho\)**: primary field; it directly shows the Rayleigh-Taylor interface, plume penetration, and mixing.
  * **Vertical velocity \(v\)**: secondary field; it captures buoyancy-driven motion and plume dynamics.

* **Task:** Plot these fields at 4 distinct times:
\[
t=0,\;1,\;2,\;3.
\]

* **Layout:**  
Rows = \(\rho\), \(v\).  
Columns = \(t=0,1,2,3\).

* **Saving Data:**  
Save the **full solution state** at frequent intervals, e.g.
\[
dt_{\text{save}}=0.1.
\]

* **Constraints:**
  * Save figure as: `summary_all.png`
  * **Disable colorbars**: `colorbar=false` if using Julia.
  * Use automatic scaling; no fixed colormap.
  * Plot only \(\rho\) and \(v\).

---

**5. Cross-Validation & Data Integration (Conditional)**

No external reference data or sensor files were provided.  
No relative \(L_2\) error computation required.

---

**REQUIRED COMPONENTS**

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
Shock Capturing Indicator  
Adaptive Mesh Refinement  
Increase maximum AMR level to 8  
Increase polynomial degree to 4

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
* **Observation:** The simulation executed flawlessly. The generated `summary_all.png` displays the time evolution of the Rayleigh-Taylor instability with exceptional clarity. At $t=3.0$, the density field reveals exquisite detail of the secondary Kelvin-Helmholtz instabilities (roll-ups) along the shear layers of the primary mushroom plume. The interfaces between the heavy and light fluids are razor-sharp, showing no signs of excessive numerical diffusion or smearing. Furthermore, the solution maintains perfect left-right symmetry throughout the entire time integration, which is a strong indicator of numerical stability and algorithmic correctness. *Note: Quantitative error is not applicable as no exact analytical ground truth is available for the fully nonlinear regime of this problem.*
* **Diagnosis:** High-Fidelity Resolution. The strategic adjustments to increase the polynomial degree (`polydeg=4`) and the maximum Adaptive Mesh Refinement level (`max_level=8`) have perfectly synergized. The AMR correctly tracked the density gradients, deploying high resolution exactly where the complex topological changes and shear layers occur, while the high-order DGSEM minimized dispersion errors.
* **Targeted Improvements:** 
    * **For Integrity:** None. Perfect execution.
    * **For Accuracy:** None. The physical behavior perfectly matches the expected nonlinear dynamics of the Rayleigh-Taylor instability under the specified upward gravity source term.
    * **For Details:** None. The contact discontinuities are exceptionally crisp, and there are no visible Gibbs oscillations. The shock-capturing and AMR strategies are optimally tuned.
    * **For Optimality:** None. Discontinuous Galerkin with density-based AMR is the state-of-the-art choice for this specific fluid dynamics problem.

**Verdict:** Perfect Score Achieved. No further refinement possible. The generated data is of High-Fidelity Ground Truth quality and is ready for PINN training.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Numerical DGSEM (Trixi.jl) solution of 2D compressible Euler equations with gravity source term on domain [0,0.25]x[0,1] for t in [0,3]. Polydeg=3, LobattoLegendreBasis, IndicatorHennemannGassner shock capturing on density, AMR via IndicatorLöhner on density with ControllerThreeLevel (levels 4-6). Initial conditions, piecewise pressure, tanh density, v perturbation with cos(8πx)sin(πy)^6, source term g=(0,1), periodic x / slip-wall y boundaries, and γ=1.4 all match exactly. Time integration uses RDPK3SpFSAL49 with CFL=0.5. Saves solution snapshots every dt=0.1 and produces summary_all.png (ρ and v at t=0,1,2,3). | N/A | 50 | 20 | 12 | 15 | **97** |
| 2 | Numerical DGSEM method on 2D compressible Euler equations with gravity source term. Uses polydeg=4, HennemannGassner shock capturing on density, AMR via Löhner indicator on density with max_level=8. Domain [0,0.25]x[0,1], periodic x / slip-wall y BCs, exact ICs (tanh density, piecewise p, v perturbation), tspan=[0,3]. Time integration via RDPK3SpFSAL49 with CFL stepping. Saves solution snapshots every dt=0.1. | N/A | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves multiple time_*.npz files (dt_save=0.1) under data_npz/ containing keys: 't' (scalar), 'x' (vector), 'y' (vector), 'rho' (vector), 'u' (vector), 'v' (vector), 'p' (vector). Also writes summary_all.png (2x4 layout of ρ and v fields). |
| 2 | Saves multiple time_*.npz files (dt_save=0.1) containing keys: 't' (scalar), 'x' (vector), 'y' (vector), 'rho' (vector), 'u' (vector), 'v' (vector), 'p' (vector). Also saves trixi_output/*.h5 via SaveSolutionCallback and final summary_all.png. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/RT/run4/results//CompressibleEuler2D_PeriodicReflective_RayleighTaylor_ICs/2026_11_06_1055/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/RT/run4/results//CompressibleEuler2D_PeriodicReflective_RayleighTaylor_ICs/2026_11_06_1101/` |


---
