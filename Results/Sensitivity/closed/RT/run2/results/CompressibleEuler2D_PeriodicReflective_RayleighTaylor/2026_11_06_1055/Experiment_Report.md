
# Experiment Report
*Last Updated: 2026-11-06 10:59:08*

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
The 2D compressible Euler equations with gravity in the positive \(y\)-direction are
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
\rho g\, v
\end{pmatrix},
\]
with equation of state
\[
p=(\gamma-1)\left(E-\frac{1}{2}\rho(u^2+v^2)\right).
\]

* **Domain:**
\[
(x,y)\in [0,0.25]\times[0,1.0], \qquad t\in[0,3.0].
\]

* **Boundary Conditions:**
  * \(x=0\) and \(x=0.25\): periodic.
  * \(y=0\) and \(y=1\): reflective / free-slip wall:
\[
v=0 \quad \text{at } y=0,1,
\]
with zero normal mass flux and slip tangential velocity.

* **Initial Conditions:**
  * Density:
\[
\rho(x,y,0)=\rho(y)=2.0+(1.0-2.0)\cdot \frac{1}{2}\left(1+\tanh\left(50(y-0.5)\right)\right).
\]
Equivalently,
\[
\rho(x,y,0)=2.0-\frac{1}{2}\left(1+\tanh\left(50(y-0.5)\right)\right).
\]

  * Pressure:
\[
p(x,y,0)=p(y)=
\begin{cases}
2y+1, & y<0.5,\\
y+1.5, & y\ge 0.5.
\end{cases}
\]

  * Horizontal velocity:
\[
u(x,y,0)=0.
\]

  * Vertical velocity:
\[
v(x,y,0)=-0.025\, c(x,y)\,\cos(8\pi x)\,\sin(\pi y)^6,
\]
where
\[
c(x,y)=\sqrt{\frac{\gamma p(x,y,0)}{\rho(x,y,0)}}.
\]

  * Total energy:
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

Interface location and smoothing:
\[
y_0=0.5,\qquad \alpha=50.
\]

* **Expected Physics:**
Rayleigh-Taylor instability driven by the unstable density stratification under upward gravity, with interface roll-up and growing plume/bubble structure near \(y=0.5\). Compressible wave activity is secondary; density is the primary structural field.

---

**2. Strategic Methodology**

* Can obtain an exact solution or similar?  
No. This is a genuinely 2D nonlinear compressible instability problem with nontrivial source terms, boundaries, and evolving interface deformation. No useful exact solution or Method of Characteristics reduction exists for the full problem.

* **Main method selection:**
Use **Discontinuous Galerkin Spectral Element Method in 2D plus time**.

Reason: the problem is time-dependent, compressible, inviscid, and develops increasingly sharp interface structure. DGSEM is the correct robust high-order method from the toolbox for this regime.

---

**3. Implementation Requirements (CRITICAL)**

* **Configuration:** use template defaults.

* **3.0.1 Adaptive Mesh Refinement & Shock Capturing:**
  * **AMR/Shock variable:** **Density \(\rho\)**.
  * **Justification:** The dominant coherent structure is the material interface between heavy and light fluids. Density remains the cleanest marker of the instability front, spikes, and bubbles. Velocity will become globally active and noisier as the instability develops, so refining on velocity would over-refine broad regions and reduce efficiency. Pressure is comparatively smoother because of hydrostatic initialization and acoustic redistribution. Therefore density is superior for feature-focused AMR and any indicator-driven shock/interface capturing.

* **3.1 Advisor-Mandated Adjustments (Overrides):**
  * None.

---

**4. Visualization Strategy:**

* **Field Selection (Analysis):**
  * Primary field 1: **Density \(\rho\)**.
  * Primary field 2: **Vertical momentum \(\rho v\)**.

Reason: density shows the instability morphology directly; vertical momentum shows gravity-driven motion and plume development without overcrowding the figure.

* **Task:** Plot these fields at **4 distinct time steps**.
A suitable choice:
\[
t=\{0.0,\ 1.0,\ 2.0,\ 3.0\}.
\]

* **Layout:** Grid with rows = fields, columns = time.

* **Saving Data:** Save the **full solution state** at frequent intervals, e.g.
\[
dt_{\text{save}}=0.1.
\]

* **Constraints:**
  * Save figure as: `summary_all.png`
  * If code is in Julia: **disable colorbars** with `colorbar=false`
  * Use automatic color scaling
  * Plot only the selected primary variables
  * Do not compute derived quantities

---

**5. Cross-Validation & Data Integration (Conditional)**

No external reference data or sensor files were provided. No cross-validation action required.

---

**REQUIRED COMPONENTS**

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
Shock Capturing  
Adaptive Mesh Refinement

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
**Section 1: Status Decision**
* **Status:** REJECTED_FOR_REFINEMENT

**Section 2: Quantitative Scorecard**
* **Score Integrity:** 50
* **Score Accuracy:** 20
* **Score Details:** 12
* **Score Optimality:** 15
* **TOTAL SCORE:** 97

**Section 3: Forensic Analysis & Feedback**
* **Observation:** The simulation successfully ran to completion and the logs show stable execution. Quantitative error is not applicable as no analytical ground truth is available for the fully nonlinear 2D Rayleigh-Taylor instability. The qualitative physics are excellent: the initial perturbation grows into the characteristic mushroom-shaped plume, and the symmetry is perfectly maintained throughout the domain. However, looking closely at the density field at $t=3.0$, the fine-scale structures (the secondary Kelvin-Helmholtz roll-ups on the sides of the plume) exhibit slight numerical diffusion. The contact discontinuity between the heavy and light fluids is slightly "soft" or smeared, which is a common artifact when using `polydeg=3` combined with hybrid shock capturing on an evolving interface.
* **Diagnosis:** Mild Numerical Dissipation at the contact discontinuity.
* **Targeted Improvements:**
    * **For Integrity:** None. Perfect execution.
    * **For Accuracy:** None. The macroscopic physics and boundary conditions are perfectly respected.
    * **For Details (Crucial for Euler):** To achieve perfect "Ground Truth Quality" (razor-sharp interfaces) for PINN training, we must reduce the numerical dissipation. **Proactive Fix:** Increase the polynomial degree `polydeg` from 3 to 4 to reduce the inherent numerical viscosity of the scheme. Additionally, to better resolve the cascading small scales of this inviscid instability, increase the maximum AMR level (e.g., update the controller to levels 4-7 instead of 4-6). 
    * **For Optimality:** None. DGSEM with AMR on density is the optimal choice for this problem.

* **Reason for Rejection:** We are in the 'Verification Phase'. A High Score (>90) was achieved, but we require 5 iterations of refinement to attempt a Perfect Score (100) before stopping. We must push the resolution limits to ensure the PINN training data is as sharp as computationally feasible.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Numerical simulation using 2D DGSEM (polydeg=3, LobattoLegendreBasis) on P4estMesh with AMR (IndicatorLöhner on density, ControllerThreeLevel levels 4-6) and hybrid shock capturing (IndicatorHennemannGassner on density). Solves compressible Euler equations with gravity source term (g=1.0) on domain [0,0.25]x[0,1] with periodic x / slip-wall y boundaries. Initial condition implements exact tanh density profile, piecewise linear pressure, and perturbation velocity. Time integration via RDPK3SpFSAL49 to t=3.0 with CFL stepping. Saves full solution snapshots (rho, rho_v, ...) every dt=0.1 via npz and produces summary_all.png showing rho and rho_v at t={0,1,2,3}. | N/A | 50 | 20 | 12 | 15 | **97** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves multiple time_*.npz files (dt_save=0.1) each containing keys: 't' (scalar), 'x' (vector), 'y' (vector), 'rho' (vector), 'rho_v' (vector), 'u' (vector), 'v' (vector), 'p' (vector). Also saves summary_all.png (PNG image, 2x4 layout of rho/rho_v fields). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/RT/run2/results//CompressibleEuler2D_PeriodicReflective_RayleighTaylor/2026_11_06_1055/` |


---
