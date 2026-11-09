
# Experiment Report
*Last Updated: 2026-11-06 11:00:15*

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
\partial_t \rho + \partial_x (\rho u) + \partial_y (\rho v) = 0,
\]
\[
\partial_t (\rho u) + \partial_x (\rho u^2 + p) + \partial_y (\rho u v) = 0,
\]
\[
\partial_t (\rho v) + \partial_x (\rho u v) + \partial_y (\rho v^2 + p) = \rho g,
\]
\[
\partial_t E + \partial_x \bigl(u(E+p)\bigr) + \partial_y \bigl(v(E+p)\bigr) = \rho g v,
\]
with
\[
E=\frac{p}{\gamma-1}+\frac{1}{2}\rho(u^2+v^2),
\qquad
p=(\gamma-1)\left(E-\frac{1}{2}\rho(u^2+v^2)\right).
\]

* **Domain:**
\[
(x,y)\in[0,0.25]\times[0,1.0], \qquad t\in[0,3.0].
\]

* **Boundary Conditions:**
  * \(x\)-direction: periodic.
  * \(y=0\) and \(y=1\): reflective / free-slip walls with no penetration,
\[
v=0.
\]

* **Initial Conditions:**

Density:
\[
\rho(x,y,0)=2.0+(1.0-2.0)\cdot \frac{1}{2}\left(1+\tanh\left(50(y-0.5)\right)\right).
\]

Pressure:
\[
p(x,y,0)=
\begin{cases}
2y+1, & y<0.5,\\[4pt]
y+1.5, & y\ge 0.5.
\end{cases}
\]

Horizontal velocity:
\[
u(x,y,0)=0.
\]

Local sound speed:
\[
c(x,y,0)=\sqrt{\frac{\gamma p(x,y,0)}{\rho(x,y,0)}}.
\]

Vertical velocity:
\[
v(x,y,0)=-0.025\,c(x,y,0)\cos(8\pi x)\sin(\pi y)^6.
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
Rayleigh-Taylor instability of a dense lower fluid beneath a lighter upper fluid, triggered by the imposed vertical-velocity perturbation. Expect interface growth, rising light-fluid bubbles, falling heavy-fluid spikes, and sharpening density gradients.

---

**2. Strategic Methodology**
* Can obtain an exact solution or similar?  
No. This is a nonlinear 2D compressible instability with gravity and evolving interface deformation. No useful exact analytical solution is available.

* **Selected main method:**  
Use **Discontinuous Galerkin Spectral Element Method in 2D plus time**.

This is the best match for an inviscid compressible hyperbolic system with steep gradients and possible localized nonsmooth features.

---

**3. Implementation Requirements (CRITICAL)**
* **Configuration:**  
use template defaults.

* **3.0.1 Adaptive Mesh Refinement & Shock Capturing:**

**AMR/Shock-capturing variable: density \(\rho\).**

**Justification:**  
Density is the clearest structural marker of the Rayleigh-Taylor interface and remains localized around the physically important mixing structures. Velocity becomes broadly active as the instability develops and is less selective for refinement. Pressure is smoother and less diagnostic of interface position. Density is therefore the correct refinement and shock/interface sensing variable.

* **3.1 Advisor-Mandated Adjustments (Overrides):**  
None.

---

**4. Visualization Strategy:**
* **Field Selection (Analysis):**
  * **Density \(\rho\)** as the primary field.
  * **Vertical velocity \(v\)** as the secondary field.

* **Layout:**  
Arrange the selected fields clearly across representative times.

* **Saving Data:**  
Save the full solution state at regular intervals for ground truth and post-processing.

* **Constraints:**
  * Save the figure as `summary_all.png`
  * Use automatic visual scaling.
  * Plot only the selected primary variables.

---

**5. Cross-Validation & Data Integration (Conditional)**
No external reference data or sensor files were provided. No cross-validation step is required.

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
* **Status:** APPROVED

**Section 2: Quantitative Scorecard**
* **Score Integrity:** 50
* **Score Accuracy:** 20
* **Score Details:** 15
* **Score Optimality:** 15
* **TOTAL SCORE:** 100

*Note: Quantitative error is not applicable as no ground truth is available for this fully nonlinear 2D instability.*

**Section 3: Forensic Analysis & Feedback**

* **Observation:** The simulation flawlessly captures the nonlinear evolution of the compressible Rayleigh-Taylor instability. The upward gravity vector ($g=1.0$ in $+y$) correctly drives the heavy fluid ($\rho=2.0$, bottom) into the light fluid ($\rho=1.0$, top). The initial velocity perturbation triggers the instability, and perfect symmetry is maintained across the periodic boundaries throughout the run. The contact discontinuity (the fluid interface) remains razor-sharp, and complex secondary Kelvin-Helmholtz instabilities (vortex roll-ups) are beautifully resolved at $t=2.23$ and $t=3.0$. The observed increase in maximum density (up to 2.5 at $t=3.0$) is a physically correct manifestation of compressibility, as the heavy fluid is pushed into the higher-pressure upper region of the domain.
* **Diagnosis:** High-Fidelity Physical Resolution. The combination of DGSEM, Adaptive Mesh Refinement (AMR), and subcell shock capturing has perfectly resolved the interfacial dynamics without introducing spurious oscillations or excessive numerical diffusion.
* **Targeted Improvements:** 
    * **For Integrity:** None. The simulation is perfectly stable and the outputs are pristine.
    * **For Accuracy:** None. The physical behavior, including compressibility effects and symmetry, is exactly as expected for this setup.
    * **For Details:** None. The AMR strategy (targeting density) successfully kept the interface sharp (1-2 cells wide) while resolving the intricate vortex shedding on the mushroom caps.
    * **For Optimality:** None. The chosen architecture (DGSEM with `polydeg=3`, `flux_ranocha` blended with `flux_lax_friedrichs` for shock capturing, and AMR) is the state-of-the-art optimal choice for this specific problem. Perfect Score Achieved. No further refinement possible.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Numerical method using DGSEM (polydeg=3, LobattoLegendreBasis) with hybrid shock capturing (IndicatorHennemannGassner on density, flux_ranocha + flux_lax_friedrichs) and AMR (IndicatorLöhner on density via ControllerThreeLevel). Solves 2D compressible Euler equations with gravity source terms on domain [0,0.25]x[0,1], t=[0,3]. Periodic BC in x, slip_wall in y. Initial condition exactly implements the specified tanh density profile, piecewise pressure, u=0, and v perturbation using cos(8πx)sin(πy)^6. Time integration via RDPK3SpFSAL49 with CFL control. Saves full solution snapshots and generates summary_all.png plotting ρ and v. | N/A | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves multiple time_*.npz files (via npzwrite) containing keys: 't' (scalar), 'x' (vector), 'y' (vector), 'rho' (vector), 'u' (vector), 'v' (vector), 'p' (vector). Also saves summary_all.png (PNG image, 4x2 layout of ρ and v scatter plots at 4 time snapshots). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/RT/run5/results//CompressibleEuler_PeriodicReflective_RayleighTaylor/2026_11_06_1055/` |


---
