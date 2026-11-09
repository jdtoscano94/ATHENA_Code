
# Experiment Report
*Last Updated: 2026-11-06 10:59:43*

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

The 2D compressible Euler equations with gravity source term are
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
with equation of state
\[
p=(\gamma-1)\left(E-\frac{1}{2}\rho(u^2+v^2)\right).
\]

* **Domain:**
\[
(x,y)\in [0,0.25]\times[0,1.0],\qquad t\in[0,3.0].
\]

* **Boundary Conditions:**
  * \(x\)-direction: periodic.
  * \(y=0\) and \(y=1\): reflective / free-slip wall:
    \[
    v=0,\qquad \partial_y \rho = 0,\qquad \partial_y u = 0,\qquad \partial_y p = \text{consistent with wall treatment}.
    \]

* **Initial Conditions:**

Density:
\[
\rho(x,y,0)=\rho(y)=2.0+(1.0-2.0)\cdot \frac{1}{2}\left(1+\tanh\left(50(y-0.5)\right)\right).
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
c(x,y,0)=\sqrt{\frac{\gamma p(y)}{\rho(y)}}.
\]

Vertical velocity:
\[
v(x,y,0)=-0.025\, c(x,y,0)\cos(8\pi x)\sin(\pi y)^6.
\]

Total energy:
\[
E(x,y,0)=\frac{p(x,y,0)}{\gamma-1}+\frac{1}{2}\rho(x,y,0)\left(u(x,y,0)^2+v(x,y,0)^2\right).
\]

* **Constants & Parameters:**
\[
\gamma=1.4,\qquad g=1.0,
\]
\[
\rho_{\text{heavy}}=2.0,\qquad \rho_{\text{light}}=1.0.
\]
Inviscid and non-conducting:
\[
\mu=0,\qquad k=0.
\]

* **Expected Physics:**
Rayleigh-Taylor instability from a heavy fluid below a light fluid under gravity acting in \(+y\), with interface deformation, plume growth, nonlinear roll-up, and eventual sharp gradients near the interface.

---

**2. Strategic Methodology**

* Can obtain an exact solution or similar?  
No. This is a fully nonlinear, 2D, variable-density compressible instability with gravity and nontrivial boundaries. No useful global exact solution or Method of Characteristics reduction exists for the full problem.

* **Main numerical strategy:**  
Use **Discontinuous Galerkin Spectral Element Method in 2D plus time**.

Reason: compressible Euler, nonlinear wave dynamics, evolving sharp interface gradients, and possible under-resolved steepening require a conservative high-order hyperbolic solver with robust shock/gradient handling.

---

**3. Implementation Requirements (CRITICAL)**

* **Configuration:** use template defaults.

* **3.0.1 Adaptive Mesh Refinement & Shock Capturing:**

**AMR/Shock-capturing variable: Density \(\rho\).**

**Justification:**  
The dominant coherent structure is the material interface between heavy and light fluids. Density remains the clearest marker of the Rayleigh-Taylor layer and plume structure. Velocity will become globally complex during nonlinear growth and is therefore a poor refinement indicator because it spreads activity across the whole domain. Pressure is comparatively smoother due to near-hydrostatic initialization and compressibility effects. Therefore, \(\rho\) gives the best signal-to-noise ratio for tracking the physically relevant fronts and mixing structures.

* **3.1 Advisor-Mandated Adjustments (Overrides):**
None.

---

**4. Visualization Strategy**

* **Field Selection (Analysis):**
  * **Density \(\rho\)** — primary diagnostic of the instability and mixing layer.
  * **Vertical velocity \(v\)** — captures the growth and direction of buoyancy-driven motion.

* **Task:** Plot these selected fields at **4 distinct time steps**.

Recommended times:
\[
t=\{0.0,\ 1.0,\ 2.0,\ 3.0\}.
\]

* **Layout:** Grid with rows = fields, columns = time.

* **Saving Data:** Save the full solution state at frequent intervals:
\[
dt_{\text{save}}=0.1.
\]

* **Constraints:**
  * Save figure as: `summary_all.png`
  * **Disable colorbars**: `colorbar=false` if code is in Julia.
  * Use automatic scaling; do not fix colormaps.
  * Plot only primary variables \(\rho\) and \(v\).
  * Do not compute derived quantities.

---

**5. Cross-Validation & Data Integration (Conditional)**

No external data paths or reference files were provided.  
No sensor interpolation or relative \(L_2\) error computation is required.

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

*Note: Quantitative error is not applicable as no analytical ground truth is available for this fully nonlinear 2D instability.*

**Section 3: Forensic Analysis & Feedback**

* **Observation:** The simulation successfully executes and captures the complex physics of the Rayleigh-Taylor instability. The setup is physically consistent: with gravity acting in the $+y$ direction, the heavy fluid at the bottom ($\rho=2.0$) is buoyantly unstable and accelerates into the light fluid ($\rho=1.0$) at the top. The plots clearly show the formation of the classic mushroom-shaped plumes, nonlinear roll-up, and secondary instabilities by $t=3.0$. The Adaptive Mesh Refinement (AMR) and shock capturing successfully maintain stability without crashing. However, looking closely at the density field at $t=3.0$, the material interface (contact discontinuity) between the heavy and light fluids exhibits visible smearing/diffusion. It is somewhat "soft" rather than razor-sharp.
* **Diagnosis:** Numerical Dissipation at the contact discontinuity. While the Hennemann-Gassner shock capturing is excellent for stability, it introduces inherent dissipation that smears material interfaces over time, preventing the generation of a perfect, zero-thickness ground truth for PINN training.
* **Targeted Improvements:**
    * **For Integrity:** None. The execution is flawless.
    * **For Accuracy:** None. The macroscopic physics, boundary conditions, and hydrostatic balance are perfectly respected.
    * **For Details:** To elevate the interface resolution from "Standard CFD" to "Ground Truth Quality" (Score 15), we must reduce the numerical dissipation at the interface. **Proactive Fix:** Increase the maximum AMR level (e.g., from `level=6` to `level=7` or `8`) to force the grid to be significantly finer exactly at the density gradients. Additionally, increasing the polynomial degree from `polydeg=3` to `polydeg=4` will reduce dispersion errors and better resolve the tight vortical structures in the mushroom caps.
    * **For Optimality:** None. DGSEM with AMR is the correct architectural choice for this problem.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Numerical DGSEM (Trixi.jl) solution of 2D compressible Euler equations with gravity source term on domain [0,0.25]x[0,1] for t in [0,3]. Polydeg=3, LobattoLegendreBasis, IndicatorHennemannGassner shock capturing on density, AMR via IndicatorLöhner on density with ControllerThreeLevel (levels 4-6). Initial conditions, piecewise pressure, tanh density, v perturbation with cos(8πx)sin(πy)^6, source term g=(0,1), periodic x / slip-wall y boundaries, and γ=1.4 all match exactly. Time integration uses RDPK3SpFSAL49 with CFL=0.5. Saves solution snapshots every dt=0.1 and produces summary_all.png (ρ and v at t=0,1,2,3). | N/A | 50 | 20 | 12 | 15 | **97** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves multiple time_*.npz files (dt_save=0.1) under data_npz/ containing keys: 't' (scalar), 'x' (vector), 'y' (vector), 'rho' (vector), 'u' (vector), 'v' (vector), 'p' (vector). Also writes summary_all.png (2x4 layout of ρ and v fields). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/RT/run4/results//CompressibleEuler2D_PeriodicReflective_RayleighTaylor_ICs/2026_11_06_1055/` |


---
