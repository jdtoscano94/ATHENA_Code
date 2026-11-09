
# Experiment Report
*Last Updated: 2026-11-06 11:00:12*

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
\end{bmatrix}
\]
with equation of state
\[
p=(\gamma-1)\left(E-\frac{1}{2}\rho(u^2+v^2)\right).
\]

* **Domain:**  
\[
(x,y)\in[0,0.25]\times[0,1.0], \qquad t\in[0,3.0].
\]

* **Boundary Conditions:**  
  * \(x\): periodic.  
  * \(y=0\) and \(y=1\): reflective / free-slip walls with no penetration,
\[
v=0.
\]

* **Initial Conditions:**  
Density:
\[
\rho(x,y,0)=\rho_{\text{heavy}}+(\rho_{\text{light}}-\rho_{\text{heavy}})\frac{1}{2}\left(1+\tanh\left(50(y-0.5)\right)\right),
\]
where
\[
\rho_{\text{heavy}}=2.0,\qquad \rho_{\text{light}}=1.0.
\]

Pressure:
\[
p(x,y,0)=
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
Growth of the Rayleigh-Taylor instability from the perturbed density interface, with rising light-fluid bubbles, descending heavy-fluid spikes, and increasing interface deformation and mixing.

---

**2. Strategic Methodology**
* Can obtain an exact solution or similar?  
No. This is a nonlinear 2D compressible Euler instability with gravity and evolving interface structure. No exact analytical solution is feasible.

* **Selected method:**  
**Discontinuous Galerkin Spectral Element Method in 2D plus time**

This is the best choice for a hyperbolic compressible flow with steepening gradients and interface-dominated dynamics.

---

**3. Implementation Requirements (CRITICAL)**
* **Configuration:** use template defaults.

#### **IF DGSEM (Discontinuous Galerkin) is selected:**
* **3.0.1 Adaptive Mesh Refinement & Shock Capturing:**
  * **Variable for AMR/Shock Capturing:** **Density \(\rho\)**
  * **Justification:** Density is the sharpest and most physically relevant marker of the heavy/light interface. Velocity becomes more spatially noisy as the instability develops, and pressure is smoother than the material interface. Therefore density is the correct indicator for targeted refinement and shock/interface capturing.

* **3.1 Advisor-Mandated Adjustments (Overrides):**
  * None.

---

**4. Visualization Strategy:**
* **Field Selection (Analysis):**
  * **Density \(\rho\)**
  * **Vertical momentum \(\rho v\)**

These are the most relevant primary fields for the instability: density captures the interface morphology, while vertical momentum captures buoyancy-driven motion.

* **Task:** Create an appropriate figure using these primary fields.

* **Saving Data:** Save the full solution state using template defaults for Ground Truth.

* **Constraints:**
  * Save as:
\[
\texttt{summary\_all.png}
\]
  * Use automatic plotting defaults.
  * Plot only primary variables.

---

**5. Cross-Validation & Data Integration (Conditional)**
No external data, sensors, or reference solution paths were provided. No cross-validation requirement applies.

---

**REQUIRED COMPONENTS**
**Main Method:**  
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**  
Shock Capturing  
AMR on Density Indicator

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
* **Observation:** The simulation successfully captures the nonlinear development of the Rayleigh-Taylor instability. The symmetry of the rising bubble and falling spikes is perfectly maintained even at late times ($t=3.0$), which indicates a highly stable and well-implemented numerical scheme. Furthermore, the compressibility effects are physically accurate: as the heavy fluid ($\rho=2.0$) rises into the higher-pressure region at the top of the domain ($p=2.5$), it compresses, resulting in a maximum density of $\approx 2.5$, which perfectly aligns with isentropic compression expectations. However, looking closely at the late-stage Kelvin-Helmholtz roll-ups at $t=3.0$, the thin filaments of the density interface exhibit slight smearing and diffusion. *Note: Quantitative error is not applicable as no ground truth is available for this forward simulation.*
* **Diagnosis:** Numerical Dissipation at Contact Discontinuities. The `flux_hll` surface flux is known to be overly dissipative for contact waves (such as the material density interface in this RT instability). While `polydeg=3` provides good baseline resolution, the combination of the HLL flux and the shock-capturing blending diffuses the sharpest topological features of the interface during late-stage mixing.
* **Targeted Improvements:**
    * **For Integrity:** None. Perfect execution.
    * **For Accuracy:** None. Physics and symmetry are excellent.
    * **For Details:** To achieve perfect Ground Truth quality (razor-sharp interfaces) and reach a score of 100, we must reduce numerical dissipation at the material interface. **Proactive Fix:** Switch the surface flux from `flux_hll` to **`flux_hllc`** (which is specifically designed to exactly resolve contact discontinuities). Additionally, increase **`polydeg` to 4** to further sharpen the roll-ups without adding excessive computational cost.
    * **For Optimality:** None. DGSEM with AMR on density is the optimal strategy here.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Numerical DGSEM implementation (Trixi.jl) for 2D compressible Euler equations with gravity source. Polydeg=3, LobattoLegendreBasis, HennemannGassner shock capturing on density, HLL surface flux, Ranocha/Lax-Friedrichs volume fluxes. AMR via IndicatorLöhner on density with ControllerThreeLevel. Domain [0,0.25]x[0,1], tspan=[0,3], periodic x / slip-wall y BCs. Exact match to specified IC (tanh density, piecewise p, v perturbation with cos(8πx)sin(πy)^6) and source term. Saves trixi_output via SaveSolutionCallback (template default) plus summary_all.png containing ρ and ρv fields. | N/A | 50 | 20 | 13 | 15 | **98** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves trixi_output/*.h5 via SaveSolutionCallback (full conservative/primitive state at interval=20). Also saves data_npz/time_*.npz containing keys 't' (scalar), 'x'/'y' (2D arrays), 'rho'/'u'/'v'/'p' (2D arrays). Finally saves summary_all.png (PNG image, 2x4 scatter layout of ρ and ρv at 4 time snapshots). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/RT/run1/results//CompressibleEuler2D_Periodic_Reflective_RayleighTaylor/2026_11_06_1055/` |


---
