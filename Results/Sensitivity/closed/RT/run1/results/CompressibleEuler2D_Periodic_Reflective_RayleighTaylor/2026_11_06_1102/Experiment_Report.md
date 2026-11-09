
# Experiment Report
*Last Updated: 2026-11-06 11:10:51*

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
\rho \\
\rho u \\
\rho v \\
E
\end{pmatrix}
+
\partial_x
\begin{pmatrix}
\rho u \\
\rho u^2 + p \\
\rho u v \\
u(E+p)
\end{pmatrix}
+
\partial_y
\begin{pmatrix}
\rho v \\
\rho u v \\
\rho v^2 + p \\
v(E+p)
\end{pmatrix}
=
\begin{pmatrix}
0 \\
0 \\
\rho g \\
\rho g v
\end{pmatrix},
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
\[
\rho(y)=\rho_{\text{heavy}}+(\rho_{\text{light}}-\rho_{\text{heavy}})\cdot \frac{1}{2}\left(1+\tanh\left(50(y-0.5)\right)\right),
\]
with
\[
\rho_{\text{heavy}}=2.0,\qquad \rho_{\text{light}}=1.0.
\]

\[
p(y)=
\begin{cases}
2y+1, & y<0.5,\\
y+1.5, & y\geq 0.5.
\end{cases}
\]

\[
u(x,y)=0.
\]

\[
c(x,y)=\sqrt{\frac{\gamma p(y)}{\rho(y)}}.
\]

\[
v(x,y)=-0.025\,c(x,y)\cos(8\pi x)\sin(\pi y)^6.
\]

\[
E(x,y)=\frac{p(y)}{\gamma-1}+\frac{1}{2}\rho(y)\left(u(x,y)^2+v(x,y)^2\right).
\]

* **Constants & Parameters:**
\[
\gamma=1.4,\qquad g=1.0,\qquad \mu=0,\qquad k=0.
\]

* **Expected Physics:**
Buoyancy-driven Rayleigh-Taylor growth from the perturbed density interface at \(y=0.5\), followed by nonlinear bubble/spike development and late-stage Kelvin-Helmholtz roll-up along the contact layer.

---

**2. Strategic Methodology**
* Can obtain an exact solution or similar?  
No. This is a nonlinear, forced, multidimensional compressible instability problem with no practical exact solution for the full evolution.

* Best method:  
Use **Discontinuous Galerkin Spectral Element Method in 2D plus time**.

This is the correct choice for a hyperbolic compressible flow with a sharp material interface, gravity source, and wall/periodic mixed boundaries. Trial history already confirms physical fidelity; the remaining issue is interface dissipation, not method mismatch.

---

**3. Implementation Requirements (CRITICAL)**
* **Configuration:** use template defaults.

#### **IF DGSEM (Discontinuous Galerkin) is selected:**
* **3.0.1 Adaptive Mesh Refinement & Shock Capturing:**
  * **Variable for AMR/Shock Capturing:** **Density \(\rho\)**.
  * **Justification:**  
  Density is the cleanest marker of the heavy/light material interface and remains sharply localized to the physically relevant structures. Velocity becomes increasingly oscillatory and broadly active during vortex roll-up, making it a poor refinement indicator. Pressure is smoother and less diagnostic of interface topology. Therefore, density is the superior feature variable.

* **3.1 Advisor-Mandated Adjustments (Overrides):**
  * Switch surface flux to **`flux_hllc`**.
  * Increase **`polydeg` to 4**.

---

**4. Visualization Strategy:**
* **Field Selection (Analysis):**
  * **Density \(\rho\)**
  * **Vertical momentum \(\rho v\)**

Density is the primary observable for interface integrity and mixing morphology. Vertical momentum best captures the buoyancy-driven transport and symmetry of the instability.

* **Task:** Plot these selected fields at **4 distinct time steps**.
* **Layout:** Grid with rows = selected fields, columns = time.
* **Saving Data:** Always save the **full solution state** at frequent intervals for Ground Truth.
* **Constraints:**
  * Save as **`summary_all.png`**.
  * In Julia, explicitly set **`colorbar=false`**.
  * Use automatic scaling.
  * Plot only the primary variables above.

---

**5. Cross-Validation & Data Integration (Conditional)**
No external reference data or sensor files were provided.

---

**REQUIRED COMPONENTS**

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
Shock Capturing Indicator  
Adaptive Mesh Refinement  
`flux_hllc`  
`polydeg=4`

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
* **Observation:** The simulation flawlessly captures the complex, nonlinear evolution of the Rayleigh-Taylor instability. The density field clearly exhibits the classic heavy-fluid bubble rising and light-fluid spikes falling (driven by the $+y$ gravity source). By $t=2.45$ and $t=3.0$, intricate Kelvin-Helmholtz roll-ups are beautifully resolved along the shear layers of the spikes. Crucially, the material interfaces remain razor-sharp without excessive numerical diffusion, and the left-right symmetry of the initial perturbation is perfectly preserved deep into the chaotic, nonlinear regime. Quantitative error is not applicable as no analytical ground truth is available for the full evolution of this instability.
* **Diagnosis:** High-Fidelity Physical Resolution. The strategic shift to the `flux_hllc` surface flux (which explicitly includes the contact wave missing in standard HLL) combined with an increased polynomial degree (`polydeg=4`) and density-targeted Adaptive Mesh Refinement (AMR) has completely eliminated the interfacial smearing seen in standard dissipative schemes. 
* **Targeted Improvements:** 
    * **For Integrity:** None. Perfect execution.
    * **For Accuracy:** None. Physics and symmetries are perfectly respected.
    * **For Details:** None. The interfaces are crisp and the secondary instabilities are highly detailed without spurious Gibbs oscillations.
    * **For Optimality:** None. DGSEM with HLLC and AMR is the optimal configuration for this compressible interfacial problem. 

**Conclusion:** Perfect Score Achieved. The current configuration yields exceptional, high-fidelity ground truth data suitable for PINN training. No further refinement is necessary.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Numerical DGSEM implementation (Trixi.jl) for 2D compressible Euler equations with gravity source. Polydeg=3, LobattoLegendreBasis, HennemannGassner shock capturing on density, HLL surface flux, Ranocha/Lax-Friedrichs volume fluxes. AMR via IndicatorLöhner on density with ControllerThreeLevel. Domain [0,0.25]x[0,1], tspan=[0,3], periodic x / slip-wall y BCs. Exact match to specified IC (tanh density, piecewise p, v perturbation with cos(8πx)sin(πy)^6) and source term. Saves trixi_output via SaveSolutionCallback (template default) plus summary_all.png containing ρ and ρv fields. | N/A | 50 | 20 | 13 | 15 | **98** |
| 2 | Numerical DGSEM simulation of 2D compressible Euler equations with gravity source. Uses polydeg=4, flux_hllc surface flux, HennemannGassner shock capturing on density, AMR via Löhner indicator on density. Domain [0,0.25]x[0,1], tspan=[0,3], periodic x / slip-wall y BCs. Initial condition implements exact tanh density profile, piecewise p(y), and v perturbation with cos(8πx)sin(πy)^6. Saves full solution via SaveSolutionCallback + custom NPZ snapshots; generates summary_all.png (2x4 layout of ρ and ρv at 4 times, colorbar=false). | N/A | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves trixi_output/*.h5 via SaveSolutionCallback (full conservative/primitive state at interval=20). Also saves data_npz/time_*.npz containing keys 't' (scalar), 'x'/'y' (2D arrays), 'rho'/'u'/'v'/'p' (2D arrays). Finally saves summary_all.png (PNG image, 2x4 scatter layout of ρ and ρv at 4 time snapshots). |
| 2 | Saves multiple time_*.npz files (via custom DiscreteCallback every 40 steps) containing keys: 't' (scalar), 'x' (vector), 'y' (vector), 'rho' (vector), 'u' (vector), 'v' (vector), 'p' (vector). Also saves Trixi HDF5 solution files via SaveSolutionCallback (interval=20). Final visualization artifact: summary_all.png. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/RT/run1/results//CompressibleEuler2D_Periodic_Reflective_RayleighTaylor/2026_11_06_1055/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/RT/run1/results//CompressibleEuler2D_Periodic_Reflective_RayleighTaylor/2026_11_06_1102/` |


---
