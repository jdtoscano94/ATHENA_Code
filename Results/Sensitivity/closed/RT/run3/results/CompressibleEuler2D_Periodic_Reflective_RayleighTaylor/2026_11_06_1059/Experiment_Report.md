
# Experiment Report
*Last Updated: 2026-11-06 11:03:21*

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
For the 2D compressible Euler equations with gravity acting in the positive \(y\)-direction,
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
\end{bmatrix},
\]
with equation of state
\[
p = (\gamma-1)\left(E - \frac{1}{2}\rho(u^2+v^2)\right).
\]

* **Domain:**
\[
(x,y)\in[0,0.25]\times[0,1.0], \qquad t\in[0,3.0].
\]

* **Boundary Conditions:**
  * \(x=0\) and \(x=0.25\): periodic.
  * \(y=0\) and \(y=1\): reflective / free-slip wall:
\[
v=0,
\]
with zero normal mass flux and tangential slip allowed.

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

  Local sound speed:
\[
c(x,y,0)=\sqrt{\frac{\gamma p(y)}{\rho(y)}}.
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
\gamma=1.4, \qquad g=1.0, \qquad \mu=0, \qquad k=0.
\]

* **Expected Physics:**
Rayleigh-Taylor instability driven by gravity, with interface roll-up and increasing small-scale structure near \(y=0.5\). The smoothed density interface steepens and may develop under-resolved compressible features; robustness near sharp gradients is required.

---

**2. Strategic Methodology**

* Can obtain an exact solution or similar?  
No. This is a nonlinear 2D compressible Euler instability problem with nontrivial interface growth and no tractable exact solution. Method of Characteristics is not feasible for the full 2D system with this geometry and source forcing.

* **Main method selection:**  
Use **Discontinuous Galerkin Spectral Element Method in 2D plus time**.

Reason: the problem is time-dependent, compressible, inviscid, and develops sharp density-gradient structures. DGSEM is the correct robust high-order choice from the toolbox.

---

**3. Implementation Requirements (CRITICAL)**

* **Configuration:** use template defaults.

* **3.0.1 Adaptive Mesh Refinement & Shock Capturing:**
  * **Variable for AMR/Shock Capturing:** **Density \(\rho\)**.
  * **Justification:** The dominant coherent structure is the Rayleigh-Taylor material interface, which is defined most cleanly by the density jump/smoothed transition. Velocity will become progressively more oscillatory and spatially broad as instability develops, making it a poor refinement signal. Pressure is initially hydrostatic and less sharply localized to the interface. Therefore, \(\rho\) has the best signal-to-noise ratio for tracking the physically relevant instability structures.

* **3.1 Advisor-Mandated Adjustments (Overrides):**
  * None.

---

**4. Visualization Strategy:**

* **Field Selection (Analysis):**
  * Plot **Density \(\rho\)** and **Vertical velocity \(v\)**.
  * Reason: density captures the primary instability morphology and interface evolution; vertical velocity captures the buoyancy-driven motion.

* **Task:** Plot these fields at **4 distinct time steps**.

* **Recommended times:**
\[
t=\{0.0,\ 1.0,\ 2.0,\ 3.0\}.
\]

* **Layout:**  
Rows = fields \((\rho, v)\), Columns = time snapshots.

* **Saving Data:**  
Save the **full solution state** at frequent intervals, e.g.
\[
dt_{\text{save}}=0.1.
\]

* **Constraints:**
  * Save figure as: `summary_all.png`
  * If the code is in Julia, explicitly set: `colorbar=false`
  * Use automatic scaling; no fixed colormaps
  * Plot only primary variables \(\rho\) and \(v\)

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

*Note: Quantitative error is not applicable as no ground truth is available for this nonlinear 2D instability problem. Accuracy is judged based on physical behavior, symmetry preservation, and feature resolution.*

**Section 3: Forensic Analysis & Feedback**
* **Observation:** The simulation successfully captures the macroscopic physics of the Rayleigh-Taylor instability. The symmetry of the initial perturbation is perfectly maintained throughout the run, and the classic mushroom-shaped structures develop as expected by $t=2.0$ and $t=3.0$. However, a close inspection of the density field at $t=0.0$ reveals a visible "sawtooth" or banding artifact along the horizontal interface. Additionally, at $t=3.0$, the finest scales of the secondary Kelvin-Helmholtz roll-ups on the mushroom cap exhibit slight grid-imprinting and could be sharper.
* **Diagnosis:** Projection Error / Mild Gibbs Phenomenon. The initial density interface uses a very steep $\tanh$ function (scaling factor of 50). Projecting this sharp gradient onto a relatively low-order polynomial basis (`polydeg=3`) causes sub-element oscillatory artifacts (the sawtooth pattern). As the simulation progresses, this limits the sharpness of the contact discontinuities.
* **Targeted Improvements:**
    * **For Integrity:** None. The code is perfectly stable.
    * **For Accuracy:** None. The macroscopic physics and symmetry are correct.
    * **For Details:** To achieve perfect Ground Truth quality and eliminate the initial projection artifacts, we must increase the resolving power of the scheme. **Proactive Fix: Increase `polydeg` from 3 to 4 (or 5).** Higher-order polynomials will significantly improve the spectral resolution of the steep initial gradient (smoothing out the $t=0$ sawtooth) and provide razor-sharp contact discontinuities during the late-stage roll-ups at $t=3.0$.
    * **For Optimality:** None. DGSEM with AMR is the correct architectural choice for this problem.

*Decision Logic:* We are in the 'Verification Phase'. A High Score (>90) was achieved on the first attempt, but we require up to 5 iterations of refinement to attempt a Perfect Score (100) before stopping. We will refine the polynomial degree to eliminate the visual artifacts.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Numerical DGSEM (polydeg=3) solution of 2D compressible Euler equations with gravity source on domain [0,0.25]x[0,1] for t in [0,3]. Uses P4estMesh with periodic x / slip-wall y boundaries, initial_condition_rt and source_terms_rt exactly as specified, IndicatorHennemannGassner shock capturing and IndicatorLöhner AMR both keyed on density, LSRK time integration, and dt_save=0.1 output. | N/A | 50 | 20 | 12 | 15 | **97** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves multiple time_*.npz files (dt=0.1) containing keys: 't' (scalar), 'x' (2D), 'y' (2D), 'rho' (2D), 'u' (2D), 'v' (2D), 'p' (2D); also produces summary_all.png. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/RT/run3/results//CompressibleEuler2D_Periodic_Reflective_RayleighTaylor/2026_11_06_1059/` |


---
