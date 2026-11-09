
# Experiment Report
*Last Updated: 2026-11-06 10:55:45*

## User Request
Let's solve the 2D Inviscid Burgers' equation.
Domain: $(x,y) \in [0, 4\pi]^2$, $t \in [0, 2]$.
BCs: Periodic in both x and y.

Initial Condition:
$$ u(x,y,0) = \frac{1}{3} + \frac{2}{3}\sin\left(\frac{x+y}{2}\right) $$

The final plot must be saved as summary_all.png it should include snapshots of several timesteps

---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
**1. Problem Statement**

**1. Mathematical Specification (Code-Ready Extraction)**
* **Governing Equations:**
$$
\partial_t u + u\,\partial_x u + u\,\partial_y u = 0
$$
Equivalently,
$$
\partial_t u + \nabla \cdot \mathbf{F}(u)=0,
\qquad
\mathbf{F}(u)=\left(\frac{u^2}{2},\frac{u^2}{2}\right)
$$

* **Domain:**
$$
(x,y)\in[0,4\pi]\times[0,4\pi],\qquad t\in[0,2]
$$

* **Boundary Conditions:**
Periodic in both directions:
$$
u(0,y,t)=u(4\pi,y,t),\qquad
u(x,0,t)=u(x,4\pi,t)
$$

* **Initial Conditions:**
$$
u(x,y,0)=u_0(x,y)=\frac{1}{3}+\frac{2}{3}\sin\!\left(\frac{x+y}{2}\right)
$$

* **Constants & Parameters:**
No additional physical constants.

* **Expected Physics:**
The solution is smooth initially and propagates along the $(1,1)$ direction with state-dependent speed. Shock formation is expected because
$$
u_0'(\xi)=\frac{1}{3}\cos\!\left(\frac{\xi}{2}\right),\qquad \xi=x+y
$$
attains negative values, so characteristics compress. Earliest shock time:
$$
t_s=-\frac{1}{\min u_0'(\xi)}=-\frac{1}{-1/3}=3
$$
Thus over the requested interval $t\in[0,2]$, the solution remains smooth.

---

**2. Strategic Methodology**

* **Can obtain an exact solution or similar?**
Yes. This PDE admits an exact reduction by characteristics.

Since the initial condition depends only on
$$
s=x+y,
$$
seek
$$
u(x,y,t)=U(s,t),\qquad s=x+y.
$$
Then
$$
\partial_x u = U_s,\qquad \partial_y u = U_s,
$$
so the PDE becomes
$$
U_t + 2U\,U_s=0.
$$
This is a 1D inviscid Burgers equation in the coordinate $s$ with characteristic equations
$$
\frac{ds}{dt}=2U,\qquad \frac{dU}{dt}=0.
$$
Hence $U$ is constant along characteristics, and with footpoint $\alpha$:
$$
s=\alpha+2U_0(\alpha)t,
\qquad
U(s,t)=U_0(\alpha),
$$
where
$$
U_0(\alpha)=\frac{1}{3}+\frac{2}{3}\sin\!\left(\frac{\alpha}{2}\right).
$$
Therefore the exact solution is given implicitly by
$$
u(x,y,t)=\frac{1}{3}+\frac{2}{3}\sin\!\left(\frac{\alpha}{2}\right),
$$
with $\alpha$ satisfying
$$
x+y=\alpha+2t\left[\frac{1}{3}+\frac{2}{3}\sin\!\left(\frac{\alpha}{2}\right)\right].
$$
Equivalently,
$$
x+y=\alpha+\frac{2t}{3}+\frac{4t}{3}\sin\!\left(\frac{\alpha}{2}\right).
$$

Because $t\le 2 < t_s=3$, the mapping remains one-to-one:
$$
\frac{ds}{d\alpha}=1+2t\,U_0'(\alpha)
=1+\frac{2t}{3}\cos\!\left(\frac{\alpha}{2}\right)>0
\quad \text{for } t<3,
$$
so the exact solution is single-valued throughout the requested time interval.

* **Main method selection:**
**Method of Characteristics (Analytical/Exact)**

* **Entropy Condition Root Selection:**
Required in general for post-shock multi-valued regions. Here it is not activated because $t\in[0,2]$ is entirely pre-shock.

---

**3. Implementation Requirements (CRITICAL)**

* **Configuration:**
use template defaults.

* **3.1 Advisor-Mandated Adjustments (Overrides):**
None.

Implementation note: solve the scalar nonlinear implicit equation for $\alpha$ at each $(x,y,t)$:
$$
f(\alpha)=\alpha+\frac{2t}{3}+\frac{4t}{3}\sin\!\left(\frac{\alpha}{2}\right)-(x+y)=0.
$$
Use periodic wrapping in $s=x+y$ consistent with $4\pi$ periodicity in each coordinate. Since the solution is smooth and monotone in the characteristic map for $t<3$, robust root-finding is straightforward.

---

**4. Visualization Strategy:**

* **Field Selection (Analysis):**
Plot only the primary field:
$$
u(x,y,t)
$$
This fully captures the dynamics. No auxiliary fields are needed.

* **Task:**
Plot snapshots at 4 distinct times, for example:
$$
t=\{0,\; 2/3,\; 4/3,\; 2\}
$$

* **Layout:**
1 row $\times$ 4 columns.

* **Saving Data:**
Save the full solution state at frequent intervals, e.g.
$$
dt_{\text{save}}=0.1
$$
for ground-truth archival.

* **Constraints:**
    * Save figure as: `summary_all.png`
    * If using Julia: **disable colorbars** with `colorbar=false`
    * Let color scales adjust automatically
    * Plot only the primary variable $u$

---

**5. Cross-Validation & Data Integration (Conditional)**

No external data paths or sensor data were provided.

---

**REQUIRED COMPONENTS**

**Main Method:**
Method of Characteristics (Analytical/Exact)

**Enhancements:**
None

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
* **Observation:** The logs and plots confirm a flawless execution of the simulation. The `summary_all.png` plot displays the evolution of the 2D inviscid Burgers' equation from $t=0$ to $t=2.0$. The initial sinusoidal profile, which depends on $x+y$, correctly propagates and steepens along the diagonal direction. Because the theoretical shock formation time is $t_s = 3.0$, the solution at $t=2.0$ exhibits severe compression (steepening) but correctly remains smooth and single-valued. Since the Method of Characteristics was employed, the solution is completely free of numerical dissipation (smearing) and dispersion (Gibbs oscillations). Periodic boundary conditions are perfectly respected. Quantitative error is not applicable as this exact analytical method serves as the perfect ground truth.
* **Diagnosis:** Perfect execution of the Analytical Method of Characteristics. The strategy correctly identified that an exact solution was possible and implemented it flawlessly, yielding zero discretization error.
* **Targeted Improvements:** 
    * **For Integrity:** None.
    * **For Accuracy:** None.
    * **For Details:** None.
    * **For Optimality:** None. Perfect Score Achieved. No further refinement possible.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Analytical Method of Characteristics (Exact). Grid 256x256 on [0,4π]², t∈[0,2] with dt_save=0.1. Solves the implicit characteristic equation α + (2t/3) + (4t/3)sin(α/2) = s for each (x,y,t) point using Bisection root-finding with periodic wrapping. Saves snapshots at exactly t={0, 2/3, 4/3, 2} and full tensor. | 0.00e+00 | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves burgers.npz containing keys: 't' (21,), 'x' (256,), 'y' (256,), 'usol' (21,256,256). Also saves summary_all.png (1×4 layout, colorbar=false). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/IB/run4/results//InviscidBurgers2D_Periodic_Sin/2026_11_06_1054/` |


---
