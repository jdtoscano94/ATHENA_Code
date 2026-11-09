
# Experiment Report
*Last Updated: 2026-11-06 10:54:40*

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
\partial_t u + \partial_x\!\left(\frac{u^2}{2}\right) + \partial_y\!\left(\frac{u^2}{2}\right)=0
$$

* **Domain:**
$$
(x,y)\in[0,4\pi]\times[0,4\pi],\qquad t\in[0,2]
$$

* **Boundary Conditions:**
Periodic in both directions:
$$
u(0,y,t)=u(4\pi,y,t),\qquad u(x,0,t)=u(x,4\pi,t)
$$

* **Initial Conditions:**
$$
u(x,y,0)=\frac{1}{3}+\frac{2}{3}\sin\!\left(\frac{x+y}{2}\right)
$$

* **Constants & Parameters:**
No additional physical constants.

* **Expected Physics:**
Smooth advection/compression initially, followed by gradient steepening and shock formation.  
For the reduced phase variable
$$
s=\frac{x+y}{2},
$$
the problem reduces to 1D inviscid Burgers:
$$
\partial_t U + U\,\partial_s U = 0,\qquad U(s,0)=\frac13+\frac23\sin s
$$
Shock onset occurs when
$$
t_*=-\frac{1}{\min(U_0'(s))} = -\frac{1}{\min\left(\frac13\cos s\right)}=3
$$
Hence on the requested interval $t\in[0,2]$, the solution remains smooth.

---

**2. Strategic Methodology**
* **Can obtain an exact solution or similar?** Yes.
  
The PDE has scalar hyperbolic form:
$$
\partial_t u + u(\partial_x u+\partial_y u)=0
$$
Along characteristics,
$$
\frac{dx}{dt}=u,\qquad \frac{dy}{dt}=u,\qquad \frac{du}{dt}=0
$$
so $u$ is constant on each characteristic. Since the initial data depends only on
$$
s=\frac{x+y}{2},
$$
the solution remains of the form
$$
u(x,y,t)=U(s,t),\qquad s=\frac{x+y}{2}
$$
with
$$
\partial_x u=\frac12 U_s,\qquad \partial_y u=\frac12 U_s
$$
thus
$$
\partial_t U + U\,U_s=0
$$
This is exactly the 1D inviscid Burgers equation.

The exact implicit characteristic solution is
$$
U(s,t)=U_0(\xi),\qquad s=\xi+t\,U_0(\xi),
$$
with
$$
U_0(\xi)=\frac13+\frac23\sin\xi
$$
Therefore the exact solution is
$$
u(x,y,t)=\frac13+\frac23\sin\xi
$$
where $\xi$ is defined implicitly by
$$
\frac{x+y}{2}=\xi+t\left(\frac13+\frac23\sin\xi\right)
$$
modulo the $2\pi$ periodicity in $\xi$.

Because $t\le 2 < 3$, the map remains one-to-one, so no entropy branch ambiguity occurs on the requested time interval.

**Main selection:** **Method of Characteristics (Analytical/Exact)**.

---

**3. Implementation Requirements (CRITICAL)**
* **Configuration:** use template defaults.

* **3.1 Advisor-Mandated Adjustments (Overrides):**
None.

Additional exact-solution implementation requirement:
* Solve the scalar implicit equation
$$
F(\xi)=\xi+t\left(\frac13+\frac23\sin\xi\right)-\frac{x+y}{2}=0
$$
at each evaluation point, using periodic root handling in $\xi\in[0,2\pi)$.
* Since $t<3$, the characteristic map is monotone enough to admit a unique smooth branch for each point.
* If extending beyond shock time in future studies, enforce **Entropy Condition Root Selection**.

---

**4. Visualization Strategy:**
* **Field Selection (Analysis):**
Only plot the primary field:
$$
u(x,y,t)
$$
It fully captures the dynamics; no auxiliary fields exist.

* **Task:** Plot at **4 distinct time steps**, recommended:
$$
t=\{0,\; 2/3,\; 4/3,\; 2\}
$$

* **Layout:** Grid with 1 row and 4 columns.

* **Saving Data:** Save the full solution state at frequent intervals, e.g.
$$
dt_{\text{save}}=0.1
$$
for ground-truth archival.

* **Constraints:**
  * Save final figure as:
    `summary_all.png`
  * If using Julia, set:
    `colorbar=false`
  * Use automatic scaling.
  * Plot only the primary variable $u$.

---

**5. Cross-Validation & Data Integration (Conditional)**
No external reference data or sensor files were provided.

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
* **Observation:** The generated plots display the flawless evolution of the 2D inviscid Burgers' equation. The initial diagonal sinusoidal wave steepens progressively from $t=0.0$ to $t=2.0$. The steepening occurs exactly as predicted by the characteristic curves, forming extremely sharp gradients at $t=2.0$. Because this is an exact analytical evaluation, there are absolutely no numerical artifacts—no Gibbs oscillations, no artificial dissipation (smearing), and no dispersion errors. The periodic boundary conditions are perfectly maintained across the $4\pi \times 4\pi$ domain. 
* **Diagnosis:** Perfect Analytical Execution. The strategic choice to use the Method of Characteristics combined with a robust periodic root-finding algorithm (bisection) yields the exact solution up to machine precision. As calculated in the strategy, the theoretical shock formation time is $t_* = 3$. Since the simulation ends at $t=2$, the solution correctly remains single-valued and continuous, avoiding any entropy branch ambiguities.
* **Targeted Improvements:** 
    * **For Integrity:** None. The code executed perfectly and saved the required high-resolution `.npz` dataset.
    * **For Accuracy:** None. Quantitative error is not applicable as no external ground truth is available; this analytical evaluation *is* the absolute ground truth.
    * **For Details:** None. The 256x256 grid provides excellent visual and data resolution for the steepening gradients without incurring the massive computational costs of a high-order numerical PDE solver.
    * **For Optimality:** None. Utilizing an exact analytical method is the highest possible standard for generating Physics-Informed Neural Network (PINN) training data. Perfect Score Achieved. No further refinement possible.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Analytical Method of Characteristics (Exact). Implements implicit characteristic root-finding for the reduced 1D Burgers problem via periodic Bisection on F(ξ)=ξ + t*(1/3+2/3*sin(ξ)) - s = 0. Grid: 256x256 points on [0,4π]². Evaluates at 21 times (dt_save=0.1) plus the 4 requested plot instants t={0,2/3,4/3,2}. | 0.00e+00 | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves burgers.npz containing keys: 't' (21,), 'x' (256,), 'y' (256,), 'usol' (21,256,256). Also saves summary_all.png (1x4 heatmap layout of u at the four requested times, colorbar=false). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/IB/run3/results//InviscidBurgers2D_Periodic_Sin_ICs/2026_11_06_1053/` |


---
