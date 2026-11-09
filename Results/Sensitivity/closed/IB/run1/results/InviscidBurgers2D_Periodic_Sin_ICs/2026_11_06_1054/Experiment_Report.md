
# Experiment Report
*Last Updated: 2026-11-06 10:55:03*

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
\partial_t u + \mathbf{a}(u)\cdot \nabla u = 0,
\qquad
\mathbf{a}(u) = (u,u).
$$

* **Domain:**
$$
(x,y)\in [0,4\pi]\times[0,4\pi],\qquad t\in[0,2].
$$

* **Boundary Conditions:**
Periodic in both directions:
$$
u(0,y,t)=u(4\pi,y,t),\qquad u(x,0,t)=u(x,4\pi,t).
$$

* **Initial Conditions:**
$$
u(x,y,0)=u_0(x,y)=\frac{1}{3}+\frac{2}{3}\sin\left(\frac{x+y}{2}\right).
$$

* **Constants & Parameters:**
No additional physical constants.

* **Expected Physics:**
The solution is smooth initially and depends only on
$$
s=\frac{x+y}{2}.
$$
Thus the 2D problem reduces exactly to a 1D inviscid Burgers evolution along the diagonal direction. Compression occurs where $u_0'(s)<0$, leading to gradient blow-up and shock formation. Since
$$
u_0'(s)=\frac{1}{3}\cos s,
$$
the minimum slope is $-1/3$, so the first shock time is
$$
t_s=-\frac{1}{2\min(u_0'(s))}=\frac{3}{2}.
$$
Therefore: smooth solution for $t<1.5$, shock formation at $t=1.5$, entropy solution afterward.

---

**2. Strategic Methodology**
* Can obtain an exact solution or similar?  
Yes. This is the correct route.

Because the initial condition depends only on
$$
s=\frac{x+y}{2},
$$
set
$$
u(x,y,t)=U(s,t),\qquad s=\frac{x+y}{2}.
$$
Then
$$
\partial_x u = \frac12 U_s,\qquad \partial_y u=\frac12 U_s,
$$
so
$$
\partial_t U + U\left(\frac12 U_s+\frac12 U_s\right)=\partial_t U+U\,U_s=0.
$$
Hence the 2D PDE reduces exactly to 1D inviscid Burgers:
$$
\partial_t U + U\,\partial_s U = 0,
\qquad
U(s,0)=\frac13+\frac23\sin s.
$$

Apply the Method of Characteristics:
$$
\frac{dU}{dt}=0,\qquad \frac{ds}{dt}=U.
$$
Thus along characteristics,
$$
U(s,t)=U_0(\xi),\qquad s=\xi+t\,U_0(\xi),
$$
with
$$
U_0(\xi)=\frac13+\frac23\sin\xi.
$$
Therefore the implicit exact pre-shock solution is
$$
u(x,y,t)=\frac13+\frac23\sin\xi,
$$
where $\xi$ satisfies
$$
\frac{x+y}{2}=\xi+t\left(\frac13+\frac23\sin\xi\right)
\quad (\mathrm{mod}\ 2\pi).
$$

For $t\ge t_s=3/2$, characteristics cross and the classical solution becomes multi-valued. The entropy solution must be selected.

**Main conclusion:** exact reduction + Method of Characteristics is feasible, so it is mandatory to select it.

If you use the **Method of Characteristics** require:
* **Entropy Condition Root Selection** - For post-shock multi-valued regions, the implementation must select the entropy-satisfying branch.

---

**3. Implementation Requirements (CRITICAL)**
* **Configuration:** use template defaults.

* **3.1 Advisor-Mandated Adjustments (Overrides):**
None.

---

**4. Visualization Strategy:**
* **Field Selection (Analysis):**
There is only one primary field:
$$
u(x,y,t).
$$
This is the sole dynamically relevant quantity and fully captures steepening and post-shock structure.

* **Task:** Plot this field at **4 distinct time steps**.
Recommended:
$$
t=\{0.0,\ 1.0,\ 1.5,\ 2.0\}.
$$
This captures: initial state, pre-shock steepening, shock onset, post-shock entropy solution.

* **Layout:** Grid with 1 row and 4 columns.

* **Saving Data:** Always save the full solution state at frequent intervals, e.g.
$$
dt_{\text{save}}=0.1.
$$

* **Constraints:**
    * Save as: `summary_all.png`
    * If the code is in Julia: **disable colorbars** with `colorbar=false`
    * Use automatic scaling
    * Plot only the primary variable $u$

---

**5. Cross-Validation & Data Integration (Conditional)**
No external data or sensor paths were provided.

---

**REQUIRED COMPONENTS**

**Main Method:**
Method of Characteristics (Analytical/Exact)

**Enhancements:**
Entropy Condition Root Selection

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
* **Observation:** The logs indicate a flawless execution of the analytical Method of Characteristics, successfully mapping the 2D domain to the 1D diagonal $s=(x+y)/2$. The generated plots perfectly capture the expected physics: the initial smooth sine wave steepens progressively until $t=1.5$, where the gradient becomes vertical, marking the exact theoretical onset of the shock. By $t=2.0$, a distinct, razor-sharp discontinuity is visible. Because this is an exact analytical solution with shock-fitting (Equal-Area/Rankine-Hugoniot), the shock interface is perfectly resolved (limited only by the pixel grid) with absolutely zero numerical dissipation (smearing) or Gibbs phenomena (oscillations). 
* **Diagnosis:** Perfect Exact Solution. The implementation successfully bypassed numerical discretization errors by utilizing the analytical characteristics and entropy condition root selection, yielding the highest possible fidelity for PINN ground truth data.
* **Targeted Improvements:** 
    * **For Integrity:** None.
    * **For Accuracy:** None.
    * **For Details:** None.
    * **For Optimality:** None. Perfect Score Achieved. No further refinement possible.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Analytical Method of Characteristics applied to the exact 2D-to-1D reduction of the inviscid Burgers equation along s=(x+y)/2. Uses implicit characteristic root-finding (Secant/ Order1) with shock fitting via Equal-Area/Rankine-Hugoniot reflection to enforce entropy solution. Grid 256x256, t in [0,2] with dt_save=0.1, alpha=1/3, beta=2/3. Plots generated at exactly t={0.0,1.0,1.5,2.0} (1x4 layout). | 0.00e+00 | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves summary_all.png (1x4 heatmap layout of u at target times, colorbar=false). Saves burgers.npz containing keys: 't' (21,), 'x' (256,), 'y' (256,), 'usol' (21,256,256). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/IB/run1/results//InviscidBurgers2D_Periodic_Sin_ICs/2026_11_06_1054/` |


---
