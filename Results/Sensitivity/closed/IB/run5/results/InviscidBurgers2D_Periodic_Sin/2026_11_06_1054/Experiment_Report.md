
# Experiment Report
*Last Updated: 2026-11-06 10:56:11*

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
  Solve the 2D inviscid Burgers equation for a scalar field \(u(x,y,t)\):
  $$
  \partial_t u + \partial_x\!\left(\frac{u^2}{2}\right) + \partial_y\!\left(\frac{u^2}{2}\right) = 0
  $$
  Equivalently,
  $$
  \partial_t u + u\,\partial_x u + u\,\partial_y u = 0 .
  $$

* **Domain:**  
  $$
  (x,y)\in [0,4\pi]\times[0,4\pi], \qquad t\in[0,2].
  $$

* **Boundary Conditions:**  
  Periodic in \(x\).  
  Periodic in \(y\).

* **Initial Conditions:**  
  $$
  u(x,y,0)=\frac13+\frac23\sin\!\left(\frac{x+y}{2}\right).
  $$

* **Constants & Parameters:**  
  No additional physical constants.

* **Expected Physics:**  
  Smooth nonlinear advection initially. Since the data depends only on \(s=(x+y)/2\), the problem reduces to a 1D Burgers wave along the diagonal direction. Shock formation is expected when characteristics intersect. For
  $$
  u_0(s)=\frac13+\frac23\sin s,
  $$
  we have
  $$
  \frac{d u_0}{ds}=\frac23\cos s,
  $$
  so the earliest shock time is
  $$
  t_s=-\frac{1}{\min(u_{0x}+u_{0y})}.
  $$
  Because
  $$
  u_{0x}=u_{0y}=\frac13\cos\!\left(\frac{x+y}{2}\right),
  \quad\Rightarrow\quad
  u_{0x}+u_{0y}=\frac23\cos\!\left(\frac{x+y}{2}\right),
  $$
  the minimum is \(-2/3\), hence
  $$
  t_s=\frac{3}{2}.
  $$
  Therefore the solution remains smooth for \(t<1.5\), and a shock appears before the final time \(t=2\).

---

**2. Strategic Methodology**

* **Can obtain an exact solution or similar?**  
  Yes. This is a hyperbolic conservation law, and the initial condition has special structure:
  $$
  u(x,y,0)=u_0\!\left(\frac{x+y}{2}\right).
  $$
  Thus the solution remains of the form
  $$
  u(x,y,t)=U\!\left(\frac{x+y}{2},t\right),
  $$
  where \(U(s,t)\) satisfies the 1D inviscid Burgers equation
  $$
  \partial_t U + U\,\partial_s U = 0,
  \qquad
  U(s,0)=\frac13+\frac23\sin s.
  $$
  The characteristic equations are
  $$
  \frac{ds}{dt}=U, \qquad \frac{dU}{dt}=0.
  $$
  Hence along characteristics,
  $$
  U(s,t)=U_0(\xi), \qquad s=\xi+t\,U_0(\xi),
  $$
  with
  $$
  U_0(\xi)=\frac13+\frac23\sin\xi.
  $$
  Returning to \((x,y)\),
  $$
  \frac{x+y}{2}=\xi+t\left(\frac13+\frac23\sin\xi\right),
  $$
  and therefore the implicit exact solution is
  $$
  u(x,y,t)=\frac13+\frac23\sin\xi,
  $$
  where \(\xi\) solves
  $$
  \xi+t\left(\frac13+\frac23\sin\xi\right)=\frac{x+y}{2}.
  $$
  This is the correct exact pre-shock representation. Post-shock, the entropy solution is obtained by **entropy condition root selection** in the multi-valued region. Since an exact characteristic formulation is available, it must be selected.

* **Main method selection:**  
  **Method of Characteristics (Analytical/Exact)**

---

**3. Implementation Requirements (CRITICAL)**

* **Configuration:**  
  Use template defaults.

* **3.1 Advisor-Mandated Adjustments (Overrides):**  
  None.

* **Exact-solution implementation requirement:**  
  Implement the characteristic map
  $$
  F(\xi;x,y,t)=\xi+t\left(\frac13+\frac23\sin\xi\right)-\frac{x+y}{2}=0
  $$
  and solve for \(\xi\) at each \((x,y,t)\). Then evaluate
  $$
  u(x,y,t)=\frac13+\frac23\sin\xi.
  $$
  For \(t>t_s=3/2\), if multiple roots exist, enforce **entropy condition root selection** to obtain the physically admissible entropy solution.

---

**4. Visualization Strategy:**

* **Field Selection (Analysis):**  
  Plot only the primary field:
  $$
  u(x,y,t).
  $$
  This scalar fully captures the nonlinear wave steepening and shock formation. No secondary field is needed.

* **Task:**  
  Plot \(u(x,y,t)\) at 4 distinct times, for example:
  $$
  t=\{0,\;0.75,\;1.5,\;2.0\}.
  $$
  This set captures initial state, pre-shock steepening, shock onset, and post-shock entropy solution.

* **Layout:**  
  1 row \(\times\) 4 columns.

* **Saving Data:**  
  Save the full solution state at frequent intervals, e.g.
  $$
  dt_{\text{save}}=0.1.
  $$

* **Constraints:**  
  * Save figure as: `summary_all.png`
  * If using Julia plotting: **disable colorbars** with `colorbar=false`
  * Use automatic color scaling
  * Do not plot derived quantities

---

**5. Cross-Validation & Data Integration (Conditional)**

No external reference data or sensor files were provided. No cross-validation requirement.

---

**REQUIRED COMPONENTS**

**Main Method:**
- Method of Characteristics (Analytical/Exact)

**Enhancements:**
- Entropy Condition Root Selection

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
* **Observation:** The provided plots beautifully capture the nonlinear advection and wave steepening characteristic of the 2D inviscid Burgers' equation. The initial smooth diagonal wave steepens progressively. At exactly $t=1.5$, the onset of a shock is visible, which perfectly matches the theoretical shock formation time $t_s = 1.5$ derived from the initial conditions. By $t=2.0$, a razor-sharp shock front has fully developed. Because the implementation utilizes the exact Method of Characteristics coupled with Lax-Oleinik potential minimization for entropy root selection, the shock interface is perfectly resolved (zero cells of artificial smearing) and completely free of Gibbs oscillations or numerical dissipation. 
* **Diagnosis:** Perfect Exact Solution. The methodology successfully bypasses all discretization errors associated with standard Eulerian numerical schemes.
* **Targeted Improvements:** 
    * **For Integrity:** None. The code executed flawlessly and the data structures are correctly formatted.
    * **For Accuracy:** None. The solution is analytically exact. Quantitative error is not applicable as this *is* the ground truth.
    * **For Details:** None. The shock resolution is mathematically perfect.
    * **For Optimality:** None. Using the exact characteristic map with entropy root selection is the absolute gold standard for generating PINN ground truth data for hyperbolic conservation laws. Perfect Score Achieved. No further refinement possible.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Method of Characteristics (Analytical/Exact) with entropy root selection via Lax-Oleinik potential minimization. Grid: 256x256 on [0,4π]×[0,4π], t∈[0,2] with dt_save=0.1 (21 time steps). Implements exact implicit characteristic solve F(ξ)=ξ+t*(1/3+2/3*sin(ξ))-(x+y)/2=0 and u=1/3+2/3*sin(ξ). Visualizes u at 4 times in 1×4 layout and saves summary_all.png (colorbar=false). | 0.00e+00 | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves burgers.npz containing keys: 't' (21,), 'x' (256,), 'y' (256,), 'usol' (21,256,256). Also saves summary_all.png. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/closed/IB/run5/results//InviscidBurgers2D_Periodic_Sin/2026_11_06_1054/` |


---
