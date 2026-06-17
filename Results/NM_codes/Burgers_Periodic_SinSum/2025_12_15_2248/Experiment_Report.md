
# Experiment Report
*Last Updated: 2025-12-15 22:49:56*

## User Request

Let's solve the 2D Inviscid Burgers' equation:
$$ u_t + u u_x + u u_y = 0 $$

Domain: $(x,y) \in [0, 4\pi]^2$, $t \in [0, 2]$.
BCs: Periodic in both x and y.

Initial Condition:
$$ u(x,y,0) = rac{1}{3} + rac{2}{3}\sin\left(rac{x+y}{2}ight) $$

The final plot must be saved as summary_all.png it should include snapshots of several timesteps


---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
**1. Problem Statement**

We solve the 2D inviscid Burgers’ equation with a scalar velocity field advecting itself identically in both spatial directions.

---

### 1. Mathematical Specification (Code-Ready Extraction)

**Governing Equation**

\[
\partial_t u(x,y,t) + u(x,y,t)\,\partial_x u(x,y,t) + u(x,y,t)\,\partial_y u(x,y,t) = 0,
\]
or equivalently
\[
u_t + u\,u_x + u\,u_y = 0.
\]

**Domain**

\[
(x,y) \in [0,4\pi] \times [0,4\pi], \qquad t \in [0,2].
\]

**Boundary Conditions**

\[
u(0,y,t) = u(4\pi,y,t), \qquad u(x,0,t) = u(x,4\pi,t),
\]
i.e. periodic in both \(x\) and \(y\).

**Initial Condition**

\[
u(x,y,0) = u_0(x,y) = \frac{1}{3} + \frac{2}{3}\,\sin\!\left(\frac{x+y}{2}\right).
\]

**Constants & Parameters**

- No external parameters beyond the domain extents.
- Effective characteristic speeds in \(x\) and \(y\) directions are given by \(u(x,y,t)\) itself.
- Periods: in both \(x\) and \(y\), period \(L = 4\pi\).

**Expected Physics**

- This is a first-order **nonlinear hyperbolic** scalar PDE.
- Characteristics satisfy \(\dot x = u, \ \dot y = u\), \(\dot u = 0\).
- Initial profile is smooth and strictly positive:
  \[
  u_0 \in \left[\frac{1}{3}-\frac{2}{3},\,\frac{1}{3}+\frac{2}{3}\right]
  = \left[-\frac{1}{3},\,1\right],
  \]
  so there are regions with negative and positive characteristic speeds.
- Nonlinear compression where \(\partial_n u_0 < 0\) along the characteristic direction will generate gradient steepening and eventually shock formation.
- We expect finite-time shock formation; the Method of Characteristics remains valid as an **implicit exact solution** with entropy-condition root selection beyond that time.

---

**2. Strategic Methodology**

**Feasibility of Method of Characteristics**

Rewrite the PDE in characteristic form. Consider characteristic curves \((x(t),y(t),u(t))\) such that along each curve:

\[
\frac{dx}{dt} = u,\qquad \frac{dy}{dt} = u,\qquad \frac{du}{dt} = 0.
\]

Thus,
- \(u(t) = u_0(\xi,\eta)\) is constant along the characteristic issuing from initial point \((\xi,\eta)\).
- Integrating the position ODEs,
  \[
  x(t) = \xi + t\,u_0(\xi,\eta), \qquad
  y(t) = \eta + t\,u_0(\xi,\eta).
  \]

Given a fixed \((x,y,t)\), the exact solution \(u(x,y,t)\) is determined implicitly by finding \((\xi,\eta)\) such that

\[
\begin{cases}
x = \xi + t\,u_0(\xi,\eta),\\[4pt]
y = \eta + t\,u_0(\xi,\eta),
\end{cases}
\]
with
\[
u(x,y,t) = u_0(\xi,\eta).
\]

For this specific \(u_0\),
\[
u_0(\xi,\eta) = \dfrac{1}{3} + \dfrac{2}{3}\,\sin\!\left(\dfrac{\xi+\eta}{2}\right).
\]

Introduce
\[
s = \xi + \eta, \qquad r = x + y.
\]

Along characteristics, we have
\[
x+y = (\xi+\eta) + 2t\,u_0(\xi,\eta) \quad\Rightarrow\quad
r = s + 2t\,u_0(s),
\]
where
\[
u_0(s) = \frac{1}{3} + \frac{2}{3}\,\sin\!\left(\frac{s}{2}\right).
\]

Hence \(u\) depends only on \(r = x+y\) and \(t\), and is given implicitly by
\[
u(x,y,t) = u_0(s),
\]
with \(s\) solving the scalar nonlinear equation
\[
r = s + 2t\,u_0(s)
= s + 2t\left[\frac{1}{3} + \frac{2}{3}\,\sin\!\left(\frac{s}{2}\right)\right],
\qquad r = x + y.
\]

Periodic domain: The initial profile \(u_0\) has period \(4\pi\) in the variable \((x+y)\), so the solution is naturally \(4\pi\)-periodic in both \(x\) and \(y\); characteristic mapping is taken mod \(4\pi\) as needed.

- Before shock: mapping \(s \mapsto r\) is one-to-one.
- After shock: mapping can become multi-valued. The exact solution is still encoded in the characteristic equations; the physical entropy-satisfying weak solution is obtained by **entropy condition root selection** (single-valued conservative solution branch).

Thus, **Method of Characteristics is fully feasible** and yields an exact (implicit) solution for all \(t\), including post-shock regimes via entropy-based root selection.

**Main Method (Selected)**  
**Method of Characteristics (Exact)**

Reason: scalar nonlinear hyperbolic PDE with simple characteristic structure; reduces to a 1D implicit mapping in \(s=x+y\). This is the cleanest and most physically transparent ground-truth construction. Post-shock behavior is handled by entropy-condition selection over candidate roots of the implicit characteristic mapping.

---

**3. Implementation Requirements (CRITICAL)**

**Configuration**

- Use template defaults for temporal and spatial discretization **around** the characteristic solver; practical implementation can be:
  - Discretize the domain in \((x,y)\).
  - For each grid point and each output time, solve the 1D nonlinear equation
    \[
    r = s + 2t\left[\frac{1}{3} + \frac{2}{3}\,\sin\!\left(\frac{s}{2}\right)\right],
    \quad r = x+y,
    \]
    for \(s\) (mod \(4\pi\) in the sense of periodicity).
  - Then set
    \[
    u(x,y,t) = \frac{1}{3} + \frac{2}{3}\,\sin\!\left(\frac{s}{2}\right),
    \]
    choosing the **entropy-satisfying root** where multiple \(s\) exist.

- Periodicity: enforce \(s\) in a fundamental interval, e.g. \(s \in [0,4\pi]\), using modular arithmetic for \(r\).

**3.1 Adaptive Mesh Refinement & Shock Capturing**

Not applicable here as the **primary method is analytic (MoC)**, not DGSEM. No AMR or numerical shock capturing is required.

- Variable selection for potential refinement (if a secondary numerical method is ever used):  
  - **Density-like scalar** here is just \(u\).  
  - The steepening and shocks occur in \(u\) itself, and \(u\) remains the cleanest indicator of structural features (shock fronts).  
  - Thus, if AMR were used, it should be driven by gradients of \(u\), not by any derived quantity.

**3.1 Advisor-Mandated Adjustments**

- Advisor requested none.  
  - **Advisor Overrides:** None.

---

**4. Visualization Strategy**

**Field Selection**

- Single scalar field: \(u(x,y,t)\).
- This field directly encodes all dynamics (steepening, shock formation, and propagation). No secondary field needed.

**Plotting Instructions**

- Select **4 distinct times** in the interval \(t \in [0,2]\), for example:
  \[
  t_1 = 0,\quad t_2 = \frac{2}{3},\quad t_3 = \frac{4}{3},\quad t_4 = 2.
  \]
- At each chosen time, compute \(u(x,y,t_k)\) via the characteristic mapping.

**Layout**

- Grid with:
  - **Rows:** 1 (the field \(u\)).
  - **Columns:** 4 (the four time snapshots).
- Each subplot: 2D color plot of \(u(x,y,t_k)\) on \([0,4\pi]\times[0,4\pi]\).
- Automatic scaling of colormap per plot (do not fix common color limits unless required by the template).

**Mandatory Plot Settings**

- Save final multi-panel image as:
  - `summary_all.png` (exact filename).
- If implementation is in Julia:
  - **Explicitly disable colorbars**: `colorbar=false`.
- No derived quantities (e.g., gradients, vorticity) should be plotted.

**Data Saving**

- Save the **full solution state** \(u(x,y,t)\) at frequent intervals, e.g. `dt_save = 0.1`, into files (format per template).
- These data serve as ground truth for any further analysis or comparison.

---

**5. Cross-Validation & Data Integration**

- No external data paths, sensors, or reference solutions are specified in [USER_QUESTION].
- No cross-validation is required.

---

**Required Components**

**Main Method:**

- Method of Characteristics (Exact)

**Enhancements:**

- [Coordinate Mapping] (implicit mapping \(s \mapsto r = s + 2t\,u_0(s)\) with entropy-condition root selection)

No additional enhancements (no Shock Capturing module, no Reynolds Number Continuation) are needed, since the solution is constructed analytically via characteristics.

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
# Section 1: Status Decision
*   **Status:** APPROVED

# Section 2: Quantitative Scorecard
*   **Score Integrity:** **50** / 50
*   **Score Accuracy:** **20** / 20
*   **Score Details:** **15** / 15
*   **Score Optimality:** **15** / 15
*   **TOTAL SCORE:** **100** / 100

# Section 3: Forensic Analysis & Feedback

**1. Observation**
The simulation successfully generates the **exact entropy solution** for the 2D Inviscid Burgers' equation using the Method of Characteristics.
*   **Physics:** The solution correctly follows the symmetry of the initial condition ($u$ depends on $x+y$). The positive velocity regions (yellow) travel faster than the negative/low velocity regions (purple), causing the "falling edge" of the wave to steepen.
*   **Shock Formation:** The theoretical shock formation time for this problem is $t_b = 1.5$. The plots accurately reflect this:
    *   At $t=1.33$, the gradient is extremely steep but the solution is likely still continuous (or just forming the singularity).
    *   At $t=2.0$, a clear, sharp shock has formed.
*   **Quality:** The shock front is razor-sharp (step function), limited only by the pixel resolution of the 200x200 grid. There are **no numerical artifacts** (no Gibbs oscillations, no artificial smearing) because the solution is derived analytically rather than via a discrete stencil.

**2. Diagnosis**
**Perfect Ground Truth Generation.** The use of the analytical characteristic mapping with entropy-condition root selection provides the highest possible fidelity for this problem.

**3. Targeted Improvements**
*   **None.** The solution achieves the maximum possible score. No further refinement is necessary. The data generated is suitable as "Exact Ground Truth" for training Physics-Informed Neural Networks (PINNs).


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Analytical implementation using Method of Characteristics reduced to 1D scalar problem along r = x + y. For each (x, y, t), solves the implicit equation r = s + 2 t u_0(s) with u_0(s) = 1/3 + 2/3 sin(s/2) via Lax-Oleinik functional J(s) = (r - s)^2 / (4 t) + integral u_0(s) for entropy-satisfying root selection (coarse grid min followed by Roots.find_zero on characteristic equation). Domain [0, 4π] x [0, 4π], t in [0, 2], spatial grid 200 x 200, time array with dt_save=0.1 (21 steps total including snapshots). Periodic handling implicit via periodic u_0. Visualizes 4 snapshots at t=0, 2/3, 4/3, 2 as 1x4 heatmaps. | 0.00e+00 | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves burgers.npz containing keys: 't' (21,), 'x' (200,), 'y' (200,), 'usol' (21,200,200). Also saves summary_all.png (1x4 heatmap grid of u fields). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//Burgers_Periodic_SinSum/2025_12_15_2248/` |


---
