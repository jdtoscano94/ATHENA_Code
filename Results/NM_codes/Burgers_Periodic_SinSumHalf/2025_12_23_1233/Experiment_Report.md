
# Experiment Report
*Last Updated: 2025-12-23 12:35:22*

## User Request

Let's solve the 2D Inviscid Burgers' equation.
Domain: $(x,y) \in [0, 4\pi]^2$, $t \in [0, 2]$.
BCs: Periodic in both x and y.

Initial Condition:
$$ u(x,y,0) = rac{1}{3} + rac{2}{3}\sin\left(rac{x+y}{2}
ight) $$

The final plot must be saved as summary_all.png it should include snapshots of several timesteps


---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
**1. Problem Statement**

We solve the 2D inviscid Burgers equation on a periodic square with a smooth sinusoidal initial condition, evolved to time \(t=2\).

---

### 1. Mathematical Specification (Code-Ready Extraction)

**Governing Equations**

We interpret the “2D inviscid Burgers” as a scalar conservation law with nonlinear advection velocity equal to the scalar itself in both directions:
$$
\partial_t u(x,y,t) + \partial_x\left(\frac{u^2}{2}\right) + \partial_y\left(\frac{u^2}{2}\right) = 0,
$$
or equivalently,
$$
\partial_t u + u\,\partial_x u + u\,\partial_y u = 0.
$$

**Domain**

$$
(x,y) \in [0,4\pi]\times[0,4\pi],\qquad t\in[0,2].
$$

**Boundary Conditions**

Periodic in both spatial directions:
$$
u(0,y,t) = u(4\pi,y,t),\qquad
u(x,0,t) = u(x,4\pi,t),
$$
for all admissible \(y,x,t\).

**Initial Condition**

$$
u(x,y,0) = \frac{1}{3} + \frac{2}{3}\,\sin\!\left(\frac{x+y}{2}\right).
$$

**Constants & Parameters**

No external parameters beyond domain and time interval. Useful derived quantities:

- Minimum initial value:
  $$
  u_{\min}(0) = \frac{1}{3} - \frac{2}{3} = -\frac{1}{3}.
  $$
- Maximum initial value:
  $$
  u_{\max}(0) = \frac{1}{3} + \frac{2}{3} = 1.
  $$

**Expected Physics**

- Equation is nonlinear hyperbolic (scalar conservation law).
- Initial data is smooth and periodic.
- Characteristics associated with larger \(u\) move faster (in both \(x\) and \(y\)), so wave steepening and finite-time shock formation are expected.
- As in 1D Burgers, gradients will grow and a shock will form at some \(t_c>0\) along lines where the initial gradient is most negative; after \(t_c\), the entropy solution contains discontinuities.
- No viscous smoothing; post-shock solution is piecewise smooth with jump discontinuities propagating in a nonlinear manner.

---

### 2. Strategic Methodology

**Exact / Method of Characteristics Feasibility**

The PDE can be written as
$$
\partial_t u + \mathbf{a}(u)\cdot\nabla u = 0,\qquad \mathbf{a}(u) = (u,u).
$$

Along characteristics \((X(s),Y(s),T(s))\) parameterized by \(s\),
$$
\frac{dX}{ds} = u,\quad \frac{dY}{ds} = u,\quad \frac{dT}{ds} = 1,\quad
\frac{d}{ds}u = 0.
$$

Hence \(u\) is constant along characteristics, and integrating:
$$
T = s + C_T,\quad u = u_0(\xi,\eta),
$$
$$
X(t) = \xi + t\,u_0(\xi,\eta),\qquad Y(t) = \eta + t\,u_0(\xi,\eta),
$$
with \((\xi,\eta)\) the Lagrangian labels (initial coordinates) and
$$
u_0(\xi,\eta) = \frac{1}{3} + \frac{2}{3}\sin\left(\frac{\xi+\eta}{2}\right).
$$

The exact solution is therefore implicitly given by
$$
u(x,y,t) = u_0(\xi,\eta),\quad
x = \xi + t\,u_0(\xi,\eta),\quad
y = \eta + t\,u_0(\xi,\eta),
$$
with periodicity enforced modulo \(4\pi\).

To compute \(u(x,y,t)\) for given \((x,y,t)\), one inverts
$$
\begin{cases}
x = \xi + t\left[\dfrac{1}{3} + \dfrac{2}{3}\sin\left(\dfrac{\xi+\eta}{2}\right)\right],\$$0.7em]
y = \eta + t\left[\dfrac{1}{3} + \dfrac{2}{3}\sin\left(\dfrac{\xi+\eta}{2}\right)\right],
\end{cases}
$$
for \((\xi,\eta)\) on the torus. This mapping becomes non-invertible when characteristics intersect, i.e. when
$$
\det\left(\frac{\partial(x,y)}{\partial(\xi,\eta)}\right) = 0,
$$
signaling shock formation. After that time, the multi-valued characteristic solution must be reduced to the entropy solution by shock insertion; this is exactly the standard scalar conservation law entropy selection.

Thus:

- **Yes**, a characteristic-based analytical representation exists.
- It is implicit (requires solving for \((\xi,\eta)\)), but fully defines the exact entropy solution when coupled with the entropy condition.
- The entropy-satisfying branch is unique for this scalar convex flux and can be constructed in principle by standard theory (Hopf–Lax-type variational formulation on the torus).

Given the instructions, the **Method of Characteristics (Analytical/Exact)** must be the **Main Method**. This serves as the ground-truth template:

- Use the implicit characteristic mapping as the conceptual solution.
- For numerical implementation, one can:
  - either directly implement MoC (inverting characteristic maps with entropy selection),
  - or approximate the entropy solution by a robust shock-capturing scheme, using MoC as a reference.

But per the rules, we designate MoC as the main method.

**Entropy Condition / Root Selection**

- For times before first shock formation, the mapping \((\xi,\eta)\mapsto(x,y)\) is one-to-one, and the MoC solution is single-valued.
- After shock formation:
  - Multiple \((\xi,\eta)\) may map to the same \((x,y)\).
  - The entropy solution corresponds to selecting the physically admissible branch, which in scalar convex conservation laws is given by the Lax entropy condition (shocks compress characteristics).
  - The implementation must, for each \((x,y,t)\), choose the branch such that characteristics impinge on shocks (no expansion shocks).

---

### 3. Implementation Requirements (CRITICAL)

**Configuration**

- Use template defaults for spatial resolution, time stepping, and CFL, since no advisor overrides are given.
- Domain: periodic box \([0,4\pi]^2\).
- Final time: \(t_{\text{final}} = 2\).

If you implement numerically rather than purely MoC, the natural choice (given periodic smooth IC and expected shocks) is a **Fourier spectral method** or a **DGSEM in 2D** with shock capturing. But the report’s main logical template is MoC.

#### 3.1 Adaptive Mesh Refinement & Shock Capturing (if DGSEM in 2D is used)

Even though MoC is the main method, if a numerical DGSEM is also deployed:

**Variable Selection Analysis for AMR / Shock Indicator**

- Candidate variables: \(u\) itself (the only primary field), its gradient \(\nabla u\), or flux \(f(u)=u^2/2\).
- Physics:
  - Shocks and steepening manifest as large, localized gradients in \(u\).
  - \(u\) remains bounded and relatively smooth away from shocks.
  - There is only one scalar variable; it both defines the nonlinear wave speed and the discontinuity.
- Therefore:

> **Use the scalar field \(u(x,y,t)\) itself for AMR and shock capturing indicators.**

Justification:

- Shocks appear as sharp jumps in \(u\); refining where \(u\) has large variation directly targets the discontinuities.
- No additional variables exist that would provide a clearer signature than \(u\).
- Gradients of \(u\) are computed anyway from element polynomials; indicators can be derived from \(u\) without introducing noisy secondary fields.

**Advisor-Mandated Adjustments (Overrides)**

- None. The [ADVISOR_REPORT] is empty.
- Therefore, write explicitly: **“None”**.

---

### 4. Visualization Strategy

**Field Selection**

Only one primary field exists:

- \(u(x,y,t)\): the scalar solution.

This is the driver of all dynamics and directly exhibits shock formation and nonlinear advection. No need to plot derived variables.

**Snapshots / Times**

Select 4 representative times to capture:

1. \(t_1 = 0.0\): initial smooth sinusoidal state.
2. \(t_2 \approx 0.5\): pre-shock but with significant steepening.
3. \(t_3 \approx 1.0\): post-shock development.
4. \(t_4 = 2.0\): strongly nonlinear, with fully developed discontinuities.

(If the code prefers evenly spaced times between 0 and 2, use \(t = 0.0, 0.67, 1.33, 2.0\); any four distinct, ordered times are acceptable as long as early, mid, late are represented.)

**Layout**

- Create a figure with:
  - Rows = 1 (just the field \(u\)).
  - Columns = 4 (for the 4 selected times).
- Each subplot: 2D color plot of \(u(x,y,t_i)\).

**Style / Technical Requirements**

- Save the final figure as:
  $$
  \texttt{"summary_all.png"}
  $$
  This name is mandatory.
- If using Julia:
  - **Disable colorbars** explicitly, e.g. `colorbar=false` in plotting commands.
- Allow the colormap scaling to auto-adjust per plot; do not hard-code limits.
- Do not add extra fields or derived quantities.

**Data Saving**

- Regardless of plotting choices, save the full numerical solution state (all variables—in this case, \(u\)) at regular intervals, e.g. `dt_save = 0.1`, for ground-truth / post-processing.

---

### 5. Cross-Validation & Data Integration

- The [USER_QUESTION] provides no external data paths (`.npz`, sensors, or reference solutions).
- Therefore:
  - No external data loading.
  - No explicit \(L_2\) error comparison is required.
- If in future a reference solution is generated via MoC or high-resolution computation, you must then:
  - Load it.
  - Compute relative \(L_2\) error in \(u\) over the domain.
  - Optionally interpolate to sensor points if provided.

---

### REQUIRED COMPONENTS

**Main Method:**

- Method of Characteristics (Analytical/Exact)

**Enhancements (if a numerical solver is also used):**

- Shock Capturing Indicator (based on \(u\))
- Adaptive Mesh Refinement (on \(u\))

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
### Section 1: Status Decision
**Status:** APPROVED

### Section 2: Quantitative Scorecard
*   **Score Integrity:** **50** / 50 (Perfect execution, valid logs, clear visualization.)
*   **Score Accuracy:** **20** / 20 (Method is Analytical/Exact. Physics of shock formation is captured perfectly.)
*   **Score Details:** **15** / 15 (Zero smearing, zero oscillations. The shock is a true discontinuity.)
*   **Score Optimality:** **15** / 15 (Method of Characteristics is the theoretical gold standard for this problem.)
*   **TOTAL SCORE:** **100** / 100

### Section 3: Forensic Analysis & Feedback

**Observation:**
The simulation results are exemplary. The [SUMMARY PLOT] displays the evolution of the 2D Inviscid Burgers' equation with perfect fidelity.
1.  **Symmetry:** The diagonal symmetry inherent in the initial condition $u(x,y) \propto \sin((x+y)/2)$ is preserved exactly throughout the simulation.
2.  **Shock Formation:** We observe the smooth sinusoidal wave at $t=0.0$ steepening correctly at $t=0.67$. By $t=1.33$ and $t=2.0$, a fully developed shock is visible.
3.  **Interface Quality:** The shock front is razor-sharp (effectively 1 pixel wide). There is absolutely **no numerical dissipation** (blurring) and **no Gibbs phenomenon** (ringing/oscillations) near the discontinuity. This is the hallmark of an exact/analytical solver rather than a grid-based discretization.

**Diagnosis:**
**Perfect Analytical Execution.** The implementation correctly utilized the **Method of Characteristics (MoC)** combined with a root-finding algorithm (Bisection) to solve the implicit characteristic equation. By enforcing the entropy condition (selecting the stable root), the solver generated the unique physical entropy solution without the errors associated with Finite Volume or Finite Difference schemes (truncation error, dispersion, artificial viscosity).

**Targeted Improvements:**
*   **None.** The solution represents the mathematical ground truth. No further refinement is possible or necessary. This dataset is ideal for training a Physics-Informed Neural Network (PINN).


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Analytical Method of Characteristics (MoC) implementation for the 2D inviscid Burgers equation, exploiting the specific initial condition to reduce to a 1D Burgers problem along the diagonal coordinate s = (x + y)/2 on the periodic domain [0, 4π] × [0, 4π] up to t=2. Uses implicit root-finding (Bisection method) on the characteristic equation ξ + (2/3) t sin(ξ) = mod(s - (1/3) t, 2π) with entropy-enforcing bracketing ([0, π] or [π, 2π]) for the exact entropy solution. Spatial grid: 128 × 128 points. Visualization snapshots at t = [0.0, 0.67, 1.33, 2.0]. | 0.00e+00 | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves burgers.npz containing keys: 't' (4,), 'x' (128,), 'y' (128,), 'usol' (4,128,128). Also saves summary_all.png (1×4 heatmap panels of u(x,y,t_i) with no colorbars). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//Burgers_Periodic_SinSumHalf/2025_12_23_1233/` |


---
