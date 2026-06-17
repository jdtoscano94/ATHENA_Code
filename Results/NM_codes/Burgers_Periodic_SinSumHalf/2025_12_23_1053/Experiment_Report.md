
# Experiment Report
*Last Updated: 2025-12-23 10:55:52*

## User Request

Let's solve the 2D Inviscid Burgers' equation.
Domain: $(x,y) \in [0, 4\pi]^2$, $t \in [0, 2]$.
BCs: Periodic in both x and y.

Initial Condition:
$$ u(x,y,0) = rac{1}{3} + rac{2}{3}\sin\left(rac{x+y}{2}ight) $$

The final plot must be saved as summary_all.png it should include snapshots of several timesteps


---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
**1. Problem Statement**

We solve the 2D inviscid Burgers’ equation with a *scalar* velocity field that is advected by itself:

\[
\partial_t u + u\,\partial_x u + u\,\partial_y u = 0,
\qquad (x,y)\in[0,4\pi]\times[0,4\pi],\quad t\in[0,2],
\]
with periodic boundaries and smooth sinusoidal initial data. This is a first-order nonlinear hyperbolic PDE; characteristics will intersect → gradient blowup (shock) in finite time.

---

### 1. Mathematical Specification (Code-Ready Extraction)

**Governing Equation (scalar PDE):**
\[
\boxed{
\partial_t u(x,y,t) + u(x,y,t)\,\partial_x u(x,y,t) + u(x,y,t)\,\partial_y u(x,y,t) = 0
}
\]

Equivalently, in compact form:
\[
\partial_t u + \nabla\cdot \mathbf{F}(u) = 0,\qquad 
\mathbf{F}(u) = \begin{pmatrix}\frac{1}{2}u^2 \\[4pt] \frac{1}{2}u^2 \end{pmatrix},
\]
but the advective form above is what we use for characteristics.

**Domain:**
\[
(x,y)\in[0,4\pi]\times[0,4\pi],\qquad t\in[0,2].
\]

**Boundary Conditions:**

Periodic in both directions:

- In \(x\):
\[
u(0,y,t) = u(4\pi,y,t)\quad \forall\,y\in[0,4\pi],\ t\in[0,2].
\]
- In \(y\):
\[
u(x,0,t) = u(x,4\pi,t)\quad \forall\,x\in[0,4\pi],\ t\in[0,2].
\]

**Initial Condition:**

\[
\boxed{
u(x,y,0) = u_0(x,y) = \frac{1}{3} + \frac{2}{3}\,\sin\!\left(\frac{x+y}{2}\right)
}
\]

This is smooth, periodic with period \(4\pi\) in both \(x\) and \(y\) because
\[
\sin\!\left(\frac{x+4\pi + y}{2}\right) = \sin\!\left(\frac{x+y}{2} + 2\pi\right)=\sin\!\left(\frac{x+y}{2}\right),
\]
and similarly for \(y\).

**Constants & Parameters:**

- No physical viscosity: inviscid.
- Non-dimensional, no explicit parameters.
- Numerical:
  - Final time: \(T=2\).
  - Domain size: \(L_x=L_y=4\pi\).

**Expected Physics:**

- Hyperbolic, nonlinear advection; smooth initial data.
- Compression regions develop where \(\partial_{x}u_0 + \partial_{y}u_0 < 0\).
- Finite-time gradient blowup (shock formation) along lines in \((x,y)\).
- Before first shock: exact solution via method of characteristics.
- After first shock: multiple characteristic branches → need entropy-satisfying weak solution.

---

## 2. Strategic Methodology

### 2.1 Exact Solution Feasibility: Method of Characteristics

The PDE can be written as
\[
\partial_t u + \mathbf{a}(u)\cdot \nabla u = 0,\qquad 
\mathbf{a}(u) = (u, u).
\]
Characteristics satisfy
\[
\frac{d x}{d t} = u,\qquad \frac{d y}{dt} = u,\qquad \frac{d u}{dt} = 0.
\]

Thus, along a characteristic curve parameterized by an initial point \((\xi,\eta)\) at \(t=0\):
\[
u(x(t),y(t),t) = u_0(\xi,\eta),
\]
and
\[
\frac{dx}{dt} = u_0(\xi,\eta),\quad \frac{dy}{dt} = u_0(\xi,\eta).
\]

Integrating in time:
\[
x(t) = \xi + t\,u_0(\xi,\eta),\qquad
y(t) = \eta + t\,u_0(\xi,\eta).
\]

So the exact solution is given *implicitly* by:
1. For a given \((x,y,t)\), find \((\xi,\eta)\) solving:
   \[
   x = \xi + t\,u_0(\xi,\eta),\qquad
   y = \eta + t\,u_0(\xi,\eta),
   \]
   where
   \[
   u_0(\xi,\eta) = \frac{1}{3} + \frac{2}{3}\,\sin\!\left(\frac{\xi+\eta}{2}\right).
   \]
2. Then
   \[
   \boxed{u(x,y,t) = u_0(\xi,\eta)}
   \]
   with \((\xi,\eta)\) the appropriate preimage.

It is convenient to reparameterize using
\[
s = \frac{\xi+\eta}{2},\qquad d = \frac{\xi-\eta}{2}\quad\Rightarrow\quad
\xi = s+d,\ \eta = s-d.
\]
The initial condition depends only on \(s\):
\[
u_0(\xi,\eta) = \frac{1}{3} + \frac{2}{3}\sin(s).
\]

In terms of \((s,d)\),
\[
x = s+d + t\,u_0(s),\qquad
y = s-d + t\,u_0(s),
\]
where \(u_0(s) = \frac{1}{3} + \frac{2}{3}\sin(s)\).

Invert this mapping (pre-shock, the Jacobian is nonzero) to get \((s,d)\):
\[
x + y = 2s + 2t\,u_0(s),\qquad
x - y = 2d.
\]
Thus
\[
d = \frac{x-y}{2},
\]
and \(s\) must satisfy:
\[
\boxed{
x + y = 2s + 2t\left[\frac{1}{3} + \frac{2}{3}\sin(s)\right]
}
\]
or
\[
s + t\left[\frac{1}{3} + \frac{2}{3}\sin(s)\right] = \frac{x+y}{2}.
\]

Given \((x,y,t)\), we solve this scalar nonlinear equation for \(s\) (mod \(4\pi\) periodicity), then
\[
u(x,y,t) = u_0(s) = \frac{1}{3} + \frac{2}{3}\sin(s).
\]

**Shock / multi-valued region & entropy selection:**

- The mapping \((s,d)\mapsto(x,y)\) loses invertibility (Jacobian \(\to 0\)) at shock formation:
  \[
  \frac{\partial(x+y)}{\partial s} = 2 + 2t\,u_0'(s) = 0.
  \]
- For a given \((x,y,t)\) after this time, the equation for \(s\) can have multiple roots → multi-valued classical solution.
- The correct physical (entropy-satisfying) weak solution is a single-valued function \(u(x,y,t)\) with shocks.

**MANDATORY MAIN METHOD CHOICE:**

An exact characteristic-based description exists (implicit form with root-finding in \(s\)), valid for all times. Therefore:

- **Main Method: Method of Characteristics (Analytical/Exact).**

Implementation will:
- Use the implicit characteristic relation for \(s\).
- For times before first shock: unique root.
- For times after: apply an entropy condition root selection mechanism / shock construction to enforce the entropy-satisfying weak solution (as required).

Alternative numerical PDE solvers (e.g., DGSEM, spectral) could approximate this, but they are secondary and unnecessary as the primary strategy.

---

## 3. Implementation Requirements (CRITICAL)

Assume the codebase has a Method-of-Characteristics template that handles periodicity, root finding, and entropy-consistent branch selection.

### 3.1 Configuration

- Use template defaults for:
  - Spatial resolution.
  - Time stepping / sampling.
  - Root-finding tolerances (unless template dictates otherwise).
- Enforce periodicity in \(s\) with period \(4\pi\).

For each numerical grid point \((x_i,y_j)\) and each output time \(t^n\):

1. Compute:
   \[
   S_{\text{target}} = \frac{x_i + y_j}{2}.
   \]
2. Solve for \(s\) in:
   \[
   g(s) = s + t^n\left(\frac{1}{3} + \frac{2}{3}\sin(s)\right) - S_{\text{target}} = 0,
   \]
   with \(s\) taken modulo \(4\pi\).
3. Apply entropy condition to select correct branch when multiple roots exist.
4. Set:
   \[
   u_{i,j}^n = \frac{1}{3} + \frac{2}{3}\sin(s).
   \]

### 3.2 Adaptive Mesh Refinement & Shock Capturing

For this problem, AMR and classical “shock capturing” are relevant only if you add a secondary discrete PDE solver (e.g., DGSEM) to compare with the exact MoC solution. If you do that:

- **Variable for AMR & Shock Indicator:**  
  **Use \(u\) itself.**

**Justification (Signal-to-Noise Analysis):**

- There is only one primary scalar field \(u\).
- Shocks manifest as large gradients and discontinuities in \(u\); all structure of interest is directly in \(u\).
- No auxiliary variables (density, pressure, etc.) exist that might be smoother or more localized.
- Therefore, refining on \(|\nabla u|\) or similar indicators based on \(u\) directly targets shock locations and avoids wasteful refinement.

If you only use the exact MoC method (recommended as main), AMR is not required; you can keep a uniform mesh, since we aren’t resolving shocks numerically but representing them via the characteristic construction plus entropy rule.

### 3.3 Advisor-Mandated Adjustments (Overrides)

[ADVISOR_REPORT]: None provided beyond generic template.  
So:

- **Advisor-Mandated Adjustments:** None.

Use template defaults for time integration and any numerical tolerances, unless your local environment requires otherwise.

---

## 4. Visualization Strategy

**Primary field to visualize:**

- Only field: \(u(x,y,t)\). This fully captures the dynamics (steepening and shock formation). No need for derived variables.

**Plot specification:**

- Plot \(u(x,y,t)\) at **4 distinct times** spanning \(t\in[0,2]\). For instance:
  - \(t_0 = 0.0\)
  - \(t_1 = 0.5\)
  - \(t_2 = 1.0\)
  - \(t_3 = 2.0\)

- Layout:
  - 1 row (field: \(u\)).
  - 4 columns (times \(t_0,\dots,t_3\)).
  - Each panel: 2D colormap of \(u(x,y,t_k)\).

**Strict requirements:**

- **Filename:**  
  Save the final figure as:
  \[
  \texttt{summary_all.png}
  \]
  exactly.

- **If using Julia plotting:**  
  - Explicitly set `colorbar=false` in each plot (or globally) to **disable colorbars**.
  - Allow default color scaling per panel (no manual fixed limits) so the contrast naturally adapts as shocks develop.

- **Data saving:**
  - Even though the figure only shows 4 times, **save the full solution state** \(u(x,y,t)\) at reasonably frequent intervals (e.g., every \(\Delta t_{\text{save}} = 0.1\) or as per template) for ground truth and potential post-processing.
  - Store all grid data, not only the plotted subset.

---

## 5. Cross-Validation & Data Integration

[USER_QUESTION] does **not** provide external data paths (no `.npz` reference solutions, no sensor coordinates).

- **External data loading:** None.
- **Relative \(L_2\) Error vs. reference:** Not applicable.
- **Sensor interpolation / pointwise error:** Not required.

If later you supply reference data, extend the implementation to:

- Load the reference at matching times.
- Interpolate the MoC solution to the reference grid or sensor points.
- Compute and print relative \(L_2\) errors.

---

## REQUIRED COMPONENTS

**Main Method:**
- Method of Characteristics (Analytical/Exact)

**Enhancements (if adding a secondary numerical PDE solver for comparison):**
- Shock Capturing Indicator (based on gradients of \(u\))
- AMR based on \(u\) (optional, only if a discrete solver is used in addition to MoC)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
**Section 1: Status Decision**
*   **Status:** APPROVED

**Section 2: Quantitative Scorecard**
*   **Score Integrity:** **50** (Perfect execution, logs complete, data saved, no crashes).
*   **Score Accuracy:** **20** (Method is Exact/Analytical. Physics are perfect: correct wave speed, steepening, and shock formation time at $t=1.5$).
*   **Score Details:** **15** (Razor-sharp shocks with zero smearing, no Gibbs oscillations, optimal resolution).
*   **Score Optimality:** **15** (Method of Characteristics is the mathematically optimal approach for inviscid ground truth).
*   **TOTAL SCORE:** **100**

**Section 3: Forensic Analysis & Feedback**
*   **Observation:** The simulation successfully implements the **Method of Characteristics** to generate an exact solution.
    *   **Physics Check:** The initial condition ($t=0$) shows smooth diagonal bands corresponding to $\sin((x+y)/2)$. As time progresses to $t=0.5$ and $t=0.99$, the high-velocity regions (Red, $u \approx 1$) advect faster than the low-velocity regions (Blue, $u \approx -0.33$), causing the wave front to steepen.
    *   **Shock Formation:** Theoretical breaking time is $T_b = 1.5$. The plot at $t=2.0$ correctly shows a fully formed, razor-sharp shock discontinuity where the fast characteristic branches have overtaken the slow ones.
    *   **Artifacts:** There are **zero** numerical artifacts (no smearing, no oscillations) because no grid-based discretization was used for the evolution.
*   **Diagnosis:** **Perfect Analytical Ground Truth.** The solution is exact within machine precision and root-finding tolerance.
*   **Targeted Improvements:**
    *   **None.** The solution has achieved a perfect score and represents the highest possible fidelity for PINN training data. No further refinement is possible or necessary.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Analytical Method of Characteristics via reduction to 1D periodic inviscid Burgers equation on s = (x+y)/2 mod 2π over [0, 2π), with exact implicit solution using root-finding on shifted characteristics x_target - alpha*t = x0 + beta*t*sin(x0), and post-shock entropy enforcement via domain splitting at shock location π + alpha*t, reflection across shock for right side, and u-reflection as 2*alpha - u_reflected. Handles periodicity by shifting if shock exceeds 2π. Spatial grid 256x256 on [0, 4π]x[0, 4π], time grid 200 steps from t=0 to t=2. Plots u at t≈0.0,0.5,1.0,2.0 as 1x4 heatmaps without colorbars. | 0.00e+00 | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves burgers.npz containing keys: 't' (200,), 'x' (256,), 'y' (256,), 'usol' (200,256,256). Also saves summary_all.png (visualization). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//Burgers_Periodic_SinSumHalf/2025_12_23_1053/` |


---
