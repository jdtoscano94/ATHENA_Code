
# Experiment Report
*Last Updated: 2025-12-15 23:05:17*

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

We solve a scalar 2D inviscid Burgers-type equation with a *single* scalar field transported and steepened by itself along the (1,1) direction:

\[
u_t + u\,u_x + u\,u_y = 0,
\quad (x,y)\in [0,4\pi]\times[0,4\pi],\; t\in[0,2],
\]
with periodic boundary conditions and smooth sinusoidal initial data.

---

### 1. Mathematical Specification (Code-Ready Extraction)

#### Governing Equation

\[
\boxed{
\partial_t u(x,y,t) + u(x,y,t)\,\partial_x u(x,y,t) + u(x,y,t)\,\partial_y u(x,y,t) = 0
}
\]

Equivalently,
\[
u_t + u\,u_x + u\,u_y = 0.
\]

#### Domain

\[
(x,y) \in [0,4\pi]\times[0,4\pi], \quad t \in [0,2].
\]

#### Boundary Conditions

Both directions are periodic:

- In \(x\):
  \[
  u(0,y,t) = u(4\pi,y,t), \quad \forall y\in[0,4\pi],\; t\in[0,2].
  \]
- In \(y\):
  \[
  u(x,0,t) = u(x,4\pi,t), \quad \forall x\in[0,4\pi],\; t\in[0,2].
  \]

#### Initial Condition

\[
\boxed{
u(x,y,0) = \frac{1}{3} + \frac{2}{3}\,\sin\!\left(\frac{x+y}{2}\right)
}
\]

for all \((x,y)\in[0,4\pi]\times[0,4\pi]\).

#### Constants & Parameters

- No explicit physical constants beyond the PDE coefficients.
- For reference:
  - Spatial domain size: \(L_x = L_y = 4\pi\).
  - Time interval: \(0 \le t \le 2\).

#### Expected Physics

- The advective velocity is \((u,u)\), i.e., along the diagonal direction \((1,1)\).
- Initial data is smooth and periodic in the combination \(\xi = x+y\).
- Solution will steepen and form a shock in finite time along the characteristic direction.
- Because the domain is periodic and transport is along \((1,1)\), shock formation and subsequent multi-valued characteristics require **entropy-consistent** selection of the physical branch.

---

**2. Strategic Methodology**

#### Method of Characteristics (MoC) Feasibility

The PDE can be recast in a form that reveals the characteristic structure. Define
\[
\xi = x + y,\quad \eta = x - y.
\]

Then
\[
\partial_x = \partial_\xi + \partial_\eta,\quad
\partial_y = \partial_\xi - \partial_\eta.
\]

Compute:
\[
u_x + u_y = (\partial_\xi u + \partial_\eta u) + (\partial_\xi u - \partial_\eta u)
= 2\,\partial_\xi u.
\]

Hence the PDE becomes
\[
u_t + u (u_x + u_y) = u_t + 2u\,u_\xi = 0.
\]

Crucially, there is **no** \(\eta\)-derivative: \(\eta\) is a passive parameter. So for each fixed \(\eta\), we have a **1D inviscid Burgers equation** in \(\xi\):

\[
\boxed{
u_t(\xi,\eta,t) + 2u(\xi,\eta,t)\,\partial_\xi u(\xi,\eta,t) = 0
}
\]

with 1D periodicity in \(\xi\) induced by the 2D periodic box.

Initial condition in \((\xi,\eta)\):

Original IC:
\[
u(x,y,0) = \frac{1}{3} + \frac{2}{3}\sin\!\left(\frac{x+y}{2}\right)
= \frac{1}{3} + \frac{2}{3}\sin\!\left(\frac{\xi}{2}\right),
\]
independent of \(\eta\). Denote
\[
u_0(\xi) = \frac{1}{3} + \frac{2}{3}\sin\!\left(\frac{\xi}{2}\right).
\]

For each fixed \(\eta\), the equation
\[
u_t + 2u u_\xi = 0
\]
has standard Burgers-type characteristics:

- Characteristic ODEs:
  \[
  \frac{d\xi}{dt} = 2u, \quad \frac{d\eta}{dt} = 0,\quad \frac{du}{dt} = 0.
  \]
- Along a characteristic, \(u\) is constant: \(u(t) = u_0(\xi_0)\).
- Characteristic curves in \(\xi\):
  \[
  \xi(t) = \xi_0 + 2u_0(\xi_0)\,t.
  \]

Thus the solution can be written implicitly as
\[
u(\xi,\eta,t) = u_0(\xi_0),
\]
where \(\xi_0\) satisfies
\[
\xi = \xi_0 + 2u_0(\xi_0)\,t
\quad\Longleftrightarrow\quad
\xi_0 = \xi - 2u_0(\xi_0)\,t.
\]

Because \(u_0(\xi_0)\) is explicitly known,
\[
u_0(\xi_0) = \frac{1}{3} + \frac{2}{3}\sin\!\left(\frac{\xi_0}{2}\right),
\]
the characteristic relation is
\[
\xi = \xi_0 + 2\left(\frac{1}{3} + \frac{2}{3}\sin\!\left(\frac{\xi_0}{2}\right)\right)t
= \xi_0 + \frac{2}{3}t + \frac{4}{3} t \sin\!\left(\frac{\xi_0}{2}\right).
\]

For each time \(t\) and each spatial point \((x,y)\), we compute \(\xi = x + y\) and solve the **implicit scalar equation** for \(\xi_0\):

\[
\boxed{
\xi = \xi_0 + \frac{2}{3}t + \frac{4}{3} t \sin\!\left(\frac{\xi_0}{2}\right)
}
\]

Then
\[
\boxed{
u(x,y,t) = u(\xi,\eta,t) = u_0(\xi_0)
= \frac{1}{3} + \frac{2}{3}\sin\!\left(\frac{\xi_0}{2}\right).
}
\]

- \(\eta = x - y\) does not appear in the formula; solution depends only on \(\xi=x+y\), consistent with the symmetry of the initial data and PDE.

This is a **full Method of Characteristics solution** (implicit form). For small \(t\), the map \(\xi_0 \mapsto \xi\) is invertible and single-valued. After shock formation time, the mapping becomes multi-valued; the **entropy condition** selects the physically correct weak solution branch. The MoC implementation is assumed to handle this via entropy-consistent root selection.

Given this derivation is complete and closed-form aside from solving a scalar implicit equation per point, the **Method of Characteristics is fully feasible** and exact for all times (up to and beyond shock).

#### Main Method Selection

By mandate:

- MoC derivation succeeded with explicit characteristic relations.
- Therefore, the **Main Method MUST be the Method of Characteristics (Exact)**.

No need to fall back to DG or spectral discretizations for this problem.

**Entropy Condition Root Selection:**  
Post-shock, when the equation
\[
\xi = \xi_0 + \frac{2}{3}t + \frac{4}{3} t \sin\!\left(\frac{\xi_0}{2}\right)
\]
admits multiple roots \(\xi_0\) for a given \((\xi,t)\), the implementation MUST select the entropy-satisfying branch (corresponding to the physically admissible weak solution of 1D Burgers in \(\xi\)). This is handled within the MoC solver.

---

**3. Implementation Requirements (CRITICAL)**

#### Configuration

- Use the **Method of Characteristics** with:
  - Coordinate mapping:
    \[
    \xi = x + y,\quad \eta = x - y.
    \]
  - For each grid point \((x_i,y_j)\) and time of interest \(t\), compute:
    \[
    \xi_{ij} = x_i + y_j,
    \]
    then solve for \(\xi_0\) in
    \[
    \xi_{ij} = \xi_0 + \frac{2}{3}t + \frac{4}{3} t \sin\!\left(\frac{\xi_0}{2}\right),
    \]
    and evaluate
    \[
    u_{ij}(t) = \frac{1}{3} + \frac{2}{3}\sin\!\left(\frac{\xi_0}{2}\right).
    \]

- Domain discretization, time grid, and solver details can use template defaults.

#### 3.1 Adaptive Mesh Refinement & Shock Capturing

Not applicable: we are using the Method of Characteristics, **not** DGSEM. No AMR or numerical shock-capturing mechanism is required; the solution is analytic (implicit) and exact.

If an implementation later switches to a DGSEM solver, the recommended AMR/shock indicator variable would be:

- **Variable for AMR / Shock Capturing:** \(u\) itself.
- **Justification:** In scalar Burgers, the primary coherent feature is the gradient steepening and shock in \(u\); there are no additional variables. \(u\) directly encodes the shock structure with minimal global chaotic behavior.

But under the current plan (MoC), this is informational only and not used.

#### 3.1 Advisor-Mandated Adjustments (Overrides)

- Advisor report: None.
- Therefore:

\[
\boxed{\text{Advisor-Mandated Adjustments: None}}
\]

---

**4. Visualization Strategy**

#### Field Selection

There is a single primary field:

- **\(u(x,y,t)\)** is the only and thus the relevant field to visualize.

We will plot **\(u\)** at 4 distinct time steps to capture evolution and shock formation. For example:

- \(t_1 = 0\) (initial condition, smooth sinusoid along \(x+y\)).
- \(t_2 = 0.5\)
- \(t_3 = 1.0\)
- \(t_4 = 2.0\)

(Exact choices can follow template defaults, but must include multiple times between 0 and 2.)

#### Plot Layout

- Single row (since only one field), 4 columns (one per time).
- Title or labeling indicating time for each subplot.

**Critical plotting directive (Julia-specific):**

- Save the combined figure as:
  \[
  \boxed{\texttt{"summary_all.png"}}
  \]
- When using Julia plotting, **disable colorbars** explicitly:
  - e.g., `colorbar=false` in the plotting calls.
- Let colormap scales adjust automatically; do **not** fix global min/max across time unless desired by the template.

#### Data Saving

- Save the **full numerical solution state** \(u(x,y,t)\) at regular intervals (e.g. `dt_save = 0.1`) for ground truth and possible later analysis.
- Plotting selection (4 snapshots) affects only `summary_all.png`, not the data saving cadence.

---

**5. Cross-Validation & Data Integration**

- The [USER_QUESTION] does **not** provide external data paths (no sensors, `.npz` files, or reference solution files).
- Hence:

  - No external data loading is required.
  - No relative \(L_2\) error computation against reference data is mandated.
  - No sensor interpolation is needed.

---

**REQUIRED COMPONENTS**

**Main Method:**

- **Method of Characteristics (Exact)**
- **[Coordinate Mapping]**: \((x,y) \mapsto (\xi=x+y,\ \eta=x-y)\)

**Enhancements:**

- None required (no Reynolds number, no DG shock capturing, no advisor overrides).

This strategy yields an exact (implicit) solution suitable as ground truth, with robust handling of pre- and post-shock regimes via entropy-consistent characteristic inversion.

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
**Section 1: Status Decision**
*   **Status:** APPROVED

**Section 2: Quantitative Scorecard**
*   **Score Integrity:** **50** / 50 (Perfect execution, valid data generation, clear visualization).
*   **Score Accuracy:** **20** / 20 (Physics are exact; shock formation time and location align with analytical theory).
*   **Score Details:** **15** / 15 (Razor-sharp shocks, zero numerical diffusion, zero oscillations).
*   **Score Optimality:** **15** / 15 (Method of Characteristics is the theoretical gold standard for this problem).
*   **TOTAL SCORE:** **100** / 100

**Section 3: Forensic Analysis & Feedback**

*   **Observation:**
    *   The simulation implements the **Analytical Method of Characteristics (MoC)**.
    *   **Physics Verification:** The initial condition is a smooth sine wave oriented along the diagonal $\xi = x+y$. As time progresses ($t=0.51, 0.99$), the wave steepens correctly due to the nonlinear advection term.
    *   **Shock Formation:** Theoretical shock formation time for this setup is $t_s = 1.5$. The plot at $t=0.99$ shows extreme steepening but continuity, while the plot at $t=2.0$ shows a fully formed, crisp discontinuity (shock). This perfectly matches the theoretical prediction.
    *   **Resolution:** Because the solution is analytical (root-finding), the shock interface is infinitely sharp (limited only by pixel density), with absolutely no "smearing" or "ringing" (Gibbs phenomenon) typical of grid-based methods.

*   **Diagnosis:**
    *   **Perfect Ground Truth:** The solution is mathematically exact. It correctly handles the multi-valued region of the characteristics by applying the entropy condition (equal-area rule equivalent) to select the physical shock location.

*   **Targeted Improvements:**
    *   **None.** The solution represents the highest possible fidelity for this problem. It is ready for use as training data for the PINN.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Analytical Method of Characteristics (Exact) for the 2D scalar inviscid Burgers equation u_t + u u_x + u u_y = 0, using coordinate mapping ξ = x + y (η = x - y passive), reducing to 1D Burgers u_t + 2 u u_ξ = 0 along characteristics. Solves implicit equation ξ = ξ₀ + (2/3)t + (4/3) t sin(ξ₀/2) via Bisection root-finding in phase space θ = ξ/2, with entropy-consistent shock fitting: tracks shock at θ_s = π + (1/3)t, decomposes domain into pre/post-shock regions, solves single-valued roots on expansion side, reflects compression side using equal-area rule (reflection across shock for coordinates and values). Domain [0,4π]×[0,4π] periodic, initial condition u=1/3 + 2/3 sin((x+y)/2), discretized on 128×128 spatial grid, evaluated at 100 time steps over t∈[0,2]. Plots u at t=0,0.5,1.0,2.0. | 0.00e+00 | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves 'summary_all.png' (1×4 heatmap visualization of u(x,y,t) at 4 snapshot times, colorbar=false). Saves 'burgers.npz' containing keys: 't' (100,), 'x' (128,), 'y' (128,), 'usol' (100,128,128). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//Burgers_Periodic_SinHalfSum/2025_12_15_2303/` |


---
