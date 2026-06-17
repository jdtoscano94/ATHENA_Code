
# Experiment Report
*Last Updated: 2026-01-11 12:54:47*

## User Request

Let's solve the 3D Wave Equation assuming spherical symmetry.
Equation:
$$ \frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u $$
where $\nabla^2 u = \frac{1}{r^2} \frac{\partial}{\partial r} \left( r^2 \frac{\partial u}{\partial r} \right)$.

Parameters: c = 1.0
Domain: $r \in [0, 10]$, $t \in [0, 6]$.
BCs:
- $u(r,t)$ must be finite at $r=0$.
- Homogeneous Neumann at $r=10$.

Initial Conditions:
$$ u(r,0) = \frac{e^{-10(r-5)^2}}{r} \cdot \sin\left(\frac{\pi r}{10}\right) $$
$$ \frac{\partial u}{\partial t}(r,0) = 0 $$

The final plot must be saved as summary_all.png and include snapshots of the solution at t=0, 2, 4, 6.


---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
**1. Problem Statement**

We solve the spherically symmetric 3D wave equation on a finite radial interval with a Neumann outer boundary and regularity at the origin, using an exact analytical reduction via spherical symmetry.

---

### 1. Mathematical Specification (Code-Ready Extraction)

#### Governing Equation

With wave speed \(c=1\), the spherically symmetric 3D wave equation is
\[
\frac{\partial^2 u}{\partial t^2}(r,t) \;=\; \nabla^2 u(r,t)
\]
with radial Laplacian
\[
\nabla^2 u(r,t) \;=\; \frac{1}{r^2}\,\frac{\partial}{\partial r} \left( r^2 \frac{\partial u}{\partial r}(r,t) \right),
\]
so explicitly:
\[
\boxed{
\frac{\partial^2 u}{\partial t^2}(r,t) \;=\; \frac{1}{r^2}\,\frac{\partial}{\partial r} \left( r^2 \frac{\partial u}{\partial r}(r,t) \right),\quad r\in(0,10),\ t\in(0,6).
}
\]

#### Domain

\[
r \in [0,10],\qquad t \in [0,6].
\]

#### Boundary Conditions

- Regularity at the origin: \(u(r,t)\) must remain finite at \(r=0\). For a spherically symmetric 3D wave, this implies
  \[
  \lim_{r\to 0} u(r,t) < \infty,
  \]
  and, equivalently for the reduced 1D problem (below), a Dirichlet-like condition for the transformed field.

- Homogeneous Neumann at the outer boundary:
  \[
  \left. \frac{\partial u}{\partial r}(r,t) \right|_{r=10} = 0,\quad \forall t\in[0,6].
  \]

#### Initial Conditions

Given initial displacement and zero initial velocity:
\[
u(r,0) = u_0(r) = \frac{e^{-10(r-5)^2}}{r} \,\sin\left(\frac{\pi r}{10}\right),
\]
\[
\left.\frac{\partial u}{\partial t}(r,t)\right|_{t=0} = u_t(r,0) = 0.
\]

Write these explicitly:
\[
\boxed{
u(r,0) = \frac{e^{-10(r-5)^2}}{r} \,\sin\left(\frac{\pi r}{10}\right),\quad r\in(0,10],
}
\]
\[
\boxed{
\frac{\partial u}{\partial t}(r,0) = 0,\quad r\in[0,10].
}
\]

At \(r=0\), interpret \(u(r,0)\) via its limit; numerically, you will regularize using the transformed variable below.

#### Constants & Parameters

- Wave speed:
  \[
  c = 1.0.
  \]
- Final time:
  \[
  T_{\text{final}} = 6.
  \]
- Outer radius:
  \[
  R = 10.
  \]

#### Expected Physics

- This is a radially localized, smooth “shell-like” disturbance centered around \(r=5\) (Gaussian envelope times a sinusoid).
- The 3D spherical geometry introduces geometric spreading; amplitude decays as waves move outward.
- With the Neumann condition at \(r=10\), we expect partial reflection at the outer boundary.
- Zero initial velocity leads to symmetric-in-time evolution: an initial “standing” configuration splitting into inward and outward traveling components.

---

**2. Strategic Methodology**

#### 2.1 Exact Reduction via Spherical Symmetry

Use the standard reduction for the 3D radial wave equation:
\[
v(r,t) := r\,u(r,t).
\]

Compute the equation for \(v\):

Start from
\[
u_{tt} = \frac{1}{r^2}\partial_r(r^2 u_r).
\]

With \(u = v/r\),
\[
u_r = \frac{v_r r - v}{r^2},
\]
\[
r^2 u_r = v_r r - v,
\]
\[
\partial_r(r^2 u_r) = \partial_r(v_r r - v) = v_{rr} r.
\]

Hence
\[
u_{tt} = \frac{1}{r^2}\, v_{rr} r = \frac{1}{r} v_{rr}.
\]

Left-hand side:
\[
u_{tt} = \frac{1}{r} v_{tt}.
\]

So for \(r>0\):
\[
\frac{1}{r} v_{tt} = \frac{1}{r} v_{rr}
\quad\Rightarrow\quad
\boxed{v_{tt} = v_{rr}}.
\]

Thus \(v(r,t)\) satisfies the **1D wave equation** on the interval \(r\in(0,10)\).

#### 2.2 Transformed Boundary & Initial Conditions

- At \(r=0\), finiteness of \(u\) implies \(v(0,t) = r u(r,t) \to 0\) as \(r\to 0\). So:
  \[
  \boxed{v(0,t) = 0\quad\text{(Dirichlet at } r=0\text{)}}.
  \]

- At \(r=10\), Neumann for \(u\):
  \[
  u_r(10,t) = 0
  \quad\Rightarrow\quad
  v_r(10,t) = v(10,t).
  \]
  (From \(u_r = (v_r r - v)/r^2\), impose \(u_r(10,t)=0\Rightarrow v_r(10,t) = v(10,t)/10\). But we already have:
  \[
  r^2 u_r = v_r r - v; \quad u_r(10,t)=0 \Rightarrow v_r(10,t) \cdot 10 - v(10,t) = 0 \Rightarrow v_r(10,t) = \frac{v(10,t)}{10}.
  \]
  This is a Robin condition for \(v\).)

  So boundary for \(v\) is:
  \[
  \boxed{v_r(10,t) = \frac{1}{10} v(10,t)\quad\text{(Robin at } r=10\text{)}}.
  \]

- Initial conditions for \(v\):
  \[
  v(r,0) = r u(r,0) = r \cdot \frac{e^{-10(r-5)^2}}{r} \,\sin\left(\frac{\pi r}{10}\right)
  = e^{-10(r-5)^2} \,\sin\left(\frac{\pi r}{10}\right),
  \]
  \[
  v_t(r,0) = r u_t(r,0) = 0.
  \]

So:
\[
\boxed{
v(r,0) = e^{-10(r-5)^2} \,\sin\left(\frac{\pi r}{10}\right),\quad v_t(r,0) = 0.
}
\]

This is smooth on \([0,10]\) and compatible with \(v(0,0)=0\) since \(v(0,0)=e^{-250}\sin(0)=0\).

#### 2.3 Exact Solution Class

On \((0,10)\), \(v\) satisfies
\[
v_{tt} = v_{rr},\quad v(0,t)=0,\quad v_r(10,t) = \frac{1}{10}v(10,t),
\]
with \(v(r,0)=f(r)\), \(v_t(r,0)=0\), where \(f(r)=e^{-10(r-5)^2}\sin(\pi r/10)\).

This is a **linear, second-order hyperbolic PDE** on a finite interval with mixed (Dirichlet–Robin) homogeneous linear boundary conditions. It is ideally suited to separation of variables / eigenfunction expansion:

Seek eigenmodes
\[
v(r,t) = X(r)T(t),
\]
leading to
\[
\frac{T''(t)}{T(t)} = \frac{X''(r)}{X(r)} = -\lambda,
\]
with spatial problem
\[
X''(r) + \lambda X(r) = 0, \quad X(0)=0,\quad X'(10) = \frac{1}{10}X(10).
\]

- This yields a discrete set of eigenvalues \(\lambda_n = k_n^2 > 0\), with eigenfunctions
  \[
  X_n(r) = \sin(k_n r),
  \]
  where \(k_n\) satisfy
  \[
  X(0)=0 \Rightarrow X(r)=A\sin(k r),
  \]
  \[
  X'(10) = \frac{1}{10}X(10)
  \Rightarrow k\cos(10k) = \frac{1}{10}\sin(10k)
  \Rightarrow \tan(10k) = 10k.
  \]

- Time dependence:
  \[
  T_n(t) = A_n \cos(k_n t) + B_n \sin(k_n t).
  \]

- Imposing \(v_t(r,0)=0\) forces \(B_n=0\), so:
  \[
  v(r,t) = \sum_{n=1}^\infty a_n \cos(k_n t)\,X_n(r) = \sum_{n=1}^\infty a_n \cos(k_n t)\,\sin(k_n r).
  \]

- Coefficients from initial displacement \(f(r)\):
  \[
  f(r) = \sum_{n=1}^\infty a_n \sin(k_n r),
  \]
  with inner product appropriate to this Sturm–Liouville problem (weight \(w(r)=1\) here). So
  \[
  a_n = \frac{\displaystyle \int_0^{10} f(r)\,\sin(k_n r)\,dr}{\displaystyle \int_0^{10} \sin^2(k_n r)\,dr}.
  \]

Thus we have an **analytical representation**:
\[
\boxed{
v(r,t) = \sum_{n=1}^\infty a_n \cos(k_n t)\,\sin(k_n r),\quad
k_n\ \text{solve}\ \tan(10k_n) = 10k_n,\quad
a_n = \frac{\int_0^{10} e^{-10(r-5)^2}\sin\!\left(\tfrac{\pi r}{10}\right)\sin(k_n r)\,dr}
{\int_0^{10}\sin^2(k_n r)\,dr}.
}
\]

Then
\[
\boxed{
u(r,t) = \frac{1}{r} v(r,t)
= \frac{1}{r} \sum_{n=1}^\infty a_n \cos(k_n t)\,\sin(k_n r),
\quad r>0.
}
\]
At \(r=0\), define \(u(0,t)\) via limit:
\[
u(0,t) = \lim_{r\to 0} \frac{v(r,t)}{r} = \sum_{n=1}^\infty a_n k_n \cos(k_n t),
\]
since \(\sin(k_n r) \sim k_n r\) near \(0\). In practice, this is either computed from the series or regularized numerically.

This is an **exact analytical solution via separation of variables**, which is a special case of an analytical method and must be prioritized as the main method.

#### 2.4 Method Selection

- The PDE is linear, hyperbolic, smooth initial data, finite domain, with simple linear BCs.
- We can obtain a closed-form eigenfunction expansion; no shocks, no nonlinearity.
- Therefore: **use the analytical separation-of-variables solution, implemented under the “Method of Characteristics (Analytical/Exact)” template.**

No DGSEM or other numerical scheme is needed for the primary ground truth; any numerical solver would be for secondary verification only.

---

**3. Implementation Requirements**

Use the analytical expansion for \(v\) and recover \(u\).

1. **Eigenvalues \(k_n\):**
   - Solve for \(k_n\) as the positive roots of
     \[
     \tan(10k) = 10k.
     \]
   - Truncate at some \(N_{\text{modes}}\) large enough for convergence (e.g. start with \(N_{\text{modes}} \sim 100\) and increase if needed until solution changes are negligible).

2. **Eigenfunctions:**
   - Use
     \[
     X_n(r) = \sin(k_n r).
     \]

3. **Normalization integrals:**
   - Denominator:
     \[
     D_n = \int_0^{10} \sin^2(k_n r)\,dr.
     \]
   - Numerator:
     \[
     N_n = \int_0^{10} e^{-10(r-5)^2}\sin\!\left(\frac{\pi r}{10}\right)\sin(k_n r)\,dr.
     \]
   - Coefficients:
     \[
     a_n = \frac{N_n}{D_n}.
     \]

   Both integrals should be computed numerically with high-order quadrature.

4. **Time evolution:**
   - For any \(t\),
     \[
     v(r,t) = \sum_{n=1}^{N_{\text{modes}}} a_n \cos(k_n t)\,\sin(k_n r),
     \]
     \[
     u(r,t) = \frac{v(r,t)}{r},\quad r>0.
     \]
   - At \(r=0\), evaluate via limit (e.g., small-\(r\) expansion or simply set \(u(0,t)=u(\Delta r,t)\) with very small \(\Delta r\) if needed; the exact series gives
     \(\sum a_n k_n \cos(k_n t)\)).

5. **Configuration:**
   - Follow "template defaults" for numerical quadrature and root-finding tolerances.
   - No CFL or time stepper is needed for the main method (we evaluate times directly).

#### 3.1 Adaptive Mesh Refinement & Shock Capturing

Not applicable: we are using an analytical method. No shocks, no AMR, no shock capturing.

Explicitly:
- Variable for AMR/shock capturing: **None** (not used; solution smooth and exact).

#### 3.1 Advisor-Mandated Adjustments (Overrides)

- Advisor report: none provided.
- Overrides: **None**.

---

**4. Visualization Strategy**

You need a 1D radial solution visualized at the requested times.

- **Primary field for plotting:** \(u(r,t)\).
  - This is the physical displacement field and directly answers the posed problem.
- **Time snapshots:** \(t = 0,\ 2,\ 4,\ 6\).

**Plot layout:**

- Create a figure with 1 row and 4 columns (since only one field is plotted).
  - Column 1: \(u(r,0)\) vs \(r\).
  - Column 2: \(u(r,2)\) vs \(r\).
  - Column 3: \(u(r,4)\) vs \(r\).
  - Column 4: \(u(r,6)\) vs \(r\).
- Use consistent axes for \(r \in [0,10]\).
- If using Julia, **explicitly set** `colorbar=false` (even though this is 1D line plots, keep it explicit to follow instructions).
- Let vertical scales auto-adjust per subplot (no manual limits unless needed).

**Saving:**

- Save the final composite figure as:
  \[
  \texttt{"summary_all.png"}
  \]

**Data saving:**

- Save the full solution \(u(r,t)\) (and optionally \(v(r,t)\)) over the radial grid at sufficient temporal resolution (e.g., many time samples between 0 and 6) for ground truth, even if only 4 times are plotted.

---

**5. Cross-Validation & Data Integration**

- The [USER_QUESTION] does **not** specify any external reference data, sensors, or `.npz` files.
- Therefore:
  - External data loading: **Not required**.
  - Relative \(L_2\) error to reference: **Not required**.
  - Sensor interpolation: **Not required**.

---

### REQUIRED COMPONENTS

**Main Method:**

- Method of Characteristics (Analytical/Exact)  
  (implemented as separation-of-variables eigenfunction expansion for the reduced 1D wave equation in \(v = r u\)).

**Enhancements:**

- None.

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
### Section 1: Status Decision
*   **Status:** APPROVED
*   **Reason:** Perfect Score Achieved. The solution represents the exact analytical ground truth (via eigenfunction expansion) for the spherically symmetric wave equation. The physics of geometric spreading (amplitude scaling $\propto 1/r$) and boundary reflections are captured perfectly with no numerical artifacts.

---

### Section 2: Quantitative Scorecard

*   **Score Integrity:** **50/50**
    *   Code executed successfully.
    *   Logs confirm convergence of eigenmode computation.
    *   Output files (`summary_all.png`, `wave_spherical.npz`) generated correctly.
*   **Score Accuracy:** **20/20**
    *   **Physics Check:** The solution exhibits the correct spherical wave behavior.
        *   *Inward Traveling Wave:* Amplitude increases as $r \to 0$ (due to $1/r$ factor).
        *   *Outward Traveling Wave:* Amplitude decreases as $r \to 10$.
        *   *Reflections:* The reflection at $r=0$ inverts the sign (Dirichlet behavior for the reduced variable $v=ru$), while the reflection at $r=10$ preserves the sign (Neumann behavior).
    *   **Timing:** Pulse positions match $c=1$ exactly (e.g., at $t=2$, peaks are at $5 \pm 2 = 3, 7$).
*   **Score Details:** **15/15**
    *   **Resolution:** The Gaussian pulse is smooth and well-resolved by the 150 eigenmodes.
    *   **Stability:** No Gibbs phenomenon or ringing is visible.
    *   **Efficiency:** The analytical series expansion is the most efficient method for this linear problem.
*   **Score Optimality:** **15/15**
    *   **Method Selection:** The "Method of Characteristics (Analytical/Exact)" using Separation of Variables is the theoretically optimal approach for generating ground truth data for this PDE.

*   **TOTAL SCORE:** **100/100**

---

### Section 3: Forensic Analysis & Feedback

**1. Observation**
The [SUMMARY PLOT] tells a clear physical story consistent with the 3D Wave Equation in spherical coordinates:
*   **t=0.0:** A symmetric Gaussian pulse is initialized at $r=5$.
*   **t=2.0:** The pulse splits into two counter-propagating waves.
    *   The **left-moving pulse** (at $r=3$) has grown in amplitude ($>0.15$) compared to the initial state.
    *   The **right-moving pulse** (at $r=7$) has diminished in amplitude ($<0.10$).
    *   *Physics:* This confirms the $1/r$ geometric spreading law.
*   **t=4.0:** The separation continues. The inward pulse at $r=1$ has spiked significantly in amplitude ($\approx 0.5$), while the outward pulse at $r=9$ is small.
*   **t=6.0:** Both pulses have reflected.
    *   **Inner Reflection ($r=1$):** The wave that hit the origin ($r=0$) has reflected back to $r=1$. Crucially, it has **flipped sign** (become negative). This is correct: the regularity condition at $r=0$ acts like a "fixed end" for the reduced variable $v=ru$, causing phase inversion.
    *   **Outer Reflection ($r=9$):** The wave that hit the boundary ($r=10$) has reflected back to $r=9$. It retains its **positive sign**. This is correct: the Neumann condition ($u_r=0$) acts like a "free end," preserving phase.

**2. Diagnosis**
*   **Phenomenon:** **Exact Analytical Solution.**
*   **Root Cause:** The implementation correctly utilized the transformation $v(r,t) = r u(r,t)$ to reduce the problem to a 1D wave equation with mixed boundary conditions, solved via Fourier-like series expansion.

**3. Targeted Improvements**
*   **None.** The solution is optimal and represents high-fidelity ground truth.

**4. Log Analysis (Error Norms)**
*   **Minimum achieved error:** N/A (Ground Truth Generation).
    *   *Note:* As this is the generation of the exact solution itself, there is no reference solution to calculate an error against. The logs correctly show no error metrics, only successful computation steps.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Analytical method via separation of variables / eigenfunction expansion for the reduced 1D wave equation v_tt = v_rr on [0,10] with Dirichlet at r=0 and Robin v_r(10)=v(10)/10. Eigenvalues k_n solve tan(10 k_n)=10 k_n using bisection root-finding for first N_modes=150 roots. Coefficients a_n from L2-projection of initial v(r,0)=exp(-10(r-5)^2) sin(pi r /10) onto sin(k_n r) using QuadGK quadrature. Solution evaluated as v(r,t)=sum a_n cos(k_n t) sin(k_n r), u(r,t)=v(r,t)/r for r>0 and series limit sum a_n k_n cos(k_n t) at r=0. Computed on grid N_r=500 points in r [0,10], N_t=600 points in t [0,6]. | N/A | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves summary_all.png (1x4 plot layout of u(r,t) snapshots). Saves wave_spherical.npz containing keys: 't' (600,), 'r' (500,), 'u' (600x500) where u is time-first (t,r) ordering of the full solution tensor. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//Wave_Neumann_GaussianSin/2026_01_11_1252/` |


---
