
# Experiment Report
*Last Updated: 2026-01-11 01:17:57*

## User Request

Let's solve the 2D Anisotropic Diffusion Equation with nu = 0.25:
$$ \frac{\partial u}{\partial t} = 0.25 \left( \frac{\partial^2 u}{\partial x^2} + 2 \frac{\partial^2 u}{\partial x \partial y} + \frac{\partial^2 u}{\partial y^2} \right) $$

Domain: $(x,y) \in [0, 2\pi]^2$, $t \in [0, 2]$.
BCs: Periodic in both x and y.

Initial Condition:
$$ u(x,y,0) = \sin(x+y) + \cos(x-y) $$

The final plot must be saved as summary_all.png and include snapshots of the solution field.


---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
**1. Problem Statement**

We solve the 2D anisotropic diffusion equation
\[
\frac{\partial u}{\partial t}
= 0.25\left(
\frac{\partial^2 u}{\partial x^2}
+ 2\frac{\partial^2 u}{\partial x \partial y}
+ \frac{\partial^2 u}{\partial y^2}
\right),
\quad (x,y)\in[0,2\pi]^2,\ t\in[0,2].
\]

with periodic boundary conditions and given trigonometric initial data.

---

### 1. Mathematical Specification (Code-Ready Extraction)

**Governing Equation**

\[
\boxed{
\partial_t u(x,y,t)
= \nu\left(
u_{xx}(x,y,t)
+ 2\,u_{xy}(x,y,t)
+ u_{yy}(x,y,t)
\right),\quad \nu=0.25
}
\]

Explicitly:
\[
\partial_t u = 0.25\left(u_{xx} + 2u_{xy} + u_{yy}\right).
\]

**Domain**

\[
(x,y) \in [0,2\pi]\times[0,2\pi],\qquad t\in[0,2].
\]

**Boundary Conditions**

Periodic in both directions:
\[
u(0,y,t) = u(2\pi,y,t),\quad
u(x,0,t) = u(x,2\pi,t),
\]
for all \(y\in[0,2\pi]\), \(x\in[0,2\pi]\), and \(t\in[0,2]\).

**Initial Condition**

\[
\boxed{
u(x,y,0) = \sin(x+y) + \cos(x-y)
}
\]

**Constants & Parameters**

- Diffusivity: \(\nu = 0.25\).
- Final time: \(T = 2.0\).
- Spatial domain size: \(L_x=L_y=2\pi\).

**Expected Physics**

- The PDE is **parabolic** (anisotropic diffusion).
- The operator
  \[
  \mathcal{L}u = u_{xx} + 2u_{xy} + u_{yy}
  \]
  is elliptic and diagonalizable by rotating coordinates.
- The two initial Fourier modes decouple:
  - \( \sin(x+y)\) is an eigenmode of \(\mathcal{L}\) with eigenvalue \(-4\).
  - \(\cos(x-y)\) is an eigenmode of \(\mathcal{L}\) with eigenvalue \(0\).
- Therefore:
  - The \(\sin(x+y)\) component diffuses and decays as \(e^{-1.0\,t}\).
  - The \(\cos(x-y)\) component is steady in time.

The exact solution is:
\[
\boxed{
u(x,y,t) = e^{-1.0\,t}\,\sin(x+y) + \cos(x-y)
}
\]

The previous run used \(e^{-0.5 t}\); this must be corrected to \(e^{-1.0 t}\).

---

**2. Strategic Methodology**

An exact analytical solution is available by diagonalizing the operator in rotated coordinates:

- Introduce
  \[
  \xi = x+y,\quad \eta = x-y.
  \]
- For \(u(x,y,t) = f(\xi,t)\), we have
  \[
  u_{xx} = f_{\xi\xi},\quad
  u_{xy} = f_{\xi\xi},\quad
  u_{yy} = f_{\xi\xi},
  \]
  hence
  \[
  u_{xx} + 2u_{xy} + u_{yy} = f_{\xi\xi} + 2f_{\xi\xi} + f_{\xi\xi} = 4 f_{\xi\xi} = 4 u_{\xi\xi}.
  \]
- The PDE for the \(\sin(x+y)\) mode is thus
  \[
  \partial_t f = 0.25 \cdot 4 f_{\xi\xi} = 1.0\, f_{\xi\xi},
  \]
  and since \(f(\xi,0) = \sin\xi\) is an eigenfunction of \(\partial_{\xi\xi}\) with eigenvalue \(-1\),
  \[
  f(\xi,t) = e^{-1.0\,t}\sin\xi.
  \]
- For the \(\cos(x-y)\) mode, direct computation shows \(\mathcal{L}\cos(x-y) = 0\), so it does not decay.

Thus the **Method of Characteristics / analytical exact separation in rotated coordinates** gives the closed form:
\[
u(x,y,t) = e^{-1.0\,t}\sin(x+y) + \cos(x-y).
\]

**Can obtain an exact solution?** Yes. It is above and matches the PDE, domain, and IC. The correct decay rate is \(e^{-1.0 t}\).

**Main Method choice (per rules):**

- **Method of Characteristics (Analytical/Exact)** used as the template for this exact rotated-coordinate solution.

No numerical PDE solver is needed; we directly evaluate \(u(x,y,t)\) on the grid and times of interest.

---

**3. Implementation Requirements (CRITICAL)**

**Configuration**

- Use template defaults for grid resolution, time sampling, etc., except as needed to produce clean plots.
- Evaluate the closed-form solution:
  \[
  u(x,y,t) = e^{-1.0\,t}\sin(x+y) + \cos(x-y)
  \]
  **ensuring the exponent coefficient is exactly \(-1.0 t\)** (fixing the earlier \(-0.5 t\) mistake).

Concrete code-level formula change (from Advisor):

- **Replace**
  ```pseudo
  u = exp(-0.5 * t) * sin(x + y) + cos(x - y)
  ```
- **With**
  ```pseudo
  u = exp(-1.0 * t) * sin(x + y) + cos(x - y)
  ```

No time stepping: just evaluate this expression at chosen \(t\).

**3.1 Adaptive Mesh Refinement & Shock Capturing**

Not applicable: purely smooth diffusion, exact evaluation, no AMR or shock capturing required.

**Variable Selection Analysis**

- No shocks or interfaces; the solution is globally smooth and analytically known.
- AMR/shock capturing are not used.

**3.1 Advisor-Mandated Adjustments (Overrides)**

- Advisor explicitly requests: **correct decay rate** in the analytical formula from \(e^{-0.5 t}\) to \(e^{-1.0 t}\).
- Other aspects: **None**.

So:

- Advisor overrides: **Use \(e^{-1.0 t}\) for the \(\sin(x+y)\) mode**.

---

**4. Visualization Strategy**

**Field Selection**

- Primary and only field: \(u(x,y,t)\).

It captures all dynamics: anisotropic diffusion causing decay of the \(\sin(x+y)\) pattern while the \(\cos(x-y)\) pattern persists.

**Time Snapshots**

Plot \(u(x,y,t)\) at 4 times (columns):

- \(t_1 = 0\)
- \(t_2 \approx 0.67\) (e.g., \(t=2/3\))
- \(t_3 \approx 1.33\) (e.g., \(t=4/3\))
- \(t_4 = 2.0\)

Each is a 2D heatmap of \(u(x,y,t)\).

**Layout**

- Grid: 1 row (field), 4 columns (time).
- Save figure as:
  ```text
  summary_all.png
  ```
- In Julia or similar, explicitly:
  - **Disable colorbars**: `colorbar=false`.
  - Allow automatic scaling of colormap per subplot.

**Saving Data**

- Even though plotting is from an exact formula, save full fields \(u(x,y,t)\) at a chosen temporal sampling (e.g., `dt_save = 0.1`) for possible further analysis or verification.

---

**5. Cross-Validation & Data Integration**

- No external data paths or sensors specified; no cross-validation required.
- If later provided, compute relative \(L_2\) error against any reference solution using this exact formula as the ground truth.

---

**REQUIRED COMPONENTS**

**Main Method:**

- Method of Characteristics (Analytical/Exact)

**Enhancements:**

- Advisor-mandated coefficient correction (decay rate fix in analytical formula)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
### Section 1: Status Decision
**Status:** APPROVED

---

### Section 2: Quantitative Scorecard

| Category | Score | Max | Notes |
| :--- | :---: | :---: | :--- |
| **Integrity** | **50** | 50 | Code executed successfully, data saved, plots generated correctly. |
| **Accuracy** | **20** | 20 | Exact Analytical Solution used. Physics (decay rates) verified mathematically. |
| **Details** | **15** | 15 | Resolution (256x256) is high fidelity. No numerical artifacts (analytical). |
| **Optimality** | **15** | 15 | "Method of Characteristics/Exact" is the theoretically optimal approach for ground truth. |
| **TOTAL** | **100** | 100 | **Perfect Score.** |

---

### Section 3: Forensic Analysis & Feedback

**1. Observation**
*   **Visual Physics:** The simulation correctly depicts the anisotropic nature of the diffusion.
    *   At $t=0$, the field shows a "checkerboard" interference pattern resulting from the superposition of $\sin(x+y)$ and $\cos(x-y)$.
    *   As time evolves to $t=2.0$, the cross-hatching fades, leaving a dominant diagonal wave structure oriented with slope $+1$ (lines of constant $y-x$).
*   **Mathematical Verification:**
    *   The $\sin(x+y)$ mode corresponds to the fast-decaying eigenmode (eigenvalue $\lambda = -1.0$), which is visibly damped out by $t=2.0$ ($e^{-2} \approx 0.135$).
    *   The $\cos(x-y)$ mode corresponds to the null-space eigenmode (eigenvalue $\lambda = 0$), which persists without decay.
    *   The plot at $t=2.0$ is visually dominated by $\cos(x-y)$, confirming the correct implementation of the decay coefficients.

**2. Diagnosis**
*   **Success:** The previous error (incorrect decay rate $e^{-0.5t}$) has been rectified to $e^{-1.0t}$. The solution is now mathematically exact.
*   **Methodology:** The choice to use the analytical solution rather than a numerical solver guarantees zero discretization error, providing the highest possible quality ground truth for PINN training.

**3. Targeted Improvements**
*   **None.** The solution is perfect. The data generated in `anisotropic_diffusion.npz` is ready for downstream training tasks.

**Reason for Approval:**
The run achieved a **Total Score of 100**. The solution is exact, the physics are verified, and the implementation is optimal. No further refinement is possible or necessary.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Analytical exact solution via direct evaluation of the closed-form formula u(x,y,t) = exp(-0.5 t) sin(x+y) + cos(x-y), matching the Fourier mode decay in rotated coordinates. Domain [0,2π]x[0,2π], t in [0,2]. Spatial grid 256x256, 200 time snapshots (dt≈0.01). Plots 1x4 heatmap snapshots of u at t=0, ≈0.67, ≈1.33, 2. No numerical solver used. | N/A | 50 | 0 | 15 | 5 | **70** |
| 1 | Analytical exact solution evaluation using the closed-form formula u(x,y,t) = exp(-1.0 * t) * sin(x + y) + cos(x - y), matching the rotated-coordinate separation for the anisotropic diffusion equation. Spatial grid: 256 x 256 points over [0, 2π]^2. Time sampling: 200 steps uniformly from t=0 to t=2.0. Plots 2D heatmaps of u at t=0, ≈0.67, ≈1.33, 2.0 in 1x4 layout with colorbar=false, aspect=:equal. | 0.00e+00 | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves anisotropic_diffusion.npz containing keys: 't' (200,), 'x' (256,), 'y' (256,), 'usol' (200,256,256). Also generates summary_all.png (1x4 heatmap panels of u(x,y,t_i)). |
| 1 | Saves summary_all.png (visualization figure). Saves anisotropic_diffusion.npz containing keys: 't' (200,), 'x' (256,), 'y' (256,), 'usol' (200,256,256). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//AnisotropicDiffusion_Periodic_SinPlusCos/2026_01_11_0114/` |
| 1 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//AnisotropicDiffusion_Periodic_SinPlusCos/2026_01_11_0116/` |


---
