
# Experiment Report
*Last Updated: 2026-01-11 01:00:08*

## User Request

Let's solve the following Variable-Coefficient Parabolic PDE (backward in time):

$$ \frac{\partial u}{\partial t} + \frac{1}{2} \sigma^2 x^2 \frac{\partial^2 u}{\partial x^2} + r x \frac{\partial u}{\partial x} - r u = 0 $$

Parameters:
$\sigma = 0.2$ (volatility)
$r = 0.05$ (rate)
$K = 100$

Domain:
$x \in [0, 300]$ (Space)
$t \in [0, 1]$ (Time)

Boundary/Terminal Conditions:
1. Terminal Condition at $t=1$:
   $$ u(x, 1) = \max(x - K, 0) $$
2. Boundary at $x=0$: $u(0, t) = 0$
3. Boundary at $x=300$: $u(300, t) \approx x - K e^{-r(1-t)}$

The final plot must be saved as summary_all.png and include snapshots of the solution at t=1.0, 0.75, 0.5, 0.25, 0.0.


---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
**1. Problem Statement**

We solve a **variable-coefficient parabolic PDE (backward in time)** corresponding to the Black–Scholes equation for a European call option, on a truncated asset domain.

---

### 1. Mathematical Specification (Code-Ready Extraction)

**Governing Equation (backward in time):**

Let \(u(x,t)\) denote the option price, with \(x\) the asset price and \(t\) the time (0 = initial, 1 = maturity). The PDE is:

\[
\boxed{
\frac{\partial u}{\partial t}(x,t)
+ \frac{1}{2}\sigma^2 x^2 \frac{\partial^2 u}{\partial x^2}(x,t)
+ r x \frac{\partial u}{\partial x}(x,t)
- r\,u(x,t) = 0,
\quad (x,t)\in (0,300)\times(0,1)
}
\]

**Domain:**

\[
x \in [0,300],\quad t \in [0,1].
\]

We integrate **backward in time** from \(t=1\) to \(t=0\), but numerically this is usually implemented as a **forward-in-time** evolution of \(\tau = 1 - t\). See the strategy below.

**Boundary / Terminal Conditions:**

1. **Terminal condition at maturity \(t=1\):**
   \[
   u(x,1) = \max(x - K, 0) =
   \begin{cases}
   0, & x \le K, \\
   x - K, & x > K,
   \end{cases}
   \quad x\in[0,300].
   \]

2. **Boundary at \(x=0\):**
   \[
   u(0,t) = 0, \quad t\in[0,1].
   \]

3. **Boundary at \(x=300\):** (large-asset asymptotic)
   \[
   u(300,t) \approx 300 - K e^{-r(1-t)}, 
   \quad t\in[0,1].
   \]

**Constants & Parameters:**

\[
\sigma = 0.2,\quad
r = 0.05,\quad
K = 100.
\]

**Expected Physics / Behavior:**

- This is the **Black–Scholes PDE for a European call** on a non-dividend-paying asset.
- Solution is **smooth** in \(x\) and \(t\) except for a kink in \(\partial_x u\) at \(x=K\) at \(t=1\) (payoff).
- As \(t \to 1\), \(u(x,t)\) approaches the payoff \(\max(x-K,0)\).
- As \(x \to 0\), the call price tends to 0.
- As \(x \to \infty\), \(u(x,t) \sim x - K e^{-r(1-t)}\), consistent with the upper boundary condition.

---

**2. Strategic Methodology**

### 2.1 Exact Solution Feasibility

This PDE is the **standard Black–Scholes equation** for a European call with:

- no dividends,
- constant volatility \(\sigma\),
- constant rate \(r\).

It has a **known closed-form analytical solution**:

Transforming to risk-neutral expectation / lognormal distribution, one obtains
\[
u(x,t) = C(x,t) = x\,\Phi(d_1) - K e^{-r(1-t)} \Phi(d_2),
\]
with
\[
d_1 = \frac{\ln\left(\frac{x}{K}\right) + \left(r + \frac{1}{2}\sigma^2\right)(1-t)}{\sigma\sqrt{1-t}},
\quad
d_2 = d_1 - \sigma\sqrt{1-t},
\]
where \(\Phi(\cdot)\) is the standard normal CDF.

This is an **exact analytical solution** of the stated PDE with the given terminal condition on the semi-infinite domain \(x\in(0,\infty)\); the truncated domain to \(x\in[0,300]\) is compatible (boundary at \(x=300\) matches the large-\(x\) asymptotics).

Therefore:

- Yes, we can obtain an **exact analytical solution**.
- By the instructions, we must select an exact/analytical method as the main method. We will treat it under the umbrella of **“Method of Characteristics (Analytical/Exact)”** as our base template for exact solutions.

### 2.2 Main Strategy

We will:

1. **Use the known Black–Scholes closed-form solution** as the ground-truth solution:
   \[
   u(x,t) = x\,\Phi(d_1) - K e^{-r(1-t)} \Phi(d_2),
   \]
   with the above \(d_1, d_2\).

2. Evaluate this solution directly on the discrete grid of \((x,t)\) required by the implementation.

3. Use it to generate **snapshots** at the requested times \(t = 1.0, 0.75, 0.5, 0.25, 0.0\).

This eliminates numerical errors and ensures a perfect benchmark.

We do **not** need a DG or spectral solver for this problem as a primary approach; they can be reserved as optional cross-checks if desired.

---

**3. Implementation Requirements (CRITICAL)**

### 3.0 Configuration

- Use template defaults for spatial and temporal discretization (grid resolution, time-stepping, etc.) if a numerical grid is still constructed for plotting.
- Time will be treated as the original \(t\in[0,1]\). No need to explicitly transform to \(\tau\), since we are directly evaluating the analytical formula.

### 3.1 Adaptive Mesh Refinement & Shock Capturing

- Not applicable as the **main method is exact/analytical**, and the solution is smooth (no shocks).
- If a numerical PDE solver is implemented as a secondary check (optional), the solution remains smooth and parabolic; **no AMR or shock capturing is needed**.

Thus for AMR / shock-capturing:

- Variable: **None** (no AMR; no shocks).
- Justification: The Black–Scholes solution is globally smooth in \(x\) for \(t<1\); only a slope kink at \(t=1\) in the payoff, but that is exactly represented by the analytical formula and does not require mesh adaptation.

### 3.2 Advisor-Mandated Adjustments (Overrides)

- [ADVISOR_REPORT]: None.
- **Advisor-Mandated Adjustments:** None.

---

**4. Visualization Strategy**

**Primary field to plot:**

- The only primary field is the option price:
  \[
  u(x,t)
  \]
- This variable fully captures the physical/financial behavior of interest.

**Required snapshots:**

Plot \(u(x,t)\) vs. \(x\) for:

- \(t_1 = 1.0\),
- \(t_2 = 0.75\),
- \(t_3 = 0.5\),
- \(t_4 = 0.25\),
- \(t_5 = 0.0\).

Since the requirement is for **summary_all.png** and prior instructions call for a grid with “Rows = selected fields, Columns = Time”, we will:

- Use **1 row** (field = \(u\)).
- Use **5 columns** (the five times).
- Each subplot: a 1D line plot of \(u(x,t_i)\) vs. \(x\).

**Implementation notes for plotting:**

- Filename: `summary_all.png` (exact string).
- If implemented in Julia:
  - Explicitly set `colorbar=false` (even though these are line plots, follow the requirement).
- Allow automatic scaling of the vertical axis; do not fix colormaps or color scales.
- Save **full solution state** as needed:
  - Here, the “full state” can be the analytic values of \(u(x,t)\) for all grid points in time, saved at a chosen `dt_save` (e.g., `dt_save = 0.01`), though numerically this is trivial since it is analytic.

---

**5. Cross-Validation & Data Integration**

- [USER_QUESTION] does **not** provide external data paths or sensors.
- No `.npz` files or sensor coordinates are given.
- Therefore:
  - No external data loading.
  - No relative \(L_2\) error calculation versus reference data.
  - No sensor interpolation is required.

---

**REQUIRED COMPONENTS**

**Main Method:**

- **Method of Characteristics (Analytical/Exact)**  
  (Using the closed-form Black–Scholes formula for a European call as the exact analytical solution.)

**Enhancements:**

- None.

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
### Section 1: Status Decision
*   **Status:** APPROVED

### Section 2: Quantitative Scorecard
*   **Score Integrity:** **50** / 50
*   **Score Accuracy:** **20** / 20
*   **Score Details:** **15** / 15
*   **Score Optimality:** **15** / 15
*   **TOTAL SCORE:** **100** / 100

### Section 3: Forensic Analysis & Feedback

**Observation:**
The simulation correctly implements the **Exact Analytical Solution** for the Black-Scholes equation.
*   **Physics:** The plots demonstrate the correct backward-in-time evolution of a European Call option. At maturity ($t=1.0$), the solution exhibits the characteristic "hockey stick" payoff profile ($\max(x-K, 0)$) with a sharp non-differentiable kink at the strike price $K=100$.
*   **Diffusion:** As time progresses backward to $t=0.0$, the diffusion term naturally smooths the kink at the strike price. The solution remains strictly non-negative.
*   **Boundaries:** The solution respects the boundary behaviors: $u \to 0$ as $x \to 0$, and $u$ approaches the linear asymptote $x - Ke^{-r(1-t)}$ as $x \to 300$.
*   **Artifacts:** There are no numerical artifacts (oscillations, smearing, or instability) because the solution is derived from the closed-form equation, not a discretized PDE solver.

**Diagnosis:**
**Perfect Ground Truth Generation.** The strategy correctly identified that the Black-Scholes equation admits a closed-form solution and utilized it. This eliminates discretization error entirely, providing the highest possible fidelity data for PINN training.

**Targeted Improvements:**
*   **None.** The solution is mathematically exact. No further refinement is possible or necessary. The generated data is optimal for training purposes.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Exact analytical solution using the closed-form Black-Scholes formula for European call option price u(x,t) = x Φ(d1) - K exp(-r(1-t)) Φ(d2), with Φ via erf, handling t=1 payoff and x=0 boundary. Evaluated on spatial grid x ∈ [0,300] with 301 points and temporal grid t ∈ [0,1] with 101 points (dt=0.01). Parameters: σ=0.2, r=0.05, K=100. Plots u(x,t) vs x at snapshots t=1.0,0.75,0.5,0.25,0.0 in 1x5 layout. | 0.00e+00 | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves black_scholes.npz containing keys: 't' (101,), 'x' (301,), 'usol' (101x301) (time-major order). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//BlackScholes_Dirichlet_CallPayoff/2026_01_11_0059/` |


---
