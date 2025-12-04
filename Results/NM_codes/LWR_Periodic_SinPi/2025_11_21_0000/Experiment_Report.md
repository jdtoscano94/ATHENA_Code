
# Experiment Report
*Last Updated: 2025-11-21 00:01:21*

## User Request

Let's solve the Lighthill-Whitham-Richards (LWR) Traffic Flow equation.
The domain is x \in [-1, 1] and t \in [0, 1.5] with periodic boundary conditions.

The Equation is the scalar conservation law for density ho:
\partial_t ho + \partial_x (ho (1 - ho)) = 0

The Initial Condition is a smooth density variation:
ho_0(x) = 0.5 + 0.3 \sin(\pi x)


---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
**1. Problem Statement**

- **Equation (Strong Form):**  
  \[
  \partial_t \rho(x,t) + \partial_x\big(\rho(x,t)\,(1 - \rho(x,t))\big) = 0,\quad x\in[-1,1],\ t\in[0,1.5].
  \]
  with flux \(f(\rho)=\rho(1-\rho)\).

- **Type:** 1D scalar conservation law, nonlinear, **hyperbolic**.

- **Domain / BCs:**  
  Spatial domain: \([-1,1]\), time interval: \([0,1.5]\).  
  Boundary conditions: **periodic** in \(x\).

- **Initial Condition:**  
  \[
  \rho(x,0) = \rho_0(x) = 0.5 + 0.3\sin(\pi x).
  \]

- **Expected Physics:**  
  Nonlinear advection with \(f'(\rho) = 1 - 2\rho\), possible development of steep gradients and shocks from smooth initial data; rarefaction where characteristics diverge, compression where they converge, all wrapped periodically.

---

**2. Strategic Methodology**

- **Selected Main Method:** Method of Characteristics (Analytical/Exact).

- **Rationale:** 1D scalar conservation law with smooth periodic data and concave flux admits a clean characteristic formulation and implicit exact solution, ideal for generating ground-truth.

---

**3. Mathematical Blueprint**

**3.1 Characteristic Formulation (Lagrangian Frame)**

PDE:
\[
\partial_t \rho + \partial_x f(\rho) = 0,\quad f(\rho)=\rho(1-\rho).
\]

Characteristic ODE system:
\[
\begin{cases}
\displaystyle \frac{d x}{d t} = f'(\rho) = 1 - 2\rho,\\[4pt]
\displaystyle \frac{d \rho}{d t} = 0.
\end{cases}
\]

Thus \(\rho\) is constant along characteristics and
\[
x(t) = x_0 + \big(1 - 2\rho_0(x_0)\big)\, t,
\]
with
\[
\rho(x(t),t) = \rho_0(x_0),\quad x_0\in[-1,1]\ \text{(periodic)}.
\]

Using the given initial data:
\[
\rho_0(x_0) = 0.5 + 0.3\sin(\pi x_0),
\]
so
\[
1 - 2\rho_0(x_0) 
= 1 - 2\big(0.5 + 0.3\sin(\pi x_0)\big)
= -0.6\sin(\pi x_0).
\]

Hence the characteristic map:
\[
x(t) = x_0 - 0.6\, t\,\sin(\pi x_0)\quad (\text{mod }2 \text{ periodically on }[-1,1]).
\]

**Implicit exact solution:**
\[
\rho(x,t) = \rho_0(x_0),\quad x = x_0 - 0.6\, t\,\sin(\pi x_0)\ (\text{periodic}).
\]

For each \((x,t)\), \(x_0\) is obtained by solving the nonlinear equation
\[
G(x_0; x,t) := x_0 - 0.6\, t\,\sin(\pi x_0) - x = 0
\]
on the periodic domain; then
\[
\rho(x,t) = 0.5 + 0.3\sin(\pi x_0).
\]

**3.2 Shock Formation Criterion (Rankine–Hugoniot / Multivalued Characteristics)**

Characteristics intersect when \(\partial x / \partial x_0 = 0\):
\[
\frac{\partial x}{\partial x_0}
= 1 - 0.6\, t\,\pi\cos(\pi x_0) = 0
\quad\Rightarrow\quad
t_s(x_0) = \frac{1}{0.6\pi\,\cos(\pi x_0)}.
\]

The earliest positive shock time:
\[
t_{\min} = \min_{x_0:\ \cos(\pi x_0)>0} t_s(x_0) 
= \frac{1}{0.6\pi} \approx 0.53,
\]
attained where \(\cos(\pi x_0)=1\) (e.g., \(x_0=0\) mod 2).  
Thus for \(t < t_{\min}\) the characteristic map is invertible and the implicit MoC solution is single-valued and smooth. For \(t > t_{\min}\), shocks form; one then selects the entropy solution using the **Rankine–Hugoniot condition** at discontinuities:
\[
s = \frac{f(\rho_R) - f(\rho_L)}{\rho_R - \rho_L}.
\]

For the current strategic report, the key is the pre-shock exact solution structure; post-shock, one uses the same characteristic construction plus entropy selection.

**3.3 Weak Form (for reference / cross-check)**

For any smooth periodic test function \(\varphi(x,t)\) with compact support in time \((0,1.5)\),
\[
\int_0^{1.5}\!\!\int_{-1}^{1} 
\Big( \rho\,\partial_t \varphi + f(\rho)\,\partial_x \varphi\Big)\,dx\,dt
+ \int_{-1}^{1} \rho_0(x)\,\varphi(x,0)\,dx = 0.
\]

Here shocks (when they appear) are admitted as weak solutions; entropy admissibility is enforced via the characteristic construction plus Rankine–Hugoniot.

---

**4. Implementation Requirements (CRITICAL)**

- **Configuration:** Use template defaults.

**Main Method:**
- Method of Characteristics (Exact / Implicit Mapping)

**Enhancements:**
- Coordinate Mapping (Lagrangian \(x_0 \mapsto x(t)\) with periodic wrap)
- Nonlinear Root Solver Module for \(G(x_0; x,t)=0\)
- Entropy Shock Selection via Rankine–Hugoniot (for \(t>t_{\min}\))

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
**1. Observation**
*   **Space-Time Plot (Left):** Shows a smooth color gradient initially. Characteristics converge towards $x=0$. Around $t \approx 0.53$ (the theoretical breaking time), a sharp discontinuity forms. This shock remains stationary at $x=0$ for the remainder of the simulation.
*   **Final Profile (Right):** At $t=1.5$, the density profile is a classic "sawtooth" or N-wave shape, discontinuous at $x=0$. The values are bounded within the physical range $[0.26, 0.74]$ (decayed from the initial $[0.2, 0.8]$ due to dissipation at the shock).
*   **Logs:** The log confirms the successful generation of the exact solution. No error norms are reported because this dataset **is** the reference ground truth.

**2. Diagnosis**
*   **Physics Verification:** The simulation correctly models the **LWR Traffic Flow** physics. The flux $f(\rho) = \rho(1-\rho)$ is concave, leading to shock formation when characteristics cross. The implementation correctly handles the multi-valued region by applying the equal-area rule (or equivalent Rankine-Hugoniot condition) to place the shock.
*   **Stationary Shock:** The specific initial condition $\rho_0(x) = 0.5 + 0.3\sin(\pi x)$ creates a symmetric convergence onto $x=0$, resulting in a shock with zero velocity. This is a stable, physical feature of this specific setup.

**3. Targeted Improvements**
*   **None Required.** The output represents high-fidelity ground truth data suitable for training a Physics-Informed Neural Network (PINN).

Recommendation:
Use the generated `burgers.npz` as the **Ground Truth**. When training the PINN, pay special attention to the discontinuity at $x=0$. Standard PINNs may struggle with the infinite gradient; consider using domain decomposition or adaptive sampling near $x=0$ to capture the shock accurately.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The code implements the exact Method of Characteristics (MoC) for the LWR traffic flow model via canonical transformation to inviscid Burgers' equation on [0, 2π] periodic domain. It uses nonlinear root solving (Secant method, Order1()) for the implicit characteristic map x(t) = x0 + (1-2ρ0(x0)) t, with entropy shock selection via Rankine-Hugoniot enforced geometrically (domain split at stationary shock d2=π, reflection/equal-area rule for post-shock). Physical mapping: x_can = π(x_phys + 1), t_can = π t_phys, v = 1-2ρ with IC v0 = 0.6 sin(x_can) (alpha=0, beta=0.6). N/A for Time Integrator, Polynomial Degree, Numerical Flux, Shock Capturing settings, AMR strategy as this is purely analytical (no DG solver). Matches physics (IC ρ=0.5+0.3sin(πx), periodic BCs, flux ρ(1-ρ), shock at t≈0.53), methodology (MoC exact), enhancements (root solver, periodic mapping, RH entropy). | 0.00e+00 | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saved as .npz file 'burgers.npz' in @__DIR__ containing keys [t, x, usol] (usol shape (200,256) time-major Float64 density ρ field, t/x length 200/256). Also saves 'summary_all.png' (spacetime heatmap + final slice plot). File verification printed (size, shapes, keys). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//LWR_Periodic_SinPi/2025_11_21_0000/` |


---
### Scoring Legend
* **Integrity (0-50):** Code stability, valid plots, no NaNs.
* **Accuracy (0-20):** Physics correctness, boundary conditions.
* **Details (0-15):** Oscillations, smearing, grid efficiency.
* **Optimality (0-15):** Correct numerical method choice (e.g., DG vs FD).
* **PASSING SCORE:** > 75
