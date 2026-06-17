
# Experiment Report
*Last Updated: 2026-01-10 13:11:14*

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

We must solve the 2D inviscid Burgers equation on a periodic square domain:
\[
\begin{aligned}
&\text{PDE:} && u_t + u\,u_x + u\,u_y = 0, \quad (x,y)\in[0,4\pi]^2,\ t\in[0,2],\\
&\text{BCs:} && u(x+4\pi,y,t) = u(x,y,t),\quad u(x,y+4\pi,t)=u(x,y,t),\\
&\text{IC:} && u(x,y,0) = \frac{1}{3} + \frac{2}{3}\sin\left( \frac{x+y}{2} \right).
\end{aligned}
\]
Goal: robust numerical solution up to time \(t=2\), with proper shock capture, and production of a summary plot `summary_all.png` with multiple time snapshots. Full solution data must be saved for later analysis.

---

**2. Strategic Methodology**

### Method Selection

**Main approach:** Finite Volume Method (FVM) on a uniform Cartesian grid with periodic boundaries, using a conservative formulation and a shock-capturing upwind flux.

1. **Recasting PDE for FVM**

Write the equation in conservative flux form:
\[
u_t + \partial_x f(u) + \partial_y g(u) = 0,
\]
with
\[
f(u) = \frac{1}{2}u^2,\quad g(u)=\frac{1}{2}u^2.
\]
This ensures conservation and aligns with standard finite volume discretization.

2. **Spatial Discretization Strategy**

- **Grid:**
  - Uniform Cartesian grid on \([0,4\pi]^2\).
  - Number of cells: choose \(N_x = N_y = N\) (e.g. \(N=256\) as a baseline; possibly 128 for initial tests).
  - Cell centers: \(x_i, y_j\); cell spacing: \(\Delta x = \Delta y = 4\pi / N\).

- **Finite Volume Formulation (cell averages):**
  - Evolve cell-averaged solution \(u_{i,j}(t)\) using:
    \[
    \frac{d u_{i,j}}{dt} = -\frac{1}{\Delta x}\left( F_{i+1/2,j} - F_{i-1/2,j} \right)
                            -\frac{1}{\Delta y}\left( G_{i,j+1/2} - G_{i,j-1/2} \right),
    \]
    where \(F_{i+1/2,j}\) and \(G_{i,j+1/2}\) are numerical fluxes in \(x\) and \(y\) directions, respectively.

- **Numerical Flux:**
  For inviscid Burgers with scalar state, we can use either:
  - **Rusanov (local Lax–Friedrichs) flux** – robust, simple, diffusive but stable:
    \[
    F_{i+1/2,j} = \frac{1}{2}\Big(f(u_L) + f(u_R)\Big) - \frac{1}{2} a_{i+1/2,j}(u_R - u_L),
    \]
    where \(a_{i+1/2,j} = \max(|u_L|, |u_R|)\) is the local characteristic speed; similarly for \(G_{i,j+1/2}\) using \(g\).
  - Or **Godunov flux for Burgers** (exact Riemann solver) if desired: slightly less diffusive but more complex; Rusanov is adequate as a first implementation and more robust.

  Here \(u_L, u_R\) are interface states obtained from reconstruction (see next).

- **Reconstruction:**
  - To capture shocks without excessive smearing but avoid oscillations, use:
    - **Second-order piecewise linear reconstruction** with slope limiter (e.g., minmod or MC limiter) in each direction.
    - This provides a **MUSCL-type scheme**: second-order in smooth regions, first-order near shocks for stability.
  - For each cell, compute slopes in \(x\) and \(y\) separately using limited centered differences.
  - Interface states:
    - In \(x\)-direction:
      - \(u_L = u_{i,j} + \frac{1}{2}(\partial_x u)_{i,j}\Delta x\) for interface \(i+1/2\) from left cell.
      - \(u_R = u_{i+1,j} - \frac{1}{2}(\partial_x u)_{i+1,j}\Delta x\) from right cell.
    - Analogously for \(y\)-direction using \((\partial_y u)_{i,j}\).

- **Periodic Boundary Conditions:**
  - Implement periodicity via halo/ghost layers:
    - For indices, use modular indexing or explicitly fill ghost cells:
      - e.g. cell index \(i = -1\) maps to \(i = N-1\), \(i = N\) maps to \(i = 0\), etc.
  - Ensure reconstruction and flux computation near boundaries use the periodic neighbors, so that all derivatives and slopes are consistent with periodicity.

3. **Time Integration**

- **Recommended method:** Strong-stability-preserving Runge–Kutta (SSP-RK2 or SSP-RK3).
  - **SSP-RK3** provides high robustness for hyperbolic conservation laws with limiters.
  - Each stage uses the semi-discrete spatial operator (flux divergences) with updated periodic BCs.

- **CFL Condition (explicit scheme):**
  - CFL estimate:
    \[
    \Delta t \le \text{CFL} \times \min\left(\frac{\Delta x}{\max |u|}, \frac{\Delta y}{\max |u|}\right),
    \]
    with CFL number around \(0.3\text{–}0.5\) for SSP-RK3 and second-order spatial scheme.
  - Since \(|u|\) starts in \([-\tfrac{1}{3}, 1]\) but will evolve, recompute or at least update a global max |u| each time step to maintain stability.
  - Use a global time step based on max wave speed across whole domain.

4. **Shock Formation and Resolution Strategy**

- For inviscid Burgers with sinusoidal-like initial data, shocks develop in finite time.
- Strategy:
  - Ensure limiter is active near steep gradients to avoid Gibbs-like oscillations.
  - Accept some numerical diffusion at shocks for robustness.
  - Use resolution (e.g., \(N=256\) or higher) to maintain reasonable accuracy for times up to \(t=2\).

5. **Verification / Sanity Checks (Conceptual)**

- Check:
  - Conservation: monitor spatial integral of \(u\); it should be approximately constant (modulo numerical errors).
  - Bounds: verify that \(u\) remains in a physically reasonable range (no extreme overshoots).
  - Symmetry: initial condition is symmetric in \(x\) and \(y\) in the combination \(x+y\); early-time solution should respect that symmetry qualitatively.

---

**3. Implementation Requirements**

**Advisor Adjustments:**  
No prior advisor-based changes were requested; this is the initial strategy. Still, the following should be viewed as requirements for the implementation team:

1. **Discretization Module**
   - Set up:
     - Uniform grid generation on \([0,4\pi]^2\).
     - Cell-centered storage of \(u_{i,j}\).
   - Periodic indexing utilities:
     - Functions or mechanisms to wrap indices in both directions.
   - Ghost-cell management (if used explicitly):
     - Routines to fill periodic halo cells from opposite sides of the domain at every time step or RK stage.

2. **Flux and Reconstruction Module**
   - Implement:
     - Piecewise linear reconstruction with limiter in x and y.
       - Slope limiter options (minmod as default; possibly configurable).
     - Numerical flux:
       - Rusanov flux using physical flux functions \(f(u)=u^2/2, g(u)=u^2/2\).
       - Computation of local wave speed \(a = \max(|u_L|,|u_R|)\) at each interface.
   - Spatial operator:
     - Routine that, given \(u\), returns the right-hand side \(du/dt\) via flux divergences with periodic BCs correctly applied.

3. **Time Integration Module**
   - Implement SSP-RK3 (or SSP-RK2 as backup) using the spatial operator.
   - Adaptive time step based on CFL:
     - Each step (or every few steps), compute global max \(|u|\) to set \(\Delta t\).
     - Ensure the final time \(t=2\) is reached; adjust last step to land exactly on \(t=2\) if desired.

4. **Initialization and Parameter Control**
   - Initial condition:
     - For each cell center \((x_i,y_j)\), set:
       \[
       u_{i,j}(0) = \frac{1}{3} + \frac{2}{3}\sin\left(\frac{x_i + y_j}{2}\right).
       \]
   - Input parameters:
     - Grid resolution \(N_x, N_y\).
     - CFL number.
     - Final time \(T=2\).
     - Time snapshot times for output and plotting.
     - Choice of limiter (if multiple supported).
   - Sanity outputs:
     - Initial diagnostic: max, min of \(u\) at \(t=0\).
     - Optional log of conservation measure (e.g., spatial mean of \(u\)).

5. **Data Management**
   - Save:
     - Full field \(u(x,y,t)\) at prescribed times:
       - Include at least: \(t=0\), an early pre-shock time, first shock time, intermediate times, and \(t=2\).
       - Example set: \(t = 0, 0.25, 0.5, 1.0, 1.5, 2.0\) (adjust after preliminary runs if shock develops earlier/later).
     - Storage format:
       - Use a structured format that supports 2D arrays with metadata (grid spacing, times), e.g. HDF5, NetCDF, or similar.
   - Keep:
     - Time vector.
     - Domain extents \([0, 4\pi]\times[0, 4\pi]\).
     - Any global diagnostics (max, min, integral of \(u\)).

---

**4. Visualization Strategy**

**Field Selection**

- Primary field: scalar solution \(u(x,y,t)\).
- Optionally for diagnostics (not necessarily in `summary_all.png`):
  - Difference \(u(x,y,t) - \text{mean}(u(t))\) to emphasize structures.
  - 1D slices (e.g., \(u(x,y_0,t)\)) along diagonals or axes if needed for debugging.

**Layout**

- File: `summary_all.png`.
- Content:
  - A grid of 2D contour or surface-like plots of \(u(x,y,t)\) at several times.
  - Recommended: 2×3 or 3×2 subplots, for six representative times, e.g.
    - \(t = 0\),
    - \(t = 0.25\),
    - \(t = 0.5\),
    - \(t = 1.0\),
    - \(t = 1.5\),
    - \(t = 2.0\).
  - Each subplot:
    - Shows \(u(x,y,t_k)\) over the full domain \([0,4\pi]^2\).
    - Use the same color map (or grayscale if preferred) across all subplots to allow comparison.
    - Use consistent v-limits (min, max) across all times (e.g., based on global min/max over all saved snapshots) to make shock development visually clear.
    - Add titles indicating the time, e.g., “t = 1.0”.

- **Important constraint:**  
  - Do **not** use colorbars if using Julia plotting backends, per instructions.
  - Instead, rely on consistent colormap and axis labeling; optionally note color scale in text/legend outside the axes if needed.

- Axes and annotations:
  - Label axes: “x” and “y” (possibly only on outer plots to reduce clutter).
  - Use equal aspect ratio if possible, so structures are not distorted.
  - Include a global title or figure caption-like text indicating:
    - “2D Inviscid Burgers solution, periodic \([0,4\pi]^2\), initial condition \(1/3 + (2/3)\sin((x+y)/2)\)”.

**Saving**

- Ensure:
  - `summary_all.png` is saved at sufficient resolution (e.g. 150–300 dpi) for readability.
  - All intermediate data (snapshots of \(u\) used to create the figure) are saved to disk in the chosen data format.
  - The plotting script reads from these saved data files rather than from in-memory variables only, to decouple simulation and visualization stages.

---

**REQUIRED COMPONENTS**

**Main Method:**
- **Finite Volume Method (FVM) on a uniform grid** with:
  - Conservative formulation,
  - Second-order MUSCL reconstruction with slope limiter,
  - Rusanov numerical flux (in both x and y),
  - Explicit SSP-RK3 time integration under a CFL constraint.

**Enhancements:**
1. **Slope limiter module** to handle steep gradients and shocks without oscillations.
2. **CFL-based adaptive timestep controller** to maintain stability as max |u| evolves.
3. **Periodic boundary manager** (index wrapping or ghost-cell filling) to robustly enforce BCs.
4. **Diagnostics module** for:
   - Tracking min/max, spatial mean, possibly L2 norm versus time.
5. **Data I/O module** for:
   - Saving 2D fields at selected times,
   - Storing parameters and metadata.
6. **Visualization module** for:
   - Reading stored fields,
   - Building the multi-panel summary figure,
   - Saving the final figure as `summary_all.png` without colorbars and with consistent scales.

This strategy provides a robust, shock-capturing, and computationally straightforward approach suitable for 2D inviscid Burgers on a periodic domain with the required final visualization.

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
**Section 1: Status Decision**
* **Status:** APPROVED

**Section 2: Quantitative Scorecard**
* **Score Integrity:** 50
* **Score Accuracy:** 20
* **Score Quality:** 15
* **Score Methodology:** 15
* **TOTAL SCORE:** 100

**Section 3: Feedback**
* **Observation:** The simulation successfully solves the 2D Inviscid Burgers' equation. The initial condition (diagonal bands) evolves correctly according to the characteristic velocity field $\vec{v}=(u,u)$. As expected for this problem, the smooth sine wave steepens over time due to the nonlinear advection term.
    *   **Physics Check:** The problem effectively reduces to a 1D Burgers equation along the diagonal $\xi = x+y$ with an effective advection speed of $2u$. Theoretical shock formation time is $T_b = 1 / \max(-v'(\xi)) = 1.5$. The plots show the gradient becoming vertical (shock formation) exactly around $t=1.5$, and a clear discontinuity is present at $t=2.0$.
    *   **Numerics:** The shock interface is sharp with minimal smearing, indicating the MUSCL reconstruction and Minmod limiter are working effectively. There are no visible spurious oscillations (Gibbs phenomenon).
* **Diagnosis:** The solution is physically accurate, stable, and high-quality. The methodology (FVM + MUSCL + SSP-RK3) is ideal for this hyperbolic conservation law.
* **Targeted Improvements:**
    * **General Advice:** No changes required. The current resolution ($256^2$) and scheme are sufficient for generating Ground Truth data for this specific problem setup.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Numerical Finite Volume Method (FVM) on uniform 256x256 Cartesian grid over [0,4π]^2 with periodic boundaries via mod1 indexing. Conservative formulation with f(u)=g(u)=u^2/2 fluxes. Second-order MUSCL reconstruction using minmod limiter on forward/backward differences for slopes. Rusanov numerical flux with local speed max(). SSP-RK3 time integration with adaptive CFL=0.4 timestep based on global max u , adjusted to hit target times exactly up to t=2. Initial condition exactly 1/3 + 2/3 sin((x+y)/2) at cell centers. Produces summary_all.png with 2x3 heatmaps of u at t=[0,0.25,0.5,1.0,1.5,2.0] using consistent global clims, equal aspect, no colorbars. | 0.00e+00 | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves burgers.npz containing keys: 't' (6,), 'x' (256,), 'y' (256,), 'u' (256,256,6). Also saves summary_all.png (2x3 subplot figure). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//Burgers_Periodic_SinSum/2026_01_10_1309/` |


---
