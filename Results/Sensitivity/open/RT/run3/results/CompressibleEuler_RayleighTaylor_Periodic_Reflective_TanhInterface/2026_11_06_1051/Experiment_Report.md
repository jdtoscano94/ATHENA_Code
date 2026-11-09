
# Experiment Report
*Last Updated: 2026-11-06 10:55:57*

## User Request
Let's solve the 2D Compressible Euler equations for the Rayleigh-Taylor Instability.

**Physics Parameters:**
* **Equations:** Compressible Euler ($\gamma = 1.4$).
* **Gravity:** $g = 1.0$ acting in the **positive $y$-direction** (The source term adds momentum in $+y$).
* **Inviscid:** $\mu = 0$, thermal conductivity $k = 0$.

**Domain:**
$(x,y) \in [0, 0.25] \times [0, 1.0]$
Time interval: $t \in [0, 3.0]$

**Boundary Conditions:**
* $x$-direction: **Periodic**.
* $y$-direction: **Reflective / Free-slip walls** at $y=0$ and $y=1$ (No-penetration condition $v=0$).

**Initial Conditions:**
The domain is split at the interface $y=0.5$.
* **Density ($\rho$):** A smoothed interface between $\rho_{heavy} = 2.0$ (bottom) and $\rho_{light} = 1.0$ (top).
    $$ \rho(y) = \rho_{heavy} + (\rho_{light} - \rho_{heavy}) \cdot \frac{1}{2}\left(1 + \tanh\left(50(y - 0.5)\right)\right) $$
* **Pressure ($p$):** Hydrostatic balance ($\frac{dp}{dy} = \rho g$).
    $$ p(y) = \begin{cases} 2y + 1 & \text{if } y < 0.5 \\ y + 1.5 & \text{if } y \geq 0.5 \end{cases} $$
* **Velocity:** Initialized with zero horizontal velocity ($u=0$) and a perturbation in vertical velocity ($v$):
    $$ v(x,y) = -0.025 \cdot c \cdot \cos(8\pi x) \cdot \sin(\pi y)^6 $$
    Where $c = \sqrt{\gamma p / \rho}$ is the local speed of sound.

**Output:**
Create an appropiate Figure and saved it as summary_all.png

---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
**1. Problem Statement**

2D inviscid compressible Euler simulation of the single-mode Rayleigh-Taylor Instability. Gravity acts in $+y$, placing the heavy fluid ($\rho=2.0$, bottom) in effective "overhang" of the light fluid ($\rho=1.0$, top) — an unstable configuration. A single-mode velocity perturbation seeds the interface; the goal is to capture the linear growth phase, the nonlinear bubble/spike regime, and the late-time mixing layer up to $t=3.0$.

---

**1. Mathematical Specification (Code-Ready Extraction)**

* **Governing Equations:** 2D Compressible Euler with gravitational source ($+y$):
$$
\frac{\partial}{\partial t}\begin{pmatrix} \rho \\ \rho u \\ \rho v \\ E \end{pmatrix}
+ \frac{\partial}{\partial x}\begin{pmatrix} \rho u \\ \rho u^2 + p \\ \rho u v \\ u(E+p) \end{pmatrix}
+ \frac{\partial}{\partial y}\begin{pmatrix} \rho v \\ \rho u v \\ \rho v^2 + p \\ v(E+p) \end{pmatrix}
= \begin{pmatrix} 0 \\ 0 \\ \rho g \\ \rho g\, v \end{pmatrix}
$$
with equation of state $p = (\gamma - 1)\left(E - \tfrac{1}{2}\rho(u^2 + v^2)\right)$.

* **Domain:** $(x,y) \in [0, 0.25] \times [0, 1.0]$, $t \in [0, 3.0]$.

* **Boundary Conditions:**
  * $x$-direction: Periodic.
  * $y=0$ and $y=1$: Reflective / free-slip walls (no-penetration, $v=0$).

* **Initial Conditions:**
$$
\rho(x,y) = 2.0 + (1.0 - 2.0)\cdot \frac{1}{2}\left(1 + \tanh\left(50\,(y - 0.5)\right)\right)
$$
$$
p(x,y) = \begin{cases} 2y + 1, & y < 0.5 \\ y + 1.5, & y \geq 0.5 \end{cases}
$$
$$
u(x,y) = 0
$$
$$
v(x,y) = -0.025\, c(x,y)\, \cos(8\pi x)\, \left[\sin(\pi y)\right]^6, \qquad c(x,y) = \sqrt{\frac{\gamma\, p(x,y)}{\rho(x,y)}}
$$

* **Constants & Parameters:** $\gamma = 1.4$; $g = 1.0$ (source adds momentum in $+y$); $\rho_{heavy}=2.0$, $\rho_{light}=1.0$; interface at $y=0.5$; interface smoothing parameter $50$; perturbation amplitude $0.025$; perturbation wavenumber $8\pi$ (single mode over $L_x = 0.25$); $\mu = 0$, $k = 0$.

* **Expected Physics:**
  * $t \lesssim 1$: Linear phase — exponential growth of the seeded mode; light-fluid bubbles descend ($-y$), heavy-fluid spikes ascend ($+y$).
  * $t \approx 1\text{–}2$: Nonlinear bubble/spike fronts form; Kelvin-Helmholtz roll-up on spike flanks.
  * $t \approx 2\text{–}3$: Broadening mixing layer, secondary instabilities, vortical structures; possible weak compressible shocks from spike deceleration.
  * Minor initial acoustic transient from the (standard) slight mismatch between the smoothed $\rho$ and the piecewise hydrostatic $p$; decays quickly and is dominated by RT growth.

---

**2. Strategic Methodology**

* **Exact solution feasibility: NOT FEASIBLE.** The system is a nonlinear, multi-dimensional hyperbolic PDE with a source term. The Method of Characteristics does not extend to 2D nonlinear systems (characteristics form a multi-parameter family with no closed-form envelope), and no analytical solution exists for the nonlinear RT mixing process. Ground truth must be numerical with high resolution.
* **Selection: Discontinuous Galerkin Spectral Element Method in 2D plus time.** Rationale: (i) hyperbolic system with discontinuity-sharpening interfaces — DG's locality and shock-capturing support are essential; (ii) mixed BCs (periodic in $x$, walls in $y$) are natively supported; (iii) AMR can track the evolving mixing layer efficiently; (iv) entropy-stable DG handles the strong density contrast without spurious oscillations. Fourier Spectral is rejected (non-periodic in $y$; steep gradients/mixing). Steady CG is rejected (transient problem).

---

**3. Implementation Requirements (CRITICAL)**

* **Configuration:** Use template defaults.

* **3.0.1 Adaptive Mesh Refinement & Shock Capturing — Variable Selection Analysis:**
  * **Selected variable: Density ($\rho$).**
  * *Justification:* The RTI is driven entirely by the density contrast; the material interface remains a sharp, spatially localized feature in $\rho$ for all $t$, so a density-based indicator concentrates resolution exactly on the mixing layer. **Velocity** is rejected: it becomes globally turbulent/chaotic at late times, causing AMR to refine everywhere and waste DOF. **Pressure** is rejected: it is dominated by the smooth hydrostatic background ($p \sim 1$–$2.5$) and is continuous across the interface — its gradients reflect stratification, not the instability. Density is the superior sensor for both AMR and the shock-capturing indicator for this PDE.

* **3.1 Advisor-Mandated Adjustments (Overrides):** None.

---

**4. Visualization Strategy**

* **Field Selection (Analysis):**
  * **Primary: Density ($\rho$)** — captures the interface distortion, bubbles, spikes, and mixing-layer growth (the defining RTI diagnostics).
  * **Secondary: Vertical velocity ($v$)** — a primary variable that reveals bubble/spike motion and the vortical structure of the mixing layer.
* **Task:** Plot $\rho$ and $v$ at **4 time steps: $t = 0.75,\ 1.5,\ 2.25,\ 3.0$**.
* **Layout:** Grid — 2 rows (fields: $\rho$, $v$) × 4 columns (times).
* **Saving Data:** Save the **full solution state** (all variables) at frequent intervals, `dt_save = 0.1`, for Ground Truth. Plotting selection affects only `summary_all.png`.
* **Constraints:**
  * Save as: `summary_all.png` (strictly).
  * **CRITICAL: Disable colorbars (`colorbar=false`) — code is in Julia.**
  * Do not use fixed colormaps; let scales adjust automatically per panel.
  * Plot only primary variables ($\rho$, $v$). Do not compute derived quantities (vorticity, Schlieren, etc.).

---

**5. Cross-Validation & Data Integration**

* No external data paths (sensors / reference solutions) are provided in [USER_QUESTION]. → No cross-validation required.

---

**REQUIRED COMPONENTS**

**Main Method:**
* Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
* Adaptive Mesh Refinement (Density-based indicator)
* Shock Capturing (Density-based indicator)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

**Status:** `REJECTED_FOR_REFINEMENT`

The simulation completed successfully and shows the expected Rayleigh–Taylor evolution, but the current output is not yet sufficiently sharp or well-verified for PINN ground truth. No previous trials exist, so the history-saturation stopping rule does not apply.

## Section 2: Quantitative Scorecard

- **Score Integrity:** 45/50
- **Score Accuracy:** 17/20
- **Score Details:** 10/15
- **Score Optimality:** 12/15
- **TOTAL SCORE:** **84/100**

**Error check:** `Minimum achieved error:` does not appear in the supplied stdout tail. **Quantitative error is not applicable as no ground truth is available.** The full nonlinear, two-dimensional Euler/RTI problem with gravity and wall boundaries has no feasible exact analytical solution; numerical convergence verification is therefore required.

## Section 3: Forensic Analysis & Feedback

### Observation

- The log reaches `Saving summary_all.png... Done.` with no reported crash or NaN. All eight panels contain nonblank solution fields.
- The density evolution is qualitatively credible:
  - At \(t=0.75\), the seeded single-mode interface deformation is visible.
  - At \(t=1.5\), a central heavy-fluid spike rises into the lighter fluid.
  - At \(t=2.25\) and \(t=3.0\), nonlinear spike/bubble structures and mushroom-like roll-up develop, followed by interaction with the upper part of the domain.
- Approximate left–right symmetry is retained, and there is no obvious catastrophic Gibbs instability.
- However, the late-time density interfaces and roll-up edges appear softened, while fine Kelvin–Helmholtz structure is not clearly resolved.
- **Regular horizontal white stripes obscure every panel.** Their uniformity across both density and velocity strongly suggests a plotting/mesh-line or rasterization artifact rather than physical RT structure. They prevent reliable inspection of interface thickness and the no-penetration condition at \(y=0,1\).

### Diagnosis

The principal numerical concern is **contact-interface dissipation and insufficient effective resolution**, not solver failure:

- `flux_hll` does not explicitly restore the contact wave and is more dissipative at material interfaces than HLLC.
- The Lax–Friedrichs finite-volume stabilization adds further dissipation wherever shock-capturing blending is active.
- `polydeg=3` with AMR capped at level 6 is a reasonable robust starting point, but may not resolve the late-time roll-up sharply enough for ground truth.
- The white bands are a separate **visualization defect** and must be removed before accepting the figure.

### Targeted Improvements

- **For Integrity:**
  - Disable mesh/grid-line overlays or correct the heatmap row-gap/rasterization issue causing the white stripes.
  - Add logged sanity checks for finite values and \(\min(\rho)>0\), \(\min(p)>0\); also report extrema of \(v\) at the two wall boundaries.

- **For Accuracy:**
  - Increase **`polydeg` from 3 to 5**, retaining the adaptive `StepsizeCallback` and CFL-based time-step control. This is preferable to immediately increasing the maximum AMR level and remains within the recommended polynomial-degree limit.
  - Verify convergence by comparing against the present run using interface location, mixing-layer width, total mass, and the energy/source-work balance. Preserve the user-specified initial conditions.

- **For Details:**
  - Replace **`flux_hll` with `flux_hllc`** for the surface flux to reduce contact-discontinuity smearing.
  - Retain `flux_ranocha` in the volume and the finite-volume fallback for robustness.
  - Reduce the **AMR interval from 20 to 10 steps** so refinement tracks the moving interface more reliably. Keep the maximum level at 6 initially; only consider further refinement if the \(N=5\) result still shows excessive interface thickness.
  - Do not broadly weaken shock capturing unless diagnostics show that blending is active over large smooth regions.

- **For Optimality:**
  - DGSEM with density-based AMR remains the appropriate architecture. No change to an exact method is warranted because the full nonlinear RTI evolution has no usable closed-form solution.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The code implements a 2D inviscid compressible Euler simulation of single-mode Rayleigh-Taylor instability using a numerical discontinuous Galerkin spectral element method (DGSEM) with Trixi.jl. The governing equations use gamma=1.4, gravitational source terms in +y with g=1.0, domain [0,0.25] x [0,1], t in [0,3.0], periodic x-boundaries, and free-slip/reflective y-walls. Initial conditions match the strategy: smoothed density interface with tanh(50(y-0.5)), piecewise hydrostatic pressure, u=0, and v=-0.025*c*cos(8*pi*x)*sin(pi*y)^6. The solver uses polydeg=3, P4estMesh with initial_refinement_level=3, density-based AMR via IndicatorLöhner with ControllerThreeLevel levels 4-6 and interval 20, density-based shock capturing via IndicatorHennemannGassner, HLL surface flux, entropy-conserving volume flux flux_ranocha, FV stabilization flux flux_lax_friedrichs, RDPK3SpFSAL49 time integration, and StepsizeCallback with CFL=0.5. The visualization code plots density and vertical velocity at t=0.75, 1.5, 2.25, and 3.0 in a 2x4 layout and saves summary_all.png with colorbar=false. | N/A | 45 | 17 | 10 | 12 | **84** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves NPZ snapshots in data_npz/ named time_%08.5f.npz (e.g., time_00.00000.npz through time_03.00000.npz) at t=0.0,0.1,0.2,...,3.0 plus the visualization times. Each NPZ contains keys: 't' (scalar Float64), 'x' and 'y' (1D Vector{Float64} of scattered PlotData2D node coordinates, length N_points), primitive variables 'rho','u','v','p' (1D Vector{Float64}, length N_points), conservative variables 'conservative_rho','rho_u','rho_v','rho_e' (1D Vector{Float64}, length N_points), and 'state_vector' (1D Vector{Float64} raw DG state, length N_DOFs). N_points and N_DOFs vary with AMR. Also saves Trixi HDF5 solution files in trixi_output/ via SaveSolutionCallback at dt=0.1, including initial and final states, with solution_variables=cons2cons (full conservative DG state; exact HDF5 structure is Trixi's native format and mesh/DOF-dependent). Saves summary_all.png as a PNG 2x4 panel plot: row 1 density (rho), row 2 vertical velocity (v), columns t=0.75,1.5,2.25,3.0, with colorbars disabled. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run3/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective_TanhInterface/2026_11_06_1051/` |


---
