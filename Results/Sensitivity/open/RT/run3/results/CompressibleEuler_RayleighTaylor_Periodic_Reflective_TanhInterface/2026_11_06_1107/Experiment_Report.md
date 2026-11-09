
# Experiment Report
*Last Updated: 2026-11-06 11:13:10*

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
# Scientific Strategy Report: 2D Compressible Euler — Rayleigh-Taylor Instability

---

## 1. Problem Statement

Single-mode Rayleigh-Taylor instability of the inviscid compressible Euler equations: a heavy fluid ($\rho=2.0$) rests below a light fluid ($\rho=1.0$) with effective acceleration ($g=+1.0$ in $y$) destabilizing the interface at $y=0.5$. A single-wavelength velocity perturbation seeds the instability. Target: high-fidelity ground truth on $[0,0.25]\times[0,1]$ for $t\in[0,3]$, resolving the linear growth $\to$ nonlinear spike/bubble $\to$ Kelvin-Helmholtz roll-up cascade.

---

## 2. Mathematical Specification (Code-Ready Extraction)

**Governing Equations** (2D compressible Euler with gravitational source):
$$
\partial_t \rho + \partial_x(\rho u) + \partial_y(\rho v) = 0
$$
$$
\partial_t(\rho u) + \partial_x(\rho u^2 + p) + \partial_y(\rho u v) = 0
$$
$$
\partial_t(\rho v) + \partial_x(\rho u v) + \partial_y(\rho v^2 + p) = \rho g
$$
$$
\partial_t E + \partial_x\big(u(E+p)\big) + \partial_y\big(v(E+p)\big) = \rho g v
$$
with equation of state $E = \dfrac{p}{\gamma-1} + \dfrac{1}{2}\rho(u^2+v^2)$.

**Domain:** $(x,y) \in [0, 0.25] \times [0, 1.0]$, $\quad t \in [0, 3.0]$

**Boundary Conditions:**
- $x \in \{0, 0.25\}$: Periodic
- $y = 0$ and $y = 1$: Reflective / free-slip wall (no-penetration: $v = 0$; mirror $u$, tangential momentum; reflect $\rho, p$)

**Initial Conditions** (exact, at $t=0$):
$$
\rho(x,y) = 2.0 + (1.0 - 2.0)\cdot\frac{1}{2}\left(1 + \tanh\left(50(y - 0.5)\right)\right)
$$
$$
p(x,y) = \begin{cases} 2y + 1 & \text{if } y < 0.5 \\ y + 1.5 & \text{if } y \geq 0.5 \end{cases}
$$
$$
u(x,y) = 0
$$
$$
v(x,y) = -0.025 \cdot c(y) \cdot \cos(8\pi x) \cdot \sin^6(\pi y), \qquad c(y) = \sqrt{\frac{\gamma\, p(y)}{\rho(y)}}
$$
$$
E(x,y) = \frac{p(y)}{\gamma - 1} + \frac{1}{2}\,\rho(x,y)\, v(x,y)^2
$$

**Constants & Parameters:**
| Parameter | Value |
|---|---|
| $\gamma$ | $1.4$ |
| Gravity vector $\mathbf{g}$ | $[0, +1.0]^T$ (momentum source $\rho g$ in $+y$; energy source $\rho g v$) |
| $\rho_{heavy},\ \rho_{light}$ | $2.0,\ 1.0$ |
| Interface position / smoothing | $y = 0.5$, $\tanh$ width parameter $50$ |
| Perturbation amplitude / wavenumber | $0.025$, $8\pi$ (one wavelength across $x$-domain) |
| Atwood number | $A = 1/3$ |

**Expected Physics:** Exponential linear growth of the seeded mode ($t \lesssim 1$); heavy-fluid spike ascends and light-fluid bubble descends ($t \approx 1$–$2$); nonlinear mushroom cap with Kelvin-Helmholtz roll-up vortices on the spike flanks ($t \approx 2$–$3$); spike interaction with the upper wall near $t = 3$. Contact discontinuity (material interface) is the dominant sharp feature; shear layers generate secondary KH instabilities.

---

## 3. Strategic Methodology

**Exact solution feasibility: NOT FEASIBLE.** The full nonlinear 2D Euler system with gravity, wall boundaries, and a finite-amplitude evolving interface has no closed-form solution. The Method of Characteristics is restricted to 1D/simple-wave hyperbolic problems — here the dynamics are genuinely multi-dimensional (vorticity deposition at the shear layer, KH roll-up, shocklet formation), precluding any characteristics-based exact solution. Linear RTI stability theory yields only early-time growth rates, not the full evolution. **A numerical method is mandatory.**

**Method Selection:**
- *Fourier Spectral:* Rejected — domain is non-periodic in $y$ (reflective walls).
- *DGSEM 1D:* Rejected — genuinely 2D problem.
- *Steady CG 2D:* Rejected — strongly time-dependent transient.
- **DGSEM 2D + time: SELECTED.** High-order accuracy in smooth regions, native discontinuity handling (shock capturing + AMR), exact support for periodic/wall BC combinations, and the proven architecture for this exact benchmark. The prior run (score 84) confirms the architecture is sound; the Advisor's diagnosis is contact-interface dissipation and a visualization defect — both are correctable within DGSEM.

---

## 4. Implementation Requirements (CRITICAL)

**Configuration:** Use template defaults, **EXCEPT** the following Advisor-mandated overrides (these supersede all defaults):

| Setting | Value | Rationale (Advisor) |
|---|---|---|
| `polydeg` | **5** (was 3) | Sharper late-time roll-up for ground truth |
| Surface flux | **`flux_hllc`** (was `flux_hll`) | Restores contact wave; reduces material-interface smearing |
| Volume flux | `flux_ranocha` (retain) | Entropy-conserving smooth-region accuracy |
| FV stabilization flux | `flux_lax_friedrichs` (retain) | Robustness fallback |
| AMR interval | **10 steps** (was 20) | Refinement tracks the moving interface |
| AMR levels | Three-level controller, levels 4–6, max 6 (retain) | Sufficient with $N=5$ |
| Shock capturing | Density-based indicator (retain; do **not** weaken) | KH roll-up demands active stabilization |
| Time integration | RDPK3SpFSAL49 + StepsizeCallback, CFL = 0.5 (retain) | FSAL pairing with shock-capturing blending |

**4.0.1 Adaptive Mesh Refinement & Shock Capturing — Variable Selection Analysis:**
- **Selected sensor variable: Density ($\rho$).**
- *Justification:* The RTI dynamics are driven entirely by the density stratification; the material interface is a contact discontinuity across which $\rho$ jumps by $2\times$ while remaining globally bounded and smooth elsewhere — an ideal, localized feature sensor. **Velocity is rejected**: at late times the velocity field becomes globally chaotic (KH cascade fills the mixing layer), which would scatter refinement elements across the whole domain. **Pressure is rejected**: it remains near-hydrostatic and smooth — a poor feature indicator. Density concentrates refinement exactly on the interface and roll-up edges, the features of interest.

**4.1 Advisor-Mandated Runtime Diagnostics (log at every save interval):**
- All-finite check on the full state; assert $\min(\rho) > 0$ and $\min(p) > 0$.
- Report extrema of $v$ at $y=0$ and $y=1$ (no-penetration verification: $|v|_{wall} \approx 0$).
- Report total mass $\int \rho\, dA$ (must be conserved).
- Report energy balance: $\frac{d}{dt}\int E\, dA$ vs. source work $\int \rho g v\, dA$.
- Convergence metrics vs. the prior `polydeg=3` run: interface apex location ($\rho = 1.5$ contour), mixing-layer width, total mass, energy/source-work balance.

---

## 5. Visualization Strategy

**Field Selection (Analysis):** Plot exactly **2 fields**:
1. **Density $\rho$** — the primary driver; captures interface deformation, spike/bubble, and roll-up.
2. **Vertical velocity $v$** — captures perturbation growth and spike/bubble motion.

**Task:** Plot both fields at $t = 0.75,\ 1.5,\ 2.25,\ 3.0$ (4 time steps).
**Layout:** 2 rows (fields) × 4 columns (times).

**Saving Data:** Save the **full solution state** (all conserved variables) at `dt_save = 0.1` for Ground Truth. Plotting selection affects only the summary image.

**Constraints:**
- Save as: `summary_all.png` (strictly).
- **`colorbar = false`** (Julia — mandatory).
- **CRITICAL (Advisor-mandated artifact fix):** The previous figure showed regular horizontal white stripes across all panels — a rendering defect, not physics. Render each field as a **single continuous filled heatmap** (`seriestype = :heatmap`): disable all grid lines, mesh/wireframe overlays, and cell-gap rasterization; ensure the data matrix orientation matches the $(x,y)$ axes and the rasterization resolution matches the sample grid. No blank rows/columns.
- Do not use fixed colormaps or color limits — let scales adjust automatically per panel.
- Plot only primary variables ($\rho$, $v$). Do not compute derived quantities (vorticity, Schlieren, etc.).

---

## 6. Cross-Validation & Data Integration (Conditional)

**Trigger check:** No external data paths (sensors / reference `.npz`) are present in the problem statement → **no external cross-validation applicable**.

**Internal verification (Advisor-mandated):** The diagnostics of §4.1 (positivity, wall no-penetration, mass conservation, energy/source-work balance) plus the convergence comparison against the prior `polydeg=3` baseline (interface apex, mixing width) constitute the verification protocol. Print all metrics to stdout.

---

## REQUIRED COMPONENTS

**Main Method:**
- Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
- Adaptive Mesh Refinement (Density-based indicator, three-level controller)
- Shock Capturing Indicator (density-based)
- HLLC Surface Flux (Advisor-mandated contact-resolution upgrade)
- Runtime Positivity & Conservation Diagnostics (Advisor-mandated)
- Convergence Comparison vs. `polydeg=3` Baseline (Advisor-mandated)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

* **Status:** `REJECTED_FOR_REFINEMENT`
* **Decision rationale:** The run is a substantial improvement and is suitable as a high-fidelity candidate, but it does not achieve a perfect score. The history contains only Run 0, scored **84**, so there is no previous score \(\geq 90\) and the five-run saturation condition cannot be applied. This run should therefore serve as the baseline for a focused convergence/verification refinement.

## Section 2: Quantitative Scorecard

* **Score Integrity:** **50/50**
* **Score Accuracy:** **19/20**
* **Score Details:** **13/15**
* **Score Optimality:** **15/15**
* **TOTAL SCORE:** **97/100**

**Error-log check:** No `Minimum achieved error:` line appears in the supplied stdout tail. The log shows completed solver timing output followed by successful generation and saving of `summary_all.png`. **Quantitative error is not applicable as no ground truth is available.**

## Section 3: Forensic Analysis & Feedback

### Observation

* The simulation runs to completion, saves the requested figure, and produces finite, readable fields. The previous horizontal white-stripe rendering artifact is absent.
* The density evolution is physically coherent:
  * At \(t=0.75\), the initially smoothed interface has developed a controlled single-mode deformation.
  * At \(t=1.5\), the heavy-fluid spike rises in \(+y\), consistent with the prescribed gravity and stratification.
  * At \(t=2.25\), a mushroom cap and rolled-up flank structures are visible, indicating the expected shear-driven Kelvin–Helmholtz dynamics.
  * At \(t=3.0\), the spike approaches/interacts with the upper wall and the mixing layer contains elongated roll-up structures.
* The vertical-velocity field evolves consistently with the density motion: initially localized near the interface, then extending along the spike and mixing region. There is no obvious large-scale symmetry violation, Gibbs ringing, or catastrophic boundary artifact.
* The HLLC upgrade and higher polynomial degree have visibly improved feature resolution. Nevertheless, the thinnest late-time density filaments remain somewhat soft at the supplied image resolution. The figure alone cannot certify that contact interfaces are resolved within one or two cells.
* The supplied log tail does not show the numerical values of the mandated positivity, wall-normal velocity, mass-conservation, or energy/source-work diagnostics. Successful completion alone is insufficient to verify those quantities.

### Diagnosis

The remaining limitation is **mild residual numerical dissipation/resolution uncertainty**, rather than an evident stability failure. Possible contributors are the finite-volume Lax–Friedrichs stabilization in blended elements, the AMR resolution cap, and rasterization of very thin filaments. There is no visual justification for weakening shock capturing globally.

An exact-solution override is **not warranted**: nonlinear, multidimensional Euler evolution with gravity, wall interaction, and vortical roll-up does not admit a feasible closed-form characteristics or Fourier solution. DGSEM with HLLC, entropy-conserving volume flux, AMR, and localized shock capturing is methodologically appropriate.

### Targeted Improvements

* **For Integrity:** None.
* **For Accuracy:** Use the present `polydeg=5` result as the new verification baseline. At all output times, explicitly report:
  * \(\min(\rho)\) and \(\min(p)\);
  * maximum \(|v|\) on both walls;
  * relative total-mass drift;
  * the integrated energy-balance residual against \(\int \rho g v\,dA\);
  * interface-apex position and mixing-layer width.
* **For Details:** Perform one focused p-convergence trial with **`polydeg=6`**, retaining HLLC, `flux_ranocha`, and the present AMR controller. Use **CFL \(\approx 0.4\)** for additional stability margin. Compare late-time filament thickness, apex position, and mixing width against the current run; do not revert to HLL or repeat the `polydeg=3` configuration. If the p=6 result is unchanged, examine whether finite-volume blending or the AMR level cap—not polynomial resolution—is controlling the residual smearing.
* **For Optimality:** None. The present numerical architecture is appropriate; only convergence verification and localized resolution assessment remain.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | The code implements a 2D inviscid compressible Euler simulation of single-mode Rayleigh-Taylor instability using a numerical discontinuous Galerkin spectral element method (DGSEM) with Trixi.jl. The governing equations use gamma=1.4, gravitational source terms in +y with g=1.0, domain [0,0.25] x [0,1], t in [0,3.0], periodic x-boundaries, and free-slip/reflective y-walls. Initial conditions match the strategy: smoothed density interface with tanh(50(y-0.5)), piecewise hydrostatic pressure, u=0, and v=-0.025*c*cos(8*pi*x)*sin(pi*y)^6. The solver uses polydeg=3, P4estMesh with initial_refinement_level=3, density-based AMR via IndicatorLöhner with ControllerThreeLevel levels 4-6 and interval 20, density-based shock capturing via IndicatorHennemannGassner, HLL surface flux, entropy-conserving volume flux flux_ranocha, FV stabilization flux flux_lax_friedrichs, RDPK3SpFSAL49 time integration, and StepsizeCallback with CFL=0.5. The visualization code plots density and vertical velocity at t=0.75, 1.5, 2.25, and 3.0 in a 2x4 layout and saves summary_all.png with colorbar=false. | N/A | 45 | 17 | 10 | 12 | **84** |
| 1 | The code implements a 2D compressible Euler Rayleigh-Taylor simulation using Trixi's DGSEM on a P4estMesh over [0,0.25]x[0,1] with t in [0,3]. It uses CompressibleEulerEquations2D(gamma=1.4), polydeg=5, LobattoLegendreBasis, surface flux flux_hllc, entropy-conserving volume flux flux_ranocha, FV stabilization flux_lax_friedrichs, density-based IndicatorHennemannGassner shock capturing, density-based IndicatorLöhner AMR with ControllerThreeLevel levels 4/5/6 and interval=10, and RDPK3SpFSAL49 with StepsizeCallback(cfl=0.5). Initial conditions match the strategy: smoothed heavy/light density profile, piecewise hydrostatic pressure, zero u, v perturbation -0.025*c(y)*cos(8*pi*x)*sin(pi*y)^6, and E via prim2cons. Gravity source [0,0,+rho,+rho*v] is implemented. Boundaries are periodic in x and slip walls at y=0,1. Runtime diagnostics check finiteness, positivity, wall v extrema, total mass, energy tendency vs source work, interface apex/mixing width, and polydeg=3 baseline comparison when baseline data exist. Visualization plots rho and v at t=0.75,1.5,2.25,3.0 in a 2x4 heatmap layout with colorbar=false and gap-free rasterization. | N/A | 50 | 19 | 13 | 15 | **97** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves NPZ snapshots in data_npz/ named time_%08.5f.npz (e.g., time_00.00000.npz through time_03.00000.npz) at t=0.0,0.1,0.2,...,3.0 plus the visualization times. Each NPZ contains keys: 't' (scalar Float64), 'x' and 'y' (1D Vector{Float64} of scattered PlotData2D node coordinates, length N_points), primitive variables 'rho','u','v','p' (1D Vector{Float64}, length N_points), conservative variables 'conservative_rho','rho_u','rho_v','rho_e' (1D Vector{Float64}, length N_points), and 'state_vector' (1D Vector{Float64} raw DG state, length N_DOFs). N_points and N_DOFs vary with AMR. Also saves Trixi HDF5 solution files in trixi_output/ via SaveSolutionCallback at dt=0.1, including initial and final states, with solution_variables=cons2cons (full conservative DG state; exact HDF5 structure is Trixi's native format and mesh/DOF-dependent). Saves summary_all.png as a PNG 2x4 panel plot: row 1 density (rho), row 2 vertical velocity (v), columns t=0.75,1.5,2.25,3.0, with colorbars disabled. |
| 1 | Saves NPZ snapshots under data_npz with filename pattern time_%08.5f.npz (e.g., time_00.00000.npz) using NPZ.jl, at t=0.0,0.1,...,3.0 plus the visualization times 0.75,1.5,2.25,3.0. Each .npz contains: t scalar Float64; x, y, rho, u, v, p, conservative_rho, rho_u, rho_v, rho_e as 2D arrays on the current PlotData2D visualization grid (dimensions vary with AMR and are oriented to match plotting); state_vector as a 1D vector of raw DG conservative degrees of freedom; and scalar diagnostics min_rho, min_p, total_mass, mass_drift, total_energy, energy_tendency, source_work, energy_balance_residual, interface_apex, mixing_layer_width, wall_v_y0_min, wall_v_y0_max, wall_v_y1_min, wall_v_y1_max. It also invokes Trixi SaveSolutionCallback into trixi_output with dt=0.1, saving full conservative variables (cons2cons) at initial, final, and intermediate times. Finally it saves summary_all.png as a 2x4 continuous heatmap grid of rho and v at the four requested times. It optionally reads data_npz_polydeg3 for baseline convergence comparison; no baseline files are written by this code. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run3/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective_TanhInterface/2026_11_06_1051/` |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run3/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective_TanhInterface/2026_11_06_1107/` |


---
