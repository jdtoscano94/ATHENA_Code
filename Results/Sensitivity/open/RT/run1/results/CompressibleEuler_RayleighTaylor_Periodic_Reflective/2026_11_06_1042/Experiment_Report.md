
# Experiment Report
*Last Updated: 2026-11-06 10:48:15*

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
# ATHENA Lab — Scientific Strategy Report
**Problem:** 2D Compressible Euler — Rayleigh–Taylor Instability (Refinement Run 3)
**Verdict:** Retain DGSEM architecture; execute Advisor-mandated upgrades (HLLC, polydeg=4, selective shock capturing, artifact-free rendering, full physical verification suite).

---

**1. Problem Statement**

Forward simulation of the single-mode compressible Rayleigh–Taylor instability: inviscid Euler equations (γ = 1.4) with a gravitational source (g = 1.0 in +y) on $[0,0.25]\times[0,1]$, $t\in[0,3]$. Heavy fluid (ρ = 2) lies beneath light fluid (ρ = 1) across a smoothed interface at y = 0.5; since gravity points toward the light fluid, the configuration is unstable. A single-wavelength perturbation seeds a central heavy spike rising in +y. Objective: a ground-truth-quality dataset of the full nonlinear evolution (linear growth → spike/bubble → KH roll-up → late-time mixing near the top wall).

This is a **refinement run** of the previous attempt (81/100). The Advisor's diagnosis mandates: (i) eliminate the horizontal white-stripe rendering artifact; (ii) replace the contact-diffusing HLL flux with HLLC; (iii) raise resolution to polydeg = 4; (iv) make Lax–Friedrichs blending selective; (v) supply quantitative physical verification (mass, positivity, symmetry, wall compliance, energy balance).

---

**2. Mathematical Specification (Code-Ready Extraction)**

* **Governing Equations:** 2D compressible Euler with gravity source:
$$\partial_t \begin{pmatrix} \rho \\ \rho u \\ \rho v \\ E \end{pmatrix} + \partial_x \begin{pmatrix} \rho u \\ \rho u^2 + p \\ \rho u v \\ u(E+p) \end{pmatrix} + \partial_y \begin{pmatrix} \rho v \\ \rho u v \\ \rho v^2 + p \\ v(E+p) \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ \rho g \\ \rho g v \end{pmatrix}$$
with total energy $E = \dfrac{p}{\gamma-1} + \dfrac{1}{2}\rho(u^2+v^2)$. The energy source is the gravity power $S_E = \rho g v$ (force × velocity — **not** $\rho g v^2$).

* **Domain:** $(x,y) \in [0, 0.25] \times [0, 1.0]$; $t \in [0, 3.0]$.

* **Boundary Conditions:** x: **Periodic**. y = 0 and y = 1: **Reflective / free-slip walls** (no-penetration $v = 0$, zero normal flux).

* **Initial Conditions** (interface at $y_i = 0.5$):
$$\rho(x,y,0) = 2.0 + (1.0 - 2.0)\cdot\frac{1}{2}\Big(1 + \tanh\big(50\,(y - 0.5)\big)\Big)$$
$$p(y,0) = \begin{cases} 2y + 1, & y < 0.5 \\ y + 1.5, & y \geq 0.5 \end{cases} \qquad (\text{hydrostatic } dp/dy = \rho g \text{ away from the interface})$$
$$u(x,y,0) = 0$$
$$v(x,y,0) = -0.025\; c(x,y)\, \cos(8\pi x)\, \big(\sin(\pi y)\big)^{6}, \qquad c(x,y) = \sqrt{\frac{\gamma\, p(y)}{\rho(y)}}$$
$$E(x,y,0) = \frac{p}{\gamma - 1} + \frac{1}{2}\rho\left(u^2 + v^2\right) \quad (\text{via prim2cons})$$

* **Constants & Parameters:** $\gamma = 1.4$; $g = 1.0$; gravity vector $\mathbf{g} = [0, 1.0]^T$; $\rho_{heavy} = 2.0$; $\rho_{light} = 1.0$; interface $y = 0.5$; tanh steepness $50$; perturbation amplitude $0.025$; wavenumber $8\pi$ (one full wavelength across the domain: $8\pi \cdot 0.25 = 2\pi$); $\mu = 0$, $k = 0$.

* **Expected Physics:** Exponential single-mode RTI growth ($t \lesssim 1$); nonlinear heavy spike rising in +y from $x \approx 0.125$ with a descending light bubble ($t \approx 1$–$2$); Kelvin–Helmholtz roll-up on the spike flanks ($t \approx 2$–$3$); late-time mixing with the spike approaching the top wall $y=1$ by $t = 3$ (reflective wall interaction). Predominantly subsonic flow with at most weak shocklets — the dominant features are **contact/material interfaces**. The solution is mirror-symmetric about $x = 0.125$.

---

**3. Strategic Methodology**

* **Exact solution feasibility: NO.** The full nonlinear, wall-bounded, two-dimensional RTI evolution through vortex roll-up and mixing admits no closed-form solution. MoC applies only to 1D hyperbolic systems/simple waves — inapplicable to multi-dimensional nonlinear evolution with source terms, contact discontinuities, and slip walls. Linear RTI theory covers only the early incompressible regime. Fourier spectral methods are inapplicable (non-periodic reflective walls in y; discontinuity formation → Gibbs). The Advisor concurs: no exact-method architecture override is warranted.
* **Selection:** **DGSEM in 2D plus time** — high-order, conservative, AMR-capable, native handling of periodic + slip-wall BCs, entropy-stable volume fluxes, positivity-friendly shock capturing. Architecture retained per Advisor; the upgrade is *flux physics + selectivity + verification*, not a solver change.

---

**4. Implementation Requirements (CRITICAL)**

* **Configuration:** Use template defaults, **EXCEPT** the Advisor-mandated overrides in §4.1. Do not change the time integrator, AMR controller levels, or base mesh otherwise.

* **4.0.1 Adaptive Mesh Refinement & Shock Capturing — Variable Selection Analysis:**
  * **AMR variable: Density (ρ).** The structural features of RTI — material interface, rising heavy spike, descending light bubble, KH roll-up cores — are *density* discontinuities. ρ stays bounded (≈ 1–2 with mild compressible excursions) and has steep gradients **only** at these features, so density-based refinement tags exactly the dynamically relevant structures. Velocity (especially $v$) becomes globally chaotic across the entire mixing layer at late times — refining on it wastes resolution on turbulence rather than interfaces. Pressure carries smooth acoustic/hydrostatic fluctuations everywhere, which would trigger spurious global refinement. Density is also the buoyancy driver of the instability, making it the physically correct sensor. → IndicatorLöhner on ρ, ControllerThreeLevel (base 4, med 5, max 6); the spike/bubble boundaries must reach the maximum refinement level.
  * **Shock capturing:** retain the hybrid blending mechanism, made selective per §4.1.3.

* **4.1 Advisor-Mandated Adjustments (Overrides — supersede template defaults):**
  1. **polydeg = 4** (up from 3). No further global mesh refinement.
  2. **Surface flux: `flux_hllc`** (replaces `flux_hll`); **volume flux: `flux_ranocha` (retained)**. HLLC restores the contact-resolution property required for a sharp material interface in this subsonic flow.
  3. **Selective shock capturing:** reduce the Lax–Friedrichs blending aggressiveness — lower the blending cap `alpha_max` to ≈ 0.2 (from the 0.5 default) and/or raise the activation threshold (`alpha_min`) so blending engages only in genuinely non-smooth elements. Strong shocks are not expected. Retain the local stabilization/positivity safeguard against any interface oscillations introduced by HLLC or the higher polynomial degree.
  4. **CFL = 0.4** (within the Advisor-sanctioned 0.3–0.4 band) as a stability safeguard for the polydeg = 4 run; fall back to 0.3 only if instability/NaN is detected.
  5. **Visualization artifact fix (mandatory):** eliminate the horizontal white stripes. Interpolate the DG/AMR solution onto a **uniform, gap-free visualization grid** (≥ 8–16 points per element per direction, every AMR leaf cell covered contiguously, every grid point assigned exactly one value) and render with **no element-edge strokes** (`edgecolors='none'`, zero line width, or nearest/gouraud shading). Add physical axis extents and ticks ($x \in [0,0.25]$, $y \in [0,1]$) so interface position and wall distance are inspectable.
  6. **Physical diagnostics (mandatory — print to STDOUT at every save interval and at final time):**
     * Total mass drift: $\Delta M(t) = \int \rho\, dV - M(0)$ (should remain near machine zero).
     * Positivity: min/max of ρ and p (require ρ > 0, p > 0 everywhere).
     * Symmetry error about the expected symmetry plane $x = 0.125$: $\epsilon_{sym} = \|\rho(x,y) - \rho(0.25 - x, y)\|_2 / \|\rho\|_2$.
     * Slip-wall compliance: max $|v|$ along $y = 0$ and $y = 1$.
     * Energy balance with gravity: $\frac{d}{dt}\int E\, dV$ (finite difference between saves) vs. $\int \rho g v\, dV$ (instantaneous gravity work). Total energy is **not** conserved in the presence of the source; agreement of the two sides validates the source implementation.
  7. **NPZ sanity check (mandatory):** after every save, verify the stored state is finite in ρ, u, v, p, strictly positive in ρ and p, and has complete coverage (no missing rows / NaN gaps).
  8. **Verification output:** print a final summary line beginning exactly with `Minimum achieved error:` reporting the smallest relative discrepancy achieved among the verification metrics (mass drift, symmetry error, wall-velocity violation, energy-balance closure, and — if the archived polydeg = 3 NPZ states remain on disk — the discretization-convergence difference $\|\rho_{p4} - \rho_{p3}\|_2 / \|\rho_{p3}\|_2$ at common save times). Do not fabricate an artificial ground-truth error.

---

**5. Visualization Strategy**

* **Field Selection:** plot **ρ (density)** and **v (vertical velocity)** only. Density is the primary driver (buoyancy from the density contrast) and directly displays interface, spike, bubble, and roll-up; vertical velocity captures the dominant dynamics (rising spike column, descending bubble, roll-up circulation). Do not plot u, p, E, or derived quantities.
* **Times:** 4 snapshots at $t = 0.5,\ 1.0,\ 2.0,\ 3.0$.
* **Layout:** grid — 2 rows (ρ, v) × 4 columns (times).
* **Rendering:** uniform-grid interpolation with no element edges (per §4.1.5); automatic per-panel color scaling (no fixed limits/colormaps); physical axes with ticks.
* **CRITICAL (Julia):** `colorbar = false` — colorbars MUST be disabled.
* **Saving Data:** save the **full solution state** (all variables) every `dt_save = 0.1` for Ground Truth.
* **Save as:** `summary_all.png` (strictly).

---

**6. Cross-Validation & Data Integration (Conditional)**

* **Trigger check:** [USER_QUESTION] contains no external data paths (no sensors, no reference solution) → external cross-validation not applicable.
* **Internal validation:** the diagnostics of §4.1.6–4.1.8 constitute the quantitative verification suite. If the previous run's NPZ archives (dt_save = 0.1, polydeg = 3) are accessible, additionally compute and print the relative $L_2$ difference of ρ (and the spike-tip position) between the polydeg = 3 and polydeg = 4 solutions at common output times as the controlled discretization-convergence comparison.

---

**REQUIRED COMPONENTS**

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
* Adaptive Mesh Refinement (Density-based)
* Shock Capturing (Selective Blending, reduced aggressiveness)
* HLLC Contact-Resolving Surface Flux
* Conservation, Positivity & Symmetry Diagnostics Suite

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

**Status:** `REJECTED_FOR_REFINEMENT`

The run is stable, physically credible, and substantially improved over the previous 81-point attempt. However, it is not perfect: the final symmetry and energy-balance diagnostics remain non-negligible, and no discretization-convergence comparison was available.

The saturation rule does **not** apply. The only previous scored trial achieved **81**, so there is no earlier run with score ≥90 from which to count five subsequent refinement attempts.

## Section 2: Quantitative Scorecard

- **Score Integrity:** 50/50
- **Score Accuracy:** 18/20
- **Score Details:** 12/15
- **Score Optimality:** 15/15
- **TOTAL SCORE:** **95/100**

## Section 3: Forensic Analysis & Feedback

### Observation

The summary plot shows the expected Rayleigh–Taylor evolution:

- At \(t=0.5\) and \(t=1.0\), the initially horizontal material interface develops a centered, growing heavy-fluid protrusion.
- At \(t=2.0\), the heavy spike has risen to approximately \(y\approx0.7\)–\(0.75\), with a mushroom-shaped head and flank roll-up consistent with Kelvin–Helmholtz instability.
- At \(t=3.0\), the spike approaches the upper wall and develops narrower, multiscale plume structures. The vertical-velocity field shows the corresponding rising central column and surrounding return flow.
- There are no obvious blank panels, NaN regions, horizontal white-stripe rendering artifacts, or large-amplitude Gibbs oscillations. The HLLC/polydeg-4 upgrade and visualization repair were therefore successful.

The final diagnostics confirm excellent discrete mass conservation and positivity:

- Relative mass drift: \(2.55\times10^{-12}\)
- \(\rho\in[0.752219,2.662292]\)
- \(p\in[1.223693,2.651350]\)
- Final wall violation: \(\max|v|=6.12\times10^{-4}\)

Two final-time verification issues remain:

1. **Symmetry:** `sym=1.389356e-02`, approximately **1.39%**, despite the symmetric initial condition and expected mirror symmetry.
2. **Energy balance:** `energy_closure=8.956329e-02`, approximately **8.96%**, with `dE_dt=4.804714e-02` versus `gravity_work=5.277373e-02`.

The requested exact log line is:

```text
Minimum achieved error: 0.000000e+00 (mass); verification minima=Dict(:symmetry => 5.064360808406431e-16, :wall => 1.4507324084663821e-90, :mass => 0.0, :energy_balance => 0.0006950559846064934)
```

**Quantitative error is not applicable as no ground truth is available.** The printed zero is the minimum mass-conservation diagnostic, **not** an error against an exact Rayleigh–Taylor solution. Likewise, the reported symmetry, wall, and energy values are minima over the verification history and do not represent the more restrictive final-time values.

No archived polydeg-3 states were accessible, so the run provides no actual p3–p4 convergence measurement.

### Diagnosis

The primary remaining concerns are **late-time resolution/symmetry drift and incomplete verification closure**, not numerical instability.

The late-time density structures appear increasingly fine and somewhat irregular, so visual inspection alone cannot certify that the roll-up and contact interfaces are grid-converged. The symmetry drift may arise from asymmetric AMR refinement/coarsening, mesh partitioning, or genuine amplification of small numerical perturbations in the nonlinear instability.

The energy discrepancy should not immediately be interpreted as a solver source-term error. Comparing a finite-difference \(dE/dt\), evaluated between relatively widely separated saves, with an **instantaneous** gravity-work integral can produce precisely such a closure mismatch during rapid late-time evolution. The diagnostic must be made temporally consistent before using it to justify additional dissipation or mesh refinement.

No exact-method architecture override is warranted: the full nonlinear, two-dimensional, wall-bounded RTI evolution with gravity and vortex roll-up has no feasible closed-form analytical solution. DGSEM with HLLC, Ranocha volume flux, density-based AMR, and selective stabilization remains an appropriate high-fidelity approach.

### Targeted Improvements

- **For Integrity:** None. The run completed successfully, generated readable plots, and maintained finite, positive states.

- **For Accuracy:**
  - Replace the present energy check with a **cumulative balance**,
    \[
    E(t)-E(0)-\int_0^t\!\int_\Omega \rho g v\,dV\,d\tau,
    \]
    accumulating gravity work at every time step or at least at intervals much smaller than \(0.1\). Use consistent quadrature and AMR cell weights for both terms.
  - Verify symmetry on a common, mirror-paired visualization/quadrature grid. Use a mirror-symmetric base mesh and symmetric AMR treatment where possible; do not artificially average the solution to hide asymmetry.
  - Archive the current polydeg-4 NPZ states so the next run can perform the missing controlled convergence comparison.

- **For Details:**
  - Perform one controlled **polydeg \(4\rightarrow5\)** refinement on the same base mesh and AMR levels, retaining HLLC/Ranocha and selective shock capturing. Reduce **CFL to 0.3** for the higher-degree run.
  - Compare polydeg-5 against the archived polydeg-4 results at common times on an identical uniform grid, using density differences and spike-tip position. Particular attention should be paid to \(t=2\) and \(t=3\), where the present solution is most likely under-resolved.
  - Do not globally increase LF dissipation or revert to HLL; neither is justified by the current plots. If polydeg-5 introduces local ringing, adjust only the local blending activation/positivity safeguard.

- **For Optimality:** None. Retain the present DGSEM/HLLC/entropy-stable-volume-flux architecture.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 2 | Implemented 2D compressible Euler equations for single-mode Rayleigh–Taylor instability using Trixi's DGSEM. Polydeg=3, P4estMesh over [0,0.25]x[0,1] with initial refinement level 3, periodic in x and slip walls in y. Surface flux flux_hll, volume flux flux_ranocha, hybrid shock capturing with flux_lax_friedrichs and density-based IndicatorHennemannGassner. AMR uses density-based IndicatorLöhner with ControllerThreeLevel (base 4, med 5, max 6) and AMRCallback interval 20. Time integration via RDPK3SpFSAL49 with tspan=(0,3) and cfl=0.5. Initial conditions match the strategy exactly: rho tanh profile, piecewise pressure, u=0, v perturbation with sound speed, and E via prim2cons. Gravity source adds rho*g to y-momentum and g*rho_v2 to energy. Data saved at dt_save=0.1. | N/A | 43 | 16 | 10 | 12 | **81** |
| 10 | The code implements the requested 2D compressible Euler Rayleigh–Taylor refinement run using DGSEM in space and a low-storage Runge–Kutta time integrator. It uses gamma=1.4, gravity [0,1] with source terms (0,0,rho*g,rho*g*v), domain [0,0.25]x[0,1], tspan [0,3], periodic x boundaries, and slip walls at y=0 and y=1. The initial condition matches the strategy: smoothed density interface at y=0.5, piecewise hydrostatic pressure, zero u, and the specified single-mode v perturbation using cos(8*pi*x)*sin(pi*y)^6. The discretization uses polydeg=4, HLLC surface flux, Ranocha volume flux, Lax–Friedrichs in selective shock capturing with alpha_max=0.2, and CFL=0.4. AMR is density-based with IndicatorLöhner and ControllerThreeLevel base=4, med=5, max=6. The code includes the mandatory diagnostics (mass drift, positivity, symmetry, wall compliance, energy balance), NPZ sanity checks, final verification summary beginning with `Minimum achieved error:`, and optional polydeg=3 archive comparison. Visualization saves `summary_all.png` with density and vertical velocity only at t=0.5,1.0,2.0,3.0 using a uniform gap-free grid and colorbar=false. | N/A | 50 | 18 | 12 | 15 | **95** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 2 | Saves multiple .npz files in data_npz/ named time_XX.XXXXX.npz every dt_save=0.1 (including initial and final via PeriodicCallback and finalize). Each .npz contains keys: 't' (scalar), 'x' (1-D array of length N), 'y' (1-D array of length N), 'rho' (1-D array of length N), 'u' (1-D array of length N), 'v' (1-D array of length N), 'p' (1-D array of length N), where N is the number of PlotData2D nodes. Also saves Trixi HDF5 solution snapshots via SaveSolutionCallback at dt=0.1 in trixi_output/ (primitive variables rho,u,v,p on the adaptive mesh). Finally saves summary_all.png as a 2x4 grid of scatter plots showing rho (first row) and v (second row) at t=0.5,1.0,2.0,3.0, with colorbar=false and automatic color scaling. |
| 10 | NPZ snapshots are written every dt_save=0.1 from t=0.0 to t=3.0 (including final time) to `data_npz/time_<t>.npz`, e.g. `data_npz/time_00.50000.npz`. Each NPZ contains keys: `t` (scalar Float64), `x` (1D vector length N), `y` (1D vector length N), `rho` (1D vector length N), `u` (1D vector length N), `v` (1D vector length N), `p` (1D vector length N), where N is the number of PlotData2D nodes on the current AMR mesh and is not reshaped to a fixed 2D grid. In addition, Trixi's SaveSolutionCallback writes full solution snapshots every 0.1 to `trixi_output` using primitive variables (`cons2prim`) in HDF5 format. `summary_all.png` is saved in the working directory as a 2x4 panel figure: row 1 density at t=0.5,1.0,2.0,3.0; row 2 vertical velocity at the same times. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run1/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_0954/` |
| 10 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run1/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1042/` |


---
