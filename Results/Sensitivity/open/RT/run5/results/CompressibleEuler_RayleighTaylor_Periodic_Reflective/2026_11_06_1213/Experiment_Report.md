
# Experiment Report
*Last Updated: 2026-11-06 12:55:16*

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

Refinement iteration of the 2D inviscid compressible Euler Rayleigh–Taylor instability (single mode, effective gravity in $+y$, heavy fluid below light) on $[0,0.25]\times[0,1.0]$, $t\in[0,3.0]$. Run 3 scored 97/100 with a validated DGSEM architecture. The Advisor mandated: (i) a timestep audit against the raw accepted-step history, (ii) a corrected symmetry diagnostic, and (iii) **one non-repeating resolution check** at `polydeg=6, CFL=0.30` — with promotion **conditional on material improvement**. The Critic rejected the previous plan's unconditional promotion. Corrected strategy: the production configuration **remains Run 3's `polydeg=5, CFL=0.35`** by default; the higher-degree run is executed **solely as the resolution check**; the polydeg=5 baseline is **regenerated on the identical sampling grid** (Run 3's archived scalars lack the required structural metrics); and a production selection is made only through an explicit, printed decision gate.

**1. Mathematical Specification (Code-Ready Extraction)**

* **Governing Equations:** 2D compressible Euler with gravitational source:
$$
\partial_t \begin{pmatrix}\rho\\ \rho u\\ \rho v\\ E\end{pmatrix}
+ \partial_x \begin{pmatrix}\rho u\\ \rho u^2 + p\\ \rho u v\\ u(E+p)\end{pmatrix}
+ \partial_y \begin{pmatrix}\rho v\\ \rho u v\\ \rho v^2 + p\\ v(E+p)\end{pmatrix}
= \begin{pmatrix}0\\ 0\\ \rho g\\ \rho g v\end{pmatrix},
\qquad
p = (\gamma-1)\left(E - \tfrac{1}{2}\rho(u^2+v^2)\right)
$$
* **Domain:** $(x,y) \in [0, 0.25] \times [0, 1.0]$, $t \in [0, 3.0]$
* **Boundary Conditions:** $x$: Periodic ($x=0 \equiv x=0.25$). $y=0$ and $y=1$: Reflective / free-slip walls (no-penetration $v=0$).
* **Initial Conditions:**
$$
\rho(x,y,0) = 2.0 + (1.0-2.0)\cdot\tfrac{1}{2}\left[1 + \tanh\!\left(50(y-0.5)\right)\right]
$$
$$
p(y,0) = \begin{cases} 2y + 1, & y < 0.5\\ y + 1.5, & y \geq 0.5\end{cases}
\qquad
u(x,y,0) = 0
$$
$$
v(x,y,0) = -0.025\, c(x,y)\, \cos(8\pi x)\, \sin^6(\pi y), \qquad c(x,y) = \sqrt{\gamma\, p(y,0)/\rho(x,y,0)}
$$
$$
E(x,y,0) = \frac{p(y,0)}{\gamma-1} + \tfrac{1}{2}\rho(x,y,0)\, v(x,y,0)^2
$$
* **Constants & Parameters:** $\gamma = 1.4$; $g = 1.0$ (source adds momentum in $+y$: $\mathbf{S} = (0,0,\rho g, \rho g v)^T$); $\rho_{heavy}=2.0$ (bottom), $\rho_{light}=1.0$ (top); interface $y_{int}=0.5$; smoothing $50$; perturbation amplitude $0.025$; single mode $\cos(8\pi x)$ ($\lambda = 0.25$, exactly one period across the domain).
* **Expected Physics:** RTI growth with the heavy-fluid spike rising in $+y$ at $x=0.125$ and light-fluid bubbles descending at $x=0,\,0.25$: single-mode displacement by $t\approx0.75$; spike/bubble development $t\approx1.5$; mushroom with KH roll-up $t\approx2.25$; secondary KH vortices and interaction with the top wall by $t=3.0$. Pressure remains near hydrostatic with late-time compression near $y=1$. The continuous solution is exactly mirror-symmetric about $x=0.125$ for all $t$ (exact property exploited by the symmetry diagnostic).

**2. Strategic Methodology**

* **Exact solution feasibility: NO.** The full nonlinear 2D system with gravitational source, solid walls, and a seeded single-mode perturbation has no closed-form solution. The Method of Characteristics yields exact implicit solutions only for 1D simple-wave/Riemann problems; in 2D the characteristic fields do not decouple, and the late-time dynamics (mushroom roll-up, secondary KH, wall interaction) are inherently multi-scale. Linearized analysis covers only early exponential growth. The Advisor explicitly confirmed this ("no feasible closed-form analytical solution"). A numerical method is mandatory.
* **Method selection: DGSEM in 2D plus time.** Fourier spectral is ruled out by the non-periodic $y$-walls. DGSEM provides high order for the smooth roll-ups, p4est AMR to track the interface, and indicator-based shock capturing for late-time compressions — the architecture the Advisor endorsed ("For Optimality: None").
* **Two-stage, decision-gated design (Critic-corrected):**
  * **Stage A — Production baseline (default):** Run 3's exact physics configuration (`polydeg=5`, `CFL=0.35`, HLLC, pressure-based HG shock capturing, density-based Löhner AMR 4/5/6, RDPK3SpFSAL49, `adaptive=false`), re-executed with the corrected diagnostic suite. Because Run 3's archived output contains only scalar diagnostics, Stage A **regenerates the polydeg=5 baseline** — fixed-grid $\rho, p$ samples and structural metrics on the identical sampling grid.
  * **Stage B — Resolution check (non-repeating):** `polydeg=6`, `CFL=0.30`, **everything else identical** (fluxes, indicators, AMR limits, integrator, mesh, tspan). Executed solely as the Advisor's resolution check.
  * **Decision gate:** Both stages are compared on the identical cell-centered grid at $t=2.25$ and $t=3.0$ against explicit materiality criteria. The polydeg=6 configuration is promoted **only** upon demonstrated material improvement; otherwise Run 3's polydeg=5 configuration is retained as production. The verdict is printed. Both stages render their panels during their own run, so the final `summary_all.png` is selected from already-computed data — no third run is required for either outcome.

**3. Implementation Requirements (CRITICAL)**

* **Configuration:** Use template defaults for all unlisted settings. The overrides below supersede all defaults.

* **3.0.1 Adaptive Mesh Refinement & Shock Capturing — Variable Selection Analysis:**
  * *AMR sensor: density $\rho$.* The structural features (material interface, spike, bubbles, mushroom rims) are all defined by $\rho$, which stays sharp, localized, and bounded ($\sim[0.3, 2.2]$) — never globally chaotic. Velocity is rejected: after KH onset ($t\gtrsim 2$) it becomes broadband over the whole mixing zone, wasting elements everywhere. Pressure is rejected as AMR sensor: it is dominated by the smooth hydrostatic gradient and nearly uniform along the interface. Density is strictly superior for this PDE.
  * *Shock-capturing sensor: pressure.* It must detect compressions and wall-interaction shocks (where entropy is produced), not the contact interface; a density-based indicator would deposit dissipation directly on the RTI feature of interest. Dual-sensor assignment retained per the Advisor.

* **3.1 Advisor/Critic-Mandated Adjustments (Overrides):**
  1. **Shared configuration (both stages):** P4estMesh on $[0,0.25]\times[0,1]$, periodic in $x$, slip walls at $y=0,1$; HLLC surface flux; pressure-based IndicatorHennemannGassner shock capturing (flux_ranocha DG / flux_lax_friedrichs FV); density-based IndicatorLöhner AMR, ControllerThreeLevel (base=4, med=5 @0.1, max=6 @0.3), interval=20, `adapt_initial_condition=true`; RDPK3SpFSAL49, `adaptive=false`, `tspan=(0.0, 3.0)`.
  2. **Stage A (production default):** `polydeg=5`, `CFL=0.35` — Run 3's validated configuration. Full diagnostic suite (items 4–6, 8). Save full state at `dt_save=0.1`; save fixed-grid $\rho, p$ samples at $t=2.25, 3.0$; render and store its $2\times4$ panel figure.
  3. **Stage B (resolution check only):** `polydeg=6`, `CFL=0.30` — identical in all other respects. Same diagnostic suite; same fixed-grid samples at $t=2.25, 3.0$; render and store its panel figure.
  4. **Identical-grid comparison and decision gate (Details):** Both stages sample onto the **same** cell-centered uniform grid of $512 \times 2048$ points ($\Delta x = 0.25/512$, $\Delta y = 1/2048$). Compute for each stage at $t=2.25$ and $t=3.0$: (a) mixing-zone area $A_{mix} = \#\{(x,y): 1.1 < \rho < 1.9\}$; (b) mean interfacial sharpness $S = \langle|\nabla\rho|\rangle$ over the mixing zone; (c) structural count = number of disconnected heavy-fluid pockets ($\rho > 1.9$ above $y=0.5$) plus light-fluid pockets ($\rho < 1.1$ below $y=0.5$). Print a labeled `RESOLUTION CHECK` block: metric | polydeg=5 | polydeg=6 | relative shift | material? — at both times. **Material improvement is declared if ANY of:** (M1) $A_{mix}$ shifts $>10\%$ relative at $t=3.0$; (M2) $S$ shifts $>10\%$ relative at $t=3.0$; (M3) the structural count changes or new resolved sub-structures appear in the $t=3.0$ vortex cores on the identical grid. Print the verdict explicitly: `VERDICT: MATERIAL — promoting polydeg=6, CFL=0.30 to production` **or** `VERDICT: NOT MATERIAL — retaining polydeg=5, CFL=0.35 (Run 3 configuration)`. The selected run's stored figure is written as the final `summary_all.png`. **Do not promote polydeg=6 absent a demonstrated material change.**
  5. **Timestep audit (Integrity, both stages):** Record the raw accepted-step history $(t_n, \mathrm{dt}_n)$ for **every** step. Print a labeled `TIMESTEP AUDIT` block: (a) min/median/max dt over **ordinary integration steps only**; (b) the final (possibly `tspan`-clipped) step's dt separately; (c) the initialization value separately; (d) count of steps with $\mathrm{dt} < 10^{-8}$ and whether their times coincide with AMR adaptation events (interval=20). This determines whether Run 3's $\mathrm{dt}_{min}=5.55\times10^{-17}$ was a final-step clipping artifact (expected) or a recurring AMR/StepsizeCallback pathology. CFL is changed **only** as part of the Stage B check, not blindly.
  6. **Symmetry residual fix (Accuracy, both stages):** Compute residuals on a **cell-centered** uniform $x$-grid, $x_i = (i+\tfrac{1}{2})\Delta x$, $\Delta x = 0.25/512$, **without a duplicated periodic endpoint**, using the exact mirror pairing $i \leftrightarrow N_x - 1 - i$ (reflection about $x=0.125$ under periodicity; exact since $x_{N_x-1-i} = 0.25 - x_i$). Report $R_q = \|q - Mq\|_2/\|q\|_2$ **separately for each conservative variable** $q \in \{\rho, \rho u, \rho v, \rho E\}$ at $t = 0, 0.75, 1.5, 2.25, 3.0$ in a labeled `SYMMETRY MONITOR` block. The $t=0$ residual must be at machine precision — this validates the diagnostic itself.
  7. **Symmetry contingency (conditional, per Advisor):** Only if the corrected residual for any variable still exceeds $\sim 10^{-2}$ at $t=3.0$: run a **symmetric fixed-mesh control run at comparable degrees of freedom (no AMR)** to distinguish AMR-induced symmetry breaking from physical chaotic amplification; report both residuals. Do not trigger this otherwise.
  8. **Retained diagnostics (both stages, printed at every save):** mass drift vs $t=0$ (Trixi.integrate), positivity (min $\rho$, min $p$), wall no-penetration (max $|v|$ at $y=0$ and $y=1$), and the $t=0$ renderer $x$-independence test (max mirrored deviation of rendered $\rho$, $p$).
  9. **Prohibitions (Advisor):** Do NOT switch to HLL/Lax–Friedrichs surface fluxes; do NOT add indiscriminate stabilization; do NOT change the AMR level limits; no ringing is demonstrated and contacts must remain sharp.

**4. Visualization Strategy**

* **Field Selection:** $\rho$ (primary — interface, spike, bubble, and mushroom dynamics are the RTI drivers) and $p$ (secondary — hydrostatic stratification plus late-time wall compression). Do not plot $u, v, E$.
* **Task:** Plot $\rho$ (top row) and $p$ (bottom row) at $t = 0.75, 1.5, 2.25, 3.0$ — a $2 \times 4$ grid.
* **Layout/Style:** Sample the DG solution onto the uniform Cartesian grid (no element striping, no mesh lines). **CRITICAL: disable colorbars (`colorbar=false`) — Julia.** Per-panel automatic color scaling; no fixed color ranges.
* **Saving Data:** Save the **full solution state** (all variables) at `dt_save = 0.1` (plus initial/final) for Ground Truth in the production run; save the fixed-grid $\rho, p$ arrays at $t=2.25, 3.0$ for **both** stages (comparison input).
* **Production selection:** Both stages render and store their own $2\times4$ figures during execution; after the decision gate (§3.1.4), the selected production run's figure is written as `summary_all.png`. **Save as:** `summary_all.png` (strictly).

**5. Cross-Validation & Data Integration**

* No external reference data (sensors / true data) are provided → external Relative $L_2$ comparison is **not applicable** (Advisor-confirmed: no ground truth exists for this forward nonlinear RTI run).
* Internal verification suite replaces ground truth: conservation (mass drift), positivity, wall no-penetration, machine-precision $t=0$ symmetry, per-variable symmetry residuals vs. time, the accepted-step timestep audit, and the **quantitative polydeg=5 vs. polydeg=6 identical-grid comparison** (mixing-zone area, interfacial sharpness, structural counts) — the latter serving as the convergence evidence for the smallest $t=3$ vortices.

**REQUIRED COMPONENTS**

**Main Method:**
Discontinuous Galerkin Spectral Element Method in 2D plus time

**Enhancements:**
* Shock Capturing Indicator (pressure-based Hennemann–Gassner, hybrid DG–FV)
* Adaptive Mesh Refinement (density-based Löhner, three-level controller 4/5/6)
* Decision-Gated Resolution Check (polydeg=6, CFL=0.30 vs. regenerated polydeg=5 baseline on identical grid; promotion only on material improvement)
* Accepted-Step Timestep-History Audit (Integrity diagnostic)
* Mirror-Indexed, Per-Variable, Multi-Time Symmetry-Residual Diagnostic (Accuracy diagnostic)
* Conditional Symmetric Fixed-Mesh Control Run (only if corrected symmetry residual remains significant)

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
## Section 1: Status Decision

**Status:** `REJECTED_FOR_REFINEMENT`

The selected `polydeg=6, CFL=0.30` solution is high fidelity, but the final symmetry report is not internally consistent and prevents a perfect score. This is primarily a **verification-diagnostic rejection**, not a request to change the numerical architecture.

The saturation condition is also not met: Run 2 was the first run scoring at least 90, and only Run 3 and the present run have followed it—fewer than five subsequent refinement trials.

## Section 2: Quantitative Scorecard

- **Score Integrity:** 50/50
- **Score Accuracy:** 19/20
- **Score Details:** 15/15
- **Score Optimality:** 15/15
- **TOTAL SCORE:** **99/100**

**Reference-error check:** The supplied log tail contains no `Minimum achieved error:` line. **Quantitative error is not applicable as no ground truth is available.** In particular, the reported symmetry residual is an internal diagnostic, not an error against an analytical solution.

## Section 3: Forensic Analysis & Feedback

### Observation

- The density evolution is physically coherent:
  - At \(t=0.75\), the seeded single-mode displacement is visible.
  - At \(t=1.5\), the central heavy-fluid spike and adjacent light-fluid bubbles have developed.
  - At \(t=2.25\), the spike forms a mushroom-like head with rolled-up side structures.
  - At \(t=3.0\), secondary Kelvin–Helmholtz structures and interaction with the upper wall are visible. The interfaces remain thin and highly detailed, without obvious Gibbs ringing or catastrophic grid-scale oscillations.
- Pressure retains the expected early-time hydrostatic stratification. Late-time pressure variations and upper-wall compression are consistent with the increasingly nonlinear dynamics.
- The selected run completed successfully and saved the figure. Positivity and conservation are strong:
  - Relative mass drift is approximately \(7.27\times10^{-12}\).
  - Minimum density and pressure remain positive: \(\rho_{\min}\approx0.1921\), \(p_{\min}\approx1.2083\).
  - Wall-normal velocity maxima are small, approximately \(5.63\times10^{-5}\) at \(y=0\) and \(3.73\times10^{-4}\) at \(y=1\).
- Promotion of Stage B is supported by the resolution check. At \(t=3\), mixing-zone area changes by only about \(-1.75\%\), while mean interfacial sharpness increases by approximately \(20.7\%\). Thus, the higher-degree run resolves sharper small-scale structure without materially changing the overall mixing-zone area.
- The final `max_t3_symmetry_residual=2.000000e+00` is suspicious rather than evidence of complete physical symmetry failure. The plotted solution remains approximately mirror-symmetric, and the reported even-field residuals are small: the AMR residuals for \(\rho v\) and \(\rho E\) are approximately \(5.18\times10^{-5}\) and \(1.40\times10^{-6}\), respectively.

### Diagnosis

**Likely reflection-parity bug in the symmetry diagnostic.**

Under reflection about \(x=0.125\), the conserved state transforms as

\[
(\rho,\rho u,\rho v,\rho E)
\longmapsto
(\rho,-\rho u,\rho v,\rho E).
\]

Thus, \(\rho u\) is **odd**, while \(\rho\), \(\rho v\), and \(\rho E\) are **even**. If the diagnostic computes \(\|q-Mq\|/\|q\|\) for every variable, then for an odd momentum field satisfying \(Mq=-q\),

\[
\frac{\|q-Mq\|}{\|q\|}
=
\frac{\|2q\|}{\|q\|}
=2.
\]

The value exactly equal to 2 is therefore the expected signature of applying an even-field formula to \(\rho u\). It should not be interpreted as a physical loss of symmetry. The fixed-mesh control run was consequently probably triggered by a faulty diagnostic rather than by meaningful AMR-induced asymmetry.

The raw connected-component increase is not, by itself, proof of convergence because pixel-scale fragments can inflate that count. Nevertheless, the independent sharpness improvement and the visually coherent additional roll-ups justify retaining Stage B.

No exact-method override is warranted: nonlinear two-dimensional compressible Euler with gravitational forcing, wall interaction, and Kelvin–Helmholtz roll-up does not admit a feasible closed-form, Fourier-series, or global method-of-characteristics solution.

### Targeted Improvements

- **For Integrity:** None for the solver execution. Both stages completed, the selected figure was saved, and the reported state remains finite and positive.
- **For Accuracy:** Correct the symmetry residual using parity-aware reflections:
  \[
  R_{\rho u}=\frac{\|\rho u+M(\rho u)\|}{\|\rho u\|},
  \qquad
  R_q=\frac{\|q-Mq\|}{\|q\|}
  \quad\text{for }q=\rho,\rho v,\rho E.
  \]
  Handle the initially zero \(\rho u\) norm with an absolute-residual check or a guarded denominator. Recompute these diagnostics from the archived Stage B full state; another PDE solve is unnecessary unless the corrected residuals reveal genuine asymmetry. Do not trigger further fixed-mesh control runs from the uncorrected maximum residual.
- **For Details:** Retain `polydeg=6, CFL=0.30`, HLLC, and the existing AMR limits. Further degree or mesh increases are not justified by the present evidence. For future resolution gates, apply a minimum physical-area threshold to connected components so that grid-scale fragments are not counted as newly resolved vortices.
- **For Optimality:** None. DGSEM with contact-resolving HLLC flux, pressure-based shock capturing, and density-based AMR remains well matched to this problem.

**Recommendation:** Retain Stage B as the current best production dataset. The next iteration should close the parity-aware symmetry verification rather than repeat or further refine the already successful simulation.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | The code implements the 2D compressible Euler equations with gravitational source using Trixi.jl. The spatial discretization is DGSEM with polynomial degree 3 on a P4estMesh (domain [0,0.25]×[0,1], periodic in x, reflective slip walls in y via boundary_condition_slip_wall). Shock capturing is density-based using IndicatorHennemannGassner with a hybrid DG-FV volume integral (flux_ranocha for DG, flux_lax_friedrichs for FV) and HLL surface flux. AMR is density-based using IndicatorLöhner with ControllerThreeLevel (base_level=4, med_level=5, max_level=6) and AMRCallback interval=20. The initial condition exactly matches the strategy: tanh density profile, piecewise hydrostatic pressure, zero u, and the specified single-mode v perturbation. Gravity source is (0,0,ρg,ρgv) with g=1.0. Time integration uses RDPK3SpFSAL49 with adaptive=false, StepsizeCallback cfl=0.5, and tspan=(0.0,3.0). Visualization produces summary_all.png with density (top row) and pressure (bottom row) at t=0.75, 1.5, 2.25, 3.0, with colorbars disabled and automatic color scaling. | N/A | 47 | 18 | 11 | 13 | **89** |
| 2 | The implementation faithfully reproduces the DGSEM solution of the 2D inviscid compressible Euler equations with a gravitational source for the single-mode Rayleigh–Taylor instability. Governing equations use CompressibleEulerEquations2D(γ=1.4); the initial condition is the verbatim strategy profile (smoothed density 2→1 via tanh(50(y-0.5))·0.5, piecewise pressure 2y+1 / y+1.5 at y=0.5, u=0, v=-0.025·c·cos(8πx)·sin^6(πy) with c=sqrt(γp/ρ), and energy via prim2cons giving E=p/(γ-1)+0.5ρv²). The gravity source term returns SVector(0,0,g·ρ, g·ρv), i.e. (0,0,ρg,ρgv). Discretization: P4estMesh on [0,0.25]×[1.0] with x-periodic (periodicity=(true,false)) and slip-wall BCs on :y_neg/:y_pos; DGSEM with polydeg=3, LobattoLegendreBasis. Advisor-mandated upgrades are present: surface_flux=flux_hllc, pressure-based IndicatorHennemannGassner shock capture (pressure_indicator derived from conservative state) feeding VolumeIntegralShockCapturingHG with volume_flux_dg=flux_ranocha and volume_flux_fv=flux_lax_friedrichs, while AMR uses a separate density-based IndicatorLöhner (variable=density_indicator=u[1]) with ControllerThreeLevel (base 4 / med 5 @0.1 / max 6 @0.3), interval=20, adapt_initial_condition=true. Held-fixed config is respected: RDPK3SpFSAL49 time integrator, adaptive=false, StepsizeCallback(cfl=0.5), tspan=(0.0,3.0). The full diagnostic suite is implemented and printed at every save: total mass (Trixi.integrate) with absolute/relative drift vs t=0, min/max ρ and p (positivity), dt_min/dt_max/current dt (CFL), max|v| at y=0 and y=1 (wall no-penetration) and an approximate symmetry residual ‖q-q(0.25-x)‖/‖q‖ about x=0.125. Visualization renders a clean 2×4 grid (density top row, pressure bottom row) at t=0.75,1.5,2.25,3.0 with colorbar=false, no mesh lines (framestyle=:none, axis=nothing), per-panel automatic scaling, and higher resolution size=(3200,2400) saved as summary_all.png. | N/A | 47 | 19 | 12 | 15 | **93** |
| 3 | Implementation is a numerical DGSEM solver for the 2D compressible Euler equations with gamma=1.4 and the prescribed gravity source. Domain is [0,0.25] x [0,1] with t in [0,3], periodic x-boundaries, and slip walls at y=0 and y=1. Initial conditions match the strategy: smoothed tanh density interface at y=0.5, piecewise hydrostatic pressure, u=0, v=-0.025*c*cos(8*pi*x)*sin(pi*y)^6, and total energy through prim2cons. Spatial discretization uses polydeg=5 Lobatto basis, HLLC surface flux, pressure-based Hennemann-Gassner shock capturing with flux_ranocha DG and flux_lax_friedrichs FV, and density-based Lohner AMR with ControllerThreeLevel base=4/med=5/max=6, interval=20, adapt_initial_condition=true. Time integration uses RDPK3SpFSAL49 with adaptive=false, tspan=(0.0,3.0), and CFL=0.35 via StepsizeCallback. Visualization samples the DG solution onto a uniform Cartesian grid and saves summary_all.png as a 2x4 heatmap grid of density and pressure at t=0.75,1.5,2.25,3.0 with colorbar=false. The code also performs the required t=0 renderer x-independence test and prints the final diagnostic summary for mass drift, positivity, dt extrema, wall |v|, and symmetry residual. | N/A | 49 | 19 | 14 | 15 | **97** |
| 4 | Numerical implementation of the 2D inviscid compressible Euler Rayleigh-Taylor instability using DGSEM. Stage A uses polydeg=5, CFL=0.35, HLLC surface flux, pressure-based Hennemann-Gassner shock capturing with flux_ranocha DG and flux_lax_friedrichs FV, density-based Löhner AMR with ControllerThreeLevel 4/5/6 at thresholds 0.1/0.3 and interval=20, RDPK3SpFSAL49 with adaptive=false, and tspan=(0.0,3.0) on [0,0.25]x[0,1] with periodic x and slip y walls. Stage B repeats the same configuration with polydeg=6 and CFL=0.30 as the single non-repeating resolution check. The code regenerates the polydeg=5 baseline, computes both stages on the identical 512x2048 cell-centered comparison grid at t=2.25 and t=3.0, evaluates the M1/M2/M3 materiality gate, prints the explicit RESOLUTION CHECK verdict, and selects summary_all.png from the already-rendered Stage A or Stage B 2x4 rho/p panel figure. It also implements the raw accepted-step timestep audit, per-variable mirror-indexed symmetry residuals at t=0,0.75,1.5,2.25,3.0, mass drift/positivity/wall diagnostics, and the conditional fixed-mesh symmetry control run. The initial conditions, pressure law, gravity source term, boundary topology, and required diagnostics match the strategy. | N/A | 50 | 19 | 15 | 15 | **99** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | NPZ files are saved in directory `data_npz` with filenames `time_%08.5f.npz` at dt_save=0.1 and additionally at the visualization times 0.75, 1.5, 2.25, 3.0. Each NPZ file contains keys: `t` (scalar Float64), `x` (1D array of node coordinates, length N_nodes(t)), `y` (1D array, length N_nodes(t)), `rho` (1D array, length N_nodes(t)), `rho_u` (1D array), `rho_v` (1D array), `rho_e` (1D array), and `p` (1D array, length N_nodes(t)). N_nodes varies with AMR. Additionally, `SaveSolutionCallback` writes HDF5 solution files to `trixi_output` at dt=0.1 with full conservative state. The plot `summary_all.png` is saved in the working directory, containing a 2×4 grid: top row density, bottom row pressure at the four specified times, with colorbars disabled. |
| 2 | Per-snapshot .npz files (e.g. time_00.75000.npz, time_01.50000.npz, time_02.25000.npz, time_03.00000.npz) written to <script_dir>/data_npz/ via npzwrite. Each file contains a Dict with keys: 't' (scalar Float64), 'x' (Float64 array of DG solution-node x-coordinates from PlotData2D), 'y' (Float64 array of node y-coordinates), 'rho' (Float64 array of DG-node conserved density), 'rho_u' (Float64 array, x-momentum ρu), 'rho_v' (Float64 array, y-momentum ρv), 'rho_e' (Float64 array, total energy ρE), 'p' (Float64 array, pressure from cons2prim). All field arrays (rho, rho_u, rho_v, rho_e, p, x, y) are structured on the same DG-node grid (per-element nodes × elements; flattened with vec() by the rasterizer during plotting). Save times = union of {0.0,0.1,...,3.0} plus the four visualization times {0.75,1.5,2.25,3.0}, ensuring dt_save=0.1 cadence and exact plot snapshots. Additionally, SaveSolutionCallback writes full conservative state (cons2cons) HDF5 snapshots at dt=0.1 (plus initial) into <script_dir>/trixi_output/. Final figure summary_all.png (2 rows × 4 columns, 3200×2400) is saved to <script_dir>/. |
| 3 | Saves NPZ files under data_npz named time_%08.5f.npz (e.g., time_00.00000.npz, time_00.10000.npz, ..., time_03.00000.npz) at times formed from 0.0:0.1:3.0 plus [0.75,1.5,2.25,3.0]. Each NPZ contains keys: 't' as a scalar Float64; 'x' and 'y' as Trixi PlotData2D element-block arrays; and 'rho','rho_u','rho_v','rho_e','p' as PlotData2D element-block arrays matching the x/y layout. In expected block form these are (polydeg+1, polydeg+1, n_elements) = (6,6,n_elements) or a flattened 2D equivalent, with n_elements varying under AMR. The full conservative state is saved via rho, rho_u, rho_v, rho_e, and pressure p is also exported. Additionally, SaveSolutionCallback writes full conservative snapshots to trixi_output at dt=0.1 including the initial solution. Summary visualization is saved as summary_all.png: a 2x4 heatmap grid with density in the top row and pressure in the bottom row, colorbars disabled. |
| 4 | Stage A writes into data_npz/ and trixi_output/; Stage B writes into stage_b_polydeg6/data_npz/ and stage_b_polydeg6/trixi_output/. NPZ full-state snapshots are written as data_npz/time_%08.5f.npz for Stage A and stage_b_polydeg6/data_npz/time_%08.5f.npz for Stage B, containing keys: t (scalar), polydeg (scalar Int), x, y, rho, rho_u, rho_v, rho_e, p. The x/y/rho/rho_u/rho_v/rho_e/p arrays are DG block arrays with shape normally (polydeg+1, polydeg+1, n_elements), where n_elements is AMR/time-dependent; Stage A has first two dimensions 6x6 and Stage B has 7x7. Fixed-grid comparison files are data_npz/fixed_grid_time_02.25000.npz, data_npz/fixed_grid_time_03.00000.npz, and the same under stage_b_polydeg6/data_npz/. They contain keys: t (scalar), x (vector length 512), y (vector length 2048), rho (2048x512 matrix), p (2048x512 matrix), mixing_zone_count (scalar Int), mixing_zone_area (scalar Float64), sharpness (scalar Float64), structural_count (scalar Int), upper_mixing_component_count (scalar Int). Raw timestep histories are saved as data_npz/accepted_step_history.npz and stage_b_polydeg6/data_npz/accepted_step_history.npz when steps exist, with keys: t, tprev, dt, iter, amr_event (all vectors of length N accepted steps; amr_event is Bool), initialization_t (scalar), initialization_dt (scalar). Trixi SaveSolutionCallback also writes full conservative-state HDF5 solution snapshots at dt=0.1 into trixi_output/ and stage_b_polydeg6/trixi_output/. Figures saved are summary_stage_a_polydeg5.png for Stage A, stage_b_polydeg6/summary.png for Stage B, and the selected production panel copied to summary_all.png; all are 2x4 rho/p PNG panels at t=0.75,1.5,2.25,3.0 with size 3200x2400. If the conditional symmetry control triggers, it additionally writes the control run's time_*.npz snapshots and Trixi HDF5 output under symmetric_fixed_control_polydeg<deg>_level<level>/. |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1051/` |
| 2 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1109/` |
| 3 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1132/` |
| 4 | `/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Results/Sensitivity/open/RT/run5/results//CompressibleEuler_RayleighTaylor_Periodic_Reflective/2026_11_06_1213/` |


---
