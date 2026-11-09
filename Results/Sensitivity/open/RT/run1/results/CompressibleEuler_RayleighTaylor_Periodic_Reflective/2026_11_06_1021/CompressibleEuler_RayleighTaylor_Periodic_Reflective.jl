# %% [markdown]

# ## Cell 1: Solver and Discretization Initialization
#
# ## Framework: Method of Lines (MOL) for Hyperbolic Systems
# Initializes the core components to solve conservation laws of the form $\partial_t \mathbf{u} + \nabla \cdot \mathbf{f}(\mathbf{u}) = \mathbf{0}$.
#
# 1.  **Time Integration (`OrdinaryDiffEqLowStorageRK`)**:
#     * Implements **Low-Storage Runge-Kutta (LSRK)** schemes.
#     * Optimized for high-order discretizations where the state vector $\mathbf{U}$ is large, minimizing memory registers while maintaining stability regions for explicit time-stepping ($dt$).
#
# 2.  **Spatial Discretization (`Trixi`)**:
#     * Provides the spatial operator $\mathcal{L}(\mathbf{U})$ for the semi-discrete form $\frac{d\mathbf{U}}{dt} = \mathcal{L}(\mathbf{U})$.
#     * Handles high-order discontinuous Galerkin (DGSEM) or Finite Volume operators, Riemann solvers (flux reconstruction), and adaptive mesh refinement (AMR).
#
# 3.  **Diagnostics (`Plots`, `NPZ`, `Statistics`)**:
#     * Utilities for computing integral norms ($L_2, L_\infty$) and exporting solution snapshots for surrogate modeling or analysis.
#
# %%
using OrdinaryDiffEqLowStorageRK
using Trixi
using Plots
using NPZ
import Printf
using Statistics
import DiffEqCallbacks: PeriodicCallback
# Reinitialize Polyester's per-thread task buffers for the active thread count.
# Without this, Trixi's threaded kernels throw `UndefRefError` in ManualMemory.jl
# when Julia is launched with multiple threads (--threads > 1).
Trixi.Polyester.reset_threads!()
# %% [markdown]

# ## Cell 2: Physics Model Definition (Compressible Navier-Stokes)
# ## Governing Equations: Inviscid Compressible Euler System
#
# Defines the two-dimensional compressible Euler equations with ideal-gas
# equation of state and gamma = 1.4. No viscous or thermal-diffusion terms
# are required; the gravitational source is defined with the initial conditions.

# %%
equations = CompressibleEulerEquations2D(1.4)
# %% [markdown]

# ## Cell 3: Initial Conditions and Source Terms (Lid-Driven Cavity)
# ## Problem Setup: Single-Mode Rayleigh–Taylor Instability
#
# Initializes heavy fluid below light fluid across a smoothed interface at
# y = 0.5. A piecewise-hydrostatic pressure and a single-mode vertical
# velocity perturbation seed the instability. Gravity acts in +y.

# %%
@inline function initial_condition_rayleigh_taylor(x, t,
        equations::CompressibleEulerEquations2D)
    x_coord, y = x

    rho = 2.0 + (1.0 - 2.0) * 0.5 *
          (1.0 + tanh(50.0 * (y - 0.5)))
    p = y < 0.5 ? 2.0 * y + 1.0 : y + 1.5

    u = 0.0
    sound_speed = sqrt(equations.gamma * p / rho)
    v = -0.025 * sound_speed * cos(8 * pi * x_coord) * sin(pi * y)^6

    return prim2cons(SVector(rho, u, v, p), equations)
end

@inline function source_terms_gravity(u, x, t,
        equations::CompressibleEulerEquations2D)
    g = 1.0
    rho, rho_v1, rho_v2, rho_e = u

    # Energy source g * rho * v = g * rho_v2.
    return SVector(0.0, 0.0, g * rho, g * rho_v2)
end

@inline function density_indicator(u, equations)
    return u[1]
end
# %% [markdown]

# ## Cell 4: Spatial Discretization & Hybrid Shock Capturing
#
# ## Methodology: Nodal Discontinuous Galerkin (DGSEM)
# Constructs the spatial operator $\mathcal{L}(\mathbf{U})$ using the **Discontinuous Galerkin Spectral Element Method (DGSEM)**.
#
# 1.  **Function Space (`LobattoLegendreBasis`)**:
#     * Approximates solutions using tensor-product polynomials of degree $N=3$ on **Legendre-Gauss-Lobatto (LGL)** nodes.
#     * This nodal basis satisfies the **Summation-by-Parts (SBP)** property, essential for proving discrete entropy stability.
#
# 2.  **Shock Capturing Strategy (Hybrid DG-FV)**:
#     * **Problem**: High-order polynomials suffer from Gibbs oscillations near discontinuities (shocks).
#     * **Solution**: A convex blending of a high-order DG flux and a robust low-order Finite Volume (FV) flux via `volume_integral`.
#     * **Blending Formula**:
#         $$
#         \mathbf{f}_{numerical} = (1 - \alpha) \mathbf{f}_{DG}^{EC} + \alpha \mathbf{f}_{FV}^{Dissipative}
#         $$
#     * **High-Order Flux ($\mathbf{f}_{DG}^{EC}$)**: Uses `flux_ranocha`, an entropy-conserving/stable flux that preserves thermodynamic consistency in smooth regions.
#     * **Low-Order Flux ($\mathbf{f}_{FV}^{Dissipative}$)**: Uses `flux_lax_friedrichs`, a highly dissipative flux to stabilize shocks.
#
# 3.  **Interface Flux (`surface_flux=flux_hll`)**:
#     * **Riemann Solver**: Harten-Lax-van Leer (HLL).
#     * **Role**: Computes the unique numerical flux across element interfaces where the solution is discontinuous.
#     * **Physics**: An approximate Riemann solver that estimates the fastest signal velocities (wave speeds). It is robust and positivity-preserving, making it ideal for compressible flows with strong shocks.
#
# 4.  **Indicator Function (`IndicatorHennemannGassner`)**:
#     * Dynamically computes the blending coefficient $\alpha \in [0, 1]$ element-wise.
#      * **Mechanism**: Monitors the spectral decay of energy in the highest modes of the polynomial expansion. If energy in high modes is large (indicating a discontinuity), $\alpha \to 1$ (activates FV sub-cells).
#
# %%
polydeg = 4
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.2,
                                         alpha_min=0.01,
                                         alpha_smooth=true,
                                         variable=density_indicator)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=flux_ranocha,
                                                 volume_flux_fv=flux_lax_friedrichs)
dg = DGSEM(polydeg=polydeg, surface_flux=flux_hllc,
           volume_integral=volume_integral)
# %% [markdown]

# ## Cell 5: Computational Domain & Adaptive Mesh Topology
#
# ## Methodology: Forest-of-Octrees (`p4est`)
# Defines the spatial domain $\Omega$ and the underlying mesh topology using the **`p4est`** library. This framework represents the mesh as a forest of dynamic quadtrees (in 2D), enabling efficient, scalable parallel adaptive mesh refinement (AMR).
#
# 1.  **Geometric Domain**:
#     * Physical domain $\Omega = [0, 1] \times [0, 1]$ (Unit Square).
#     * **Macro-Element Structure**: `trees_per_dimension=(1, 1)` constructs the root connectivity as a single large quadtree covering the entire domain.
#
# 2.  **Discretization Topology (`P4estMesh`)**:
#     * **$h$-Refinement**: The mesh supports dynamic recursive splitting of elements.
#     * **Initial Resolution**: `initial_refinement_level=3` applies uniform recursive refinement to the root tree. The initial grid consists of $2^3 \times 2^3 = 64$ elements.
#     * **Polydeg**: Associates the spectral basis degree $N$ (from Cell 4) with each leaf element of the tree.
#
# 3.  **Boundary Topology**:
#     * `periodicity=(false, false)`: Disables periodic wrapping. This treats the domain boundaries $\partial \Omega$ as physical walls, necessitating explicit boundary conditions (Dirichlet/Neumann) for the Navier-Stokes operator.
#
# %%
coordinates_min = (0.0, 0.0)
coordinates_max = (0.25, 1.0)

mesh = P4estMesh((1, 1), polydeg=polydeg,
                 coordinates_min=coordinates_min,
                 coordinates_max=coordinates_max,
                 initial_refinement_level=3,
                 periodicity=(true, false)) # Periodic in x; walls in y
# %% [markdown]

# ## Cell 6: General Boundary Condition Specification & System Assembly
#
# ## 1. Boundary Condition Methodology
# Defines the physical constraints at the domain boundaries using a flexible, function-based interface. This approach allows for arbitrary, spatially varying boundary profiles for general fluid dynamics problems.
#
# * **Function-Based Definitions (Closures)**:
#     Boundary values are defined as closures `(x, t, equations) -> values`. This allows BCs to be time-dependent (e.g., oscillating walls) or spatially varying (e.g., parabolic inlet profiles).
#     * **Momentum (`NoSlip`)**: Implements Dirichlet conditions for velocity $\mathbf{v} = \mathbf{v}_{target}(x,t)$.
#         * *Example 1*: A moving boundary (shear driver) where $\mathbf{v} \neq 0$.
#         * *Example 2*: A static wall where $\mathbf{v} = 0$.
#     * **Energy (`Adiabatic` vs `Isothermal`)**:
#         * *Adiabatic*: Neumann condition ($\nabla T \cdot \mathbf{n} = 0$) for insulated walls.
#         * *Isothermal*: Dirichlet condition ($T = T_{wall}$) for heated/cooled walls.
#
# * **Physics Wrappers**:
#     Raw functions are wrapped in physics-specific containers (e.g., `BoundaryConditionNavierStokesWall`) to bundle momentum and energy constraints into a single object usable by the parabolic solver.
#
# ## 2. Topology-Physics Mapping (`Dict`)
# Maps the geometric boundary tags (defined in the Mesh Construction step) to the physical boundary objects.
# * **Inviscid (Hyperbolic) Dictionary**: Defines the base state for the Euler operator (typically `slip_wall` for impermeability).
# * **Viscous (Parabolic) Dictionary**: Defines the viscous corrections (No-Slip/Heat Flux).
# * **Generalization**: Keys (e.g., `:x_pos`, `:y_neg`) must match the boundary names defined by the mesh generator (P4est or Unstructured).
#
# ## 3. Semidiscretization: Constructing the Operator $\mathcal{L}(\mathbf{U})$
# This step fuses the Topology (Mesh), Physics (Equations), and Numerics (DG Solver) into the global semi-discrete operator. The framework selects the appropriate container based on the physics regime:
#
# ### **Case A: Pure Hyperbolic (Euler)**
# $$ \frac{d\mathbf{U}}{dt} = \mathcal{L}_{hyp}(\mathbf{U}) $$
# * **Constructor**: `SemidiscretizationHyperbolic`
# * **Inputs**: Requires only the base `equations` (Euler) and standard boundary conditions.
# * **Role**: Assembles the inviscid spatial operator using the selected Riemann solver (e.g., HLL).
#
# ### **Case B: Hyperbolic-Parabolic (Navier-Stokes)**
# $$ \frac{d\mathbf{U}}{dt} = \mathcal{L}_{hyp}(\mathbf{U}) + \mathcal{L}_{par}(\mathbf{U}, \nabla \mathbf{U}) $$
# * **Constructor**: `SemidiscretizationHyperbolicParabolic`
# * **Inputs**: Requires both `equations` (Euler) and `equations_parabolic` (Diffusion).
# * **Role**:
#     1.  Computes the inviscid residuals.
#     2.  Computes the viscous residuals (gradients + fluxes) using the scheme defined in Cell 2 (e.g., Bassi-Rebay 1).
#     3.  Applies the split boundary conditions (Slip for Euler, No-Slip for Diffusion).
#
# Free-slip, no-penetration walls at y = 0 and y = 1.
# The x-direction is periodic through the P4estMesh connectivity.
boundary_conditions = Dict(
    :y_neg => boundary_condition_slip_wall,
    :y_pos => boundary_condition_slip_wall
)

semi = SemidiscretizationHyperbolic(mesh, equations,
                                    initial_condition_rayleigh_taylor, dg;
                                    boundary_conditions=boundary_conditions,
                                    source_terms=source_terms_gravity)
# %% [markdown]

# ## Cell 7: Dynamic Adaptive Mesh Refinement (AMR) Strategy
#
# ## Methodology: Feature-Driven $h$-Refinement
# Implements an automated feedback loop that dynamically adjusts the local grid resolution ($h$) based on the evolving flow physics. This minimizes computational cost by focusing degrees of freedom only where needed (e.g., boundary layers, vortices).
#
# 1.  **The Sensor: A Posteriori Error Indicator (`IndicatorLöhner`)**
#     * **Algorithm**: Uses the **Löhner Error Estimator**, a classic FEM/DG metric for compressible flows.
#     * **Mechanism**: It estimates the local smoothness of the solution by computing a normalized second derivative (curvature) of the **Velocity Magnitude**.
#     * **Role**: Acts as a "feature detector," identifying regions with high gradients (shear layers) or discontinuities that require higher resolution to resolve.
#
# 2.  **The Controller: Hysteresis Logic (`ControllerThreeLevel`)**
#     * Maps the continuous indicator value $\epsilon$ to discrete mesh operations (Refine/Coarsen/Keep) using a **Three-Level Hysteresis** approach to prevent "chattering" (rapid toggling between levels).
#     * **Refine Condition ($h \to h/2$)**: If error $\epsilon > 0.3$ (`max_threshold`). Splits the element into 4 children (quadtree).
#     * **Coarsen Condition ($h \to 2h$)**: If error $\epsilon < 0.1$ (`med_threshold`). Merges 4 children into 1 parent.
#     * **Deadband**: If $0.1 \le \epsilon \le 0.3$, the mesh is frozen to ensure stability.
#     * **Bounds**: Enforces hard constraints on resolution levels ($L_{min}=4, L_{max}=6$) to guarantee memory limits.
#
# 3.  **The Actuator: Temporal Intervention (`AMRCallback`)**
#     * **Trigger**: Interrupts the time-integration loop every 20 timesteps (`interval=20`).
#     * **Action**: Executes the mesh topology changes and performs the **L2-Projection** (interpolation) of the solution polynomial $\mathbf{U}$ from the old mesh to the new mesh.
#
# %%
amr_indicator = IndicatorLöhner(semi, variable=density_indicator)
# 2. Controller
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=4,
                                      med_level=5, med_threshold=0.1,
                                      max_level=6, max_threshold=0.3)
# 3. Callback
amr_callback = AMRCallback(semi, amr_controller,
                           interval=20,
                           adapt_initial_condition=true)
# %% [markdown]

# ## Cell 8: Data I/O & Surrogate Interoperability
# ## Strategy: Cross-Language Serialization (.npz)
#
# Saves the complete primitive state `(rho, u, v, p)` and the spatial
# coordinates every `dt_save = 0.1` time units for ground-truth data.
# `PlotData2D` reconstructs the adaptive DG solution on a structured grid.
# %%
npz_out = joinpath(@__DIR__, "data_npz")
mkpath(npz_out)
dt_save = 0.1
# Verification state is retained across callback invocations. The final
# summary cell uses verification_minima without inventing a ground truth.
initial_total_mass = Ref(NaN)
previous_energy_time = Ref(NaN)
previous_total_energy = Ref(NaN)
last_npz_save_time = Ref(NaN)
verification_minima = Dict{Symbol, Float64}()
@inline diagnostic_mass(u, equations) = u[1]
@inline diagnostic_total_energy(u, equations) = u[4]
# g = 1 and rho * g * v = g * rho_v2.
@inline diagnostic_gravity_work(u, equations) = u[3]
function record_verification_error!(name, value)
    if isfinite(value)
        verification_minima[name] =
            min(get(verification_minima, name, Inf), abs(value))
    end
    return nothing
end
function verify_npz_file(filename)
    data = npzread(filename)
    for key in ("t", "x", "y", "rho", "u", "v", "p")
        haskey(data, key) || error("NPZ sanity check: missing key $key in $filename")
    end
    n = length(data["rho"])
    complete_coverage = n > 0 &&
                        length(data["u"]) == n &&
                        length(data["v"]) == n &&
                        length(data["p"]) == n &&
                        length(data["x"]) == n &&
                        length(data["y"]) == n
    finite_state = all(key -> all(isfinite, data[key]),
                       ("rho", "u", "v", "p"))
    finite_coordinates = all(isfinite, data["x"]) &&
                         all(isfinite, data["y"])
    positive_state = all(value -> value > 0, data["rho"]) &&
                     all(value -> value > 0, data["p"])
    if !(complete_coverage && finite_state && finite_coordinates &&
         positive_state)
        error("NPZ sanity check failed for $filename: " *
              "coverage=$complete_coverage, finite_state=$finite_state, " *
              "finite_coordinates=$finite_coordinates, " *
              "positive_rho_p=$positive_state")
    end
    return nothing
end
function mirror_symmetry_error(x, y, rho)
    # Pair DG/plot nodes with their images under x -> 0.25 - x.
    # Rounding only supplies a tolerant dictionary key for coordinates
    # generated independently on opposite sides of the symmetric mesh.
    node_lookup = Dict{Tuple{Float64, Float64}, Int}()
    for k in eachindex(x)
        key = (round(x[k], digits=10), round(y[k], digits=10))
        node_lookup[key] = k
    end
    numerator = 0.0
    denominator = 0.0
    matched_nodes = 0
    for k in eachindex(x)
        mirror_key = (round(0.25 - x[k], digits=10),
                      round(y[k], digits=10))
        if haskey(node_lookup, mirror_key)
            mirror_k = node_lookup[mirror_key]
            difference = rho[k] - rho[mirror_k]
            numerator += difference^2
            denominator += rho[k]^2
            matched_nodes += 1
        end
    end
    if matched_nodes == 0 || denominator == 0
        return NaN
    end
    return sqrt(numerator / denominator)
end
function print_physical_diagnostics!(integrator, pd, rho, v, p)
    semi = integrator.p
    t = integrator.t
    # Use the DG quadrature for integral quantities rather than treating
    # nonuniformly spaced visualization nodes as a uniform grid.
    total_mass = Trixi.integrate(diagnostic_mass, integrator.u, semi)
    total_energy = Trixi.integrate(diagnostic_total_energy, integrator.u, semi)
    gravity_work = Trixi.integrate(diagnostic_gravity_work, integrator.u, semi)
    if !isfinite(initial_total_mass[])
        initial_total_mass[] = total_mass
    end
    mass_drift = total_mass - initial_total_mass[]
    relative_mass_drift = abs(mass_drift) /
                          max(abs(initial_total_mass[]), eps(Float64))
    x = vec(pd.x)
    y = vec(pd.y)
    rho_values = vec(rho)
    v_values = vec(v)
    p_values = vec(p)
    rho_min, rho_max = extrema(rho_values)
    p_min, p_max = extrema(p_values)
    symmetry_error = mirror_symmetry_error(x, y, rho_values)
    wall_velocity = 0.0
    max_abs_v = 0.0
    wall_samples = 0
    wall_tolerance = 1.0e-10
    for k in eachindex(y)
        max_abs_v = max(max_abs_v, abs(v_values[k]))
        if abs(y[k]) <= wall_tolerance ||
           abs(y[k] - 1.0) <= wall_tolerance
            wall_velocity = max(wall_velocity, abs(v_values[k]))
            wall_samples += 1
        end
    end
    if wall_samples == 0
        wall_velocity = NaN
        relative_wall_violation = NaN
    else
        relative_wall_violation = wall_velocity /
                                  max(max_abs_v, eps(Float64))
    end
    if isfinite(previous_energy_time[]) && t > previous_energy_time[]
        dE_dt = (total_energy - previous_total_energy[]) /
                (t - previous_energy_time[])
        energy_closure = abs(dE_dt - gravity_work) /
                         max(abs(dE_dt), abs(gravity_work), eps(Float64))
    else
        dE_dt = NaN
        energy_closure = NaN
    end
    record_verification_error!(:mass, relative_mass_drift)
    record_verification_error!(:symmetry, symmetry_error)
    record_verification_error!(:wall, relative_wall_violation)
    record_verification_error!(:energy_balance, energy_closure)
    println(Printf.@sprintf(
        "Diagnostics t=%.5f: dM=%.6e, rel_dM=%.6e, rho=[%.6e,%.6e], p=[%.6e,%.6e], sym=%.6e, wall_max_abs_v=%.6e, dE_dt=%.6e, gravity_work=%.6e, energy_closure=%.6e",
        t, mass_drift, relative_mass_drift,
        rho_min, rho_max, p_min, p_max, symmetry_error,
        wall_velocity, dE_dt, gravity_work, energy_closure))
    previous_energy_time[] = t
    previous_total_energy[] = total_energy
    return nothing
end
function save_npz_file(integrator)
    semi = integrator.p
    t = integrator.t
    # PeriodicCallback and its finalizer can both reach t_end.
    if isfinite(last_npz_save_time[]) &&
       abs(t - last_npz_save_time[]) <= 1.0e-10
        return nothing
    end
    pd = PlotData2D(integrator.u, semi, solution_variables=cons2prim)
    rho = pd.data[1]
    u = pd.data[2]
    v = pd.data[3]
    p = pd.data[4]
    filename = joinpath(npz_out, Printf.@sprintf("time_%08.5f.npz", t))
    npzwrite(filename, Dict("t" => t, "x" => pd.x, "y" => pd.y,
                            "rho" => rho, "u" => u, "v" => v, "p" => p))
    # Re-read the actual archive so the check covers serialization as well
    # as the in-memory PlotData2D reconstruction.
    verify_npz_file(filename)
    println("Saved and NPZ-verified: $(basename(filename))")
    print_physical_diagnostics!(integrator, pd, rho, v, p)
    last_npz_save_time[] = t
    return nothing
end
# Fixed-time saves at t = 0.0, 0.1, ..., 3.0. PeriodicCallback can skip
# an affect exactly at t_end, so explicitly serialize the final state.
function finalize_npz_save(callback, u, t, integrator)
    if integrator.t >= 3.0 - 8 * eps(3.0)
        save_npz_file(integrator)
    end
    return nothing
end
save_npz_callback = PeriodicCallback(save_npz_file, dt_save;
                                     initial_affect=true,
                                     save_positions=(false, false),
                                     finalize=finalize_npz_save)
# %% [markdown]


# ## Cell 9: ODE System Formulation & Time-Loop Control
#
# ## 1. Method of Lines (MOL) Finalization
# Converts the spatially discretized PDE and boundary conditions into a large system of Ordinary Differential Equations (ODEs).
# $$
# \frac{d \mathbf{U}}{dt} = \mathcal{L}(\mathbf{U}, t), \quad \mathbf{U}(0) = \mathbf{U}_0
# $$
# * **`semidiscretize`**: Assembles the global state vector $\mathbf{U}$ and the right-hand side operator $\mathcal{L}$, preparing the system for the LSRK integrator defined in Cell 1.
# * **Time Domain**: `tspan = (0.0, 3.0)` defines the simulation window $t \in [0, T_{end}]$.
#
# ## 2. Stability Control (CFL Condition)
# Explicit time-stepping requires strict adherence to the **Courant-Friedrichs-Lewy (CFL)** condition for stability.
# * **`StepsizeCallback`**: Dynamically adapts the timestep $\Delta t_n$ based on the maximum wave speed in the domain.
#     $$
#     \Delta t_n = \text{CFL} \cdot \min_{\Omega} \left( \frac{h}{|\mathbf{v}| + c} \right)
#     $$
#     Here, we set `cfl=0.5` to ensure the method remains within the stability region of the Runge-Kutta scheme.
#
# ## 3. Observability & Interventions (`CallbackSet`)
# Aggregates the diagnostic and control agents into the main solver loop:
# * **`AnalysisCallback`**: Computes in-situ functional quantities (Entropy, Kinetic Energy, $L_2/L_\infty$ errors) to monitor physical validity.
# * **`SaveSolutionCallback`**: Performs full-state serialization (HDF5-compatible) for restarts.
# * **`AMRCallback`**: (From Cell 7) Triggers mesh adaptation.
# %%
tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)
summary_callback = SummaryCallback()
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
alive_callback = AliveCallback(analysis_interval = analysis_interval)
# Save full-state snapshots at the required physical-time interval.
trixi_out = joinpath(@__DIR__, "trixi_output")
save_solution = SaveSolutionCallback(dt=dt_save,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     output_directory=trixi_out,
                                     solution_variables=cons2prim)
stepsize_callback = StepsizeCallback(cfl=0.4)
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        amr_callback,
                        save_npz_callback,stepsize_callback)
# %% [markdown]

# ## Cell 10: Time Integration Loop
# 
# ## Methodology: Explicit Time-Stepping
# Evolves the semi-discrete system $\frac{d\mathbf{U}}{dt} = \mathcal{L}(\mathbf{U})$ from $t_0$ to $t_{end}$.
# 
# ### 1. The Numerical Integrator: `RDPK3SpFSAL49`
# * **Algorithm**: A 3rd-order, Low-Storage Runge-Kutta (LSRK) method with 4 stages.
# * **Selection Rationale**:
#     * **Memory Efficiency**: As an LSRK method, it minimizes the number of state vector registers required ($2N$ storage vs $4N+$ for classic RK4), which is critical for high-DOF spectral element methods.
#     * **Stability**: Optimized for spectra containing both imaginary eigenvalues (advection) and negative real eigenvalues (diffusion).
# 
# ### 2. Adaptive Control Strategy
# * **`adaptive=false`**: We disable the ODE solver's internal error-based adaptivity (PID control).
#     * **Reasoning**: For PDEs, local truncation error control is computationally expensive and often poor at handling shocks.
#     * **Alternative**: We rely entirely on the `StepsizeCallback` (Cell 9), which controls $\Delta t$ based on the physical **CFL condition** (wave propagation speed).

# %%
sol = solve(ode, RDPK3SpFSAL49(); 
            dt=1.0,             # 
            adaptive=false,     # 
            callback=callbacks, 
            saveat = 0.01, 
            maxiters = 1e7);

# %% [markdown]

# ## Cell 11: Data Regularization (Rasterization)
#
# ## Methodology: Nearest-Neighbor Lattice Projection
# Converts the unstructured, scattered dataset $\mathcal{D} = \{(x_k, y_k, z_k)\}$ generated by the DGSEM quadrature nodes onto a regular Euclidean lattice (image) $\mathbf{G} \in \mathbb{R}^{N_y \times N_x}$.
#
# ### 1. The Problem: Unstructured vs. Structured
# * **Source**: The solver outputs data on **Legendre-Gauss-Lobatto (LGL)** nodes within adaptive quadtrees. The spatial distribution is highly non-uniform ($h$-refinement clusters points in shock regions).
# * **Target**: Visualization tools and standard Convolutional Neural Networks (CNNs) require uniform Cartesian grids (matrices).
#
# ### 2. The Algorithm: 0-th Order Binning
# We employ a high-speed **Point-to-Grid Binning** strategy (0-th order interpolation).
# * **Grid Construction**: We define a target resolution $(N_x, N_y)$ covering the domain $\Omega = [0, 1]^2$.
# * **Index Mapping**: For every Lagrangian point $(x_k, y_k)$, we compute its discrete Eulerian coordinates $(i, j)$:
#     $$
#     i = \left\lfloor \frac{x_k - x_{min}}{\Delta x} \right\rfloor + 1, \quad j = \left\lfloor \frac{y_k - y_{min}}{\Delta y} \right\rfloor + 1
#     $$
# * **Assignment**: The value $z_k$ is assigned to the pixel $G_{ji}$.
#     * *Note*: This method is essentially a "Splatting" technique. It is computationally cheaper than high-order polynomial interpolation (rendering `PlotData2D`) but provides a rapid approximation for real-time monitoring.
# %%
function rasterize_data(x, y, z, nx=512, ny=512)
    x_min, x_max = 0.0, 0.25
    y_min, y_max = 0.0, 1.0
    nx > 0 && ny > 0 || error("Visualization grid dimensions must be positive")

    x_source = Float64.(vec(x))
    y_source = Float64.(vec(y))
    z_source = Float64.(vec(z))
    n_source = length(z_source)

    n_source > 0 || error("Cannot rasterize an empty solution")
    length(x_source) == n_source || error("x/z data lengths do not match")
    length(y_source) == n_source || error("y/z data lengths do not match")
    all(isfinite, x_source) && all(isfinite, y_source) &&
        all(isfinite, z_source) ||
        error("Cannot rasterize non-finite solution data")

    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny

    # There are no NaN padding rows: every pixel is filled below.
    grid = fill(NaN, ny, nx)
    nearest_distance = fill(Inf, ny, nx)

    # Seed occupied pixels with the source node nearest to each pixel center.
    # If several nodes occupy one pixel, this avoids order-dependent overwrites.
    for k in 1:n_source
        i = clamp(floor(Int, (x_source[k] - x_min) / dx) + 1, 1, nx)
        j = clamp(floor(Int, (y_source[k] - y_min) / dy) + 1, 1, ny)
        x_center = x_min + (i - 0.5) * dx
        y_center = y_min + (j - 0.5) * dy
        distance = (x_source[k] - x_center)^2 +
                   (y_source[k] - y_center)^2

        if distance < nearest_distance[j, i]
            nearest_distance[j, i] = distance
            grid[j, i] = z_source[k]
        end
    end

    # Multi-source nearest-neighbor propagation supplies an interpolated value
    # to pixels between DG visualization nodes, including every AMR leaf-cell
    # region. Each previously empty pixel is assigned exactly once.
    queue_i = Int[]
    queue_j = Int[]
    sizehint!(queue_i, nx * ny)
    sizehint!(queue_j, nx * ny)

    for j in 1:ny, i in 1:nx
        if !isnan(grid[j, i])
            push!(queue_i, i)
            push!(queue_j, j)
        end
    end

    head = 1
    while head <= length(queue_i)
        i = queue_i[head]
        j = queue_j[head]
        head += 1
        value = grid[j, i]

        for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
            neighbor_i = i + di
            neighbor_j = j + dj
            if 1 <= neighbor_i <= nx && 1 <= neighbor_j <= ny &&
               isnan(grid[neighbor_j, neighbor_i])
                grid[neighbor_j, neighbor_i] = value
                push!(queue_i, neighbor_i)
                push!(queue_j, neighbor_j)
            end
        end
    end

    all(isfinite, grid) ||
        error("Uniform visualization grid reconstruction left gaps")
    return grid
end
# %% [markdown]

# ## Cell 12: Post-Processing & Temporal Diagnostics
#
# ## Methodology: Lagrangian Data Visualization
# Visualizes the raw discrete solution trajectory $\mathbf{U}(t)$ at distinct temporal snapshots to validate the physical consistency of the simulation.
#
# ### 1. Data Representation (Scatter vs. Contour)
# * **Approach**: We utilize **Scatter Plots** of the raw quadrature nodes rather than contour interpolation.
# * **Mathematical Motivation**: By plotting the exact nodal values $(x_k, y_k, u_k)$, we expose the underlying spectral element discretization (GLL nodes). This reveals the effectiveness of the AMR (clustering of points in shock regions) and prevents "smoothing artifacts" that might hide grid-scale oscillations (Gibbs phenomenon).
#
# ### 2. Phase Space Trajectory
# We extract snapshots at $t_0, t_{1/3}, t_{2/3}, t_{end}$ to monitor the evolution of the primitive variables vector $\mathbf{W}$:
# $$
# \mathbf{W} = [\rho, u, v, p]^T
# $$
# * **Density ($\rho$)**: Monitored to verify mass conservation and check for vacuum states (positivity preservation).
# * **Velocity ($u, v$)**: Visualizes the development of the primary vortex and boundary layer separation.
# * **Pressure ($p$)**: Checks for acoustic wave reflections at boundaries.
#
# ### 3. Composite Output (`summary_all.png`)
# The final artifact is a $4 \times 4$ matrix aggregation of these fields. This image serves as the primary "Observation" ($O_{final}$) for the ATHENA Advisor agent to diagnose the success or failure of the simulation.
# """
println("Generating Rayleigh–Taylor summary plots...")
plots_array = []
plot_times = (0.5, 1.0, 2.0, 3.0)
# Eight samples per finest level-6 AMR element in each direction.
vis_nx, vis_ny = 512, 512
vis_x = range(0.25 / (2 * vis_nx),
              0.25 - 0.25 / (2 * vis_nx); length=vis_nx)
vis_y = range(1.0 / (2 * vis_ny),
              1.0 - 1.0 / (2 * vis_ny); length=vis_ny)
style = (
    aspect_ratio = :equal,
    xlims = (0.0, 0.25),
    ylims = (0.0, 1.0),
    xticks = ([0.0, 0.125, 0.25], ["0", "0.125", "0.25"]),
    yticks = 0.0:0.2:1.0,
    xlabel = "x",
    ylabel = "y",
    framestyle = :box,
    grid = false,
    legend = false,
    colorbar = false,
    linewidth = 0,       # No element-edge strokes.
    margin = 2Plots.mm
)
# Field-major ordering: four density panels, then four vertical-velocity panels.
for (field, field_label) in (("rho", "ρ"), ("v", "v"))
    for t_plot in plot_times
        filename = joinpath(npz_out,
                            Printf.@sprintf("time_%08.5f.npz", t_plot))
        println(filename)
        verify_npz_file(filename)
        data = npzread(filename)
        # No clims: every field and snapshot receives automatic color scaling.
        field_grid = rasterize_data(data["x"], data["y"], data[field],
                                    vis_nx, vis_ny)
        t_val = data["t"]
        push!(plots_array,
              heatmap(vis_x, vis_y, field_grid;
                      title="$(field_label), t=$(round(t_val, digits=2))",
                      style...))
    end
end
println("Saving summary_all.png...")
final_plot = plot(plots_array..., layout=(2, 4),
                  size=(2000, 1200), margin=5Plots.mm)
savefig(final_plot, joinpath(@__DIR__, "summary_all.png"))
# The heavy/light threshold is the midpoint of the initial density contrast.
function spike_tip_position(data)
    rho = vec(data["rho"])
    y = vec(data["y"])
    heavy_nodes = rho .>= 1.5
    return any(heavy_nodes) ? maximum(y[heavy_nodes]) : NaN
end
function compare_polydeg3_archives!()
    archive_directories = String[]
    # Never interpret the newly written polydeg=4 directory as an archive.
    for path in readdir(@__DIR__; join=true)
        name = lowercase(basename(path))
        if isdir(path) && abspath(path) != abspath(npz_out) &&
           ((occursin("polydeg", name) && occursin("3", name)) ||
            occursin("p3", name))
            push!(archive_directories, path)
        end
    end
    for archive_directory in archive_directories
        compared = false
        for save_index in 0:round(Int, 3.0 / dt_save)
            t = save_index * dt_save
            filename = Printf.@sprintf("time_%08.5f.npz", t)
            p4_filename = joinpath(npz_out, filename)
            p3_filename = joinpath(archive_directory, filename)
            if isfile(p4_filename) && isfile(p3_filename)
                p4_data = npzread(p4_filename)
                p3_data = npzread(p3_filename)
                required_keys = ("x", "y", "rho")
                if !all(key -> haskey(p4_data, key) && haskey(p3_data, key),
                        required_keys)
                    continue
                end
                # Compare both solutions on the same gap-free lattice.
                rho_p4 = rasterize_data(p4_data["x"], p4_data["y"],
                                        p4_data["rho"], 256, 256)
                rho_p3 = rasterize_data(p3_data["x"], p3_data["y"],
                                        p3_data["rho"], 256, 256)
                denominator = sqrt(sum(abs2, rho_p3))
                if denominator > 0
                    discrepancy = sqrt(sum(abs2, rho_p4 - rho_p3)) /
                                  denominator
                    record_verification_error!(:discretization_convergence,
                                               discrepancy)
                    tip_p4 = spike_tip_position(p4_data)
                    tip_p3 = spike_tip_position(p3_data)
                    println(Printf.@sprintf(
                        "Discretization comparison t=%.5f: relative_rho_L2=%.6e, spike_tip_p4=%.6e, spike_tip_p3=%.6e, abs_tip_difference=%.6e",
                        t, discrepancy, tip_p4, tip_p3,
                        abs(tip_p4 - tip_p3)))
                    compared = true
                end
            end
        end
        if compared
            return nothing
        end
    end
    println("No accessible archived polydeg=3 NPZ states with common save times; " *
            "skipping discretization-convergence comparison.")
    return nothing
end
function print_verification_summary()
    best_metric = :none
    best_value = Inf
    for (metric, value) in verification_minima
        if isfinite(value) && value < best_value
            best_metric = metric
            best_value = value
        end
    end
    if isfinite(best_value)
        println("Minimum achieved error: ",
                Printf.@sprintf("%.6e", best_value),
                " (", best_metric, "); verification minima=",
                verification_minima)
    else
        println("Minimum achieved error: unavailable " *
                "(no finite verification metrics)")
    end
    return nothing
end
compare_polydeg3_archives!()
println("Done.")
print_verification_summary()