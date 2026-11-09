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

# Reinitialize Polyester's per-thread task buffers for the active thread count.
# Without this, Trixi's threaded kernels throw `UndefRefError` in ManualMemory.jl
# when Julia is launched with multiple threads (--threads > 1).
Trixi.Polyester.reset_threads!()

# %% [markdown]

# ## Cell 2: Physics Model Definition (Compressible Navier-Stokes)
# ## Governing Equations: Inviscid Compressible Euler System
#
# Defines the two-dimensional compressible Euler equations for conservation
# of mass, momentum, and total energy:
# $$
# \frac{\partial \mathbf{u}}{\partial t}
# + \nabla \cdot \mathbf{f}_{inv}(\mathbf{u}) = \mathbf{S}(\mathbf{u}).
# $$
#
# * **Equation of State**: Ideal gas with $\gamma = 1.4$.
# * **Physics Regime**: Inviscid and non-conducting ($\mu = 0$, $k = 0$),
#   so no parabolic diffusion model is required.
# * **Source Term**: The gravitational body force for the Rayleigh–Taylor
#   instability is supplied separately to the semidiscretization.

# %%
equations = CompressibleEulerEquations2D(1.4)
# %% [markdown]

# ## Cell 3: Initial Conditions and Source Terms (Lid-Driven Cavity)
# ## Problem Setup: Compressible Rayleigh–Taylor Instability
#
# Initializes a heavy fluid below a light fluid, with the material interface
# smoothed around $y = 0.5$. A single-mode vertical-velocity perturbation
# seeds the instability. Gravity acts in the positive $y$ direction.
#
# The primitive state $(\rho, u, v, p)$ is converted to the conserved state
# $(\rho, \rho u, \rho v, E)$ required by the Euler solver.

# %%
@inline function initial_condition_rti(x, t, equations::CompressibleEulerEquations2D)
    x_coord, y = x

    # Heavy fluid ($\rho = 2$) below the interface; light fluid ($\rho = 1$) above.
    rho = 2.0 - 0.5 * (1.0 + tanh(50.0 * (y - 0.5)))

    # Continuous piecewise pressure profile.
    p = y < 0.5 ? 2.0 * y + 1.0 : y + 1.5

    u = 0.0
    sound_speed = sqrt(equations.gamma * p / rho)
    v = -0.025 * sound_speed * cos(8.0 * pi * x_coord) * sin(pi * y)^6

    # prim2cons forms E = p / (gamma - 1) + rho * (u^2 + v^2) / 2.
    return prim2cons(SVector(rho, u, v, p), equations)
end

@inline function source_terms_gravity(u, x, t,
                                      equations::CompressibleEulerEquations2D)
    g = 1.0
    rho = u[1]
    rho_v = u[3]

    # Since rho_v = rho * v, the final component is rho * g * v.
    return SVector(0.0, 0.0, g * rho, g * rho_v)
end

# Density sensor shared by shock capturing and adaptive mesh refinement.
@inline function density_indicator(u, equations)
    return u[1]
end
# %% [markdown]

# ## Cell 4: Spatial Discretization & Hybrid Shock Capturing
# ## Methodology: Nodal Discontinuous Galerkin (DGSEM)
# Constructs the spatial operator $\mathcal{L}(\mathbf{U})$ using the
# **Discontinuous Galerkin Spectral Element Method (DGSEM)**.
#
# 1. **Function Space (`LobattoLegendreBasis`)**:
#     * Approximates solutions using tensor-product polynomials of degree
#       $N=5$ on Legendre-Gauss-Lobatto (LGL) nodes.
#     * This nodal basis satisfies the Summation-by-Parts (SBP) property.
#
# 2. **Shock Capturing Strategy (Hybrid DG-FV)**:
#     * Blends the entropy-conserving high-order DG flux `flux_ranocha`
#       with the robust finite-volume flux `flux_lax_friedrichs`.
#     * The density-based indicator retains shock protection, while the
#       blend-fraction cap reduces dissipation along smooth interfaces.
#
# 3. **Interface Flux (`surface_flux=flux_hllc`)**:
#     * HLLC explicitly restores the contact wave, reducing numerical
#       dissipation of the density interface and shear layers.
#
# 4. **Indicator Function (`IndicatorHennemannGassner`)**:
#     * Computes an element-wise blending coefficient from the spectral
#       decay of the density indicator.
#     * `alpha_max=0.5` limits local FV blending instead of disabling
#       shock capturing globally.
#
# %%
polydeg = 5
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_indicator)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=flux_ranocha,
                                                 volume_flux_fv=flux_lax_friedrichs)
dg = DGSEM(polydeg=polydeg,
           surface_flux=flux_hllc,
           volume_integral=volume_integral)
# %% [markdown]


# ## Cell 5: Computational Domain & Adaptive Mesh Topology
# ## Methodology: Forest-of-Octrees (`p4est`)
# Defines the RTI domain using a dynamic quadtree mesh that supports AMR.
#
# * **Domain**: $\Omega = [0, 0.25] \times [0, 1]$.
# * **Topology**: One root tree with uniform initial refinement level 3.
# * **Boundaries**: Periodic in $x$; nonperiodic in $y$, where reflective/
#   free-slip wall conditions will be applied.

# %%
coordinates_min = (0.0, 0.0)
coordinates_max = (0.25, 1.0)

mesh = P4estMesh((1, 1), polydeg=polydeg,
                 coordinates_min=coordinates_min,
                 coordinates_max=coordinates_max,
                 initial_refinement_level=3,
                 periodicity=(true, false))

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
# ## Boundary Conditions and Hyperbolic System Assembly
#
# The $x$-boundaries are periodic through the mesh topology. At $y=0$ and
# $y=1$, `boundary_condition_slip_wall` imposes reflective/free-slip Euler
# wall conditions: zero normal velocity with mirrored density, tangential
# momentum, and energy.
#
# The inviscid spatial operator is assembled together with the RTI initial
# condition and gravitational source term. No parabolic operator is required.

# %%
boundary_conditions = Dict(
    :y_neg => boundary_condition_slip_wall,
    :y_pos => boundary_condition_slip_wall
)

semi = SemidiscretizationHyperbolic(mesh, equations,
                                    initial_condition_rti, dg;
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
#     * **Mechanism**: It estimates the local smoothness of the solution by computing a normalized second derivative (curvature) of the **density**.
#     * **Role**: Acts as a "feature detector," identifying the density interface, filaments, and other RTI structures that require higher resolution.
#
# 2.  **The Controller: Hysteresis Logic (`ControllerThreeLevel`)**
#     * Maps the continuous indicator value $\epsilon$ to discrete mesh operations (Refine/Coarsen/Keep) using a **Three-Level Hysteresis** approach to prevent "chattering" (rapid toggling between levels).
#     * **Refine Condition ($h \to h/2$)**: If error $\epsilon > 0.3$ (`max_threshold`). Splits the element into 4 children (quadtree).
#     * **Coarsen Condition ($h \to 2h$)**: If error $\epsilon < 0.01$ (`med_threshold`). Merges 4 children into 1 parent.
#     * **Deadband**: If $0.01 \le \epsilon \le 0.3$, the mesh is frozen to ensure stability and prevent premature interface coarsening.
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
                                      med_level=5, med_threshold=0.01,
                                      max_level=6, max_threshold=0.3)
# 3. Callback
amr_callback = AMRCallback(semi, amr_controller,
                           interval=20,
                           adapt_initial_condition=true)
# %% [markdown]


# ## Cell 8: Data I/O & Surrogate Interoperability
# ## Strategy: Cross-Language Serialization (.npz)
# Exports the complete conserved state on a structured PlotData2D grid.
# Ground-truth snapshots are written every 0.1 time units, with additional
# snapshots at the requested visualization times. Each file also contains
# coordinates, time, AMR element metadata, output-grid metadata, and diagnostics.

# %%
using DiffEqCallbacks: PresetTimeCallback

npz_out = joinpath(@__DIR__, "data_npz")
mkpath(npz_out)

dt_save = 0.1
visualization_times = [0.75, 1.5, 2.25, 3.0]
npz_save_times = sort(unique(vcat(collect(0.0:dt_save:3.0),
                                  visualization_times)))

# State used to accumulate diagnostics between output times.
rti_initial_mass = Ref(NaN)
rti_initial_energy = Ref(NaN)
rti_previous_time = Ref(NaN)
rti_previous_work_rate = Ref(NaN)
rti_accumulated_work = Ref(0.0)

function reset_rti_diagnostics!()
    rti_initial_mass[] = NaN
    rti_initial_energy[] = NaN
    rti_previous_time[] = NaN
    rti_previous_work_rate[] = NaN
    rti_accumulated_work[] = 0.0
    return nothing
end
reset_rti_diagnostics!()

# Extract element-wise AMR metadata without relying on solver-specific
# refinement-level fields. For this single-tree p4est mesh, each refinement
# level halves both physical element widths.
function rti_amr_metadata(semi)
    _, _, _, cache = Trixi.mesh_equations_solver_cache(semi)
    node_coordinates = cache.elements.node_coordinates
    n_elements = size(node_coordinates, 4)

    levels = zeros(Int, n_elements)
    centers_x = zeros(n_elements)
    centers_y = zeros(n_elements)
    sizes_x = zeros(n_elements)
    sizes_y = zeros(n_elements)

    for element in 1:n_elements
        x_nodes = view(node_coordinates, 1, :, :, element)
        y_nodes = view(node_coordinates, 2, :, :, element)
        x_min, x_max = extrema(x_nodes)
        y_min, y_max = extrema(y_nodes)

        hx = x_max - x_min
        hy = y_max - y_min
        centers_x[element] = 0.5 * (x_min + x_max)
        centers_y[element] = 0.5 * (y_min + y_max)
        sizes_x[element] = hx
        sizes_y[element] = hy

        level_x = log2((coordinates_max[1] - coordinates_min[1]) / hx)
        level_y = log2((coordinates_max[2] - coordinates_min[2]) / hy)
        levels[element] = round(Int, 0.5 * (level_x + level_y))
    end

    return (levels=levels,
            centers_x=centers_x, centers_y=centers_y,
            sizes_x=sizes_x, sizes_y=sizes_y)
end

function rti_diagnostics(u_ode, semi)
    mesh, eq, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    u = Trixi.wrap_array(u_ode, semi)

    min_rho = Inf
    min_pressure = Inf
    max_abs_v_y0 = 0.0
    max_abs_v_y1 = 0.0
    wall_tolerance = 1.0e-12

    for element in Trixi.eachelement(solver, cache)
        for j in Trixi.eachnode(solver), i in Trixi.eachnode(solver)
            u_local = Trixi.get_node_vars(u, eq, solver, i, j, element)
            rho_local = u_local[1]
            rho_u_local = u_local[2]
            rho_v_local = u_local[3]
            energy_local = u_local[4]

            pressure_local = (eq.gamma - 1) *
                             (energy_local -
                              0.5 * (rho_u_local^2 + rho_v_local^2) /
                              rho_local)
            v_local = rho_v_local / rho_local
            y_node = cache.elements.node_coordinates[2, i, j, element]

            min_rho = min(min_rho, rho_local)
            min_pressure = min(min_pressure, pressure_local)

            if abs(y_node - coordinates_min[2]) <= wall_tolerance
                max_abs_v_y0 = max(max_abs_v_y0, abs(v_local))
            elseif abs(y_node - coordinates_max[2]) <= wall_tolerance
                max_abs_v_y1 = max(max_abs_v_y1, abs(v_local))
            end
        end
    end

    # Trixi's DG quadrature includes the physical element Jacobian.
    total_mass = Trixi.integrate(
        (u_local, i, j, element, equations, solver) -> u_local[1],
        u, mesh, eq, solver, cache)
    total_energy = Trixi.integrate(
        (u_local, i, j, element, equations, solver) -> u_local[4],
        u, mesh, eq, solver, cache)

    # g=1 and rho_v=rho*v, so this is integral(rho*g*v) dV.
    gravitational_work_rate = Trixi.integrate(
        (u_local, i, j, element, equations, solver) -> u_local[3],
        u, mesh, eq, solver, cache)

    return (min_rho=min_rho,
            min_pressure=min_pressure,
            max_abs_v_y0=max_abs_v_y0,
            max_abs_v_y1=max_abs_v_y1,
            total_mass=total_mass,
            total_energy=total_energy,
            gravitational_work_rate=gravitational_work_rate)
end

function save_npz_file(integrator)
    semi = integrator.p
    t = integrator.t

    # Keep the conserved variables: rho, rho*u, rho*v, and E.
    pd = PlotData2D(integrator.u, semi; solution_variables=cons2cons)

    rho = getindex.(pd.data, 1)
    rho_u = getindex.(pd.data, 2)
    rho_v = getindex.(pd.data, 3)
    total_energy = getindex.(pd.data, 4)

    diagnostics = rti_diagnostics(integrator.u, semi)

    if isnan(rti_previous_time[])
        rti_initial_mass[] = diagnostics.total_mass
        rti_initial_energy[] = diagnostics.total_energy
        rti_accumulated_work[] = 0.0
    else
        dt_diagnostic = t - rti_previous_time[]
        if dt_diagnostic > 0
            rti_accumulated_work[] +=
                0.5 * dt_diagnostic *
                (rti_previous_work_rate[] +
                 diagnostics.gravitational_work_rate)
        end
    end

    rti_previous_time[] = t
    rti_previous_work_rate[] = diagnostics.gravitational_work_rate

    mass_drift = abs(diagnostics.total_mass - rti_initial_mass[]) /
                 abs(rti_initial_mass[])
    energy_change = diagnostics.total_energy - rti_initial_energy[]
    energy_work_residual = energy_change - rti_accumulated_work[]
    metadata = rti_amr_metadata(semi)

    filename = joinpath(npz_out, Printf.@sprintf("time_%08.5f.npz", t))

    npzwrite(filename, Dict{String, Any}(
        "t" => t,
        "time" => t,
        "x" => pd.x,
        "y" => pd.y,
        "rho" => rho,
        "rho_u" => rho_u,
        "rho_v" => rho_v,
        "E" => total_energy,
        "dt_save" => dt_save,
        "polydeg" => polydeg,
        "domain_min" => collect(coordinates_min),
        "domain_max" => collect(coordinates_max),
        "output_grid_shape" => collect(size(pd.x)),
        "output_grid_conservative_shape" => collect(size(rho)),
        "amr_element_count" => length(metadata.levels),
        "amr_level_min" => minimum(metadata.levels),
        "amr_level_max" => maximum(metadata.levels),
        "amr_levels" => metadata.levels,
        "amr_element_centers_x" => metadata.centers_x,
        "amr_element_centers_y" => metadata.centers_y,
        "amr_element_sizes_x" => metadata.sizes_x,
        "amr_element_sizes_y" => metadata.sizes_y,
        "min_rho" => diagnostics.min_rho,
        "min_pressure" => diagnostics.min_pressure,
        "mass_drift" => mass_drift,
        "wall_max_abs_v_y0" => diagnostics.max_abs_v_y0,
        "wall_max_abs_v_y1" => diagnostics.max_abs_v_y1,
        "total_mass" => diagnostics.total_mass,
        "total_energy_integral" => diagnostics.total_energy,
        "energy_change" => energy_change,
        "gravitational_work_rate" => diagnostics.gravitational_work_rate,
        "gravitational_work" => rti_accumulated_work[],
        "energy_work_residual" => energy_work_residual))

    Printf.@printf("RTI diagnostics t=%.5f: min_rho=%.6e, min_p=%.6e, mass_drift=%.6e, max_abs_v_y0=%.6e, max_abs_v_y1=%.6e, dE=%.6e, W_g=%.6e, dE-W_g=%.6e\n",
                   t, diagnostics.min_rho, diagnostics.min_pressure,
                   mass_drift, diagnostics.max_abs_v_y0,
                   diagnostics.max_abs_v_y1, energy_change,
                   rti_accumulated_work[], energy_work_residual)
    println("Saved full conservative state (rho, rho_u, rho_v, E), coordinates, time, and AMR/output-grid metadata: $(basename(filename))")
    return nothing
end

println("NPZ output confirmed: full conservative state is saved at dt_save=$dt_save, including coordinates, times, and AMR/output-grid metadata.")

# The callback inserts the requested times into the integrator's time stops.
save_npz_callback = PresetTimeCallback(npz_save_times, save_npz_file;
                                       save_positions=(false, false))
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
# Fixed physical-time interval for output
dt_save = 0.1

# Save the complete conserved state at fixed physical-time intervals.
trixi_out = joinpath(@__DIR__, "trixi_output")

save_solution = SaveSolutionCallback(dt=dt_save,
                                     save_initial_solution=true,
                                     output_directory=trixi_out,
                                     solution_variables=cons2cons)
stepsize_callback = StepsizeCallback(cfl=0.5)
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        amr_callback,
                        save_npz_callback,stepsize_callback)
# %% [markdown]

# ## Cell 10: Time Integration Loop
# ## Methodology: Explicit Time-Stepping
# Evolves the semi-discrete system from $t_0$ to $t_{end}$ using the
# low-storage Runge-Kutta method RDPK3SpFSAL49. Internal error-based
# adaptivity remains disabled; the StepsizeCallback enforces CFL control.
#
# After the polydeg=5 deliverable run, one polydeg=3 HLLC verification run
# is performed with identical AMR. Interface diagnostics are based on the
# density field: the amplitude uses the vertical extent of the rho=1.5
# contour neighborhood, and the mixing zone uses 1.05 < rho < 1.95.
# The early-time growth rate is a sanity check, not an exact error norm.

# %%
reset_rti_diagnostics!()
println("Starting polydeg=5 HLLC deliverable run...")

sol = solve(ode, RDPK3SpFSAL49();
            dt=1.0,
            adaptive=false,
            callback=callbacks,
            saveat=0.01,
            maxiters=1e7)

# Density-based measures used for the resolution comparison.
function rti_interface_measures(x, y, rho)
    x_vec = vec(x)
    y_vec = vec(y)
    rho_vec = vec(rho)

    finite_mask = isfinite.(x_vec) .& isfinite.(y_vec) .&
                  isfinite.(rho_vec)

    # A narrow density band tracks the deforming material interface.
    contour_mask = finite_mask .& (abs.(rho_vec .- 1.5) .<= 0.05)
    if count(contour_mask) >= 2
        y_contour = y_vec[contour_mask]
        interface_amplitude =
            0.5 * (maximum(y_contour) - minimum(y_contour))
    else
        interface_amplitude = NaN
    end

    # The broader band measures the vertical extent of the mixing zone.
    mixing_mask = finite_mask .& (rho_vec .> 1.05) .&
                  (rho_vec .< 1.95)
    if count(mixing_mask) >= 2
        y_mixing = y_vec[mixing_mask]
        mixing_zone_extent = maximum(y_mixing) - minimum(y_mixing)
    else
        mixing_zone_extent = NaN
    end

    return (interface_amplitude=interface_amplitude,
            mixing_zone_extent=mixing_zone_extent)
end

function rti_fit_growth_rate(history; t_min=0.2, t_max=0.75)
    selected = filter(history) do entry
        t_min - 1.0e-12 <= entry.t <= t_max + 1.0e-12 &&
        isfinite(entry.interface_amplitude) &&
        entry.interface_amplitude > 0
    end

    if length(selected) < 2
        return NaN
    end

    times = [entry.t for entry in selected]
    log_amplitudes =
        log.([entry.interface_amplitude for entry in selected])

    # Least-squares fit of log(amplitude) = constant + sigma*t.
    design_matrix = hcat(ones(length(times)), times)
    coefficients = design_matrix \ log_amplitudes
    return coefficients[2]
end

function rti_measure_at_time(history, target_time)
    index = argmin([abs(entry.t - target_time) for entry in history])
    return history[index]
end

# Construct the polydeg=3 verification discretization. Only the polynomial
# degree changes; HLLC, HG blending, AMR, CFL, and physics remain identical.
verification_polydeg = 3
verification_basis = LobattoLegendreBasis(verification_polydeg)
verification_indicator_sc =
    IndicatorHennemannGassner(equations, verification_basis,
                              alpha_max=0.5,
                              alpha_min=0.001,
                              alpha_smooth=true,
                              variable=density_indicator)
verification_volume_integral =
    VolumeIntegralShockCapturingHG(verification_indicator_sc;
                                   volume_flux_dg=flux_ranocha,
                                   volume_flux_fv=flux_lax_friedrichs)
verification_dg =
    DGSEM(polydeg=verification_polydeg,
          surface_flux=flux_hllc,
          volume_integral=verification_volume_integral)

verification_mesh =
    P4estMesh((1, 1), polydeg=verification_polydeg,
              coordinates_min=coordinates_min,
              coordinates_max=coordinates_max,
              initial_refinement_level=3,
              periodicity=(true, false))

verification_semi =
    SemidiscretizationHyperbolic(verification_mesh, equations,
                                 initial_condition_rti, verification_dg;
                                 boundary_conditions=boundary_conditions,
                                 source_terms=source_terms_gravity)

verification_amr_indicator =
    IndicatorLöhner(verification_semi, variable=density_indicator)
verification_amr_controller =
    ControllerThreeLevel(verification_semi, verification_amr_indicator,
                         base_level=4,
                         med_level=5, med_threshold=0.01,
                         max_level=6, max_threshold=0.3)
verification_amr_callback =
    AMRCallback(verification_semi, verification_amr_controller,
                interval=20,
                adapt_initial_condition=true)

verification_measure_times =
    sort(unique(vcat(collect(0.0:dt_save:3.0), visualization_times)))
verification_history = NamedTuple[]

function save_verification_measure(integrator)
    pd = PlotData2D(integrator.u, integrator.p;
                    solution_variables=cons2cons)
    rho = getindex.(pd.data, 1)
    measures = rti_interface_measures(pd.x, pd.y, rho)

    push!(verification_history,
          (t=integrator.t,
           interface_amplitude=measures.interface_amplitude,
           mixing_zone_extent=measures.mixing_zone_extent))
    return nothing
end

verification_measure_callback =
    PresetTimeCallback(verification_measure_times,
                       save_verification_measure;
                       save_positions=(false, false))

verification_ode = semidiscretize(verification_semi, tspan)
verification_summary = SummaryCallback()
verification_analysis =
    AnalysisCallback(verification_semi, interval=analysis_interval)
verification_alive =
    AliveCallback(analysis_interval=analysis_interval)
verification_output =
    joinpath(@__DIR__, "trixi_output_polydeg3")
verification_save_solution =
    SaveSolutionCallback(dt=dt_save,
                         save_initial_solution=true,
                         output_directory=verification_output,
                         solution_variables=cons2cons)
verification_stepsize = StepsizeCallback(cfl=0.5)

verification_callbacks =
    CallbackSet(verification_summary,
                verification_analysis,
                verification_alive,
                verification_save_solution,
                verification_amr_callback,
                verification_measure_callback,
                verification_stepsize)

println("Starting polydeg=3 HLLC resolution-verification run with identical AMR...")
verification_sol = solve(verification_ode, RDPK3SpFSAL49();
                         dt=1.0,
                         adaptive=false,
                         callback=verification_callbacks,
                         saveat=0.01,
                         maxiters=1e7)

# Recover the same measures from the deliverable's saved full-state files.
deliverable_history = NamedTuple[]
for t_target in verification_measure_times
    filename = joinpath(npz_out,
                        Printf.@sprintf("time_%08.5f.npz", t_target))
    isfile(filename) || error("Missing deliverable snapshot: $filename")

    data = npzread(filename)
    measures = rti_interface_measures(data["x"], data["y"], data["rho"])
    push!(deliverable_history,
          (t=t_target,
           interface_amplitude=measures.interface_amplitude,
           mixing_zone_extent=measures.mixing_zone_extent))
end

deliverable_growth = rti_fit_growth_rate(deliverable_history)
verification_growth = rti_fit_growth_rate(verification_history)
linear_growth_rate = sqrt((1.0 / 3.0) * 1.0 * (8.0 * pi))

deliverable_t3 = rti_measure_at_time(deliverable_history, 3.0)
verification_t3 = rti_measure_at_time(verification_history, 3.0)

Printf.@printf("Resolution verification, early-time growth fit over 0.2 <= t <= 0.75: sigma_polydeg5=%.6e, sigma_polydeg3=%.6e, linear_RTI_sanity=%.6e\n",
               deliverable_growth, verification_growth,
               linear_growth_rate)
Printf.@printf("Resolution verification at t=3: polydeg5 amplitude=%.6e, mixing_extent=%.6e; polydeg3 amplitude=%.6e, mixing_extent=%.6e\n",
               deliverable_t3.interface_amplitude,
               deliverable_t3.mixing_zone_extent,
               verification_t3.interface_amplitude,
               verification_t3.mixing_zone_extent)
println("Resolution verification complete; polydeg=5 remains the deliverable in sol and data_npz.")
# %% [markdown]


# ## Cell 11: Data Regularization (Rasterization)
# ## Methodology: Regular-Grid Rasterization for Visualization
# Projects the scattered PlotData2D samples onto a regular grid covering
# $\Omega=[0,0.25]\times[0,1]$. Samples falling into the same pixel are
# averaged. Empty pixels are completed from the nearest populated neighboring
# pixel, measured in physical distance, so the plotting grid contains no
# NaN holes or empty rows/columns that could appear as striping artifacts.
#
# This regularization is used only for rendering. The full conservative
# solution remains stored in the original NPZ snapshots.

# %%
function rasterize_data(x, y, z, nx=200, ny=800)
    nx > 0 && ny > 0 || throw(ArgumentError("Grid dimensions must be positive"))

    x_values = vec(x)
    y_values = vec(y)
    z_values = vec(z)

    length(x_values) == length(y_values) == length(z_values) ||
        throw(DimensionMismatch("x, y, and z must contain the same number of samples"))

    x_min, x_max = coordinates_min[1], coordinates_max[1]
    y_min, y_max = coordinates_min[2], coordinates_max[2]

    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny

    sums = zeros(Float64, ny, nx)
    counts = zeros(Int, ny, nx)

    # Average all samples assigned to a pixel instead of allowing the last
    # scattered sample to overwrite earlier samples.
    for k in eachindex(z_values)
        xk = x_values[k]
        yk = y_values[k]
        zk = z_values[k]

        if !(isfinite(xk) && isfinite(yk) && isfinite(zk))
            continue
        end
        if xk < x_min || xk > x_max || yk < y_min || yk > y_max
            continue
        end

        i = clamp(floor(Int, (xk - x_min) / dx) + 1, 1, nx)
        j = clamp(floor(Int, (yk - y_min) / dy) + 1, 1, ny)

        sums[j, i] += Float64(zk)
        counts[j, i] += 1
    end

    grid = fill(NaN, ny, nx)
    for j in 1:ny, i in 1:nx
        if counts[j, i] > 0
            grid[j, i] = sums[j, i] / counts[j, i]
        end
    end

    count(isfinite, grid) > 0 ||
        error("No finite solution samples lie inside the rasterization domain")

    # Complete empty pixels by nearest-neighbor propagation. Values are
    # copied rather than diffusively averaged across multiple pixels.
    while any(isnan, grid)
        previous = copy(grid)
        missing_before = count(isnan, previous)

        for j in 1:ny, i in 1:nx
            isnan(previous[j, i]) || continue

            best_distance = Inf
            best_value = NaN

            for jj in max(1, j - 1):min(ny, j + 1)
                for ii in max(1, i - 1):min(nx, i + 1)
                    if ii == i && jj == j
                        continue
                    end

                    value = previous[jj, ii]
                    isfinite(value) || continue

                    distance = ((ii - i) * dx)^2 + ((jj - j) * dy)^2
                    if distance < best_distance
                        best_distance = distance
                        best_value = value
                    end
                end
            end

            if isfinite(best_value)
                grid[j, i] = best_value
            end
        end

        count(isnan, grid) < missing_before ||
            error("Unable to complete the rasterized plotting grid")
    end

    return grid
end
# %% [markdown]


# ## Cell 12: Post-Processing & Temporal Diagnostics
# ## Post-Processing: Rayleigh–Taylor Summary
# Visualizes density and vertical velocity at the four requested times.
# Rows correspond to fields and columns correspond to time. Each panel uses
# automatic color scaling, disables colorbars, and annotates its field extrema.
# Heatmaps of the completed regular grids avoid mesh/cell-edge overlays and
# scattered-node rendering artifacts.

# %%
println("Generating RTI summary plots...")

# Load the exact visualization snapshots created by the NPZ callback.
snapshots = []
for t_target in visualization_times
    filename = joinpath(npz_out,
                        Printf.@sprintf("time_%08.5f.npz", t_target))
    isfile(filename) || error("Missing solution snapshot: $filename")
    push!(snapshots, npzread(filename))
end

# Pixel-center coordinates match the bins used by rasterize_data.
nx_plot, ny_plot = 200, 800
x_plot = range(coordinates_min[1] +
               0.5 * (coordinates_max[1] - coordinates_min[1]) / nx_plot,
               coordinates_max[1] -
               0.5 * (coordinates_max[1] - coordinates_min[1]) / nx_plot,
               length=nx_plot)
y_plot = range(coordinates_min[2] +
               0.5 * (coordinates_max[2] - coordinates_min[2]) / ny_plot,
               coordinates_max[2] -
               0.5 * (coordinates_max[2] - coordinates_min[2]) / ny_plot,
               length=ny_plot)

style = (
    seriestype = :heatmap,
    aspect_ratio = :auto,
    xlims = (0.0, 0.25),
    ylims = (0.0, 1.0),
    xticks = ([0.0, 0.125, 0.25], ["0", "0.125", "0.25"]),
    yticks = 0.0:0.25:1.0,
    xlabel = "x",
    ylabel = "y",
    framestyle = :box,
    grid = false,
    legend = false,
    colorbar = false,
    tickfontsize = 7,
    guidefontsize = 8,
    titlefontsize = 9,
    margin = 3Plots.mm
)

plots_array = []

# Field-major ordering gives density in the first row and vertical velocity
# in the second row, with time increasing across the columns.
for field in (:rho, :v)
    for (t_target, data) in zip(visualization_times, snapshots)
        x = vec(data["x"])
        y = vec(data["y"])

        if field === :rho
            values = vec(data["rho"])
            field_label = "ρ"
        else
            # Vertical velocity from the saved conserved variables.
            values = vec(data["rho_v"] ./ data["rho"])
            field_label = "v"
        end

        finite_values = filter(isfinite, values)
        isempty(finite_values) &&
            error("No finite values for $field_label at t=$t_target")
        field_min, field_max = extrema(finite_values)

        grid_values = rasterize_data(x, y, values, nx_plot, ny_plot)

        panel = heatmap(x_plot, y_plot, grid_values;
                        title="$field_label, t=$(round(t_target, digits=2))",
                        style...)

        # No fixed color map or color limits are supplied: Plots determines
        # the color scaling independently from each panel's field values.
        extrema_label = Printf.@sprintf("min = %.3e   max = %.3e",
                                        field_min, field_max)
        annotate!(panel, 0.125, 0.965,
                  Plots.text(extrema_label, 7, :black))

        push!(plots_array, panel)
    end
end

println("Saving summary_all.png...")
final_plot = plot(plots_array..., layout=(2, 4),
                  size=(2400, 1600), dpi=300, margin=5Plots.mm)
savefig(final_plot, joinpath(@__DIR__, "summary_all.png"))
println("Done.")
