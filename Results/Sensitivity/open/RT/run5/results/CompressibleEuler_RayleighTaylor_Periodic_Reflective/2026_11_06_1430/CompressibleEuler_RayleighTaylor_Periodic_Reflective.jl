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
using DiffEqCallbacks
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
# ## Governing Equations: Hyperbolic System
#
# The Rayleigh–Taylor instability is modeled using the inviscid,
# non-conducting compressible Euler equations:
#
# ∂ₜ𝐮 + ∇ ⋅ 𝐟(𝐮) = 𝐒(𝐮)
#
# `CompressibleEulerEquations2D` defines conservation of mass, momentum,
# and total energy using the ideal-gas equation of state with γ = 1.4.
# The gravitational source term is defined separately in Cell 3.

# %%
equations = CompressibleEulerEquations2D(1.4)
# %% [markdown]

# ## Cell 3: Initial Conditions and Source Terms (Lid-Driven Cavity)
# ## Problem Setup: Rayleigh–Taylor Instability
#
# Initializes a heavy fluid below a light fluid, with a smoothed density
# interface at y = 0.5 and the prescribed piecewise hydrostatic pressure.
# A single-mode vertical velocity perturbation seeds the instability.
#
# Gravity acts in the positive y-direction with g = 1.0. The conservative
# source vector is therefore (0, 0, ρg, ρgv).

# %%
@inline function initial_condition_rayleigh_taylor(x, t,
        equations::CompressibleEulerEquations2D)
    x_pos, y_pos = x

    rho = 2.0 + (1.0 - 2.0) * 0.5 *
          (1.0 + tanh(50.0 * (y_pos - 0.5)))

    p = y_pos < 0.5 ? 2.0 * y_pos + 1.0 : y_pos + 1.5

    u = 0.0
    c = sqrt(equations.gamma * p / rho)
    v = -0.025 * c * cos(8.0 * pi * x_pos) * sin(pi * y_pos)^6

    return prim2cons(SVector(rho, u, v, p), equations)
end

@inline function source_terms_gravity(u, x, t,
        equations::CompressibleEulerEquations2D)
    g = 1.0
    rho, rho_v1, rho_v2, rho_e = u

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
#     * Approximates solutions using tensor-product polynomials of degree $N=5$ on **Legendre-Gauss-Lobatto (LGL)** nodes.
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
# 3.  **Interface Flux (`surface_flux=flux_hllc`)**:
#     * **Riemann Solver**: Harten-Lax-van Leer-Contact (HLLC).
#     * **Role**: Computes the numerical flux across element interfaces.
#     * **Physics**: Restores the contact wave to reduce numerical dissipation
#       at the material interfaces driving the Rayleigh–Taylor instability.
#
# 4.  **Indicator Function (`IndicatorHennemannGassner`)**:
#     * Dynamically computes the blending coefficient $\alpha \in [0, 1]$ element-wise.
#     * **Mechanism**: Monitors the spectral decay of pressure in the highest
#       polynomial modes. Pressure-based shock detection avoids activating the
#       diffusive FV flux merely because the density contact is steep. Density
#       remains the independent AMR indicator.
#
# %%
@inline function pressure_indicator(u, equations::CompressibleEulerEquations2D)
    rho, rho_v1, rho_v2, rho_e = u
    return (equations.gamma - 1) *
           (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) / rho)
end

function rti_build_dg(polydeg)
    basis = LobattoLegendreBasis(polydeg)
    indicator_sc = IndicatorHennemannGassner(equations, basis,
                                             alpha_max=1.0,
                                             alpha_min=0.001,
                                             alpha_smooth=true,
                                             variable=pressure_indicator)
    volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                     volume_flux_dg=flux_ranocha,
                                                     volume_flux_fv=flux_lax_friedrichs)
    return DGSEM(polydeg=polydeg, surface_flux=flux_hllc,
                 volume_integral=volume_integral)
end

# Stage A production default; Stage B reuses rti_build_dg with polydeg=6.
polydeg = 5
dg = rti_build_dg(polydeg)
# %% [markdown]

# ## Cell 5: Computational Domain & Adaptive Mesh Topology
# ## Methodology: Forest-of-Octrees (`p4est`)
#
# Defines the Rayleigh–Taylor domain Ω = [0, 0.25] × [0, 1] using an
# adaptive quadtree mesh. The template's single root tree and initial
# refinement level are retained.
#
# Boundary topology:
# * x-direction: periodic, connecting x = 0 and x = 0.25.
# * y-direction: nonperiodic, leaving y = 0 and y = 1 as physical
#   boundaries for the reflective free-slip wall conditions.
# %%
coordinates_min = (0.0, 0.0)
coordinates_max = (0.25, 1.0)

function rti_build_mesh(degree)
    return P4estMesh((1, 1), polydeg=degree,
                     coordinates_min=coordinates_min,
                     coordinates_max=coordinates_max,
                     initial_refinement_level=3,
                     periodicity=(true, false))
end

# Stage A default; Stage B constructs an independent mesh with the same topology.
mesh = rti_build_mesh(polydeg)
# %% [markdown]

# ## Cell 6: General Boundary Condition Specification & System Assembly
# ## Boundary Conditions and Hyperbolic System Assembly
#
# The x-boundaries are periodic through the mesh topology. At y = 0 and
# y = 1, `boundary_condition_slip_wall` imposes reflective free-slip walls:
# the normal velocity is mirrored in the boundary state, while density,
# tangential velocity, and pressure are retained.
#
# The pure Euler operator is assembled with the Rayleigh–Taylor initial
# condition and the gravitational source term.
# %%
boundary_conditions = Dict(
    :y_neg => boundary_condition_slip_wall,
    :y_pos => boundary_condition_slip_wall
)

function rti_build_semi(mesh_stage, dg_stage)
    return SemidiscretizationHyperbolic(mesh_stage, equations,
                                        initial_condition_rayleigh_taylor,
                                        dg_stage;
                                        boundary_conditions=boundary_conditions,
                                        source_terms=source_terms_gravity)
end

# Stage A default; Stage B constructs an independent semidiscretization.
semi = rti_build_semi(mesh, dg)
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
rti_amr_interval = 20

function rti_build_amr(semi_stage)
    indicator = IndicatorLöhner(semi_stage, variable=density_indicator)
    controller = ControllerThreeLevel(semi_stage, indicator,
                                      base_level=4,
                                      med_level=5, med_threshold=0.1,
                                      max_level=6, max_threshold=0.3)
    return AMRCallback(semi_stage, controller,
                       interval=rti_amr_interval,
                       adapt_initial_condition=true)
end

# Used by the raw accepted-step recorder to tag scheduled AMR interventions.
@inline function rti_amr_event_due(integrator)
    return integrator.iter > 0 &&
           integrator.iter % rti_amr_interval == 0
end

# Stage A default; Stage B rebuilds these objects from its own semidiscretization.
amr_callback = rti_build_amr(semi)
# %% [markdown]

# ## Cell 8: Data I/O & Surrogate Interoperability
# ## Strategy: Cross-Language Serialization (.npz)
#
# Saves the complete conservative state (ρ, ρu, ρv, ρE) every 0.1 time
# units. Additional output at t = 0.75 and t = 2.25 ensures that all
# requested visualization times are represented exactly. Pressure is also
# exported for the required density/pressure plots.
# %%
npz_out = joinpath(@__DIR__, "data_npz")
mkpath(npz_out)
dt_save = 0.1
visualization_times = [0.75, 1.5, 2.25, 3.0]
rti_symmetry_times = [0.0, visualization_times...]
rti_comparison_times = [2.25, 3.0]
rti_comparison_nx = 512
rti_comparison_ny = 2048
rti_event_merge_ulps = 10.0
rti_event_reached_atol = 1.0e-12
@inline function rti_event_ulp_tolerance(t)
    return rti_event_merge_ulps * eps(Float64) * max(1.0, abs(t))
end
@inline function rti_same_event(t, other)
    return abs(t - other) <= max(rti_event_ulp_tolerance(t),
                                 rti_event_ulp_tolerance(other))
end
function rti_dedup_event_times(times; final_time=3.0)
    final = Float64(final_time)
    isfinite(final) ||
        throw(ArgumentError("The final event time must be finite"))
    candidates = Float64[final]
    for value in times
        t = Float64(value)
        isfinite(t) ||
            throw(ArgumentError("Scheduled event times must be finite"))
        push!(candidates, t)
    end
    sort!(candidates)
    merged = Float64[]
    for t in candidates
        if isempty(merged) || !rti_same_event(t, merged[end])
            push!(merged, t)
        elseif t == final || merged[end] == final
            # Preserve an exact canonical final time for the regression gate.
            merged[end] = final
        end
        # Otherwise retain the existing canonical representative.
    end
    return merged
end
# A remaining distance at this scale is event-time roundoff, not a CFL step.
@inline function rti_event_reached(t, event_time)
    tolerance = max(rti_event_reached_atol,
                    rti_event_ulp_tolerance(t),
                    rti_event_ulp_tolerance(event_time))
    return abs(event_time - t) <= tolerance
end
requested_save_times = vcat(collect(0.0:dt_save:3.0),
                            visualization_times)
# Shared union of all requested output events and the final time. AMR remains
# iteration-scheduled and therefore contributes no additional time events.
rti_scheduled_event_times = rti_dedup_event_times(requested_save_times;
                                                  final_time=3.0)
# Canonicalize each output schedule against that same union so separate
# callbacks cannot reintroduce ULP-separated copies of an event.
function rti_canonical_event_subset(times, canonical_times)
    subset = Float64[]
    for value in times
        t = Float64(value)
        index = argmin(abs.(canonical_times .- t))
        event = canonical_times[index]
        rti_same_event(event, t) ||
            error("Missing canonical representative for event t=$t")
        push!(subset, event)
    end
    sort!(subset)
    unique!(subset) # Exact duplicates after canonicalization only.
    return subset
end
save_times = rti_canonical_event_subset(requested_save_times,
                                        rti_scheduled_event_times)
# Use exactly the required output events for saveat as well. In particular,
# do not introduce a finer independent solution-save schedule.
rti_solution_save_times = save_times
# Each run owns its history and output state. In particular, Stage B must not
# overwrite Stage A's accepted steps, snapshots, or symmetry residuals.
mutable struct RTIStageDiagnostics
    label::String
    polydeg::Int
    npz_dir::String
    initial_mass::Float64
    initialization_t::Float64
    initialization_dt::Float64
    accepted_steps::Vector{NamedTuple}
    amr_events::Vector{NamedTuple}
    snapshot_files::Dict{Float64,String}
    symmetry_residuals::Dict{Float64,Any}
    fixed_grid_samples::Dict{Float64,Any}
    final_diagnostics::Any
end
function rti_new_stage_diagnostics(label::AbstractString, degree::Integer,
                                   directory::AbstractString)
    mkpath(directory)
    return RTIStageDiagnostics(String(label), Int(degree), String(directory),
                               NaN, NaN, NaN, NamedTuple[], NamedTuple[],
                               Dict{Float64,String}(),
                               Dict{Float64,Any}(),
                               Dict{Float64,Any}(), nothing)
end
rti_default_diagnostics =
    rti_new_stage_diagnostics("stage_a_polydeg5", polydeg, npz_out)
rti_final_diagnostics = Ref{Any}(nothing)
@inline function rti_mass_integrand(u, args...)
    return u[1]
end
function rti_total_mass(u_ode, semi_local)
    mesh_local, equations_local, solver, cache =
        Trixi.mesh_equations_solver_cache(semi_local)
    u = Trixi.wrap_array(u_ode, mesh_local, equations_local, solver, cache)
    # Use DG quadrature and mesh metric terms, rather than a nodal sum.
    return Trixi.integrate(rti_mass_integrand, u, mesh_local, equations_local,
                           solver, cache)
end
function rti_initialize_dt!(integrator, state, t)
    state.initialization_t = t
    state.initialization_dt = integrator.dt
    return nothing
end
function rti_record_dt!(integrator, state; amr_enabled=true)
    # Initialization is not an accepted integration step.
    if integrator.iter == 0 && integrator.t == integrator.tprev
        if !isfinite(state.initialization_dt)
            rti_initialize_dt!(integrator, state, integrator.t)
        end
        return nothing
    end
    # The accepted dt is the time difference, so a tspan-clipped final step
    # remains visible instead of being replaced by the next proposed dt.
    dt = integrator.t - integrator.tprev
    if !isfinite(dt) || dt <= 0.0
        dt = integrator.dt
    end
    if integrator.t > 0.0 && isfinite(dt) && dt > 0.0
        amr_event = amr_enabled && rti_amr_event_due(integrator)
        push!(state.accepted_steps,
              (t=integrator.t, tprev=integrator.tprev, dt=dt,
               iter=integrator.iter, amr_event=amr_event))
        if amr_event
            push!(state.amr_events,
                  (t=integrator.t, iter=integrator.iter))
        end
    end
    return nothing
end
function rti_build_dt_callback(state; amr_enabled=true)
    affect! = integrator -> rti_record_dt!(integrator, state;
                                           amr_enabled=amr_enabled)
    initialize = function (callback, u, t, integrator)
        rti_initialize_dt!(integrator, state, t)
    end
    return DiscreteCallback((u, t, integrator) -> true, affect!;
                            initialize=initialize,
                            save_positions=(false, false))
end
function rti_wall_max_abs_v(y, v, wall_y)
    y_vec = vec(y)
    v_vec = vec(v)
    length(y_vec) == length(v_vec) ||
        throw(DimensionMismatch("Wall coordinates and vertical velocity must have matching lengths"))
    isempty(y_vec) &&
        throw(ArgumentError("Wall diagnostic received empty coordinate data"))

    distances = abs.(y_vec .- wall_y)
    tolerance = 1.0e-10
    mask = distances .<= tolerance
    # Retain a fallback for floating-point variations in boundary coordinates.
    if !any(mask)
        mask = distances .<= minimum(distances) + tolerance
    end
    return maximum(abs.(v_vec[mask]))
end
end
function rti_snapshot_file(state, target_time)
    for (t, filename) in state.snapshot_files
        if isapprox(t, target_time; rtol=0.0, atol=1.0e-10)
            return filename
        end
    end
    error("Missing snapshot for $(state.label) at t = $target_time")
end
# Save-time symmetry diagnostics require the rasterization routines before
# the time-integration cell is executed.
function rti_linear_bracket(nodes, value)
    n = length(nodes)
    n > 0 || throw(ArgumentError("Cannot interpolate an empty node sequence"))
    if n == 1
        return 1, 1, 0.0
    end
    left = clamp(searchsortedlast(nodes, value), 1, n - 1)
    right = left + 1
    spacing = nodes[right] - nodes[left]
    weight = spacing > 0.0 ?
             clamp((value - nodes[left]) / spacing, 0.0, 1.0) : 0.0
    return left, right, weight
end
function rti_snapshot_polydeg(data, degree=nothing)
    degree !== nothing && return Int(degree)
    if haskey(data, "polydeg")
        value = data["polydeg"]
        return value isa Number ? Int(value) : Int(only(value))
    end
    return polydeg
end
# PlotData2D normally supplies three-dimensional element blocks. Also accept
# flattened serializations using the polynomial degree of the current stage.
function rti_element_blocks(array, degree::Integer)
    ndims(array) == 3 && return array
    ndims(array) == 2 ||
        throw(ArgumentError("Expected DG element-block plot data"))
    nrows, ncols = size(array)
    candidates = (degree, degree + 1, 2 * degree,
                  2 * degree + 1, 2 * (degree + 1))
    for n in candidates
        if n >= 2 && nrows == n * n
            return reshape(array, n, n, ncols)
        elseif n >= 2 && nrows == n && ncols % n == 0
            return reshape(array, n, n, div(ncols, n))
        end
    end
    throw(ArgumentError("Could not recover DG visualization element blocks"))
end
rti_element_blocks(array) = rti_element_blocks(array, polydeg)
function rti_grid_axes(nx, ny; cell_centered=false)
    if cell_centered
        dx = 0.25 / nx
        dy = 1.0 / ny
        # No duplicated periodic endpoint: x_i + x_(Nx+1-i) = 0.25.
        x_grid = [(i - 0.5) * dx for i in 1:nx]
        y_grid = [(j - 0.5) * dy for j in 1:ny]
    else
        x_grid = collect(range(0.0, 0.25, length=nx))
        y_grid = collect(range(0.0, 1.0, length=ny))
    end
    return x_grid, y_grid
end
function rti_sample_uniform_grid(data, variable;
                                 nx=256, ny=1024,
                                 degree=nothing, cell_centered=false)
    (nx > 1 && ny > 1) ||
        throw(ArgumentError("Uniform grid dimensions must exceed one"))
    degree_local = rti_snapshot_polydeg(data, degree)
    x_blocks = rti_element_blocks(data["x"], degree_local)
    y_blocks = rti_element_blocks(data["y"], degree_local)
    z_blocks = rti_element_blocks(data[variable], degree_local)
    size(x_blocks) == size(y_blocks) == size(z_blocks) ||
        throw(ArgumentError("Coordinates and solution data must have matching shapes"))
    x_grid, y_grid = rti_grid_axes(nx, ny; cell_centered=cell_centered)
    grid = fill(NaN, ny, nx)
    filled = falses(ny, nx)
    for element in axes(x_blocks, 3)
        x_block = view(x_blocks, :, :, element)
        y_block = view(y_blocks, :, :, element)
        x_min, x_max = extrema(x_block)
        y_min, y_max = extrema(y_block)
        # Determine which block dimension represents x.
        x_first = view(x_blocks, :, 1, element)
        x_second = view(x_blocks, 1, :, element)
        first_spread = maximum(x_first) - minimum(x_first)
        second_spread = maximum(x_second) - minimum(x_second)
        if first_spread >= second_spread
            x_nodes = collect(x_first)
            y_nodes = collect(view(y_blocks, 1, :, element))
            z_block = Array(view(z_blocks, :, :, element))
        else
            x_nodes = collect(x_second)
            y_nodes = collect(view(y_blocks, :, 1, element))
            z_block = permutedims(Array(view(z_blocks, :, :, element)))
        end
        x_order = sortperm(x_nodes)
        y_order = sortperm(y_nodes)
        x_nodes = x_nodes[x_order]
        y_nodes = y_nodes[y_order]
        z_block = z_block[x_order, y_order]
        padding = 1.0e-12
        i_first = max(1, searchsortedfirst(x_grid, x_min - padding))
        i_last = min(nx, searchsortedlast(x_grid, x_max + padding))
        j_first = max(1, searchsortedfirst(y_grid, y_min - padding))
        j_last = min(ny, searchsortedlast(y_grid, y_max + padding))
        for j in j_first:j_last
            y_left, y_right, wy = rti_linear_bracket(y_nodes, y_grid[j])
            for i in i_first:i_last
                filled[j, i] && continue
                x_left, x_right, wx =
                    rti_linear_bracket(x_nodes, x_grid[i])
                z00 = z_block[x_left, y_left]
                z10 = z_block[x_right, y_left]
                z01 = z_block[x_left, y_right]
                z11 = z_block[x_right, y_right]
                lower = (1.0 - wx) * z00 + wx * z10
                upper = (1.0 - wx) * z01 + wx * z11
                grid[j, i] = (1.0 - wy) * lower + wy * upper
                filled[j, i] = true
            end
        end
    end
    missing_count = count(.!filled)
    missing_count == 0 ||
        error("Uniform DG sampling left $missing_count grid points uncovered")
    all(isfinite, grid) ||
        error("Uniform DG sampling produced non-finite values")
    return x_grid, y_grid, grid
end
# Parity-aware reflection residual on the complete cell-centered grid:
# even: ||q - Mq||_2 / ||q||_2
# odd:  ||q + Mq||_2 / ||q||_2
function rti_mirror_residual(grid; parity=:even, norm_epsilon=1.0e-14)
    ny, nx = size(grid)
    iseven(nx) || throw(ArgumentError("Mirror pairing requires an even Nx"))
    parity in (:even, :odd) ||
        throw(ArgumentError("parity must be :even or :odd"))
    numerator = 0.0
    denominator = 0.0
    for j in 1:ny, i in 1:div(nx, 2)
        q_left = grid[j, i]
        q_right = grid[j, nx + 1 - i]
        deviation = parity === :odd ?
                    q_left + q_right : q_left - q_right
        numerator += 2.0 * deviation^2
        denominator += q_left^2 + q_right^2
    end
    absolute_residual = sqrt(numerator)
    field_norm = sqrt(denominator)
    # In particular, rho*u is identically zero at t=0.
    if field_norm < norm_epsilon
        return absolute_residual
    end
    return absolute_residual / field_norm
end
function rti_save_symmetry_residuals!(state, data, t)
    sampling_options = (nx=rti_comparison_nx,
                        ny=rti_comparison_ny,
                        degree=state.polydeg,
                        cell_centered=true)
    rho_sample = rti_sample_uniform_grid(data, "rho"; sampling_options...)
    rho_u_sample = rti_sample_uniform_grid(data, "rho_u";
                                           sampling_options...)
    rho_v_sample = rti_sample_uniform_grid(data, "rho_v";
                                           sampling_options...)
    rho_e_sample = rti_sample_uniform_grid(data, "rho_e";
                                           sampling_options...)
    residuals = (
        rho=rti_mirror_residual(rho_sample[3]; parity=:even),
        rho_u=rti_mirror_residual(rho_u_sample[3]; parity=:odd),
        rho_v=rti_mirror_residual(rho_v_sample[3]; parity=:even),
        rho_e=rti_mirror_residual(rho_e_sample[3]; parity=:even)
    )
    state.symmetry_residuals[t] = residuals
    return residuals
end

# Return complete variable arrays for either PlotData2D representation:
# a vector of variable arrays, or one conservative SVector per plot node.
function rti_complete_plot_fields(pd, variable_count)
    field_shape = size(pd.y)
    size(pd.x) == field_shape ||
        throw(DimensionMismatch("PlotData2D x/y coordinate shapes do not match"))
    field_length = prod(field_shape)
    data = pd.data

    if data isa AbstractVector && length(data) == variable_count &&
       all(field -> field isa AbstractArray && length(field) == field_length,
           data)
        return [reshape(data[i], field_shape) for i in 1:variable_count]
    end

    length(data) == field_length ||
        throw(DimensionMismatch("PlotData2D data and coordinates have incompatible sizes"))
    return [reshape(map(state -> state[i], data), field_shape)
            for i in 1:variable_count]
end

function rti_save_npz_file(integrator, state)
    semi_local = integrator.p
    t = integrator.t
    pd_cons = PlotData2D(integrator.u, semi_local;
                         solution_variables=cons2cons)
    rho, rho_u, rho_v, rho_e = rti_complete_plot_fields(pd_cons, 4)

    # Recover primitive fields from complete conservative nodal arrays.
    # This avoids selecting one four-component state from pd_prim.data.
    p = (equations.gamma - 1) .*
        (rho_e .- 0.5 .* (rho_u.^2 .+ rho_v.^2) ./ rho)
    y = vec(pd_cons.y)
    v_vec = vec(rho_v ./ rho)
    length(v_vec) == length(y) ||
        throw(DimensionMismatch("Vertical velocity and y coordinates must have matching lengths"))
    mass = rti_total_mass(integrator.u, semi_local)
    if t == 0.0 || !isfinite(state.initial_mass)
        state.initial_mass = mass
    end
    if t == 0.0 && !isfinite(state.initialization_dt)
        state.initialization_t = t
        state.initialization_dt = integrator.dt
    end
    mass_drift = mass - state.initial_mass
    relative_mass_drift = mass_drift / state.initial_mass
    rho_min, rho_max = extrema(rho)
    p_min, p_max = extrema(p)
    wall_bottom = rti_wall_max_abs_v(y, v_vec, 0.0)
    wall_top = rti_wall_max_abs_v(y, v_vec, 1.0)
    positivity_pass = rho_min > 0.0 && p_min > 0.0
    dt_last = NaN
    if !isempty(state.accepted_steps)
        dt_last = state.accepted_steps[end].dt
    end
    snapshot_data = Dict{String,Any}(
        "t" => t,
        "polydeg" => state.polydeg,
        "x" => pd_cons.x,
        "y" => pd_cons.y,
        "rho" => rho,
        "rho_u" => rho_u,
        "rho_v" => rho_v,
        "rho_e" => rho_e,
        "p" => p
    )
    filename = joinpath(state.npz_dir,
                        Printf.@sprintf("time_%08.5f.npz", t))
    npzwrite(filename, snapshot_data)
    state.snapshot_files[t] = filename
    # Use the complete archived state for the required diagnostic at every
    # save event, not only at the five post-processing times.
    symmetry = rti_save_symmetry_residuals!(state, snapshot_data, t)
    state.final_diagnostics = (
        t=t,
        mass=mass,
        mass_drift=mass_drift,
        relative_mass_drift=abs(relative_mass_drift),
        rho_min=rho_min,
        rho_max=rho_max,
        p_min=p_min,
        p_max=p_max,
        dt_last=dt_last,
        initialization_dt=state.initialization_dt,
        accepted_step_count=length(state.accepted_steps),
        max_abs_v_y0=wall_bottom,
        max_abs_v_y1=wall_top
    )
    if state === rti_default_diagnostics
        rti_final_diagnostics[] = state.final_diagnostics
    end
    println("Saved: $(state.label)/$(basename(filename))")
    Printf.@printf("Diagnostics t=%.5f: mass=%.16e, mass_drift=%.3e, relative_mass_drift=%.3e\n",
                   t, mass, mass_drift, relative_mass_drift)
    Printf.@printf("  rho_min=%.6e, rho_max=%.6e, p_min=%.6e, p_max=%.6e\n",
                   rho_min, rho_max, p_min, p_max)
    println("  positivity of rho/p: ", positivity_pass ? "PASS" : "FAIL")
    Printf.@printf("  accepted_steps=%d, dt_last=%.3e, max_abs_v_y0=%.3e, max_abs_v_y1=%.3e\n",
                   length(state.accepted_steps), dt_last,
                   wall_bottom, wall_top)
    Printf.@printf("  parity-aware symmetry: R_rho=%.16e, R_rho_u=%.16e, R_rho_v=%.16e, R_rho_E=%.16e, max=%.16e\n",
                   symmetry.rho, symmetry.rho_u, symmetry.rho_v,
                   symmetry.rho_e,
                   max(symmetry.rho, symmetry.rho_u,
                       symmetry.rho_v, symmetry.rho_e))
    return nothing
end
# Compatibility wrapper for the Stage A default callback.
function save_npz_file(integrator)
    return rti_save_npz_file(integrator, rti_default_diagnostics)
end
function rti_build_save_callback(state)
    return PresetTimeCallback(save_times,
                              integrator -> rti_save_npz_file(integrator, state);
                              save_positions=(false, false))
end
function rti_build_io_callbacks(state; amr_enabled=true)
    return CallbackSet(rti_build_dt_callback(state;
                                             amr_enabled=amr_enabled),
                       rti_build_save_callback(state))
end
# Return statistics without discarding the raw history. Only the last raw
# accepted step may be excused, and only when it lands exactly on final_time.
function rti_timestep_audit(state; final_time=3.0)
    steps = state.accepted_steps
    ordinary_dts = Float64[]
    nonfinal_tiny_steps = NamedTuple[]
    for i in 1:max(length(steps) - 1, 0)
        step = steps[i]
        push!(ordinary_dts, step.dt)
        if step.dt < 1.0e-8
            push!(nonfinal_tiny_steps, step)
        end
    end
    tiny_steps = filter(step -> step.dt < 1.0e-8, steps)
    tiny_amr_count = count(step -> step.amr_event, tiny_steps)
    final_is_tiny = !isempty(steps) && steps[end].dt < 1.0e-8
    final_at_tspan = !isempty(steps) && steps[end].t == final_time
    regression_pass = !isempty(steps) &&
                      isempty(nonfinal_tiny_steps) &&
                      (!final_is_tiny || final_at_tspan)
    return (
        accepted_step_count=length(steps),
        ordinary_step_count=length(ordinary_dts),
        ordinary_min=isempty(ordinary_dts) ? NaN : minimum(ordinary_dts),
        ordinary_median=isempty(ordinary_dts) ? NaN : median(ordinary_dts),
        ordinary_max=isempty(ordinary_dts) ? NaN : maximum(ordinary_dts),
        final_t=isempty(steps) ? NaN : steps[end].t,
        final_dt=isempty(steps) ? NaN : steps[end].dt,
        final_is_tiny=final_is_tiny,
        final_at_tspan=final_at_tspan,
        initialization_t=state.initialization_t,
        initialization_dt=state.initialization_dt,
        tiny_count=length(tiny_steps),
        tiny_amr_count=tiny_amr_count,
        tiny_steps=tiny_steps,
        nonfinal_tiny_count=length(nonfinal_tiny_steps),
        nonfinal_tiny_amr_count=
            count(step -> step.amr_event, nonfinal_tiny_steps),
        nonfinal_tiny_steps=nonfinal_tiny_steps,
        regression_pass=regression_pass,
        amr_events=state.amr_events
    )
end
# Stage A defaults; later stages use rti_build_io_callbacks with their own state.
rti_dt_callback = rti_build_dt_callback(rti_default_diagnostics)
save_npz_preset_callback = rti_build_save_callback(rti_default_diagnostics)
save_npz_callback = CallbackSet(rti_dt_callback, save_npz_preset_callback)
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
#     Here, we set `cfl=0.35` to ensure the method remains within the stability region of the Runge-Kutta scheme.
#
# ## 3. Observability & Interventions (`CallbackSet`)
# Aggregates the diagnostic and control agents into the main solver loop:
# * **`AnalysisCallback`**: Computes in-situ functional quantities (Entropy, Kinetic Energy, $L_2/L_\infty$ errors) to monitor physical validity.
# * **`SaveSolutionCallback`**: Performs full-state serialization (HDF5-compatible) for restarts.
# * **`AMRCallback`**: (From Cell 7) Triggers mesh adaptation.
# %%
tspan = (0.0, 3.0)
analysis_interval = 100
function rti_build_stage_callbacks(semi_stage, amr_stage, io_stage;
                                   cfl, output_directory)
    summary = SummaryCallback()
    analysis = AnalysisCallback(semi_stage, interval=analysis_interval)
    alive = AliveCallback(analysis_interval=analysis_interval)

    # Retain Trixi's output initialization, file counter, and save affect,
    # but disable its independent step-interval save trigger. In particular,
    # do not use SaveSolutionCallback(dt=...), whose PeriodicCallback would
    # accumulate a separate, ULP-shifted sequence of time events.
    save_template = SaveSolutionCallback(
        interval=typemax(Int),
        save_initial_solution=false,
        save_final_solution=false,
        output_directory=output_directory,
        solution_variables=cons2cons)

    # Use exactly the same canonical event representatives as the NPZ output.
    # save_times includes t=0, dt_save output, visualization times, and t=3.
    save = PresetTimeCallback(save_times, save_template.affect!;
                              save_positions=(false, false))

    stepsize = StepsizeCallback(cfl=cfl)

    # save_template performs initialization/finalization only; save performs
    # all scheduled output. AMR remains iteration-scheduled, and the CFL
    # callback remains last so it sees any mesh changed by AMR.
    callback_set = if amr_stage === nothing
        # Used only by the conditional symmetric fixed-mesh control run.
        CallbackSet(summary, analysis, alive, save_template, save,
                    io_stage, stepsize)
    else
        CallbackSet(summary, analysis, alive, save_template, save,
                    amr_stage, io_stage, stepsize)
    end

    return (callbacks=callback_set,
            summary_callback=summary,
            analysis_callback=analysis,
            alive_callback=alive,
            save_solution=save,
            stepsize_callback=stepsize)
end
# Stage A: regenerated Run 3 production baseline.
ode = semidiscretize(semi, tspan)
trixi_out = joinpath(@__DIR__, "trixi_output")
stage_a_callback_parts = rti_build_stage_callbacks(
    semi, amr_callback, save_npz_callback;
    cfl=0.35, output_directory=trixi_out)
# Retain the template's default variable names for the time-integration cell.
summary_callback = stage_a_callback_parts.summary_callback
analysis_callback = stage_a_callback_parts.analysis_callback
alive_callback = stage_a_callback_parts.alive_callback
save_solution = stage_a_callback_parts.save_solution
stepsize_callback = stage_a_callback_parts.stepsize_callback
callbacks = stage_a_callback_parts.callbacks
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
function rti_roundoff_event_target(integrator)
    t = integrator.t
    tprev = integrator.tprev
    t > tprev || return nothing
    for event in rti_scheduled_event_times
        # An event already reached on the preceding step must not be fired
        # again or cause time to be snapped backward on a later step.
        event <= tprev && continue
        if event > t && !rti_event_reached(t, event)
            break
        end
        if event != t && rti_event_reached(t, event)
            return event
        end
    end
    return nothing
end
function rti_snap_roundoff_event!(integrator)
    target = rti_roundoff_event_target(integrator)
    target === nothing && return nothing
    # The state difference over at most 1e-12 time units is roundoff-level
    # event bookkeeping. Snapping t before the remaining callbacks makes all
    # scheduled callbacks observe the same canonical event and prevents a
    # separate accepted step whose only purpose is crossing that gap.
    integrator.t = target
    return nothing
end
function rti_add_roundoff_event_scheduling(stage_callbacks)
    event_callback = DiscreteCallback(
        (u, t, integrator) ->
            rti_roundoff_event_target(integrator) !== nothing,
        rti_snap_roundoff_event!;
        save_positions=(false, false))
    # Preserve the existing callback ordering, with event canonicalization
    # first. Thus AMR remains iteration-scheduled and StepsizeCallback remains
    # last after any AMR mesh modification.
    return CallbackSet(stage_callbacks.continuous_callbacks...,
                       event_callback,
                       stage_callbacks.discrete_callbacks...)
end
function rti_solve_stage(ode_stage, stage_callbacks)
    scheduled_callbacks =
        rti_add_roundoff_event_scheduling(stage_callbacks)
    solution = solve(ode_stage, RDPK3SpFSAL49();
                     dt=1.0,
                     adaptive=false,
                     callback=scheduled_callbacks,
                     saveat=rti_solution_save_times,
                     tstops=filter(t -> t > tspan[1],
                                   rti_scheduled_event_times),
                     maxiters=1e7)
    isempty(solution.t) && error("Time integration returned no saved times")
    isapprox(solution.t[end], tspan[2]; rtol=0.0, atol=1.0e-12) ||
        error("Time integration stopped at t = $(solution.t[end]), expected $(tspan[2])")
    return solution
end
function rti_run_independent_stage(label, degree, cfl;
                                   fixed_refinement_level=nothing)
    fixed_mesh = fixed_refinement_level !== nothing
    mesh_stage = if fixed_mesh
        # A uniform P4est mesh is mirror-symmetric and receives no AMR.
        P4estMesh((1, 1), polydeg=degree,
                  coordinates_min=coordinates_min,
                  coordinates_max=coordinates_max,
                  initial_refinement_level=fixed_refinement_level,
                  periodicity=(true, false))
    else
        rti_build_mesh(degree)
    end
    dg_stage = rti_build_dg(degree)
    semi_stage = rti_build_semi(mesh_stage, dg_stage)
    amr_stage = fixed_mesh ? nothing : rti_build_amr(semi_stage)
    output_root = joinpath(@__DIR__, label)
    stage_npz_dir = joinpath(output_root, "data_npz")
    stage_trixi_dir = joinpath(output_root, "trixi_output")
    diagnostics = rti_new_stage_diagnostics(label, degree, stage_npz_dir)
    io_callbacks = rti_build_io_callbacks(diagnostics;
                                          amr_enabled=!fixed_mesh)
    callback_parts = rti_build_stage_callbacks(
        semi_stage, amr_stage, io_callbacks;
        cfl=cfl, output_directory=stage_trixi_dir)
    ode_stage = semidiscretize(semi_stage, tspan)
    println("Running $label: polydeg=$degree, CFL=$cfl, AMR=$(!fixed_mesh)")
    solution = rti_solve_stage(ode_stage, callback_parts.callbacks)
    return (
        label=label,
        polydeg=degree,
        cfl=cfl,
        semi=semi_stage,
        sol=solution,
        diagnostics=diagnostics,
        npz_dir=stage_npz_dir,
        trixi_dir=stage_trixi_dir,
        figure_file=joinpath(output_root, "summary.png"),
        fixed_refinement_level=fixed_refinement_level,
        state_dofs=length(solution.u[end])
    )
end
# Recover the conservative-state size from archived element-block data. This
# is used only to size the conditional fixed-mesh diagnostic control.
function rti_archived_stage_b_state_dofs(data, degree)
    rho = data["rho"]
    n_elements = 0
    if ndims(rho) == 3
        n_elements = size(rho, 3)
    elseif ndims(rho) == 2
        nrows, ncols = size(rho)
        candidates = (degree, degree + 1, 2 * degree,
                      2 * degree + 1, 2 * (degree + 1))
        for n in candidates
            if n >= 2 && nrows == n * n
                n_elements = ncols
                break
            elseif n >= 2 && nrows == n && ncols % n == 0
                n_elements = div(ncols, n)
                break
            end
        end
    end
    n_elements > 0 ||
        error("Could not recover Stage B's archived element count")
    return 4 * n_elements * (degree + 1)^2
end
# Archive-first verification closure. The five snapshots required for the
# corrected symmetry residuals also include every requested plotting time.
function rti_load_archived_stage_b()
    label = "stage_b_polydeg6"
    degree = 6
    output_root = joinpath(@__DIR__, label)
    stage_npz_dir = joinpath(output_root, "data_npz")
    stage_trixi_dir = joinpath(output_root, "trixi_output")
    isdir(stage_npz_dir) || return nothing
    diagnostics = rti_new_stage_diagnostics(label, degree, stage_npz_dir)
    for target_time in rti_symmetry_times
        filename = joinpath(
            stage_npz_dir,
            Printf.@sprintf("time_%08.5f.npz", target_time))
        isfile(filename) || return nothing
        data = npzread(filename)
        archived_degree = data["polydeg"]
        archived_degree = archived_degree isa Number ?
                          Int(archived_degree) : Int(only(archived_degree))
        archived_degree == degree || return nothing
        diagnostics.snapshot_files[target_time] = filename
    end
    final_data = npzread(diagnostics.snapshot_files[3.0])
    state_dofs = rti_archived_stage_b_state_dofs(final_data, degree)
    # final_diagnostics remains uninitialized here; the post-processing cell
    # recomputes mass drift, positivity, and wall diagnostics from the archive.
    return (
        label=label,
        polydeg=degree,
        cfl=0.30,
        semi=nothing,
        sol=nothing,
        diagnostics=diagnostics,
        npz_dir=stage_npz_dir,
        trixi_dir=stage_trixi_dir,
        figure_file=joinpath(output_root, "summary.png"),
        fixed_refinement_level=nothing,
        state_dofs=state_dofs
    )
end
# Load the previous run solely as a baseline. Its files remain in the original
# stage_b_polydeg6 directory and are not overwritten by the corrected run.
rti_archived_stage_b = rti_load_archived_stage_b()
rti_stage_b_baseline = rti_archived_stage_b
if rti_stage_b_baseline === nothing
    println("Previous Stage B archive unavailable; the fresh run will be checked against the specified numerical baselines")
else
    println("Retained previous Stage B archive solely for baseline comparison")
end
println("Re-running the identical Stage B configuration after the event-scheduling fix: polydeg=6, CFL=0.30")
rti_stage_b = rti_run_independent_stage(
    "stage_b_polydeg6_event_fixed", 6, 0.30)
# This iteration has no resolution refinement or Stage A production run.
rti_resolution_runs = (rti_stage_b,)
# Conditional diagnostic only. Cell 12 may call this exclusively after the
# corrected parity-aware residuals establish genuine asymmetry.
function rti_run_symmetric_fixed_mesh_control(reference_run)
    degree = reference_run.polydeg
    target_state_dofs = reference_run.state_dofs
    target_state_dofs > 0 ||
        error("The reference run has no valid conservative-state size")
    # Select the uniform level whose total conservative-state size is closest
    # to the reference run's final size. Main-run AMR limits remain unchanged.
    levels = collect(4:6)
    candidate_state_dofs = [
        4 * (2^level)^2 * (degree + 1)^2 for level in levels
    ]
    level = levels[argmin(abs.(candidate_state_dofs .- target_state_dofs))]
    label = "symmetric_fixed_control_polydeg$(degree)_level$(level)"
    println("SYMMETRY CONTINGENCY: running one fixed-mesh control without AMR")
    control = rti_run_independent_stage(label, degree, reference_run.cfl;
                                        fixed_refinement_level=level)
    Printf.@printf("Fixed-mesh control: level=%d, target_state_dofs=%d, control_state_dofs=%d\n",
                   level, target_state_dofs, control.state_dofs)
    return control
end
# %% [markdown]

# ## Cell 11: Data Regularization (Rasterization)
# ## Methodology: Uniform Cartesian Sampling of DG Data
#
# Shared bracketing, element-block, grid-axis, uniform-sampling, and
# parity-aware residual routines are defined in Cell 8 so that they are
# available during every save callback. This cell retains the structural and
# post-processing diagnostics built on those routines.

# Maximum mirrored deviation, used by the t=0 renderer test.
function rti_max_mirrored_deviation(grid)
    ny, nx = size(grid)
    iseven(nx) || throw(ArgumentError("Mirror pairing requires an even Nx"))
    maximum_deviation = 0.0
    for j in 1:ny, i in 1:div(nx, 2)
        maximum_deviation = max(maximum_deviation,
                                abs(grid[j, i] - grid[j, nx + 1 - i]))
    end
    return maximum_deviation
end

# Backward-compatible name; the corrected test uses exact mirror pairs.
rti_max_x_deviation(grid) = rti_max_mirrored_deviation(grid)
# Four-connected component sizes. Periodic x-connectivity prevents a pocket
# crossing x=0/0.25 from being counted as two distinct physical structures.
function rti_component_sizes(mask; periodic_x=true)
    ny, nx = size(mask)
    visited = falses(ny, nx)
    sizes = Int[]
    stack = Int[]
    for start in eachindex(mask)
        if !mask[start] || visited[start]
            continue
        end
        visited[start] = true
        empty!(stack)
        push!(stack, start)
        component_size = 0
        while !isempty(stack)
            index = pop!(stack)
            component_size += 1
            j = mod(index - 1, ny) + 1
            i = div(index - 1, ny) + 1
            for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
                ii = i + di
                jj = j + dj
                if periodic_x
                    ii = mod(ii - 1, nx) + 1
                elseif ii < 1 || ii > nx
                    continue
                end
                if jj < 1 || jj > ny
                    continue
                end
                neighbor = jj + (ii - 1) * ny
                if mask[neighbor] && !visited[neighbor]
                    visited[neighbor] = true
                    push!(stack, neighbor)
                end
            end
        end
        push!(sizes, component_size)
    end
    sort!(sizes, rev=true)
    return sizes
end
function rti_structural_metrics(rho, x_grid, y_grid)
    ny, nx = size(rho)
    length(x_grid) == nx && length(y_grid) == ny ||
        throw(ArgumentError("Structural metric axes do not match rho"))
    mixing_mask = falses(ny, nx)
    heavy_mask = falses(ny, nx)
    light_mask = falses(ny, nx)
    upper_mixing_mask = falses(ny, nx)
    for j in 1:ny, i in 1:nx
        value = rho[j, i]
        in_mixing_zone = 1.1 < value < 1.9
        mixing_mask[j, i] = in_mixing_zone
        heavy_mask[j, i] = value > 1.9 && y_grid[j] > 0.5
        light_mask[j, i] = value < 1.1 && y_grid[j] < 0.5
        upper_mixing_mask[j, i] = in_mixing_zone && y_grid[j] > 0.5
    end
    mixing_zone_count = count(mixing_mask)
    dx = 0.25 / nx
    dy = 1.0 / ny
    # Central differences in periodic x and in interior y; one-sided
    # differences at the two wall-adjacent cell centers.
    sharpness_sum = 0.0
    for j in 1:ny, i in 1:nx
        if mixing_mask[j, i]
            i_left = i == 1 ? nx : i - 1
            i_right = i == nx ? 1 : i + 1
            drho_dx = (rho[j, i_right] - rho[j, i_left]) / (2.0 * dx)
            drho_dy = if j == 1
                (rho[2, i] - rho[1, i]) / dy
            elseif j == ny
                (rho[ny, i] - rho[ny - 1, i]) / dy
            else
                (rho[j + 1, i] - rho[j - 1, i]) / (2.0 * dy)
            end
            sharpness_sum += hypot(drho_dx, drho_dy)
        end
    end
    sharpness = mixing_zone_count > 0 ?
                sharpness_sum / mixing_zone_count : NaN
    cell_area = dx * dy
    # A census component must cover at least 16 raster cells. Applying the
    # threshold to component area, rather than raw pixel count alone, prevents
    # grid-scale fragments from being interpreted as resolved vortices.
    minimum_component_area = 16.0 * cell_area
    heavy_sizes = filter(
        n -> n * cell_area >= minimum_component_area,
        rti_component_sizes(heavy_mask; periodic_x=true))
    light_sizes = filter(
        n -> n * cell_area >= minimum_component_area,
        rti_component_sizes(light_mask; periodic_x=true))
    upper_mixing_sizes = filter(
        n -> n * cell_area >= minimum_component_area,
        rti_component_sizes(upper_mixing_mask; periodic_x=true))
    return (
        mixing_zone_count=mixing_zone_count,
        mixing_zone_area=mixing_zone_count * dx * dy,
        sharpness=sharpness,
        minimum_component_area=minimum_component_area,
        heavy_pocket_count=length(heavy_sizes),
        light_pocket_count=length(light_sizes),
        structural_count=length(heavy_sizes) + length(light_sizes),
        heavy_pocket_sizes=heavy_sizes,
        light_pocket_sizes=light_sizes,
        # Deterministic, area-filtered density-based proxy for additional
        # resolved substructures in the upper mushroom/vortex region.
        upper_mixing_component_count=length(upper_mixing_sizes),
        upper_mixing_component_sizes=upper_mixing_sizes
    )
end
function rti_store_comparison_sample!(run, target_time,
                                      x_grid, y_grid, rho_grid, p_grid)
    state = run.diagnostics
    metrics = rti_structural_metrics(rho_grid, x_grid, y_grid)
    sample = (x=x_grid, y=y_grid, rho=rho_grid, p=p_grid, metrics=metrics)
    state.fixed_grid_samples[target_time] = sample
    filename = joinpath(state.npz_dir,
                        Printf.@sprintf("fixed_grid_time_%08.5f.npz",
                                        target_time))
    npzwrite(filename, Dict(
        "t" => target_time,
        "x" => x_grid,
        "y" => y_grid,
        "rho" => rho_grid,
        "p" => p_grid,
        "mixing_zone_count" => metrics.mixing_zone_count,
        "mixing_zone_area" => metrics.mixing_zone_area,
        "sharpness" => metrics.sharpness,
        "structural_count" => metrics.structural_count,
        "upper_mixing_component_count" =>
            metrics.upper_mixing_component_count
    ))
    return sample
end
# Fill missing residuals for archive-only runs without overwriting the
# complete per-save history recorded by rti_save_npz_file. Comparison samples
# are still generated at the two requested comparison times.
function rti_compute_stage_diagnostic_data!(run;
                                            times=rti_symmetry_times,
                                            store_comparison=true)
    state = run.diagnostics
    for target_time in times
        residuals_available =
            haskey(state.symmetry_residuals, target_time)
        needs_comparison = store_comparison &&
                           any(t -> isapprox(target_time, t;
                                             rtol=0.0, atol=1.0e-10),
                               rti_comparison_times)

        # Fresh runs already computed these residuals in their save callback.
        if residuals_available && !needs_comparison
            continue
        end

        filename = rti_snapshot_file(state, target_time)
        data = npzread(filename)
        sampling_options = (nx=rti_comparison_nx,
                            ny=rti_comparison_ny,
                            degree=run.polydeg,
                            cell_centered=true)

        rho_sample = rti_sample_uniform_grid(data, "rho";
                                             sampling_options...)
        x_grid, y_grid, rho_grid = rho_sample
        nx = length(x_grid)
        for i in 1:nx
            isapprox(x_grid[i] + x_grid[nx + 1 - i], 0.25;
                     rtol=0.0, atol=8.0 * eps(Float64)) ||
                error("Cell-centered mirror grid pairing is not exact")
        end

        if !residuals_available
            rho_u_sample = rti_sample_uniform_grid(data, "rho_u";
                                                   sampling_options...)
            rho_v_sample = rti_sample_uniform_grid(data, "rho_v";
                                                   sampling_options...)
            rho_e_sample = rti_sample_uniform_grid(data, "rho_e";
                                                   sampling_options...)
            residuals = (
                rho=rti_mirror_residual(rho_grid; parity=:even),
                rho_u=rti_mirror_residual(rho_u_sample[3]; parity=:odd),
                rho_v=rti_mirror_residual(rho_v_sample[3]; parity=:even),
                rho_e=rti_mirror_residual(rho_e_sample[3]; parity=:even)
            )
            state.symmetry_residuals[target_time] = residuals
        end

        if needs_comparison
            p_sample = rti_sample_uniform_grid(data, "p";
                                               sampling_options...)
            rti_store_comparison_sample!(run, target_time,
                                         x_grid, y_grid, rho_grid,
                                         p_sample[3])
        end
    end
    return state
end
# %% [markdown]

# ## Cell 12: Post-Processing & Temporal Diagnostics
# ## Rayleigh–Taylor Density and Pressure Visualization
#
# Creates a 2 × 4 summary figure at t = 0.75, 1.5, 2.25, and 3.0.
# Every panel is one heatmap of a uniformly sampled Cartesian array.
# The t = 0 renderer test uses this same sampling path.
# %%
function rti_print_timestep_audit(run)
    state = run.diagnostics
    audit = rti_timestep_audit(state)
    steps = state.accepted_steps
    println("\nTIMESTEP AUDIT: ", run.label)
    Printf.@printf("  raw accepted steps=%d; ordinary steps=%d; scheduled AMR events=%d (interval=%d)\n",
                   audit.accepted_step_count, audit.ordinary_step_count,
                   length(audit.amr_events), rti_amr_interval)
    Printf.@printf("  ordinary dt: min=%.16e, median=%.16e, max=%.16e\n",
                   audit.ordinary_min, audit.ordinary_median,
                   audit.ordinary_max)
    Printf.@printf("  final step separately: t=%.16e, dt=%.16e, dt<1e-8=%s\n",
                   audit.final_t, audit.final_dt,
                   string(audit.final_is_tiny))
    Printf.@printf("  initialization separately: t=%.16e, dt=%.16e\n",
                   audit.initialization_t, audit.initialization_dt)
    Printf.@printf("  steps with dt<1e-8: count=%d, coinciding with scheduled AMR events=%d\n",
                   audit.tiny_count, audit.tiny_amr_count)
    Printf.@printf("  non-final steps with dt<1e-8: count=%d, coinciding with scheduled AMR events=%d\n",
                   audit.nonfinal_tiny_count,
                   audit.nonfinal_tiny_amr_count)
    for step in audit.tiny_steps
        Printf.@printf("    tiny step: t=%.16e, tprev=%.16e, dt=%.16e, iter=%d, AMR_event=%s\n",
                       step.t, step.tprev, step.dt, step.iter,
                       string(step.amr_event))
    end
    if audit.regression_pass
        if audit.tiny_count == 0
            println("  assessment: no accepted step below 1e-8")
        else
            println("  assessment: tiny dt is confined to the final step landing exactly at t=3")
        end
        println("  REGRESSION GATE: PASS — zero non-final accepted steps with dt < 1e-8")
    else
        println("  assessment: non-final tiny steps, an invalid final step, or missing raw history")
        println("  REGRESSION GATE: FAIL — inspect the raw AMR/StepsizeCallback history")
    end
    # Preserve the unaveraged history used for every value above.
    if !isempty(steps)
        npzwrite(joinpath(state.npz_dir, "accepted_step_history.npz"), Dict(
            "t" => [step.t for step in steps],
            "tprev" => [step.tprev for step in steps],
            "dt" => [step.dt for step in steps],
            "iter" => [step.iter for step in steps],
            "amr_event" => [step.amr_event for step in steps],
            "initialization_t" => audit.initialization_t,
            "initialization_dt" => audit.initialization_dt
        ))
    end
    return audit
end
rti_maximum_residual(r) = max(r.rho, r.rho_u, r.rho_v, r.rho_e)
function rti_print_symmetry_monitor(run; times=rti_symmetry_times)
    state = run.diagnostics
    println("\nPARITY-AWARE SYMMETRY MONITOR: ", run.label)
    println("  cell-centered grid=$(rti_comparison_nx)x$(rti_comparison_ny); no duplicated periodic endpoint")
    println("  mirror pairing: i <-> Nx+1-i, reflection about x=0.125")
    println("  even fields: R_q=||q-Mq||_2/||q||_2 for rho, rho_v, rho_E")
    println("  odd field: R_rho_u=||rho_u+M(rho_u)||_2/||rho_u||_2")
    println("  zero-norm guard: absolute residual when ||q||_2 < 1.0e-14")
    println("  t | R_rho (even) | R_rho_u (odd) | R_rho_v (even) | R_rho_E (even)")
    for target_time in times
        residuals = state.symmetry_residuals[target_time]
        all(isfinite,
            (residuals.rho, residuals.rho_u,
             residuals.rho_v, residuals.rho_e)) ||
            error("Non-finite symmetry residual for $(run.label) at t=$target_time")
        Printf.@printf("  %.2f | %.16e | %.16e | %.16e | %.16e\n",
                       target_time, residuals.rho, residuals.rho_u,
                       residuals.rho_v, residuals.rho_e)
    end
    if any(t -> t == 0.0, times)
        initial_maximum =
            rti_maximum_residual(state.symmetry_residuals[0.0])
        Printf.@printf("  t=0 corrected diagnostic validation: max_residual=%.16e, tolerance=1.0e-12\n",
                       initial_maximum)
        initial_maximum <= 1.0e-12 ||
            error("The corrected t=0 symmetry diagnostic is not at machine precision")
    end
    return nothing
end
function rti_render_stage!(run)
    state = run.diagnostics
    # Use the same rasterization path for the renderer test and the panels.
    initial_snapshot = npzread(rti_snapshot_file(state, 0.0))
    initial_rho = rti_sample_uniform_grid(initial_snapshot, "rho";
                                          degree=run.polydeg)
    initial_p = rti_sample_uniform_grid(initial_snapshot, "p";
                                        degree=run.polydeg)
    rho_x_deviation = rti_max_mirrored_deviation(initial_rho[3])
    p_x_deviation = rti_max_mirrored_deviation(initial_p[3])
    Printf.@printf("Renderer t=0 x-independence (%s): max_density_deviation=%.16e, max_pressure_deviation=%.16e\n",
                   run.label, rho_x_deviation, p_x_deviation)
    sampled_snapshots = Any[]
    for target_time in visualization_times
        data = npzread(rti_snapshot_file(state, target_time))
        rho_sample = rti_sample_uniform_grid(data, "rho";
                                             degree=run.polydeg)
        p_sample = rti_sample_uniform_grid(data, "p";
                                           degree=run.polydeg)
        push!(sampled_snapshots,
              (x=rho_sample[1], y=rho_sample[2],
               rho=rho_sample[3], p=p_sample[3]))
    end
    style = (
        seriestype=:heatmap,
        aspect_ratio=:equal,
        xlims=(0.0, 0.25),
        ylims=(0.0, 1.0),
        yflip=false,
        axis=nothing,
        framestyle=:none,
        legend=false,
        colorbar=false,
        margin=2Plots.mm
    )
    plots_array = Any[]
    for (sampled, target_time) in
        zip(sampled_snapshots, visualization_times)
        push!(plots_array,
              heatmap(sampled.x, sampled.y, sampled.rho,
                      title="ρ, t=$(target_time)"; style...))
    end
    for (sampled, target_time) in
        zip(sampled_snapshots, visualization_times)
        push!(plots_array,
              heatmap(sampled.x, sampled.y, sampled.p,
                      title="p, t=$(target_time)"; style...))
    end
    stage_plot = plot(plots_array..., layout=(2, 4),
                      size=(3200, 2400), margin=5Plots.mm)
    mkpath(dirname(run.figure_file))
    savefig(stage_plot, run.figure_file)
    println("Stored $(run.label) panel figure: $(run.figure_file)")
    return run.figure_file
end
# Interpolatory quadrature weights on the actual archived element nodes.
# This supports both LGL output and finer visualization-node output while
# integrating the archived element polynomial without a rasterized mass proxy.
function rti_archive_quadrature_weights(nodes)
    n = length(nodes)
    n >= 2 || throw(ArgumentError("At least two element nodes are required"))
    lower, upper = extrema(nodes)
    upper > lower || throw(ArgumentError("Degenerate archived element nodes"))
    reference_nodes = [
        2.0 * (node - lower) / (upper - lower) - 1.0 for node in nodes
    ]
    weights = zeros(Float64, n)
    for k in 1:n
        # Coefficients of the kth Lagrange polynomial in ascending powers.
        coefficients = [1.0]
        denominator = 1.0
        for m in 1:n
            m == k && continue
            updated = zeros(Float64, length(coefficients) + 1)
            for d in 1:length(coefficients)
                updated[d] -= reference_nodes[m] * coefficients[d]
                updated[d + 1] += coefficients[d]
            end
            coefficients = updated
            denominator *= reference_nodes[k] - reference_nodes[m]
        end
        iszero(denominator) &&
            throw(ArgumentError("Repeated archived interpolation node"))
        # Integrate each monomial over the reference interval [-1,1].
        for d in 1:length(coefficients)
            if isodd(d)
                weights[k] += coefficients[d] * (2.0 / d) / denominator
            end
        end
    end
    return weights
end
function rti_archived_mass(data, degree)
    x_blocks = rti_element_blocks(data["x"], degree)
    y_blocks = rti_element_blocks(data["y"], degree)
    rho_blocks = rti_element_blocks(data["rho"], degree)
    size(x_blocks) == size(y_blocks) == size(rho_blocks) ||
        throw(ArgumentError("Archived mass arrays have inconsistent shapes"))
    total_mass = 0.0
    for element in axes(x_blocks, 3)
        x_first = view(x_blocks, :, 1, element)
        x_second = view(x_blocks, 1, :, element)
        first_spread = maximum(x_first) - minimum(x_first)
        second_spread = maximum(x_second) - minimum(x_second)
        if first_spread >= second_spread
            x_nodes = collect(x_first)
            y_nodes = collect(view(y_blocks, 1, :, element))
            rho_block = Array(view(rho_blocks, :, :, element))
        else
            x_nodes = collect(x_second)
            y_nodes = collect(view(y_blocks, :, 1, element))
            rho_block = permutedims(
                Array(view(rho_blocks, :, :, element)))
        end
        x_order = sortperm(x_nodes)
        y_order = sortperm(y_nodes)
        x_nodes = x_nodes[x_order]
        y_nodes = y_nodes[y_order]
        rho_block = rho_block[x_order, y_order]
        x_weights = rti_archive_quadrature_weights(x_nodes)
        y_weights = rti_archive_quadrature_weights(y_nodes)
        hx = x_nodes[end] - x_nodes[1]
        hy = y_nodes[end] - y_nodes[1]
        element_mass = 0.0
        for j in eachindex(y_weights), i in eachindex(x_weights)
            element_mass += rho_block[i, j] *
                            x_weights[i] * y_weights[j]
        end
        # Reference-element Jacobian for an axis-aligned quadtree element.
        total_mass += 0.25 * hx * hy * element_mass
    end
    return total_mass
end
function rti_recompute_archived_diagnostics!(run)
    state = run.diagnostics
    final_time = tspan[2]
    initial_data = npzread(rti_snapshot_file(state, 0.0))
    final_data = npzread(rti_snapshot_file(state, final_time))
    initial_mass = rti_archived_mass(initial_data, run.polydeg)
    final_mass = rti_archived_mass(final_data, run.polydeg)
    state.initial_mass = initial_mass
    mass_drift = final_mass - initial_mass
    rho_min, rho_max = extrema(final_data["rho"])
    p_min, p_max = extrema(final_data["p"])
    # The archive contains conservative variables, so recover v=rho_v/rho.
    y = vec(final_data["y"])
    v = vec(final_data["rho_v"] ./ final_data["rho"])
    wall_bottom = rti_wall_max_abs_v(y, v, 0.0)
    wall_top = rti_wall_max_abs_v(y, v, 1.0)
    dt_last = isempty(state.accepted_steps) ?
              NaN : state.accepted_steps[end].dt
    state.final_diagnostics = (
        t=final_time,
        mass=final_mass,
        mass_drift=mass_drift,
        relative_mass_drift=abs(mass_drift / initial_mass),
        rho_min=rho_min,
        rho_max=rho_max,
        p_min=p_min,
        p_max=p_max,
        dt_last=dt_last,
        initialization_dt=state.initialization_dt,
        accepted_step_count=length(state.accepted_steps),
        max_abs_v_y0=wall_bottom,
        max_abs_v_y1=wall_top
    )
    return nothing
end
function rti_print_final_diagnostics(run)
    diagnostics = run.diagnostics.final_diagnostics
    diagnostics === nothing &&
        error("Final diagnostics were not recorded for $(run.label)")
    println("\nFinal diagnostic summary recomputed from archived full state: ",
            run.label)
    Printf.@printf("  t=%.5f, mass_drift=%.6e, relative_mass_drift=%.6e, min_rho=%.6e, min_p=%.6e\n",
                   diagnostics.t, diagnostics.mass_drift,
                   diagnostics.relative_mass_drift,
                   diagnostics.rho_min, diagnostics.p_min)
    Printf.@printf("  max_abs_v_y0=%.6e, max_abs_v_y1=%.6e, accepted_steps=%d\n",
                   diagnostics.max_abs_v_y0, diagnostics.max_abs_v_y1,
                   diagnostics.accepted_step_count)
    return nothing
end
function rti_postprocess_stage!(run)
    println("\nPost-processing ", run.label)
    rti_compute_stage_diagnostic_data!(run)
    rti_recompute_archived_diagnostics!(run)
    rti_render_stage!(run)
    rti_print_symmetry_monitor(run)
    if isempty(run.diagnostics.accepted_steps)
        println("\nTIMESTEP AUDIT: ", run.label,
                " skipped; archive-only snapshots contain no accepted-step history")
    else
        rti_print_timestep_audit(run)
    end
    rti_print_final_diagnostics(run)
    return nothing
end
function rti_maybe_run_symmetry_control(runs)
    reference_run = nothing
    largest_residual = -Inf
    for run in runs
        residuals = run.diagnostics.symmetry_residuals[3.0]
        residual = rti_maximum_residual(residuals)
        if residual > largest_residual
            largest_residual = residual
            reference_run = run
        end
    end
    if largest_residual <= 1.0e-2
        Printf.@printf("\nSYMMETRY CONTINGENCY: not triggered; largest corrected t=3 residual=%.6e <= 1.0e-2\n",
                       largest_residual)
        return nothing
    end
    Printf.@printf("\nSYMMETRY CONTINGENCY: triggered by %s with max t=3 residual=%.6e > 1.0e-2\n",
                   reference_run.label, largest_residual)
    control = rti_run_symmetric_fixed_mesh_control(reference_run)
    rti_compute_stage_diagnostic_data!(control;
                                       times=[0.0, 3.0],
                                       store_comparison=false)
    rti_print_symmetry_monitor(control; times=[0.0, 3.0])
    reference = reference_run.diagnostics.symmetry_residuals[3.0]
    fixed = control.diagnostics.symmetry_residuals[3.0]
    println("\nSYMMETRY CONTROL COMPARISON at t=3.0: AMR reference vs symmetric fixed mesh")
    for (name, amr_residual, fixed_residual) in (
        ("rho", reference.rho, fixed.rho),
        ("rho_u", reference.rho_u, fixed.rho_u),
        ("rho_v", reference.rho_v, fixed.rho_v),
        ("rho_E", reference.rho_e, fixed.rho_e)
    )
        Printf.@printf("  %s: AMR=%.16e, fixed_mesh=%.16e\n",
                       name, amr_residual, fixed_residual)
    end
    return control
end
function rti_relative_shift(baseline, candidate)
    if baseline == 0
        return candidate == 0 ? 0.0 : NaN
    end
    return (candidate - baseline) / abs(baseline)
end
rti_material_label(final_time, flag) =
    final_time ? string(flag) : "informational"
function rti_resolution_decision(stage_a, stage_b)
    println("\nRESOLUTION CHECK")
    println("  identical cell-centered grid: $(rti_comparison_nx)x$(rti_comparison_ny)")
    println("  metric | polydeg=5 | polydeg=6 | relative shift | material?")
    m1 = false
    m2 = false
    m3_count = false
    m3_new = false
    for target_time in rti_comparison_times
        metrics_a = stage_a.diagnostics.fixed_grid_samples[target_time].metrics
        metrics_b = stage_b.diagnostics.fixed_grid_samples[target_time].metrics
        final_time = isapprox(target_time, 3.0;
                              rtol=0.0, atol=1.0e-12)
        area_shift = rti_relative_shift(metrics_a.mixing_zone_count,
                                        metrics_b.mixing_zone_count)
        sharpness_shift = rti_relative_shift(metrics_a.sharpness,
                                             metrics_b.sharpness)
        structure_shift = rti_relative_shift(metrics_a.structural_count,
                                             metrics_b.structural_count)
        substructure_shift =
            rti_relative_shift(metrics_a.upper_mixing_component_count,
                               metrics_b.upper_mixing_component_count)
        if final_time
            m1 = isfinite(area_shift) && abs(area_shift) > 0.10
            m2 = isfinite(sharpness_shift) && abs(sharpness_shift) > 0.10
            m3_count = metrics_a.structural_count !=
                       metrics_b.structural_count
            # Deterministic identical-grid proxy for additional resolved
            # substructures in the upper mushroom/vortex region.
            m3_new = metrics_b.upper_mixing_component_count >
                     metrics_a.upper_mixing_component_count
        end
        println("  t=$(target_time)")
        Printf.@printf("  A_mix [# cells] | %d | %d | %+.6e | %s\n",
                       metrics_a.mixing_zone_count,
                       metrics_b.mixing_zone_count, area_shift,
                       rti_material_label(final_time, m1))
        Printf.@printf("  A_mix physical area | %.16e | %.16e | %+.6e | %s\n",
                       metrics_a.mixing_zone_area,
                       metrics_b.mixing_zone_area, area_shift,
                       rti_material_label(final_time, m1))
        Printf.@printf("  S mean |grad(rho)| | %.16e | %.16e | %+.6e | %s\n",
                       metrics_a.sharpness, metrics_b.sharpness,
                       sharpness_shift,
                       rti_material_label(final_time, m2))
        Printf.@printf("  structural count | %d | %d | %+.6e | %s\n",
                       metrics_a.structural_count,
                       metrics_b.structural_count, structure_shift,
                       rti_material_label(final_time, m3_count))
        Printf.@printf("  upper-vortex mixing components | %d | %d | %+.6e | %s\n",
                       metrics_a.upper_mixing_component_count,
                       metrics_b.upper_mixing_component_count,
                       substructure_shift,
                       rti_material_label(final_time, m3_new))
    end
    m3 = m3_count || m3_new
    material = m1 || m2 || m3
    println("  M1 (A_mix shift >10% at t=3): ", m1)
    println("  M2 (S shift >10% at t=3): ", m2)
    println("  M3 (structural count change or new upper-vortex components): ",
            m3)
    if material
        println("VERDICT: MATERIAL — promoting polydeg=6, CFL=0.30 to production")
    else
        println("VERDICT: NOT MATERIAL — retaining polydeg=5, CFL=0.35 (Run 3 configuration)")
    end
    return (material=material, m1=m1, m2=m2, m3=m3)
end
function rti_compare_stage_b_baseline(run, baseline)
    println("\nSTAGE B BASELINE COMPARISON")
    if baseline === nothing
        println("  Previous archive unavailable; checking fresh diagnostics against the specified numerical references")
        println("  Rendered-field comparison: SKIP (no archived Stage B fields)")

        current = run.diagnostics.final_diagnostics
        current === nothing &&
            error("Final diagnostics are unavailable for $(run.label)")
        current_symmetry = rti_maximum_residual(
            run.diagnostics.symmetry_residuals[3.0])

        # These references are rounded numerical baselines. Their tolerances
        # include the supplied reference precision as well as roundoff-level
        # reproduction error.
        comparisons = (
            ("relative_mass_drift", current.relative_mass_drift,
             7.25e-12, 1.0e-12, 5.0e-2),
            ("min_rho", current.rho_min,
             0.1920849, 1.0e-6, 1.0e-6),
            ("min_p", current.p_min,
             1.208287, 1.0e-6, 1.0e-6),
            ("max_abs_v_y0", current.max_abs_v_y0,
             5.63e-5, 1.0e-7, 1.0e-3),
            ("max_abs_v_y1", current.max_abs_v_y1,
             3.73e-4, 1.0e-6, 1.0e-3),
            ("t3_symmetry", current_symmetry,
             1.126302e-4, 1.0e-8, 1.0e-4)
        )

        comparison_pass = current.rho_min > 0.0 && current.p_min > 0.0
        println("  positivity of fresh archived rho/p: ",
                comparison_pass ? "PASS" : "FAIL")

        for (name, value, target, atol, rtol) in comparisons
            metric_pass = isfinite(value) &&
                          isapprox(value, target; atol=atol, rtol=rtol)
            comparison_pass = comparison_pass && metric_pass
            Printf.@printf("  %s: fresh=%.16e, reference=%.16e, |delta|=%.3e, %s\n",
                           name, value, target, abs(value - target),
                           metric_pass ? "PASS" : "FAIL")
        end

        println("STAGE B BASELINE COMPARISON (numerical reference fallback): ",
                comparison_pass ? "PASS" : "FAIL")
        return comparison_pass
    end
    # Only t=3 is needed for the baseline symmetry residual. Mass, positivity,
    # and wall diagnostics are recomputed from the archived t=0/t=3 states.
    rti_compute_stage_diagnostic_data!(baseline;
                                       times=[3.0],
                                       store_comparison=false)
    rti_recompute_archived_diagnostics!(baseline)
    current = run.diagnostics.final_diagnostics
    reference = baseline.diagnostics.final_diagnostics
    current_symmetry = rti_maximum_residual(
        run.diagnostics.symmetry_residuals[3.0])
    reference_symmetry = rti_maximum_residual(
        baseline.diagnostics.symmetry_residuals[3.0])
    comparisons = (
        ("relative_mass_drift", current.relative_mass_drift,
         reference.relative_mass_drift, 1.0e-12, 5.0e-2),
        ("min_rho", current.rho_min, reference.rho_min,
         1.0e-8, 1.0e-8),
        ("min_p", current.p_min, reference.p_min,
         1.0e-8, 1.0e-8),
        ("max_abs_v_y0", current.max_abs_v_y0,
         reference.max_abs_v_y0, 1.0e-9, 1.0e-5),
        ("max_abs_v_y1", current.max_abs_v_y1,
         reference.max_abs_v_y1, 1.0e-9, 1.0e-5),
        ("t3_symmetry", current_symmetry, reference_symmetry,
         1.0e-8, 1.0e-5)
    )
    comparison_pass = current.rho_min > 0.0 && current.p_min > 0.0
    println("  positivity of fresh archived rho/p: ",
            comparison_pass ? "PASS" : "FAIL")
    for (name, value, target, atol, rtol) in comparisons
        metric_pass = isfinite(value) &&
                      isapprox(value, target; atol=atol, rtol=rtol)
        comparison_pass = comparison_pass && metric_pass
        Printf.@printf("  %s: fresh=%.16e, baseline=%.16e, |delta|=%.3e, %s\n",
                       name, value, target, abs(value - target),
                       metric_pass ? "PASS" : "FAIL")
    end
    # Use the renderer's default sampling resolution and endpoint-inclusive
    # grid, rather than the higher-resolution symmetry diagnostic grid.
    for target_time in visualization_times
        fresh_data = npzread(rti_snapshot_file(run.diagnostics, target_time))
        baseline_data =
            npzread(rti_snapshot_file(baseline.diagnostics, target_time))
        for variable in ("rho", "p")
            fresh_grid = rti_sample_uniform_grid(
                fresh_data, variable; degree=run.polydeg)[3]
            baseline_grid = rti_sample_uniform_grid(
                baseline_data, variable; degree=baseline.polydeg)[3]
            if size(fresh_grid) != size(baseline_grid)
                comparison_pass = false
                println("  rendered $variable at t=$target_time: FAIL " *
                        "(grid dimensions differ)")
                continue
            end
            difference = maximum(abs.(fresh_grid .- baseline_grid))
            scale = max(1.0, maximum(abs.(baseline_grid)))
            tolerance = 1.0e-10 + 1.0e-8 * scale
            field_pass = isfinite(difference) && difference <= tolerance
            comparison_pass = comparison_pass && field_pass
            Printf.@printf("  rendered %s, t=%.2f: max_abs_delta=%.3e, tolerance=%.3e, %s\n",
                           variable, target_time, difference, tolerance,
                           field_pass ? "PASS" : "FAIL")
        end
    end
    println("STAGE B BASELINE COMPARISON: ",
            comparison_pass ? "PASS" : "FAIL")
    return comparison_pass
end
println("Generating Stage B parity-aware verification and summary plot...")
# Only the fresh Stage B production dataset is post-processed. No Stage A
# solve or resolution-refinement decision is performed in this iteration.
rti_postprocess_stage!(rti_stage_b)
# Enforce the regression gate on the fresh run's unaveraged accepted steps.
rti_stage_b_audit = rti_timestep_audit(rti_stage_b.diagnostics;
                                       final_time=tspan[2])
println("\nTIMESTEP REGRESSION GATE: ",
        rti_stage_b_audit.regression_pass ? "PASS" : "FAIL")
rti_stage_b_audit.regression_pass ||
    error("Timestep regression gate failed; inspect accepted_step_history.npz")
rti_stage_b_baseline_status =
    rti_compare_stage_b_baseline(rti_stage_b, rti_stage_b_baseline)
rti_stage_b_baseline_status === true ||
    error("The fresh Stage B run did not reproduce the required Stage B baseline")
# This gate consumes only the newly computed parity-aware residuals. It cannot
# use the obsolete even-formula rho_u residual that previously produced 2.0.
rti_symmetry_control =
    rti_maybe_run_symmetry_control((rti_stage_b,))
rti_selected_run = rti_stage_b
isfile(rti_selected_run.figure_file) ||
    error("Selected Stage B figure is missing: $(rti_selected_run.figure_file)")
summary_path = joinpath(@__DIR__, "summary_all.png")
cp(rti_selected_run.figure_file, summary_path; force=true)
println("\nSelected production run: ", rti_selected_run.label)
println("Selected production configuration: polydeg=$(rti_selected_run.polydeg), CFL=$(rti_selected_run.cfl)")
println("Stage B retained verbatim; no resolution-refinement verdict was applied")
println("Saved Stage B figure as summary_all.png")
selected_diagnostics = rti_selected_run.diagnostics.final_diagnostics
selected_symmetry =
    rti_selected_run.diagnostics.symmetry_residuals[3.0]
println("Final selected-run diagnostic summary:")
Printf.@printf("  t=%.5f, mass_drift=%.6e, relative_mass_drift=%.6e, min_rho=%.6e, min_p=%.6e\n",
               selected_diagnostics.t,
               selected_diagnostics.mass_drift,
               selected_diagnostics.relative_mass_drift,
               selected_diagnostics.rho_min,
               selected_diagnostics.p_min)
Printf.@printf("  max_abs_v_y0=%.6e, max_abs_v_y1=%.6e, max_t3_parity_aware_symmetry_residual=%.6e\n",
               selected_diagnostics.max_abs_v_y0,
               selected_diagnostics.max_abs_v_y1,
               rti_maximum_residual(selected_symmetry))
println("Done.")