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
polydeg = 3
basis = LobattoLegendreBasis(polydeg)

@inline function pressure_indicator(u, equations::CompressibleEulerEquations2D)
    rho, rho_v1, rho_v2, rho_e = u
    return (equations.gamma - 1) *
           (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) / rho)
end

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=1.0,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=pressure_indicator)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=flux_ranocha,
                                                 volume_flux_fv=flux_lax_friedrichs)
dg = DGSEM(polydeg=polydeg, surface_flux=flux_hllc,
           volume_integral=volume_integral)
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

mesh = P4estMesh((1, 1), polydeg=polydeg,
                 coordinates_min=coordinates_min,
                 coordinates_max=coordinates_max,
                 initial_refinement_level=3,
                 periodicity=(true, false))
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
# Saves the complete conservative state (ρ, ρu, ρv, ρE) every 0.1 time
# units. Additional output at t = 0.75 and t = 2.25 ensures that all
# requested visualization times are represented exactly. Pressure is also
# exported for the required density/pressure plots.
# %%
npz_out = joinpath(@__DIR__, "data_npz")
mkpath(npz_out)

dt_save = 0.1
visualization_times = [0.75, 1.5, 2.25, 3.0]
save_times = sort(unique(vcat(collect(0.0:dt_save:3.0),
                              visualization_times)))

# Diagnostic state; these do not modify the solver or its configuration.
rti_initial_mass = Ref{Float64}(NaN)
rti_dt_min = Ref{Float64}(Inf)
rti_dt_max = Ref{Float64}(0.0)

@inline function rti_mass_integrand(u, args...)
    return u[1]
end

function rti_total_mass(u_ode, semi)
    mesh, equations_local, solver, cache =
        Trixi.mesh_equations_solver_cache(semi)
    u = Trixi.wrap_array(u_ode, mesh, equations_local, solver, cache)

    # Use the DG quadrature and mesh metric terms, rather than a nodal sum.
    return Trixi.integrate(rti_mass_integrand, u, mesh, equations_local,
                           solver, cache)
end

function rti_record_dt!(integrator)
    if integrator.t > 0.0
        dt = integrator.t - integrator.tprev
        if !isfinite(dt) || dt <= 0.0
            dt = integrator.dt
        end
        if isfinite(dt) && dt > 0.0
            rti_dt_min[] = min(rti_dt_min[], dt)
            rti_dt_max[] = max(rti_dt_max[], dt)
        end
    end
    return nothing
end

function rti_wall_max_abs_v(y, v, wall_y)
    distances = abs.(y .- wall_y)
    tolerance = 1.0e-10
    mask = distances .<= tolerance

    # Retain a fallback for floating-point variations in boundary coordinates.
    if !any(mask)
        mask = distances .<= minimum(distances) + tolerance
    end
    return maximum(abs.(v[mask]))
end

function rti_symmetry_residual(x, y, state)
    # Pair scattered AMR samples in common bins. The mirror of bin i is
    # nx + 1 - i, corresponding to x -> 0.25 - x about x = 0.125.
    nx, ny = 16, 64
    sums = zeros(Float64, 4, ny, nx)
    counts = zeros(Int, ny, nx)

    for k in eachindex(x)
        if isfinite(x[k]) && isfinite(y[k])
            i = clamp(floor(Int, (x[k] / 0.25) * nx) + 1, 1, nx)
            j = clamp(floor(Int, y[k] * ny) + 1, 1, ny)
            counts[j, i] += 1
            for n in 1:4
                sums[n, j, i] += state[n][k]
            end
        end
    end

    numerator = 0.0
    denominator = 0.0
    for j in 1:ny, i in 1:div(nx, 2)
        i_mirror = nx + 1 - i
        count_left = counts[j, i]
        count_right = counts[j, i_mirror]

        if count_left > 0 && count_right > 0
            for n in 1:4
                q_left = sums[n, j, i] / count_left
                q_right = sums[n, j, i_mirror] / count_right
                numerator += (q_left - q_right)^2
                denominator += 0.5 * (q_left^2 + q_right^2)
            end
        end
    end

    return denominator > 0.0 ? sqrt(numerator / denominator) : NaN
end

function save_npz_file(integrator)
    semi = integrator.p
    t = integrator.t

    # Full conservative state for Ground Truth.
    pd_cons = PlotData2D(integrator.u, semi;
                         solution_variables=cons2cons)
    rho = getindex.(pd_cons.data, 1)
    rho_u = getindex.(pd_cons.data, 2)
    rho_v = getindex.(pd_cons.data, 3)
    rho_e = getindex.(pd_cons.data, 4)

    # Pressure is a primary visualization variable in the strategy.
    pd_prim = PlotData2D(integrator.u, semi;
                         solution_variables=cons2prim)
    p = getindex.(pd_prim.data, 4)

    x = vec(pd_cons.x)
    y = vec(pd_cons.y)
    rho_vec = vec(rho)
    rho_u_vec = vec(rho_u)
    rho_v_vec = vec(rho_v)
    rho_e_vec = vec(rho_e)
    v_vec = vec(getindex.(pd_prim.data, 3))

    mass = rti_total_mass(integrator.u, semi)
    if t == 0.0 || !isfinite(rti_initial_mass[])
        rti_initial_mass[] = mass
    end
    mass_drift = mass - rti_initial_mass[]
    relative_mass_drift = mass_drift / rti_initial_mass[]

    rho_min, rho_max = extrema(rho)
    p_min, p_max = extrema(p)
    wall_bottom = rti_wall_max_abs_v(y, v_vec, 0.0)
    wall_top = rti_wall_max_abs_v(y, v_vec, 1.0)
    symmetry = rti_symmetry_residual(
        x, y, (rho_vec, rho_u_vec, rho_v_vec, rho_e_vec))

    dt_last = NaN
    if t > 0.0
        dt_last = t - integrator.tprev
        if !isfinite(dt_last) || dt_last <= 0.0
            dt_last = integrator.dt
        end
    end
    dt_min = isfinite(rti_dt_min[]) ? rti_dt_min[] : NaN
    dt_max = rti_dt_max[] > 0.0 ? rti_dt_max[] : NaN

    filename = joinpath(npz_out,
                        Printf.@sprintf("time_%08.5f.npz", t))

    npzwrite(filename, Dict("t" => t,
                            "x" => pd_cons.x,
                            "y" => pd_cons.y,
                            "rho" => rho,
                            "rho_u" => rho_u,
                            "rho_v" => rho_v,
                            "rho_e" => rho_e,
                            "p" => p))

    println("Saved: $(basename(filename))")
    Printf.@printf("Diagnostics t=%.5f: mass=%.16e, mass_drift=%.3e, relative_mass_drift=%.3e\n",
                   t, mass, mass_drift, relative_mass_drift)
    Printf.@printf("  rho_min=%.6e, rho_max=%.6e, p_min=%.6e, p_max=%.6e\n",
                   rho_min, rho_max, p_min, p_max)
    Printf.@printf("  dt_last=%.3e, dt_min=%.3e, dt_max=%.3e, max_abs_v_y0=%.3e, max_abs_v_y1=%.3e, symmetry_residual=%.3e\n",
                   dt_last, dt_min, dt_max, wall_bottom, wall_top, symmetry)
    return nothing
end

# Observe every accepted step for timestep extrema, then save at preset times.
rti_dt_callback = DiscreteCallback((u, t, integrator) -> true,
                                   rti_record_dt!;
                                   save_positions=(false, false))
save_npz_preset_callback = PresetTimeCallback(save_times, save_npz_file;
                                               save_positions=(false, false))
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
# Save the complete conservative state for Ground Truth and restarts.
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
# ## Methodology: Complete Rasterization for Scattered DG Data
#
# Samples are averaged into uniform bins on Ω = [0, 0.25] × [0, 1].
# Empty bins are then filled by separable linear interpolation: first
# vertically within populated columns, then horizontally across any
# remaining gaps. Unlike nearest-neighbor splatting with NaN pixels, this
# produces a complete image without transparent rows or mesh-like stripes.
# The interpolation is visualization-only and does not modify the saved
# full-state Ground Truth data.
# %%
function rasterize_data(x, y, z, nx=256, ny=1024)
    (nx > 0 && ny > 0) || throw(ArgumentError("Raster dimensions must be positive"))
    (length(x) == length(y) == length(z)) ||
        throw(ArgumentError("x, y, and z must have equal lengths"))

    x_min, x_max = 0.0, 0.25
    y_min, y_max = 0.0, 1.0
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny

    sums = zeros(Float64, ny, nx)
    counts = zeros(Int, ny, nx)

    # Average all finite samples falling into each pixel.
    for k in eachindex(x)
        xk, yk, zk = x[k], y[k], z[k]
        if isfinite(xk) && isfinite(yk) && isfinite(zk) &&
           x_min <= xk <= x_max && y_min <= yk <= y_max
            i = clamp(floor(Int, (xk - x_min) / dx) + 1, 1, nx)
            j = clamp(floor(Int, (yk - y_min) / dy) + 1, 1, ny)
            sums[j, i] += zk
            counts[j, i] += 1
        end
    end

    grid = fill(NaN, ny, nx)
    for j in 1:ny, i in 1:nx
        if counts[j, i] > 0
            grid[j, i] = sums[j, i] / counts[j, i]
        end
    end
    any(isfinite, grid) || error("No finite samples inside the plotting domain")

    # Fill one-dimensional gaps without overwriting measured/bin-averaged
    # pixels. Constant extrapolation is used only at the two endpoints.
    function interpolate_gaps!(values)
        valid = findall(isfinite, values)
        isempty(valid) && return nothing

        first_valid = first(valid)
        last_valid = last(valid)

        if first_valid > 1
            values[1:(first_valid - 1)] .= values[first_valid]
        end
        if last_valid < length(values)
            values[(last_valid + 1):end] .= values[last_valid]
        end

        for n in 1:(length(valid) - 1)
            left = valid[n]
            right = valid[n + 1]
            if right > left + 1
                value_left = values[left]
                value_right = values[right]
                for k in (left + 1):(right - 1)
                    weight = (k - left) / (right - left)
                    values[k] = (1 - weight) * value_left +
                                weight * value_right
                end
            end
        end
        return nothing
    end

    # Vertical pass: populate complete columns where samples exist.
    for i in 1:nx
        interpolate_gaps!(view(grid, :, i))
    end

    # Horizontal pass: fill columns that contained no original samples.
    # At least one populated column guarantees every row can be completed.
    for j in 1:ny
        interpolate_gaps!(view(grid, j, :))
    end

    all(isfinite, grid) || error("Failed to construct a complete raster")
    return grid
end
# %% [markdown]


# ## Cell 12: Post-Processing & Temporal Diagnostics
# ## Rayleigh–Taylor Density and Pressure Visualization
#
# Creates a 2 × 4 summary figure at t = 0.75, 1.5, 2.25, and 3.0.
# The top row shows density and the bottom row shows pressure. Each panel
# uses automatic color scaling, with no fixed colormap or colorbar.
# %%
println("Generating Rayleigh–Taylor summary plots...")
plot_times = [0.75, 1.5, 2.25, 3.0]
snapshots = []
for t_target in plot_times
    filename = joinpath(npz_out,
                        Printf.@sprintf("time_%08.5f.npz", t_target))
    isfile(filename) || error("Missing required snapshot at t = $t_target: $filename")
    push!(snapshots, npzread(filename))
end
style = (
    seriestype = :heatmap,
    aspect_ratio = :equal,
    xlims = (0.0, 0.25),
    ylims = (0.0, 1.0),
    yflip = false,
    axis = nothing,
    framestyle = :none,
    legend = false,
    colorbar = false,
    margin = 2Plots.mm
)

plots_array = []

# Top row: density at the four requested times. Each heatmap receives its
# own data range, so color scaling remains automatic and panel-local.
for (data, t_target) in zip(snapshots, plot_times)
    x = vec(data["x"])
    y = vec(data["y"])
    rho_grid = rasterize_data(x, y, vec(data["rho"]))

    x_grid = range(0.0, 0.25, length=size(rho_grid, 2))
    y_grid = range(0.0, 1.0, length=size(rho_grid, 1))

    push!(plots_array,
          heatmap(x_grid, y_grid, rho_grid,
                  title="ρ, t=$(t_target)"; style...))
end

# Bottom row: pressure at the same four times, with no mesh or element
# boundaries and no explicitly prescribed colormap or color limits.
for (data, t_target) in zip(snapshots, plot_times)
    x = vec(data["x"])
    y = vec(data["y"])
    p_grid = rasterize_data(x, y, vec(data["p"]))

    x_grid = range(0.0, 0.25, length=size(p_grid, 2))
    y_grid = range(0.0, 1.0, length=size(p_grid, 1))

    push!(plots_array,
          heatmap(x_grid, y_grid, p_grid,
                  title="p, t=$(t_target)"; style...))
end

println("Saving summary_all.png...")
final_plot = plot(plots_array..., layout=(2, 4),
                  size=(3200, 2400), margin=5Plots.mm)
savefig(final_plot, joinpath(@__DIR__, "summary_all.png"))
println("Done.")
