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
# ## Governing Equations: Inviscid Hyperbolic System
# Defines the two-dimensional compressible Euler equations:
# $$
# \frac{\partial \mathbf{u}}{\partial t} + \nabla \cdot \mathbf{f}(\mathbf{u})
# = \mathbf{S}(\mathbf{u}),
# $$
# where the gravitational source term is defined in the initial-condition/source-term cell.
#
# * **Inviscid Flux**: Conservation of mass, momentum, and total energy.
# * **Equation of State**: Ideal gas with $\gamma = 1.4$.
# * **Parabolic Terms**: None; the Rayleigh–Taylor configuration uses $\mu = 0$
#   and zero heat conduction.

# %%
equations = CompressibleEulerEquations2D(1.4)
# %% [markdown]

# ## Cell 3: Initial Conditions and Source Terms (Lid-Driven Cavity)
# ## Problem Setup: Single-Mode Rayleigh–Taylor Instability
# Initializes a heavy fluid below a light fluid, with a smoothed material
# interface at $y = 0.5$. A single-mode vertical-velocity perturbation seeds
# the instability. Gravity acts in the positive $y$-direction.
#
# The primitive state $(\rho, u, v, p)$ is converted to the conservative
# state required by the compressible Euler solver.

# %%
@inline function initial_condition_rayleigh_taylor_instability(
        x, t, equations::CompressibleEulerEquations2D)
    x_coord, y = x

    # Heavy fluid ($\rho = 2$) below the interface and light fluid
    # ($\rho = 1$) above it.
    rho = 2.0 + (1.0 - 2.0) * 0.5 * (1.0 + tanh(50.0 * (y - 0.5)))

    # Piecewise hydrostatic pressure background.
    p = y < 0.5 ? 2.0 * y + 1.0 : y + 1.5

    sound_speed = sqrt(equations.gamma * p / rho)
    u = 0.0
    v = -0.025 * sound_speed * cos(8.0 * pi * x_coord) * sin(pi * y)^6

    return prim2cons(SVector(rho, u, v, p), equations)
end

@inline function source_terms_gravity(u, x, t,
                                      equations::CompressibleEulerEquations2D)
    g = 1.0
    rho, rho_v1, rho_v2, rho_e = u

    # Vertical momentum source $\rho g$ and energy source $\rho g v$.
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
#     * **Role**: Computes the numerical flux across element interfaces where the solution is discontinuous.
#     * **Physics**: Explicitly restores the contact wave, reducing dissipation at the material interface while retaining robust shock resolution.
#     * **Physics**: An approximate Riemann solver that estimates the fastest signal velocities (wave speeds). It is robust and positivity-preserving, making it ideal for compressible flows with strong shocks.
#
# 4.  **Indicator Function (`IndicatorHennemannGassner`)**:
#     * Dynamically computes the blending coefficient $\alpha \in [0, 1]$ element-wise.
#      * **Mechanism**: Monitors the spectral decay of energy in the highest modes of the polynomial expansion. If energy in high modes is large (indicating a discontinuity), $\alpha \to 1$ (activates FV sub-cells).
#
# %%
polydeg = 5
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=1.0,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_indicator)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=flux_ranocha,
                                                 volume_flux_fv=flux_lax_friedrichs)
dg = DGSEM(polydeg=polydeg, surface_flux=flux_hllc, volume_integral=volume_integral)
# %% [markdown]


# ## Cell 5: Computational Domain & Adaptive Mesh Topology
# ## Methodology: Forest-of-Octrees (`p4est`)
# Defines the Rayleigh–Taylor domain and its adaptive quadtree topology.
#
# * **Domain**: $\Omega = [0, 0.25] \times [0, 1]$.
# * **Initial Resolution**: `initial_refinement_level=3` provides the template's
#   initial $8 \times 8$ element grid; subsequent resolution changes are handled by AMR.
# * **Boundary Topology**: Periodic in $x$, with physical wall boundaries at
#   $y=0$ and $y=1$.

# %%
coordinates_min = (0.0, 0.0)
coordinates_max = (0.25, 1.0)

mesh = P4estMesh((1, 1), polydeg=polydeg,
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max,
                 initial_refinement_level=3, # Start coarser, let AMR refine
                 periodicity=(true, false)) # Periodic in x; walls in y
# %% [markdown]

# ## Cell 6: General Boundary Condition Specification & System Assembly
# ## Boundary Conditions and Hyperbolic System Assembly
# The mesh supplies periodic boundaries in the $x$-direction. At $y=0$ and
# $y=1$, `boundary_condition_slip_wall` enforces reflective, free-slip walls
# with zero normal velocity. No viscous boundary conditions are required.
#
# `SemidiscretizationHyperbolic` combines the mesh, compressible Euler
# equations, DGSEM solver, initial condition, wall boundaries, and gravity
# source into the semi-discrete operator.

# %%
boundary_conditions = Dict(
    :y_neg => boundary_condition_slip_wall,
    :y_pos => boundary_condition_slip_wall
)

semi = SemidiscretizationHyperbolic(mesh, equations,
                                    initial_condition_rayleigh_taylor_instability, dg;
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
#     * **Mechanism**: It estimates the local smoothness of the solution by computing a normalized second derivative (curvature) of the **density**, concentrating refinement on the material interface and roll-up edges.
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
#     * **Trigger**: Interrupts the time-integration loop every 10 timesteps (`interval=10`).
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
                           interval=10,
                           adapt_initial_condition=true)
# %% [markdown]


# ## Cell 8: Data I/O & Surrogate Interoperability
# ## Strategy: Cross-Language Serialization (.npz)
# Exports complete solution snapshots for ground-truth storage and subsequent
# visualization. Conservative variables contain the evolved state, while
# primitive variables provide direct access to (rho, u, v, p).
#
# Output is scheduled every `dt_save = 0.1`. The visualization times
# t = 0.75, 1.5, 2.25, 3.0 are added to the schedule so they are available
# without reducing the ground-truth saving cadence.
# %%
npz_out = joinpath(@__DIR__, "data_npz")
mkpath(npz_out)
dt_save = 0.1
npz_visualization_times = [0.75, 1.5, 2.25, 3.0]
npz_save_times = sort(unique(vcat(collect(0.0:dt_save:3.0),
                                  npz_visualization_times)))
# Index 1 is t=0, which is written by the callback initializer.
npz_next_save_index = Ref(2)

# State retained between saves for conservation and energy-balance diagnostics.
diagnostic_initial_mass = Ref(NaN)
diagnostic_previous_time = Ref(NaN)
diagnostic_previous_energy = Ref(NaN)

# Quadrature integrands for Trixi.integrate. Since g=1, rho*g*v equals rho_v.
@inline diagnostic_mass(u, equations) = u[1]
@inline diagnostic_energy(u, equations) = u[4]
@inline diagnostic_source_work(u, equations) = u[3]

diagnostic_scalar(x::Number) = Float64(x)
diagnostic_scalar(x) = Float64(only(x))

function interface_metrics(rho_values, y_values)
    rho_vec = vec(rho_values)
    y_vec = vec(y_values)

    # Highest heavy-fluid point, approximating the apex of the rho=1.5 contour.
    heavy = rho_vec .>= 1.5
    apex = any(heavy) ? maximum(y_vec[heavy]) : NaN

    # Vertical extent of the intermediate-density mixing region.
    mixed = (rho_vec .> 1.05) .& (rho_vec .< 1.95)
    width = any(mixed) ? maximum(y_vec[mixed]) - minimum(y_vec[mixed]) : 0.0

    return apex, width
end

function wall_velocity_extrema(y_values, v_values, wall_y)
    y_vec = vec(y_values)
    v_vec = vec(v_values)
    distances = abs.(y_vec .- wall_y)

    indices = findall(d -> d <= 1.0e-10, distances)
    if isempty(indices)
        # Fall back to the closest plotted wall plane if coordinates differ by
        # more than roundoff from the exact wall location.
        closest_distance = minimum(distances)
        indices = findall(d -> d <= closest_distance + 1.0e-12, distances)
    end

    return extrema(v_vec[indices])
end

function compare_polydeg3_baseline(t, current)
    baseline_dir = joinpath(@__DIR__, "data_npz_polydeg3")
    isdir(baseline_dir) || return nothing

    files = sort([joinpath(baseline_dir, file) for file in readdir(baseline_dir)
                  if endswith(file, ".npz")])
    isempty(files) && return nothing

    times = Float64[diagnostic_scalar(npzread(file)["t"]) for file in files]
    index = argmin(abs.(times .- t))
    abs(times[index] - t) <= 1.0e-8 || return nothing

    data = npzread(files[index])
    baseline_apex, baseline_width = interface_metrics(data["rho"], data["y"])

    Printf.@printf("  polydeg=3 comparison: Δapex=%.6e, Δmixing_width=%.6e\n",
                   current.interface_apex - baseline_apex,
                   current.mixing_layer_width - baseline_width)

    if haskey(data, "total_mass")
        baseline_mass = diagnostic_scalar(data["total_mass"])
        Printf.@printf("  polydeg=3 comparison: Δmass=%.6e\n",
                       current.total_mass - baseline_mass)
    end

    if haskey(data, "energy_tendency") && haskey(data, "source_work")
        baseline_energy_tendency = diagnostic_scalar(data["energy_tendency"])
        baseline_source_work = diagnostic_scalar(data["source_work"])
        Printf.@printf("  polydeg=3 comparison: ΔdE/dt=%.6e, Δsource_work=%.6e\n",
                       current.energy_tendency - baseline_energy_tendency,
                       current.source_work - baseline_source_work)
    end

    if haskey(data, "energy_balance_residual")
        baseline_residual = diagnostic_scalar(data["energy_balance_residual"])
        Printf.@printf("  polydeg=3 comparison: Δenergy_residual=%.6e\n",
                       current.energy_balance_residual - baseline_residual)
    end

    if !haskey(data, "total_mass") || !haskey(data, "energy_tendency")
        println("  Legacy polydeg=3 snapshot lacks saved integral diagnostics; " *
                "only the available convergence metrics are compared.")
    end

    return nothing
end

function save_npz_file(integrator)
    semi = integrator.p
    t = integrator.t

    @assert all(isfinite, integrator.u) "Non-finite conservative state at t=$t"

    # Reconstruct all conservative and primitive variables on the same grid.
    pd_cons = PlotData2D(integrator.u, semi, solution_variables=cons2cons)
    pd_prim = PlotData2D(integrator.u, semi, solution_variables=cons2prim)

    conservative_rho = getindex.(pd_cons.data, 1)
    rho_u = getindex.(pd_cons.data, 2)
    rho_v = getindex.(pd_cons.data, 3)
    rho_e = getindex.(pd_cons.data, 4)

    rho = getindex.(pd_prim.data, 1)
    u = getindex.(pd_prim.data, 2)
    v = getindex.(pd_prim.data, 3)
    p = getindex.(pd_prim.data, 4)

    rho_min = minimum(rho)
    p_min = minimum(p)
    @assert rho_min > 0.0 "Non-positive density at t=$t: min(rho)=$rho_min"
    @assert p_min > 0.0 "Non-positive pressure at t=$t: min(p)=$p_min"

    # Use the DG quadrature, rather than point sums on the nonuniform plot grid.
    mass = Trixi.integrate(diagnostic_mass, integrator.u, semi)
    energy = Trixi.integrate(diagnostic_energy, integrator.u, semi)
    source_work = Trixi.integrate(diagnostic_source_work, integrator.u, semi)

    if !isfinite(diagnostic_initial_mass[])
        diagnostic_initial_mass[] = mass
    end
    mass_drift = mass - diagnostic_initial_mass[]

    energy_tendency = NaN
    if isfinite(diagnostic_previous_time[]) && t > diagnostic_previous_time[]
        energy_tendency = (energy - diagnostic_previous_energy[]) /
                          (t - diagnostic_previous_time[])
    end
    energy_balance_residual = energy_tendency - source_work

    bottom_wall_v = wall_velocity_extrema(pd_prim.y, v, 0.0)
    top_wall_v = wall_velocity_extrema(pd_prim.y, v, 1.0)
    interface_apex, mixing_layer_width = interface_metrics(rho, pd_prim.y)

    current_metrics = (interface_apex=interface_apex,
                       mixing_layer_width=mixing_layer_width,
                       total_mass=mass,
                       energy_tendency=energy_tendency,
                       source_work=source_work,
                       energy_balance_residual=energy_balance_residual)

    Printf.@printf("RTI diagnostics t=%.5f: min(rho)=%.6e, min(p)=%.6e\n",
                   t, rho_min, p_min)
    Printf.@printf("  wall v extrema: y=0 [%.6e, %.6e], y=1 [%.6e, %.6e]\n",
                   bottom_wall_v[1], bottom_wall_v[2],
                   top_wall_v[1], top_wall_v[2])
    Printf.@printf("  mass=%.12e, mass drift=%.6e\n", mass, mass_drift)
    Printf.@printf("  energy=%.12e, dE/dt=%.6e, source work=%.6e, residual=%.6e\n",
                   energy, energy_tendency, source_work, energy_balance_residual)
    Printf.@printf("  interface apex=%.6e, mixing-layer width=%.6e\n",
                   interface_apex, mixing_layer_width)

    compare_polydeg3_baseline(t, current_metrics)

    filename = joinpath(npz_out, Printf.@sprintf("time_%08.5f.npz", t))

    npzwrite(filename, Dict("t" => t,
                            "x" => pd_prim.x,
                            "y" => pd_prim.y,
                            "rho" => rho,
                            "u" => u,
                            "v" => v,
                            "p" => p,
                            "conservative_rho" => conservative_rho,
                            "rho_u" => rho_u,
                            "rho_v" => rho_v,
                            "rho_e" => rho_e,
                            "state_vector" => copy(integrator.u),
                            "min_rho" => rho_min,
                            "min_p" => p_min,
                            "total_mass" => mass,
                            "mass_drift" => mass_drift,
                            "total_energy" => energy,
                            "energy_tendency" => energy_tendency,
                            "source_work" => source_work,
                            "energy_balance_residual" => energy_balance_residual,
                            "interface_apex" => interface_apex,
                            "mixing_layer_width" => mixing_layer_width,
                            "wall_v_y0_min" => bottom_wall_v[1],
                            "wall_v_y0_max" => bottom_wall_v[2],
                            "wall_v_y1_min" => top_wall_v[1],
                            "wall_v_y1_max" => top_wall_v[2]))

    diagnostic_previous_time[] = t
    diagnostic_previous_energy[] = energy

    println("Saved: $(basename(filename))")
    return nothing
end

function condition_save_npz(u, t, integrator)
    index = npz_next_save_index[]
    index <= length(npz_save_times) || return false

    return t >= npz_save_times[index] - 1.0e-12
end

function affect_save_npz!(integrator)
    save_npz_file(integrator)

    # Advance past every scheduled time reached by the current integrator step.
    while npz_next_save_index[] <= length(npz_save_times) &&
          integrator.t >= npz_save_times[npz_next_save_index[]] - 1.0e-12
        npz_next_save_index[] += 1
    end

    return nothing
end

function init_save_npz(cb, u, t, integrator)
    npz_next_save_index[] = 2
    diagnostic_initial_mass[] = NaN
    diagnostic_previous_time[] = NaN
    diagnostic_previous_energy[] = NaN
    save_npz_file(integrator)
end

save_npz_callback = DiscreteCallback(condition_save_npz,
                                     affect_save_npz!,
                                     save_positions=(false, false),
                                     initialize=init_save_npz)


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
# Save the complete conservative state at the ground-truth cadence.
dt_save = 0.1
trixi_out = joinpath(@__DIR__, "trixi_output")

save_solution = SaveSolutionCallback(dt = dt_save,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     output_directory = trixi_out,
                                     solution_variables = cons2cons)
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
# ## Methodology: Gap-Free Nearest-Neighbor Lattice Projection
# Projects the scattered DG visualization samples onto a regular grid for
# continuous heatmap rendering. The default raster dimensions follow the
# native sampled coordinate counts, avoiding unnecessary downsampling.
#
# Each occupied pixel retains the value of its nearest sample. Remaining
# pixels are filled from the nearest occupied pixels using a multi-source
# grid traversal. Consequently, the returned matrix contains no NaN entries,
# empty rows, or empty columns. Matrix rows correspond to y and columns to x.

function rasterize_data(x, y, z, nx=nothing, ny=nothing)
    x_values = vec(x)
    y_values = vec(y)
    z_values = vec(z)

    @assert length(x_values) == length(y_values) == length(z_values)
    @assert !isempty(z_values)
    @assert all(isfinite, x_values)
    @assert all(isfinite, y_values)
    @assert all(isfinite, z_values)

    # Match the native sample density unless explicit dimensions are supplied.
    # Rounding merges coordinates repeated at shared element boundaries.
    if nx === nothing
        nx = max(2, length(unique(map(value -> round(value, digits=12),
                                      x_values))))
    end
    if ny === nothing
        ny = max(2, length(unique(map(value -> round(value, digits=12),
                                      y_values))))
    end
    @assert nx >= 2 && ny >= 2

    x_min, x_max = 0.0, 0.25
    y_min, y_max = 0.0, 1.0
    x_grid = collect(range(x_min, x_max, length=nx))
    y_grid = collect(range(y_min, y_max, length=ny))
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)

    # Retain the closest sample assigned to each pixel instead of averaging
    # across a discontinuous material interface.
    grid = fill(NaN, ny, nx)
    nearest_distance = fill(Inf, ny, nx)

    for k in eachindex(x_values)
        i = clamp(round(Int, (x_values[k] - x_min) / dx) + 1, 1, nx)
        j = clamp(round(Int, (y_values[k] - y_min) / dy) + 1, 1, ny)
        distance = (x_values[k] - x_grid[i])^2 +
                   (y_values[k] - y_grid[j])^2

        if distance < nearest_distance[j, i]
            nearest_distance[j, i] = distance
            grid[j, i] = z_values[k]
        end
    end

    any(isfinite, grid) || error("Rasterization received no valid samples.")

    # Multi-source breadth-first traversal fills every unoccupied pixel from
    # a neighboring occupied pixel. Every pixel is enqueued at most once.
    queue_i = Vector{Int}(undef, nx * ny)
    queue_j = Vector{Int}(undef, nx * ny)
    head = 1
    tail = 0

    for j in 1:ny, i in 1:nx
        if isfinite(grid[j, i])
            tail += 1
            queue_i[tail] = i
            queue_j[tail] = j
        end
    end

    while head <= tail
        i = queue_i[head]
        j = queue_j[head]
        head += 1

        for (ii, jj) in ((i - 1, j), (i + 1, j),
                         (i, j - 1), (i, j + 1))
            if 1 <= ii <= nx && 1 <= jj <= ny && !isfinite(grid[jj, ii])
                grid[jj, ii] = grid[j, i]
                tail += 1
                queue_i[tail] = ii
                queue_j[tail] = jj
            end
        end
    end

    @assert all(isfinite, grid) "Gap-free rasterization failed."
    return x_grid, y_grid, grid
end
# %% [markdown]


# ## Cell 12: Post-Processing & Temporal Diagnostics
# ## Post-Processing: Rayleigh–Taylor Diagnostics
# Visualizes density and vertical velocity at
# t = 0.75, 1.5, 2.25, 3.0 as continuous filled heatmaps.
# Each panel uses automatic color scaling without mesh or grid overlays.
# The final artifact is a 2 × 4 grid with fields in rows and times
# in columns, saved as `summary_all.png`.

# %%
println("Generating Rayleigh–Taylor summary plots...")

plot_times = [0.75, 1.5, 2.25, 3.0]
files = sort([joinpath(npz_out, f) for f in readdir(npz_out)
              if endswith(f, ".npz")])
isempty(files) && error("No NPZ solution snapshots were found in $(npz_out).")

# Select the saved snapshot closest to each requested visualization time.
file_times = [npzread(file)["t"] for file in files]
selected_files = [files[argmin(abs.(file_times .- target_time))]
                  for target_time in plot_times]

plots_array = []

style = (
    seriestype = :heatmap,
    aspect_ratio = :equal,
    xlims = (0.0, 0.25),
    ylims = (0.0, 1.0),
    yflip = false,
    grid = false,
    axis = nothing,
    framestyle = :none,
    legend = false,
    colorbar = false,
    margin = 2Plots.mm
)

# Row 1 contains density; row 2 contains vertical velocity.
for (field, label) in (("rho", "ρ"), ("v", "v"))
    for filename in selected_files
        data = npzread(filename)
        t_val = data["t"]
        x = vec(data["x"])
        y = vec(data["y"])
        values = vec(data[field])

        # Rows of field_grid correspond to y_grid and columns to x_grid.
        # The rasterizer returns a completely finite matrix with no gaps.
        x_grid, y_grid, field_grid = rasterize_data(x, y, values)

        push!(plots_array, plot(x_grid, y_grid, field_grid;
                                title="$(label), t=$(round(t_val, digits=2))",
                                style...))
    end
end

println("Saving summary_all.png...")
final_plot = plot(plots_array..., layout=(2, 4),
                  size=(2000, 1200), margin=5Plots.mm)
savefig(final_plot, joinpath(@__DIR__, "summary_all.png"))
println("Done.")
