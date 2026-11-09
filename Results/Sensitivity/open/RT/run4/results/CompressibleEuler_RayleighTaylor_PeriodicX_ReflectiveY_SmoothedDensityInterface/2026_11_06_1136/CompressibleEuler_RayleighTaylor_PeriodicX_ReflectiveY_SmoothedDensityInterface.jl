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
# ## Governing Equations: 2D Compressible Euler System
#
# Defines the inviscid conservation laws for mass, momentum, and total energy:
# $$
# \frac{\partial \mathbf{u}}{\partial t}
# + \nabla \cdot \mathbf{f}(\mathbf{u}) = \mathbf{S}(\mathbf{u}).
# $$
# The ideal-gas equation of state uses $\gamma = 1.4$. No viscous or
# heat-conduction fluxes are included ($\mu = k = 0$). The gravitational
# source $\mathbf{S}$ is defined with the initial conditions in Cell 3.

# %%
equations = CompressibleEulerEquations2D(1.4)
# %% [markdown]

# ## Cell 3: Initial Conditions and Source Terms (Lid-Driven Cavity)
# ## Problem Setup: Rayleigh–Taylor Instability
#
# Heavy fluid ($\rho=2$) initially lies below light fluid ($\rho=1$),
# with a smoothed interface at $y=0.5$. The piecewise pressure profile is
# continuous at the interface. A single-mode vertical-velocity perturbation
# seeds the instability and vanishes at both horizontal walls.
#
# Gravity acts in the $+y$ direction, contributing $\rho g$ to the
# vertical-momentum equation and $\rho g v$ to the energy equation.

# %%
@inline function initial_condition_rti(x, t, equations::CompressibleEulerEquations2D)
    x_coord, y = x

    # Smoothed heavy-to-light density transition
    rho = 2.0 + (1.0 - 2.0) * 0.5 *
          (1.0 + tanh(50.0 * (y - 0.5)))

    # Continuous piecewise hydrostatic pressure
    p = y < 0.5 ? 2.0 * y + 1.0 : y + 1.5

    # Single-mode velocity perturbation
    v1 = 0.0
    sound_speed = sqrt(equations.gamma * p / rho)
    v2 = -0.025 * sound_speed * cos(8.0 * pi * x_coord) *
         sin(pi * y)^6

    # prim2cons gives E = p/(gamma-1) + rho*(v1^2+v2^2)/2
    return prim2cons(SVector(rho, v1, v2, p), equations)
end

@inline function source_terms_gravity(u, x, t,
                                      equations::CompressibleEulerEquations2D)
    g = 1.0
    rho, rho_v1, rho_v2, rho_e = u

    # Energy source: rho*g*v2 = g*rho_v2
    return SVector(0.0, 0.0, g * rho, g * rho_v2)
end

# Density variable used by both AMR and shock-capturing indicators
@inline function density_indicator(u, equations)
    return u[1]
end

# ## Cell 4: Spatial Discretization & Hybrid Shock Capturing
#
# ## Methodology: Nodal Discontinuous Galerkin (DGSEM)
# Constructs the spatial operator $\mathcal{L}(\mathbf{U})$ using the **Discontinuous Galerkin Spectral Element Method (DGSEM)**.
#
# 1.  **Function Space (`LobattoLegendreBasis`)**:
#     * Approximates solutions using tensor-product polynomials of degree $N$ on **Legendre-Gauss-Lobatto (LGL)** nodes ($N=5$ primary; $N=3$ verification).
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
# 3.  **Interface Flux (`surface_flux=flux_hllc`)**:
#     * **Riemann Solver**: Harten-Lax-van Leer with Contact (HLLC).
#     * **Role**: Computes the numerical flux across element interfaces while explicitly resolving the contact/shear wave.
#     * **Physics**: Retains the robust acoustic-wave estimates of HLL but reduces dissipation at the density interface and shear structures central to the RTI.
#
# 4.  **Indicator Function (`IndicatorHennemannGassner`)**:
#     * Dynamically computes the blending coefficient $\alpha \in [0, 1]$ element-wise.
#      * **Mechanism**: Monitors the spectral decay of energy in the highest modes of the polynomial expansion. If energy in high modes is large (indicating a discontinuity), $\alpha \to 1$ (activates FV sub-cells).
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
# Retain the template topology and default initial refinement
mesh = P4estMesh((1, 1), polydeg=polydeg,
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max,
                 periodicity=(true, false)) # Periodic in x; physical walls in y
# %% [markdown]

# ## Cell 6: General Boundary Condition Specification & System Assembly
# ## Boundary Conditions and Hyperbolic Semidiscretization
#
# The mesh provides periodic connectivity in the x-direction. At y=0 and y=1,
# `boundary_condition_slip_wall` imposes reflective free-slip Euler walls:
# zero normal velocity with unrestricted tangential velocity.
#
# The semidiscretization contains only the compressible Euler operator and
# the gravitational source; no viscous boundary conditions or parabolic
# operator are required.

# %%
boundary_conditions = Dict(
    :y_neg => boundary_condition_slip_wall,
    :y_pos => boundary_condition_slip_wall
)

semi = SemidiscretizationHyperbolic(mesh, equations,
                                    initial_condition_rti, dg;
                                    boundary_conditions=boundary_conditions,
                                    source_terms=source_terms_gravity)

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
# ## Strategy: Fixed-Time Conservative-State and AMR Serialization (.npz)
#
# Exports the complete conservative state
# $\mathbf{u} = [\rho, \rho v_1, \rho v_2, \rho E]^T$
# every `dt_save = 0.1` time units. In addition to the reconstructed fields,
# each file contains the raw DG state and leaf-element coordinates, bounds,
# and AMR levels. These metadata permit reconstruction on the actual adaptive
# mesh without assuming a conforming structured grid. Separate output
# directories prevent the p=5 and p=3 runs from overwriting each other.
# %%
npz_root = joinpath(@__DIR__, "data_npz")
dt_save = 0.1

function make_npz_callbacks(output_directory, run_polydeg)
    mkpath(output_directory)

    function save_snapshot(integrator)
        semi = integrator.p
        t = integrator.t

        # Retain reconstructed fields for interoperability and quick inspection.
        pd = PlotData2D(integrator.u, semi, solution_variables=cons2cons)
        states = pd.data
        rho    = collect(getindex.(states, 1))
        rho_v1 = collect(getindex.(states, 2))
        rho_v2 = collect(getindex.(states, 3))
        rho_e  = collect(getindex.(states, 4))
        x = collect(pd.x)
        y = collect(pd.y)

        # Raw layout: (variable, node_x, node_y, leaf_element).
        u_cons = copy(Trixi.wrap_array(integrator.u, semi))
        node_coordinates = copy(semi.cache.elements.node_coordinates)
        n_elements = size(u_cons, 4)
        @assert size(node_coordinates, 4) == n_elements

        domain_min = [Float64(coordinates_min[1]),
                      Float64(coordinates_min[2])]
        domain_max = [Float64(coordinates_max[1]),
                      Float64(coordinates_max[2])]
        domain_size = domain_max .- domain_min

        # Bounds rows are xmin, xmax, ymin, ymax. Since each p4est refinement
        # halves both element dimensions, the bounds also encode the AMR level.
        element_bounds = Matrix{Float64}(undef, 4, n_elements)
        element_levels = Vector{Int}(undef, n_elements)
        for element in 1:n_elements
            xs = view(node_coordinates, 1, :, :, element)
            ys = view(node_coordinates, 2, :, :, element)
            element_bounds[1, element] = minimum(xs)
            element_bounds[2, element] = maximum(xs)
            element_bounds[3, element] = minimum(ys)
            element_bounds[4, element] = maximum(ys)

            dx = element_bounds[2, element] - element_bounds[1, element]
            dy = element_bounds[4, element] - element_bounds[3, element]
            level_estimate = 0.5 * (log2(domain_size[1] / dx) +
                                    log2(domain_size[2] / dy))
            element_levels[element] = round(Int, level_estimate)
        end

        filename = joinpath(output_directory,
                            Printf.@sprintf("time_%08.5f.npz", t))
        npzwrite(filename, Dict{String, Any}(
            "t" => Float64(t),
            "polydeg" => Int(run_polydeg),
            "x" => x,
            "y" => y,
            "rho" => rho,
            "rho_v1" => rho_v1,
            "rho_v2" => rho_v2,
            "rho_e" => rho_e,
            "u_cons" => u_cons,
            "leaf_node_coordinates" => node_coordinates,
            "leaf_element_bounds" => element_bounds,
            "element_levels" => element_levels,
            "element_ids" => collect(1:n_elements),
            "n_elements" => n_elements,
            "n_nodes_1d" => size(u_cons, 2),
            "domain_min" => domain_min,
            "domain_max" => domain_max,
            "periodicity" => [1, 0],
            "amr_controller_levels" => [4, 5, 6],
            "amr_interval" => 20,
            "amr_indicator_component" => 1,
            "shock_indicator_component" => 1))
        println("Saved: $(filename)")
        return nothing
    end

    save_callback = PeriodicCallback(save_snapshot, dt_save;
                                     initial_affect=true,
                                     save_positions=(false, false))
    final_callback = PresetTimeCallback([3.0], save_snapshot;
                                        save_positions=(false, false))
    return save_callback, final_callback
end

# Primary p=5 archive; the companion run constructs its own p=3 callbacks.
npz_out = joinpath(npz_root, "p$(polydeg)")
save_npz_callback, save_final_npz_callback = make_npz_callbacks(npz_out, polydeg)


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
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)
alive_callback = AliveCallback(analysis_interval=analysis_interval)
dt_save = 0.1

# Keep native Trixi restart/mesh output separate for each polynomial degree.
trixi_root = joinpath(@__DIR__, "trixi_output")
trixi_out = joinpath(trixi_root, "p$(polydeg)")
save_solution = SaveSolutionCallback(dt=dt_save,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     output_directory=trixi_out,
                                     solution_variables=cons2cons)
stepsize_callback = StepsizeCallback()
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        amr_callback,
                        save_npz_callback,
                        save_final_npz_callback,
                        stepsize_callback)

# Construct otherwise identical callbacks for the companion verification run.
# The supplied semidiscretization and AMR callback must belong to that run.
function make_run_callbacks(run_semi, run_amr_callback, run_polydeg)
    run_npz_out = joinpath(npz_root, "p$(run_polydeg)")
    run_save_npz_callback, run_save_final_npz_callback =
        make_npz_callbacks(run_npz_out, run_polydeg)

    run_trixi_out = joinpath(trixi_root, "p$(run_polydeg)")
    run_save_solution = SaveSolutionCallback(dt=dt_save,
                                             save_initial_solution=true,
                                             save_final_solution=true,
                                             output_directory=run_trixi_out,
                                             solution_variables=cons2cons)

    return CallbackSet(SummaryCallback(),
                       AnalysisCallback(run_semi,
                                        interval=analysis_interval),
                       AliveCallback(analysis_interval=analysis_interval),
                       run_save_solution,
                       run_amr_callback,
                       run_save_npz_callback,
                       run_save_final_npz_callback,
                       StepsizeCallback())
end
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
function log_run_checks(label, run_sol, run_semi)
    final_time = run_sol.t[end]
    reached_t_end = isapprox(final_time, 3.0; atol=1.0e-10, rtol=0.0)

    # Raw layout: (variable, node_x, node_y, leaf_element).
    u_cons = Trixi.wrap_array(run_sol.u[end], run_semi)
    fields_finite = all(isfinite, u_cons)

    min_rho = Inf
    min_pressure = Inf
    gamma = run_semi.equations.gamma

    for element in axes(u_cons, 4), j in axes(u_cons, 3), i in axes(u_cons, 2)
        rho    = u_cons[1, i, j, element]
        rho_v1 = u_cons[2, i, j, element]
        rho_v2 = u_cons[3, i, j, element]
        rho_e  = u_cons[4, i, j, element]

        pressure = (gamma - 1) *
                   (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) / rho)
        min_rho = min(min_rho, rho)
        min_pressure = min(min_pressure, pressure)
    end

    println("[$label] successful arrival at t = 3.0: ", reached_t_end,
            " (final t = ", final_time, ", retcode = ", run_sol.retcode, ")")
    println("[$label] all fields finite (no NaN/Inf): ", fields_finite)
    println("[$label] minimum rho = ", min_rho,
            "; min rho > 0: ", min_rho > 0)
    println("[$label] minimum p = ", min_pressure,
            "; min p > 0: ", min_pressure > 0)

    return (reached_t_end=reached_t_end,
            fields_finite=fields_finite,
            min_rho=min_rho,
            min_pressure=min_pressure)
end

println("Starting p=5 primary RTI run...")
sol = solve(ode, RDPK3SpFSAL49();
            dt=1.0e-5, callback=callbacks, adaptive=false);

# Preserve explicit primary-run handles for certification and post-processing.
polydeg_p5 = polydeg
semi_p5 = semi
ode_p5 = ode
sol_p5 = sol
checks_p5 = log_run_checks("p=5 primary", sol_p5, semi_p5)

# Companion p=3 run: identical physics, fluxes, AMR, and time-step control.
polydeg_p3 = 3
basis_p3 = LobattoLegendreBasis(polydeg_p3)
indicator_sc_p3 = IndicatorHennemannGassner(equations, basis_p3,
                                            alpha_max=0.5,
                                            alpha_min=0.001,
                                            alpha_smooth=true,
                                            variable=density_indicator)
volume_integral_p3 = VolumeIntegralShockCapturingHG(
    indicator_sc_p3;
    volume_flux_dg=flux_ranocha,
    volume_flux_fv=flux_lax_friedrichs)
dg_p3 = DGSEM(polydeg=polydeg_p3,
              surface_flux=flux_hllc,
              volume_integral=volume_integral_p3)

mesh_p3 = P4estMesh((1, 1), polydeg=polydeg_p3,
                    coordinates_min=coordinates_min,
                    coordinates_max=coordinates_max,
                    periodicity=(true, false))
semi_p3 = SemidiscretizationHyperbolic(mesh_p3, equations,
                                       initial_condition_rti, dg_p3;
                                       boundary_conditions=boundary_conditions,
                                       source_terms=source_terms_gravity)

amr_indicator_p3 = IndicatorLöhner(semi_p3, variable=density_indicator)
amr_controller_p3 = ControllerThreeLevel(semi_p3, amr_indicator_p3,
                                         base_level=4,
                                         med_level=5, med_threshold=0.1,
                                         max_level=6, max_threshold=0.3)
amr_callback_p3 = AMRCallback(semi_p3, amr_controller_p3,
                              interval=20,
                              adapt_initial_condition=true)

ode_p3 = semidiscretize(semi_p3, tspan)
callbacks_p3 = make_run_callbacks(semi_p3, amr_callback_p3, polydeg_p3)

println("Starting p=3 companion verification run...")
sol_p3 = solve(ode_p3, RDPK3SpFSAL49();
               dt=1.0e-5, callback=callbacks_p3, adaptive=false);
checks_p3 = log_run_checks("p=3 companion", sol_p3, semi_p3)
# %% [markdown]


# ## Cell 11: Data Regularization (Rasterization)
# ## Methodology: Leaf-Element DG Interpolation on a Uniform Grid
#
# Each archived leaf element owns a tensor-product LGL polynomial. For every
# uniform-grid point, the containing leaf element is identified from its
# physical bounds and the conservative state is evaluated by two successive
# one-dimensional Lagrange interpolations. No neighbor averaging, smoothing,
# or nearest-neighbor splatting is used. Consequently, hanging nodes and
# element-edge strokes cannot introduce gaps or stripes in the sampled data.
#
# The default 512 x 2048 grid has square cells on [0, 0.25] x [0, 1].
# Cell centers avoid ambiguous sampling exactly on element interfaces.
# %%
function lagrange_interpolation_matrix(nodes, points)
    n = length(nodes)
    weights = Matrix{Float64}(undef, n, length(points))

    for (point_index, point) in enumerate(points)
        for a in 1:n
            weight = 1.0
            for b in 1:n
                if b != a
                    weight *= (point - nodes[b]) / (nodes[a] - nodes[b])
                end
            end
            weights[a, point_index] = weight
        end
    end

    return weights
end

function interpolate_amr_snapshot(data; nx=512, ny=2048)
    u_cons = data["u_cons"]
    node_coordinates = data["leaf_node_coordinates"]
    element_bounds = data["leaf_element_bounds"]
    domain_min = data["domain_min"]
    domain_max = data["domain_max"]

    @assert size(u_cons, 1) == 4 "Expected four conservative variables."
    @assert size(u_cons, 2) == size(u_cons, 3) "Expected a tensor-product DG element."

    n = size(u_cons, 2)
    n_elements = size(u_cons, 4)
    @assert size(node_coordinates, 4) == n_elements
    @assert size(element_bounds, 2) == n_elements

    x_min, y_min = Float64(domain_min[1]), Float64(domain_min[2])
    x_max, y_max = Float64(domain_max[1]), Float64(domain_max[2])
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    isapprox(dx, dy; atol=1.0e-14, rtol=1.0e-12) ||
        error("The common visualization grid must use square cells.")

    # Cell-centered coordinates permit gapless heatmap rendering.
    x_grid = [x_min + (i - 0.5) * dx for i in 1:nx]
    y_grid = [y_min + (j - 0.5) * dy for j in 1:ny]

    # Layout: (y, x, conservative variable).
    u_grid = Array{Float64}(undef, ny, nx, 4)
    fill!(u_grid, NaN)
    covered = falses(ny, nx)

    for element in 1:n_elements
        # LGL coordinates of this leaf element in physical space.
        x_nodes = collect(view(node_coordinates, 1, :, 1, element))
        y_nodes = collect(view(node_coordinates, 2, 1, :, element))

        # Only grid points inside this leaf are evaluated from its polynomial.
        i_first = searchsortedfirst(x_grid, element_bounds[1, element])
        i_last  = searchsortedlast(x_grid, element_bounds[2, element])
        j_first = searchsortedfirst(y_grid, element_bounds[3, element])
        j_last  = searchsortedlast(y_grid, element_bounds[4, element])

        if i_first > i_last || j_first > j_last
            continue
        end

        i_range = i_first:i_last
        j_range = j_first:j_last
        nx_local = length(i_range)
        ny_local = length(j_range)

        wx = lagrange_interpolation_matrix(x_nodes, view(x_grid, i_range))
        wy = lagrange_interpolation_matrix(y_nodes, view(y_grid, j_range))

        # First interpolate in x for each retained y-node. Layout of temporary:
        # (y-node, local x-point, conservative variable).
        temporary = Array{Float64}(undef, n, nx_local, 4)
        for variable in 1:4, b in 1:n, local_i in 1:nx_local
            value = 0.0
            for a in 1:n
                value += wx[a, local_i] * u_cons[variable, a, b, element]
            end
            temporary[b, local_i, variable] = value
        end

        # Complete the tensor-product interpolation in y.
        for variable in 1:4, local_i in 1:nx_local, local_j in 1:ny_local
            value = 0.0
            for b in 1:n
                value += wy[b, local_j] * temporary[b, local_i, variable]
            end

            i = i_range[local_i]
            j = j_range[local_j]
            u_grid[j, i, variable] = value
        end

        for local_j in 1:ny_local, local_i in 1:nx_local
            covered[j_range[local_j], i_range[local_i]] = true
        end
    end

    all(covered) ||
        error("Leaf-element reconstruction left uncovered uniform-grid points.")

    rho = view(u_grid, :, :, 1)
    vertical_velocity = u_grid[:, :, 3] ./ rho

    return (x=x_grid,
            y=y_grid,
            u=u_grid,
            rho=rho,
            v=vertical_velocity,
            dx=dx,
            dy=dy,
            cell_area=dx * dy)
end
# %% [markdown]


# ## Cell 12: Post-Processing & Temporal Diagnostics
# ## Rayleigh–Taylor Visualization and Polynomial-Degree Certification
#
# Density and vertical velocity are rendered as gapless heatmaps from the
# p=5 run at t = 0.5, 1.0, 2.0, and 3.0. The p=3 and p=5 archives are also
# compared on the same uniform grid at t = 1.0, 2.0, and 3.0. No smoothing,
# fixed colormap, fixed color limits, or artificial reference error is used.
# %%
function snapshot_time(data)
    value = data["t"]
    return value isa Number ? Float64(value) : Float64(only(value))
end

function snapshot_at_time(directory, target_time)
    files = sort([joinpath(directory, f) for f in readdir(directory)
                  if endswith(f, ".npz")])
    isempty(files) && error("No NPZ solution snapshots were found in $directory.")

    file_times = [snapshot_time(npzread(file)) for file in files]
    idx = argmin(abs.(file_times .- target_time))
    isapprox(file_times[idx], target_time; atol=1.0e-6, rtol=0.0) ||
        error("No saved snapshot found at t = $target_time in $directory.")

    return npzread(files[idx])
end

# Find the y-extremes of the rho=1.5 interface by linearly locating its
# crossings along every uniform-grid x-column.
function interface_metrics(grid)
    rho = grid.rho
    y = grid.y
    crossings = Float64[]

    for i in axes(rho, 2)
        for j in 1:(size(rho, 1) - 1)
            r_lower = rho[j, i]
            r_upper = rho[j + 1, i]
            if !(isfinite(r_lower) && isfinite(r_upper))
                continue
            end

            d_lower = r_lower - 1.5
            d_upper = r_upper - 1.5

            if d_lower == 0.0
                push!(crossings, y[j])
            end

            if d_lower * d_upper < 0.0
                fraction = (1.5 - r_lower) / (r_upper - r_lower)
                push!(crossings,
                      y[j] + fraction * (y[j + 1] - y[j]))
            end
        end

        if isfinite(rho[end, i]) && rho[end, i] == 1.5
            push!(crossings, y[end])
        end
    end

    if isempty(crossings)
        return (bubble_tip=NaN, spike_tip=NaN, mixing_width=NaN)
    end

    bubble_tip = minimum(crossings)
    spike_tip = maximum(crossings)
    return (bubble_tip=bubble_tip,
            spike_tip=spike_tip,
            mixing_width=spike_tip - bubble_tip)
end

function relative_l2_difference(field, reference)
    numerator = sqrt(sum(abs2, field .- reference))
    denominator = sqrt(sum(abs2, reference))
    return denominator > 0.0 ? numerator / denominator : NaN
end

function uniform_grid_kinetic_energy(grid)
    rho = grid.u[:, :, 1]
    rho_v1 = grid.u[:, :, 2]
    rho_v2 = grid.u[:, :, 3]

    return grid.cell_area *
           sum((rho_v1.^2 .+ rho_v2.^2) ./ (2.0 .* rho))
end

# Integrate kinetic energy directly on each archived leaf element using its
# LGL quadrature weights and affine physical Jacobian.
function archive_kinetic_energy(data)
    u = data["u_cons"]
    bounds = data["leaf_element_bounds"]
    n = size(u, 2)
    weights = LobattoLegendreBasis(n - 1).weights
    energy = 0.0

    for element in axes(u, 4)
        jacobian = ((bounds[2, element] - bounds[1, element]) / 2.0) *
                   ((bounds[4, element] - bounds[3, element]) / 2.0)

        for j in 1:n, i in 1:n
            rho = u[1, i, j, element]
            kinetic_density =
                (u[2, i, j, element]^2 + u[3, i, j, element]^2) /
                (2.0 * rho)
            energy += kinetic_density * weights[i] * weights[j] * jacobian
        end
    end

    return energy
end

function kinetic_energy_history(directory)
    files = sort([joinpath(directory, f) for f in readdir(directory)
                  if endswith(f, ".npz")])
    times = Float64[]
    energies = Float64[]

    for file in files
        data = npzread(file)
        push!(times, snapshot_time(data))
        push!(energies, archive_kinetic_energy(data))
    end

    return times, energies
end

println("Generating Rayleigh–Taylor summary plots...")

target_times = (0.5, 1.0, 2.0, 3.0)
primary_grids = [
    interpolate_amr_snapshot(snapshot_at_time(npz_out, target_time))
    for target_time in target_times
]

style = (
    aspect_ratio = :equal,
    xlims = (0.0, 0.25),
    ylims = (0.0, 1.0),
    axis = nothing,
    framestyle = :none,
    legend = false,
    colorbar = false,
    margin = 2Plots.mm
)

density_plots = []
velocity_plots = []

for (target_time, grid) in zip(target_times, primary_grids)
    # Heatmaps use contiguous cells and therefore have no element-edge
    # strokes or AMR-patch gaps. Each panel receives automatic color scaling.
    push!(density_plots,
          heatmap(grid.x, grid.y, grid.rho;
                  title="ρ, t=$(round(target_time, digits=2))",
                  style...))

    push!(velocity_plots,
          heatmap(grid.x, grid.y, grid.v;
                  title="v, t=$(round(target_time, digits=2))",
                  style...))
end

# Row-major ordering: four density panels, then four velocity panels.
plots_array = vcat(density_plots, velocity_plots)

println("Saving summary_all.png...")
final_plot = plot(plots_array..., layout=(2, 4),
                  size=(2000, 1200), margin=5Plots.mm)
savefig(final_plot, joinpath(@__DIR__, "summary_all.png"))

npz_out_p3 = joinpath(npz_root, "p3")

println("Kinetic energy versus time for both archived runs:")
for (label, directory) in (("p=3", npz_out_p3), ("p=5", npz_out))
    times, energies = kinetic_energy_history(directory)
    println("  $label:")
    for (time, energy) in zip(times, energies)
        Printf.@printf("    t = %.2f, KE = %.12e\n", time, energy)
    end
end

println("Common-grid p=3/p=5 certification; p=5 is the reference:")
for target_time in (1.0, 2.0, 3.0)
    primary_index = findfirst(
        time -> isapprox(time, target_time; atol=1.0e-6, rtol=0.0),
        target_times)
    primary_index === nothing &&
        error("Primary grid is missing t = $target_time.")

    grid_p5 = primary_grids[primary_index]
    grid_p3 = interpolate_amr_snapshot(
        snapshot_at_time(npz_out_p3, target_time))

    metrics_p3 = interface_metrics(grid_p3)
    metrics_p5 = interface_metrics(grid_p5)
    ke_p3 = uniform_grid_kinetic_energy(grid_p3)
    ke_p5 = uniform_grid_kinetic_energy(grid_p5)
    l2_rho = relative_l2_difference(grid_p3.rho, grid_p5.rho)
    l2_v = relative_l2_difference(grid_p3.v, grid_p5.v)

    Printf.@printf("\nt = %.1f\n", target_time)
    Printf.@printf(
        "  p=3: bubble tip y = %.8f, spike tip y = %.8f, " *
        "mixing width = %.8f, KE = %.12e\n",
        metrics_p3.bubble_tip, metrics_p3.spike_tip,
        metrics_p3.mixing_width, ke_p3)
    Printf.@printf(
        "  p=5: bubble tip y = %.8f, spike tip y = %.8f, " *
        "mixing width = %.8f, KE = %.12e\n",
        metrics_p5.bubble_tip, metrics_p5.spike_tip,
        metrics_p5.mixing_width, ke_p5)
    Printf.@printf(
        "  relative L2 differences versus p=5: rho = %.8e, v = %.8e\n",
        l2_rho, l2_v)
end

println("Done.")
