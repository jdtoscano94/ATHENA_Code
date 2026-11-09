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
#     * Approximates solutions using tensor-product polynomials of degree $N$ on **Legendre-Gauss-Lobatto (LGL)** nodes ($N=7$ primary; $N=5$ companion; archived $N=3$ trend).
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
polydeg = 7
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
                 initial_refinement_level=3,
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
# ## Methodology: Density-Driven AMR with Local Shear Augmentation
# Density remains the primary AMR variable. The controller uses interval=10
# and tightened Löhner thresholds so filaments are refined before roll-up.
# Density-keyed refinement remains hard-capped at level 6.
#
# At t=1.5, a secondary vorticity indicator is activated. The indicator
# evaluates omega = d(v)/dx - d(u)/dy only at nodes with intermediate
# densities 1.1 <= rho <= 1.9. Thus, chaotic velocity outside the mixing
# layer cannot request refinement. Only this local shear controller can
# request level 7.
#
# level_shift=1 raises the companion's intermediate density target from
# level 5 to level 6. Shock capturing and CFL-controlled time integration
# are unchanged.
# %%
# Element-wise vorticity controller used only inside the mixing layer.
# The vorticity is made dimensionless with the local element length and
# velocity scale before applying the three-level thresholds.
struct VorticityShearController
    base_level::Int
    med_level::Int
    max_level::Int
    med_threshold::Float64
    max_threshold::Float64
    density_min::Float64
    density_max::Float64
end
struct MixingLayerShearController{D, S}
    density_controller::D
    shear_controller::S
    active::Base.RefValue{Bool}
    # Level metadata required by Trixi's AMR/analysis reporting.
    base_level::Int
    med_level::Int
    max_level::Int
end
# Compute target levels from the masked vorticity. P4est elements are
# axis-aligned rectangles, so the physical derivatives follow from the
# reference derivative matrix and the element widths.
function Trixi.get_element_variables!(element_variables,
                                      controller::VorticityShearController,
                                      semi, u_ode, t, args...)
    u = Trixi.wrap_array(u_ode, semi)
    node_coordinates = semi.cache.elements.node_coordinates
    derivative_matrix = semi.solver.basis.derivative_matrix
    n = size(u, 2)
    for element in eachindex(element_variables)
        element_variables[element] = controller.base_level
        x_coordinates = view(node_coordinates, 1, :, :, element)
        y_coordinates = view(node_coordinates, 2, :, :, element)
        width = maximum(x_coordinates) - minimum(x_coordinates)
        height = maximum(y_coordinates) - minimum(y_coordinates)
        x_derivative_scale = 2.0 / width
        y_derivative_scale = 2.0 / height
        length_scale = min(width, height)
        shear_measure = 0.0
        for j in 1:n, i in 1:n
            rho = u[1, i, j, element]
            if !isfinite(rho) ||
               rho < controller.density_min ||
               rho > controller.density_max
                continue
            end
            safe_rho = max(rho, 1.0e-12)
            horizontal_velocity = u[2, i, j, element] / safe_rho
            vertical_velocity = u[3, i, j, element] / safe_rho
            # Accumulate d(v)/d(xi) and d(u)/d(eta). The remaining
            # derivative terms are not needed for the 2D vorticity.
            dv_dxi = 0.0
            du_deta = 0.0
            for k in 1:n
                rho_x = max(u[1, k, j, element], 1.0e-12)
                rho_y = max(u[1, i, k, element], 1.0e-12)
                dv_dxi += derivative_matrix[i, k] *
                          (u[3, k, j, element] / rho_x)
                du_deta += derivative_matrix[j, k] *
                           (u[2, i, k, element] / rho_y)
            end
            vorticity = x_derivative_scale * dv_dxi -
                        y_derivative_scale * du_deta
            velocity_scale = max(
                sqrt(horizontal_velocity^2 + vertical_velocity^2),
                1.0)
            normalized_vorticity =
                abs(vorticity) * length_scale / velocity_scale
            if isfinite(normalized_vorticity)
                shear_measure = max(shear_measure, normalized_vorticity)
            end
        end
        if shear_measure > controller.max_threshold
            element_variables[element] = controller.max_level
        elseif shear_measure > controller.med_threshold
            element_variables[element] = controller.med_level
        end
    end
    return element_variables
end
# Preserve every density-controller target. After activation, take the
# maximum with the masked vorticity target; consequently only the local
# shear path can elevate an element to level 7.
function Trixi.get_element_variables!(element_variables,
                                      controller::MixingLayerShearController,
                                      semi, u_ode, t, args...)
    Trixi.get_element_variables!(element_variables,
                                 controller.density_controller,
                                 semi, u_ode, t, args...)
    if !controller.active[]
        return element_variables
    end
    shear_levels = similar(element_variables)
    Trixi.get_element_variables!(shear_levels,
                                 controller.shear_controller,
                                 semi, u_ode, t, args...)
    @inbounds for element in eachindex(element_variables)
        element_variables[element] = max(element_variables[element],
                                         shear_levels[element])
    end
    return element_variables
end
function make_amr_callback(run_semi; shear_aware=true, level_shift=0)
    level_shift in (0, 1) ||
        throw(ArgumentError("level_shift must be either 0 or 1"))

    amr_interval = 10
    density_loehner = IndicatorLöhner(
        run_semi, variable=density_indicator)

    # Match base_level to the initial level-3 mesh. The density controller
    # remains bounded at level 6 for both baseline and companion runs.
    base_level = 3
    density_max_level = 6
    shear_level_cap = 7
    med_level = min(5 + level_shift, density_max_level)
    shear_med_level = min(density_max_level + level_shift,
                          shear_level_cap)
    density_controller = ControllerThreeLevel(
        run_semi, density_loehner,
        base_level=base_level,
        med_level=med_level, med_threshold=0.05,
        max_level=density_max_level, max_threshold=0.15)
    controller = density_controller
    activation_callback = nothing
    shear_start_time = Inf
    if shear_aware
        shear_start_time = 1.5
        shear_active = Ref(false)
        # Only elements containing intermediate-density nodes can receive
        # shear refinement. The density controller remains capped at 6;
        # this controller alone has a level-7 maximum.
        shear_controller = VorticityShearController(
            base_level,
            shear_med_level,
            shear_level_cap,
            0.03,
            0.10,
            1.1,
            1.9)
        controller = MixingLayerShearController(
            density_controller, shear_controller, shear_active,
            base_level, med_level, shear_level_cap)
        function activate_shear!(integrator)
            shear_active[] = true
            return nothing
        end
        # Force a stop at t=1.5 so shear-aware refinement is enabled
        # before/during mushroom roll-up. Cell 9 places this callback
        # immediately before the AMR callback.
        activation_callback = PresetTimeCallback(
            [shear_start_time], activate_shear!;
            save_positions=(false, false))
    end
    amr_callback = AMRCallback(
        run_semi, controller,
        interval=amr_interval,
        adapt_initial_condition=true,
        adapt_initial_condition_only_refine=true)
    amr_configuration = (
        shear_aware=shear_aware,
        max_level=shear_aware ? shear_level_cap : density_max_level,
        levels=(base_level, med_level, density_max_level),
        interval=amr_interval,
        activation_callback=activation_callback,
        shear_start_time=shear_start_time,
        level_shift=level_shift)
    return amr_callback, amr_configuration
end

# Define the primary objects referenced by the NPZ archiving and callback
# assembly cells. Companion runs construct their own independent objects.
amr_callback, amr_configuration = make_amr_callback(
    semi; shear_aware=true, level_shift=0)


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
function make_npz_callbacks(output_directory, run_polydeg;
                            run_amr_configuration=(shear_aware=false,
                                                   max_level=6,
                                                   levels=(4, 5, 6),
                                                   interval=10))
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
            # Record the actual AMR configuration. A value of one for
            # amr_shear_aware denotes conditional mixing-layer-targeted
            # shear augmentation; zero denotes baseline density-only AMR.
            "amr_controller_levels" => collect(run_amr_configuration.levels),
            "amr_interval" => Int(run_amr_configuration.interval),
            "amr_max_level" => Int(run_amr_configuration.max_level),
            "amr_shear_aware" => Int(run_amr_configuration.shear_aware),
            "amr_targeted_to_mixing_layer" => Int(run_amr_configuration.shear_aware),
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
# Primary p=7 archive; companion and conditional runs construct their own callbacks.
npz_out = joinpath(npz_root, "p$(polydeg)")
save_npz_callback, save_final_npz_callback =
    make_npz_callbacks(npz_out, polydeg;
                       run_amr_configuration=amr_configuration)

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
stepsize_callback = StepsizeCallback(cfl=0.5)
dt_save = 0.1
# Keep native Trixi restart/mesh output separate for each polynomial degree.
trixi_root = joinpath(@__DIR__, "trixi_output")
trixi_out = joinpath(trixi_root, "p$(polydeg)")
save_solution = SaveSolutionCallback(dt=dt_save,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     output_directory=trixi_out,
                                     solution_variables=cons2cons)
# Activate mixing-layer shear refinement before AMR adaptation. The
# conditional retains compatibility with density-only configurations.
primary_amr_callbacks = if amr_configuration.activation_callback === nothing
    (amr_callback,)
else
    (amr_configuration.activation_callback, amr_callback)
end
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        primary_amr_callbacks...,
                        stepsize_callback,
                        save_npz_callback,
                        save_final_npz_callback)
# Construct otherwise identical callbacks for companion runs. Cell 7 supplies
# density-keyed AMR and, when enabled, its shear activation callback. The
# conditional below also keeps this helper compatible with older configurations.
function make_run_callbacks(run_semi, run_amr_callback, run_polydeg;
                            run_amr_configuration=(shear_aware=false,
                                                   max_level=6,
                                                   levels=(4, 5, 6),
                                                   interval=10,
                                                   activation_callback=nothing,
                                                   shear_start_time=Inf,
                                                   level_shift=0),
                            run_id="p$(run_polydeg)")
    run_npz_out = joinpath(npz_root, run_id)
    run_save_npz_callback, run_save_final_npz_callback =
        make_npz_callbacks(run_npz_out, run_polydeg;
                           run_amr_configuration=run_amr_configuration)
    run_trixi_out = joinpath(trixi_root, run_id)
    run_save_solution = SaveSolutionCallback(
        dt=dt_save,
        save_initial_solution=true,
        save_final_solution=true,
        output_directory=run_trixi_out,
        solution_variables=cons2cons)
    run_stepsize_callback = StepsizeCallback(cfl=0.5)
    run_activation_callback = nothing
    if hasproperty(run_amr_configuration, :activation_callback)
        run_activation_callback =
            run_amr_configuration.activation_callback
    end
    amr_callbacks = if run_activation_callback === nothing
        (run_amr_callback,)
    else
        (run_activation_callback, run_amr_callback)
    end
    return CallbackSet(SummaryCallback(),
                       AnalysisCallback(run_semi,
                                        interval=analysis_interval),
                       AliveCallback(analysis_interval=analysis_interval),
                       run_save_solution,
                       amr_callbacks...,
                       run_stepsize_callback,
                       run_save_npz_callback,
                       run_save_final_npz_callback)
end
initial_dt = 1.0e-5
# npz_out is the existing p=7 archive created by earlier runs. The new
# companion uses a different directory and cannot overwrite that baseline.
npz_out_p7_baseline = npz_out
npz_out_p7_h_refined = joinpath(npz_root, "p7_h_refined")
npz_out_p7 = npz_out_p7_h_refined
primary_run_label = "p=7 (density AMR + mixing-layer shear AMR, local h-refined)"
# Check the fixed NPZ save schedule without loading the large state arrays.
function npz_archive_has_all_save_times(directory)
    isdir(directory) || return false
    n_saves = round(Int, tspan[2] / dt_save)
    for save_index in 0:n_saves
        save_time = save_index * dt_save
        filename = joinpath(
            directory,
            Printf.@sprintf("time_%08.5f.npz", save_time))
        isfile(filename) || return false
    end
    return true
end
function solve_rti_ode(run_ode, run_callbacks)
    return solve(run_ode, RDPK3SpFSAL49();
                 adaptive=false,
                 dt=initial_dt,
                 save_everystep=false,
                 callback=run_callbacks)
end
# Construct and execute a run with identical equations, fluxes, shock
# capturing, mesh topology, and time integration. Only the requested AMR
# configuration and output identifier change.
function run_rti_case(run_polydeg; shear_aware=true, level_shift=0,
                      run_id="p$(run_polydeg)")
    run_basis = LobattoLegendreBasis(run_polydeg)
    run_indicator_sc = IndicatorHennemannGassner(
        equations, run_basis,
        alpha_max=0.5,
        alpha_min=0.001,
        alpha_smooth=true,
        variable=density_indicator)
    run_volume_integral = VolumeIntegralShockCapturingHG(
        run_indicator_sc;
        volume_flux_dg=flux_ranocha,
        volume_flux_fv=flux_lax_friedrichs)
    run_dg = DGSEM(polydeg=run_polydeg,
                   surface_flux=flux_hllc,
                   volume_integral=run_volume_integral)
    run_mesh = P4estMesh((1, 1), polydeg=run_polydeg,
                         initial_refinement_level=3,
                         coordinates_min=coordinates_min,
                         coordinates_max=coordinates_max,
                         periodicity=(true, false))
    run_semi = SemidiscretizationHyperbolic(
        run_mesh, equations, initial_condition_rti, run_dg;
        boundary_conditions=boundary_conditions,
        source_terms=source_terms_gravity)
    run_amr_callback, run_amr_configuration =
        make_amr_callback(run_semi;
                          shear_aware=shear_aware,
                          level_shift=level_shift)
    run_ode = semidiscretize(run_semi, tspan)
    run_callbacks = make_run_callbacks(
        run_semi, run_amr_callback, run_polydeg;
        run_amr_configuration=run_amr_configuration,
        run_id=run_id)
    println("Starting ", run_id,
            " with polydeg = ", run_polydeg,
            ", shear-aware AMR = ", run_amr_configuration.shear_aware,
            ", controller level shift = ", level_shift,
            ", AMR interval = ", run_amr_configuration.interval,
            ", AMR maximum level = ", run_amr_configuration.max_level,
            ", initial dt = ", initial_dt)
    run_sol = solve_rti_ode(run_ode, run_callbacks)
    run_checks = (retcode=run_sol.retcode,
                  t_end=Float64(run_sol.t[end]))
    # A returned solution is not sufficient: explicitly reject early
    # termination or a stalled/incomplete run before post-processing.
    isapprox(run_checks.t_end, tspan[2]; atol=1.0e-10, rtol=0.0) ||
        error("The $run_id solve did not reach t = $(tspan[2]); " *
              "it stopped at t = $(run_checks.t_end) with retcode " *
              "$(run_checks.retcode).")
    run_npz_directory = joinpath(npz_root, run_id)
    npz_archive_has_all_save_times(run_npz_directory) ||
        error("The $run_id NPZ archive is incomplete after the solve; " *
              "resolution diagnostics and visualization require all save times.")
    println("Finished ", run_id,
            " at t = ", run_checks.t_end,
            " with retcode ", run_checks.retcode)
    return (sol=run_sol,
            semi=run_semi,
            ode=run_ode,
            dg=run_dg,
            mesh=run_mesh,
            amr_configuration=run_amr_configuration,
            npz_directory=run_npz_directory,
            checks=run_checks)
end
# Reuse the archived p=7 result as the resolution baseline. If it is unavailable,
# generate a shear-aware level_shift=0 baseline once so that the comparison
# remains self-contained. No p=3 or p=5 run is required for this refinement.
p7_baseline_run = nothing
if npz_archive_has_all_save_times(npz_out_p7_baseline)
    println("Reusing complete archived p=7 baseline snapshots in ",
            npz_out_p7_baseline)
else
    println("No complete archived p=7 baseline was found; ",
            "running the shear-aware p=7 baseline fallback.")
    p7_baseline_run = run_rti_case(
        7;
        shear_aware=true,
        level_shift=0,
        run_id="p7_strengthened_baseline")
    npz_out_p7_baseline = p7_baseline_run.npz_directory
end
npz_archive_has_all_save_times(npz_out_p7_baseline) ||
    error("The p=7 baseline NPZ archive is incomplete; " *
          "the fixed-p=7 resolution comparison cannot be produced.")
# Execute the companion at the same polynomial degree. Cell 7 raises the
# intermediate density-controller target from level 5 to level 6 while the
# global density controller remains capped at level 6. From t=1.5, the masked
# mixing-layer shear controller can additionally request local level 7.
p7_h_refined_run = run_rti_case(
    7;
    shear_aware=true,
    level_shift=1,
    run_id="p7_h_refined")
# The new companion is the primary archive for visualization. The archived or
# fallback baseline remains available separately for Cell 10 diagnostics.
sol = p7_h_refined_run.sol
sol_p7 = sol
semi_p7 = p7_h_refined_run.semi
ode_p7 = p7_h_refined_run.ode
checks_p7 = p7_h_refined_run.checks
npz_out_p7_h_refined = p7_h_refined_run.npz_directory
npz_out_p7 = npz_out_p7_h_refined
npz_out_p7_primary = npz_out_p7
println("Using ", primary_run_label,
        " as the primary archive: ", npz_out_p7)
println("Fixed-p=7 baseline archive: ", npz_out_p7_baseline)
println("Finished primary p=7 local-h-refined run at t = ",
        checks_p7.t_end, " with retcode ", checks_p7.retcode)
# %% [markdown]

# ## Cell 10: either reuses or creates the p=3 archive. Never silently omit p=3.
# Both fixed-p=7 runs must have complete archives before any common-grid
# resolution comparison is attempted. No p=3 or p=5 run is required.
for (label, directory) in (
        ("p=7 baseline", npz_out_p7_baseline),
        (primary_run_label, npz_out_p7_h_refined))
    npz_archive_has_all_save_times(directory) ||
        error("The $label NPZ archive in $directory is incomplete; " *
              "the fixed-p=7 local-h-refinement comparison cannot be produced.")
end

# Construct this list only when called, after Cell 9 has resolved the baseline
# and companion archive directories.
function current_archive_runs()
    return Tuple{String, String}[
        ("p=7 baseline", npz_out_p7_baseline),
        (primary_run_label, npz_out_p7_h_refined)
    ]
end
function print_grid_metrics(label, grid)
    metrics = interface_metrics(grid)
    kinetic_energy = uniform_grid_kinetic_energy(grid)
    Printf.@printf(
        "  %s: bubble tip y = %.8f, spike tip y = %.8f, mixing width = %.8f, KE = %.12e\n",
        label, metrics.bubble_tip, metrics.spike_tip,
        metrics.mixing_width, kinetic_energy)
    return nothing
end
function log_archive_checks(label, directory)
    data = snapshot_at_time(directory, 3.0)
    final_time = snapshot_time(data)
    reached_t_end = isapprox(final_time, 3.0; atol=1.0e-10, rtol=0.0)
    u_cons = data["u_cons"]
    fields_finite = all(isfinite, u_cons)
    min_rho = Inf
    min_pressure = Inf
    gamma = equations.gamma
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
            " (final archived t = ", final_time, ")")
    println("[$label] all fields finite (no NaN/Inf): ", fields_finite)
    println("[$label] minimum rho = ", min_rho,
            "; min rho > 0: ", min_rho > 0)
    println("[$label] minimum p = ", min_pressure,
            "; min p > 0: ", min_pressure > 0)
    return nothing
end
# Called at the end of Cell 12 after all interpolation and diagnostic
# functions exist. The legacy function name is retained for call compatibility,
# but this routine performs resolution diagnostics, not exact-solution certification.
function run_certification_and_audit(primary_grids, target_times)
    archive_runs = current_archive_runs()

    println("Kinetic energy histories for the fixed-p=7 baseline " *
            "and local-h-refined companion:")
    for (label, directory) in archive_runs
        times, energies = kinetic_energy_history(directory)
        println("  $label:")
        for (time, energy) in zip(times, energies)
            Printf.@printf("    t = %.2f, KE = %.12e\n", time, energy)
        end
    end

    println("\nCommon-grid fixed-p=7 local-h-refinement comparison " *
            "on the 512 x 2048 grid.")
    println("These are resolution-difference diagnostics, " *
            "not errors against an exact solution.")
    println("The p=7 baseline is only a comparison reference; " *
            "neither archive is certified exact ground truth.")

    for target_time in (2.0, 3.0)
        primary_index = findfirst(
            time -> isapprox(time, target_time; atol=1.0e-6, rtol=0.0),
            target_times)
        primary_index === nothing &&
            error("Primary p=7 grid is missing t = $target_time.")

        # Reuse the already interpolated companion grid and interpolate the
        # baseline onto exactly the same cell-centered uniform grid.
        grid_h_refined = primary_grids[primary_index]
        grid_baseline = interpolate_amr_snapshot(
            snapshot_at_time(npz_out_p7_baseline, target_time))

        Printf.@printf("\nt = %.1f\n", target_time)
        print_grid_metrics("p=7 baseline", grid_baseline)
        print_grid_metrics(primary_run_label, grid_h_refined)

        # Normalize by the baseline field. These values measure sensitivity
        # between resolutions and must not be reported as ground-truth errors.
        l2_rho = relative_l2_difference(grid_h_refined.rho,
                                        grid_baseline.rho)
        l2_v = relative_l2_difference(grid_h_refined.v,
                                      grid_baseline.v)
        Printf.@printf(
            "  p=7 h-refined versus baseline relative L2 resolution differences: rho = %.8e, v = %.8e\n",
            l2_rho, l2_v)
    end

    # Keep the integrity and positivity audit near the end of stdout.
    println("\nFinal integrity and positivity audit:")
    for (label, directory) in archive_runs
        log_archive_checks(label, directory)
    end
    println("Done.")
    return nothing
end

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
# ## Rayleigh–Taylor Visualization and Fixed-p=7 Resolution Diagnostics
#
# The p=7 local-h-refined run with density-keyed and mixing-layer-restricted
# shear-aware AMR is plotted at t = 0.5, 1.0, 2.0, and 3.0. The p=7 baseline
# and companion are compared on the same uniform grid at t = 2.0 and 3.0.
# No smoothing, fixed colormap, fixed color limits, or artificial reference error is used.
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
# Shear-aware AMR is active from t=1.5; its density-masked indicator is
# restricted to the mixing layer, and only this local path can request level 7.
npz_out_p7 = npz_out_p7_h_refined

println("Shear-aware AMR active from t=1.5; local mixing-layer level-7 refinement enabled.")
println("Primary fixed-p=7 local-h-refined archive: ", npz_out_p7)
println("Generating Rayleigh–Taylor summary plots...")
target_times = (0.5, 1.0, 2.0, 3.0)
primary_grids = [
    interpolate_amr_snapshot(snapshot_at_time(npz_out_p7, target_time))
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
summary_path = joinpath(@__DIR__, "summary_all.png")
savefig(final_plot, summary_path)
isfile(summary_path) ||
    error("summary_all.png was not created at $summary_path.")
filesize(summary_path) > 0 ||
    error("summary_all.png was created but is empty at $summary_path.")
println("Saved and verified summary_all.png: ", summary_path)
# Print fixed-p=7 baseline-versus-h-refined resolution diagnostics at t=2
# and t=3, followed by finiteness, arrival-time, density, and pressure audits.
run_certification_and_audit(primary_grids, target_times)