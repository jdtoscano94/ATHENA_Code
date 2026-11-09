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
#
# 3.  **Interface Flux (`surface_flux=flux_hllc`)**:
#     * **Riemann Solver**: Harten-Lax-van Leer with Contact (HLLC).
#     * **Role**: Computes the numerical flux across element interfaces where the solution is discontinuous.
#     * **Physics**: An approximate Riemann solver that resolves the contact wave while remaining robust for compressible flows.
#
# 4.  **Indicator Function (`IndicatorHennemannGassner`)**:
#     * Dynamically computes the blending coefficient $\alpha \in [0, 1]$ element-wise.
#      * **Mechanism**: Monitors the spectral decay of energy in the highest modes of the polynomial expansion. If energy in high modes is large (indicating a discontinuity), $\alpha \to 1$ (activates FV sub-cells).
#
# %%
polydeg = 5
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.2,
                                         alpha_min=1.0e-3,
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
#     * **Role**: Acts as a "feature detector," using density curvature to identify the material interface, shear layers, and roll-up structures that require higher resolution.
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
# Wrap the supported controller while forwarding its fields for AMRCallback.
# No abstract controller supertype is required for get_element_variables!
# dispatch or for the generic AMRCallback interface.
struct MirrorPairedAMRController{C}
    base_controller::C
end

function Base.getproperty(controller::MirrorPairedAMRController, name::Symbol)
    name === :base_controller && return getfield(controller, :base_controller)
    return getproperty(getfield(controller, :base_controller), name)
end
# Construct the mirror partner of every leaf from its physical bounds. Keys use
# level-6 dyadic coordinates, so pairing does not depend on leaf storage order.
function mirror_amr_leaf_partners(semi)
    coordinates = semi.cache.elements.node_coordinates
    n_elements = size(coordinates, 4)
    n_elements > 0 || error("Mirror-paired AMR: the mesh has no leaf elements")
    scale = 64 # 2^max_level, with max_level = 6
    leaf_lookup = Dict{NTuple{3, Int}, Int}()
    bounds = Vector{NTuple{4, Float64}}(undef, n_elements)
    levels = Vector{Int}(undef, n_elements)
    for element in 1:n_elements
        x_min, x_max = extrema(view(coordinates, 1, :, :, element))
        y_min, y_max = extrema(view(coordinates, 2, :, :, element))
        width = x_max - x_min
        height = y_max - y_min
        width > 0.0 && height > 0.0 ||
            error("Mirror-paired AMR: degenerate leaf element $element")
        level = round(Int, log2(1.0 / height))
        tolerance = 1.0e-12
        isapprox(height, 1.0 / 2.0^level; atol=tolerance, rtol=0.0) &&
            isapprox(width, 0.25 / 2.0^level;
                     atol=tolerance, rtol=0.0) ||
            error("Mirror-paired AMR: cannot determine the level of leaf $element")
        ix_min = round(Int, scale * x_min / 0.25)
        iy_min = round(Int, scale * y_min)
        key = (level, ix_min, iy_min)
        haskey(leaf_lookup, key) &&
            error("Mirror-paired AMR: duplicate geometric leaf key $key")
        leaf_lookup[key] = element
        bounds[element] = (x_min, x_max, y_min, y_max)
        levels[element] = level
    end
    partners = zeros(Int, n_elements)
    for element in 1:n_elements
        x_min, x_max, y_min, y_max = bounds[element]
        level = levels[element]
        # m([x_min,x_max]×[y_min,y_max]) =
        # [0.25-x_max,0.25-x_min]×[y_min,y_max].
        mirror_ix_min = scale - round(Int, scale * x_max / 0.25)
        mirror_iy_min = round(Int, scale * y_min)
        mirror_key = (level, mirror_ix_min, mirror_iy_min)
        partner = get(leaf_lookup, mirror_key, 0)
        partner > 0 ||
            error("Mirror-paired AMR post-rebalance check failed: " *
                  "leaf $element with bounds $(bounds[element]) and level " *
                  "$level has no identical-size mirror partner")
        partner != element ||
            error("Mirror-paired AMR: leaf $element unexpectedly maps to itself")
        partner_bounds = bounds[partner]
        tolerance = 1.0e-12
        same_mirror_geometry =
            isapprox(partner_bounds[1], 0.25 - x_max;
                     atol=tolerance, rtol=0.0) &&
            isapprox(partner_bounds[2], 0.25 - x_min;
                     atol=tolerance, rtol=0.0) &&
            isapprox(partner_bounds[3], y_min;
                     atol=tolerance, rtol=0.0) &&
            isapprox(partner_bounds[4], y_max;
                     atol=tolerance, rtol=0.0) &&
            levels[partner] == level
        same_mirror_geometry ||
            error("Mirror-paired AMR post-rebalance check failed: " *
                  "leaves $element and $partner differ in size, level, or " *
                  "mirror geometry")
        partners[element] = partner
    end
    return partners
end
function pair_mirror_amr_decisions!(element_variables, semi)
    partners = mirror_amr_leaf_partners(semi)
    n_elements = length(partners)
    for key in (:indicator, :refine, :coarsen)
        haskey(element_variables, key) ||
            error("Mirror-paired AMR: controller output is missing $key")
    end
    indicator_values = element_variables[:indicator]
    refine = element_variables[:refine]
    coarsen = element_variables[:coarsen]
    length(indicator_values) == n_elements &&
        length(refine) == n_elements &&
        length(coarsen) == n_elements ||
        error("Mirror-paired AMR: controller arrays do not match the leaf count")
    for element in 1:n_elements
        partner = partners[element]
        element < partner || continue # Treat each unordered pair exactly once.
        paired_indicator = max(indicator_values[element],
                               indicator_values[partner])
        indicator_values[element] = paired_indicator
        indicator_values[partner] = paired_indicator
        # For equal-level partners, this reduction is equivalent to applying
        # ControllerThreeLevel to the paired maximum indicator. In particular,
        # a low-indicator partner cannot coarsen when its mirror must be kept.
        paired_refine = refine[element] || refine[partner]
        paired_coarsen = !paired_refine &&
                         coarsen[element] && coarsen[partner]
        refine[element] = paired_refine
        refine[partner] = paired_refine
        coarsen[element] = paired_coarsen
        coarsen[partner] = paired_coarsen
    end
    return element_variables
end
function Trixi.get_element_variables!(element_variables,
                                      controller::MirrorPairedAMRController,
                                      u, semi)
    Trixi.get_element_variables!(element_variables,
                                 getfield(controller, :base_controller),
                                 u, semi)
    return pair_mirror_amr_decisions!(element_variables, semi)
end
amr_indicator = IndicatorLöhner(semi, variable=density_indicator)
base_amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                           base_level=4,
                                           med_level=5, med_threshold=0.1,
                                           max_level=6, max_threshold=0.3)
amr_controller = MirrorPairedAMRController(base_amr_controller)
base_amr_callback = AMRCallback(semi, amr_controller,
                                interval=20,
                                adapt_initial_condition=true)
# The standard callback still performs all projection, rebalancing, and
# reinitialization. The wrapper only adds the mandatory topology assertion.
function mirror_paired_amr_affect!(integrator)
    base_amr_callback.affect!(integrator)
    mirror_amr_leaf_partners(integrator.p)
    return nothing
end
function mirror_paired_amr_initialize(callback, u, t, integrator)
    if base_amr_callback.initialize !== nothing
        base_amr_callback.initialize(base_amr_callback, u, t, integrator)
    end
    mirror_amr_leaf_partners(integrator.p)
    return nothing
end
amr_callback = DiscreteCallback(base_amr_callback.condition,
                                mirror_paired_amr_affect!;
                                initialize=mirror_paired_amr_initialize,
                                save_positions=base_amr_callback.save_positions)
# %% [markdown]

# ## Cell 8: Data I/O & Surrogate Interoperability
# ## Strategy: Cross-Language Serialization (.npz)
#
# Saves the complete primitive state `(rho, u, v, p)` and the spatial
# coordinates every `dt_save = 0.1` time units for ground-truth data.
# `PlotData2D` reconstructs the adaptive DG solution on a structured grid.
# %%
npz_out = joinpath(@__DIR__, "data_npz_polydeg5")
mkpath(npz_out)
dt_save = 0.1

comparison_nx, comparison_ny = 256, 1024
comparison_x_axis = range(0.25 / (2 * comparison_nx),
                          0.25 - 0.25 / (2 * comparison_nx);
                          length=comparison_nx)
comparison_y_axis = range(1.0 / (2 * comparison_ny),
                          1.0 - 1.0 / (2 * comparison_ny);
                          length=comparison_ny)
archive_save_times =
    [index * dt_save for index in 0:round(Int, 3.0 / dt_save)]

initial_total_mass = Ref(NaN)
initial_total_energy = Ref(NaN)
previous_energy_time = Ref(NaN)
previous_gravity_work_rate = Ref(NaN)
cumulative_gravity_work = Ref(0.0)
last_npz_save_time = Ref(NaN)
verification_minima = Dict{Symbol, Float64}()
verification_latest = Dict{Symbol, Float64}()
spike_tip_times = Float64[]
spike_tip_positions = Float64[]

@inline diagnostic_mass(u, equations) = u[1]
@inline diagnostic_total_energy(u, equations) = u[4]
# g = 1 and rho * g * v = g * rho_v2.
@inline diagnostic_gravity_work(u, equations) = u[3]

function record_verification_error!(name, value)
    verification_latest[name] = value
    if isfinite(value)
        verification_minima[name] =
            min(get(verification_minima, name, Inf), abs(value))
    end
    return nothing
end
function initialize_energy_balance_state!(integrator)
    semi = integrator.p
    initial_total_energy[] =
        Trixi.integrate(diagnostic_total_energy, integrator.u, semi)
    previous_gravity_work_rate[] =
        Trixi.integrate(diagnostic_gravity_work, integrator.u, semi)
    previous_energy_time[] = integrator.t
    cumulative_gravity_work[] = 0.0
    return nothing
end
function initialize_energy_balance_callback(callback, u, t, integrator)
    initialize_energy_balance_state!(integrator)
    return nothing
end
# Accumulate ∫ rho*g*v dV dt with a trapezoidal update on every accepted
# time step. Each spatial quadrature uses the mesh active at that endpoint.
function accumulate_gravity_work!(integrator)
    if !isfinite(previous_energy_time[]) ||
       !isfinite(previous_gravity_work_rate[])
        integrator.t == 0.0 ||
            error("Cumulative energy balance was not initialized at t=0")
        initialize_energy_balance_state!(integrator)
        return nothing
    end
    t = integrator.t
    dt = t - previous_energy_time[]
    dt >= 0.0 || error("Cumulative energy balance encountered reversed time")
    current_gravity_work_rate =
        Trixi.integrate(diagnostic_gravity_work, integrator.u, integrator.p)
    if dt > 0.0
        cumulative_gravity_work[] +=
            0.5 * dt * (previous_gravity_work_rate[] +
                        current_gravity_work_rate)
    end
    previous_energy_time[] = t
    previous_gravity_work_rate[] = current_gravity_work_rate
    return nothing
end
energy_balance_condition(u, t, integrator) = true
energy_balance_callback = DiscreteCallback(
    energy_balance_condition, accumulate_gravity_work!;
    initialize=initialize_energy_balance_callback,
    save_positions=(false, false))
npz_scalar(value) = value isa Number ? value : first(value)
metadata_bytes(text) = collect(codeunits(text))

function raw_dg_state(integrator)
    semi = integrator.p
    u = Trixi.wrap_array(integrator.u, semi)
    coordinates = semi.cache.elements.node_coordinates

    ndims(u) == 4 && size(u, 1) == 4 ||
        error("Raw DG archive: expected a four-variable 2D DG state")
    ndims(coordinates) == 4 && size(coordinates, 1) == 2 ||
        error("Raw DG archive: invalid node-coordinate storage")
    size(u, 2) == size(coordinates, 2) &&
        size(u, 3) == size(coordinates, 3) &&
        size(u, 4) == size(coordinates, 4) ||
        error("Raw DG archive: coordinates and solution have different layouts")

    # Both views have (node_i, node_j, element) storage, so flattening them
    # preserves the exact correspondence between coordinates and conservative
    # variables. No solution values are altered or symmetrized.
    x = Float64.(vec(view(coordinates, 1, :, :, :)))
    y = Float64.(vec(view(coordinates, 2, :, :, :)))
    conserved = ntuple(
        variable -> Float64.(vec(view(u, variable, :, :, :))), 4)

    all(field -> length(field) == length(x), conserved) &&
        length(y) == length(x) ||
        error("Raw DG archive: inconsistent flattened array lengths")
    return x, y, conserved
end

function interpolate_conserved_grid(x, y, fields;
                                    nx=comparison_nx,
                                    ny=comparison_ny)
    iseven(nx) || error("The mirror-comparison grid requires an even nx")
    nx > 0 && ny > 0 || error("Comparison-grid dimensions must be positive")

    x_source = Float64.(vec(x))
    y_source = Float64.(vec(y))
    field_sources = ntuple(variable -> Float64.(vec(fields[variable])), 4)
    n_source = length(field_sources[1])

    n_source > 0 || error("Cannot interpolate an empty DG state")
    length(x_source) == n_source && length(y_source) == n_source ||
        error("Comparison-grid interpolation: inconsistent coordinate lengths")
    all(field -> length(field) == n_source, field_sources) ||
        error("Comparison-grid interpolation: inconsistent field lengths")
    all(isfinite, x_source) && all(isfinite, y_source) &&
        all(field -> all(isfinite, field), field_sources) ||
        error("Comparison-grid interpolation: non-finite input")

    dx = 0.25 / nx
    dy = 1.0 / ny
    source_index = zeros(Int, ny, nx)
    nearest_distance = fill(Inf, ny, nx)

    # Every field at a target cell uses the same source DG node. This avoids
    # introducing variable-dependent interpolation artifacts.
    for k in 1:n_source
        i = clamp(floor(Int, x_source[k] / dx) + 1, 1, nx)
        j = clamp(floor(Int, y_source[k] / dy) + 1, 1, ny)
        x_center = (i - 0.5) * dx
        y_center = (j - 0.5) * dy
        distance = (x_source[k] - x_center)^2 +
                   (y_source[k] - y_center)^2
        if distance < nearest_distance[j, i]
            nearest_distance[j, i] = distance
            source_index[j, i] = k
        end
    end

    # Deterministic nearest-seed propagation fills cells between DG nodes.
    queue_i = Int[]
    queue_j = Int[]
    sizehint!(queue_i, nx * ny)
    sizehint!(queue_j, nx * ny)
    for j in 1:ny, i in 1:nx
        if source_index[j, i] != 0
            push!(queue_i, i)
            push!(queue_j, j)
        end
    end
    isempty(queue_i) && error("Comparison-grid interpolation found no seeds")

    head = 1
    while head <= length(queue_i)
        i = queue_i[head]
        j = queue_j[head]
        head += 1
        for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
            neighbor_i = i + di
            neighbor_j = j + dj
            if 1 <= neighbor_i <= nx && 1 <= neighbor_j <= ny &&
               source_index[neighbor_j, neighbor_i] == 0
                source_index[neighbor_j, neighbor_i] = source_index[j, i]
                push!(queue_i, neighbor_i)
                push!(queue_j, neighbor_j)
            end
        end
    end
    all(index -> index > 0, source_index) ||
        error("Comparison-grid interpolation left gaps")

    grids = ntuple(_ -> Matrix{Float64}(undef, ny, nx), 4)
    for j in 1:ny, i in 1:nx
        source = source_index[j, i]
        for variable in 1:4
            grids[variable][j, i] = field_sources[variable][source]
        end
    end
    return grids
end

function mirror_symmetry_error(grids)
    length(grids) == 4 || error("Symmetry metric requires four variables")
    grid_shape = size(grids[1])
    all(grid -> size(grid) == grid_shape, grids) ||
        error("Symmetry metric: inconsistent grid shapes")
    ny, nx = grid_shape
    nx > 0 && ny > 0 && iseven(nx) ||
        error("Symmetry metric: invalid common-grid dimensions")

    numerator = 0.0
    denominator = 0.0
    for variable in 1:4
        grid = grids[variable]
        for j in 1:ny, i in 1:nx
            value = grid[j, i]
            mirror_value = grid[j, nx + 1 - i]
            # rho, rho_v, and E are even; rho_u is odd under reflection.
            residual = variable == 2 ? value + mirror_value :
                       value - mirror_value
            numerator += residual^2
            denominator += value^2
        end
    end
    denominator > 0.0 || return NaN
    return sqrt(numerator / denominator)
end

function archive_spike_tip_position(rho_grid, y_coordinates)
    size(rho_grid, 1) == length(y_coordinates) ||
        error("Spike-tip archive: inconsistent grid and y coordinates")
    tip = NaN
    for j in axes(rho_grid, 1)
        if any(value -> value >= 1.5, view(rho_grid, j, :))
            tip = isnan(tip) ? y_coordinates[j] :
                  max(tip, y_coordinates[j])
        end
    end
    return tip
end

function archive_metadata(save_index)
    return Dict{String, Any}(
        "polydeg" => polydeg,
        "mesh" => metadata_bytes(
            "P4estMesh [0,0.25]x[0,1]; initial_refinement_level=3; " *
            "periodic_x=true; slip_walls_y=true"),
        "amr_policy" => metadata_bytes(
            "mirror-paired IndicatorLoehner(rho); " *
            "ControllerThreeLevel(base=4,med=5,max=6," *
            "med_threshold=0.1,max_threshold=0.3); interval=20"),
        "cfl" => 0.3,
        "dt_save" => dt_save,
        "save_times" => copy(archive_save_times),
        "save_index" => save_index,
        "comparison_nx" => comparison_nx,
        "comparison_ny" => comparison_ny,
        "initial_refinement_level" => 3,
        "amr_base_level" => 4,
        "amr_med_level" => 5,
        "amr_max_level" => 6,
        "amr_interval" => 20,
        "alpha_max" => 0.2,
        "domain_min" => [0.0, 0.0],
        "domain_max" => [0.25, 1.0],
        "mirror_plane_x" => 0.125,
        "surface_flux" => metadata_bytes("HLLC"),
        "volume_flux" => metadata_bytes("Ranocha"),
        "time_integrator" => metadata_bytes("RDPK3SpFSAL49"),
        "raw_layout" => metadata_bytes(
            "raw_x/raw_y and raw conserved values use identical DG-node order"),
        "grid_layout" => metadata_bytes(
            "grid arrays are (y,x) at comparison_x/comparison_y cell centers"))
end

function verify_npz_file(filename)
    data = npzread(filename)
    required_keys = (
        "t", "x", "y", "rho", "u", "v", "p",
        "raw_x", "raw_y", "raw_rho", "raw_rho_u", "raw_rho_v",
        "raw_energy", "comparison_x", "comparison_y",
        "grid_rho", "grid_rho_u", "grid_rho_v", "grid_energy",
        "spike_tip", "polydeg", "mesh", "amr_policy", "cfl",
        "save_times", "dt_save", "comparison_nx", "comparison_ny",
        "save_index")
    for key in required_keys
        haskey(data, key) ||
            error("NPZ sanity check: missing key $key in $filename")
    end

    n = length(data["rho"])
    primitive_coverage = n > 0 &&
        all(key -> length(data[key]) == n, ("x", "y", "u", "v", "p"))

    raw_n = length(data["raw_rho"])
    raw_coverage = raw_n > 0 &&
        all(key -> length(data[key]) == raw_n,
            ("raw_x", "raw_y", "raw_rho", "raw_rho_u",
             "raw_rho_v", "raw_energy"))

    grid_shape = (comparison_ny, comparison_nx)
    grid_keys = ("grid_rho", "grid_rho_u", "grid_rho_v", "grid_energy")
    grid_coverage =
        all(key -> size(data[key]) == grid_shape, grid_keys) &&
        length(data["comparison_x"]) == comparison_nx &&
        length(data["comparison_y"]) == comparison_ny

    finite_state = all(key -> all(isfinite, data[key]),
        ("rho", "u", "v", "p", "raw_rho", "raw_rho_u", "raw_rho_v",
         "raw_energy", grid_keys...))
    finite_coordinates = all(key -> all(isfinite, data[key]),
        ("x", "y", "raw_x", "raw_y", "comparison_x", "comparison_y"))

    raw_rho = data["raw_rho"]
    raw_rho_u = data["raw_rho_u"]
    raw_rho_v = data["raw_rho_v"]
    raw_pressure = (equations.gamma - 1) .*
        (data["raw_energy"] .-
         0.5 .* (raw_rho_u.^2 .+ raw_rho_v.^2) ./ raw_rho)
    positive_state =
        all(value -> value > 0, data["rho"]) &&
        all(value -> value > 0, data["p"]) &&
        all(value -> value > 0, raw_rho) &&
        all(value -> value > 0, raw_pressure) &&
        all(value -> value > 0, data["grid_rho"])

    saved_times = Float64.(vec(data["save_times"]))
    save_times_valid = length(saved_times) == length(archive_save_times) &&
        all(index -> isapprox(saved_times[index], archive_save_times[index];
                              atol=1.0e-12, rtol=0.0),
            eachindex(archive_save_times))
    save_index = Int(npz_scalar(data["save_index"]))
    metadata_valid =
        Int(npz_scalar(data["polydeg"])) == polydeg &&
        Float64(npz_scalar(data["cfl"])) == 0.3 &&
        Float64(npz_scalar(data["dt_save"])) == dt_save &&
        Int(npz_scalar(data["comparison_nx"])) == comparison_nx &&
        Int(npz_scalar(data["comparison_ny"])) == comparison_ny &&
        0 <= save_index < length(archive_save_times) &&
        save_times_valid &&
        length(data["mesh"]) > 0 && length(data["amr_policy"]) > 0

    spike_tip = Float64(npz_scalar(data["spike_tip"]))
    spike_valid = isfinite(spike_tip) && 0.0 <= spike_tip <= 1.0
    time_valid = isfinite(Float64(npz_scalar(data["t"])))

    if !(primitive_coverage && raw_coverage && grid_coverage &&
         finite_state && finite_coordinates && positive_state &&
         metadata_valid && spike_valid && time_valid)
        error("NPZ sanity check failed for $filename: " *
              "primitive_coverage=$primitive_coverage, " *
              "raw_coverage=$raw_coverage, grid_coverage=$grid_coverage, " *
              "finite_state=$finite_state, " *
              "finite_coordinates=$finite_coordinates, " *
              "positive_state=$positive_state, metadata=$metadata_valid, " *
              "spike_tip=$spike_valid, time=$time_valid")
    end
    return nothing
end
function npz_plot_axis(values)
    if ndims(values) == 1
        return Float64.(values)
    elseif ndims(values) == 2 &&
           (size(values, 1) == 1 || size(values, 2) == 1)
        return Float64.(vec(values))
    end
    return nothing
end
function npz_plot_node_arrays(pd)
    x_raw = collect(pd.x)
    y_raw = collect(pd.y)
    data = pd.data
    # PlotData2DTriangulated stores one primitive state vector per plot node.
    # Extract variable i from every state; data[i] would instead select node i.
    # The length check also supports equivalent flattened nodal storage.
    if !(eltype(data) <: Number) &&
       length(data) == length(x_raw) &&
       length(data) == length(y_raw)
        fields = ntuple(i -> Float64.(getindex.(data, i)), 4)
        n_values = length(x_raw)
        all(field -> length(field) == n_values, fields) ||
            error("NPZ serialization: invalid triangulated primitive fields")
        # vec uses the same column-major node ordering for coordinates and fields.
        values = ntuple(i -> vec(fields[i]), 4)
        return Float64.(vec(x_raw)), Float64.(vec(y_raw)), values
    end
    # Structured PlotData2D stores the primitive variables as separate fields.
    fields = ntuple(i -> Float64.(collect(data[i])), 4)
    field_shape = size(fields[1])
    all(field -> size(field) == field_shape, fields) ||
        error("NPZ serialization: inconsistent primitive field layouts; " *
              "size.(pd.data)=$(size.(data))")
    n_values = length(fields[1])
    # Node-wise coordinates and fields have identical storage ordering.
    if size(x_raw) == field_shape && size(y_raw) == field_shape
        values = ntuple(i -> vec(fields[i]), 4)
        return Float64.(vec(x_raw)), Float64.(vec(y_raw)), values
    end
    # Structured PlotData2D objects can store only the coordinate axes.
    # Expand both axes before flattening so every field value receives the
    # coordinate pair at the same matrix index.
    x_axis = npz_plot_axis(x_raw)
    y_axis = npz_plot_axis(y_raw)
    if x_axis !== nothing && y_axis !== nothing && length(field_shape) == 2
        nx = length(x_axis)
        ny = length(y_axis)
        if field_shape == (ny, nx)
            # Heatmap convention: rows correspond to y, columns to x.
            x_full = repeat(reshape(x_axis, 1, nx), ny, 1)
            y_full = repeat(reshape(y_axis, ny, 1), 1, nx)
        elseif field_shape == (nx, ny)
            # Dimension-major convention: rows correspond to x, columns to y.
            x_full = repeat(reshape(x_axis, nx, 1), 1, ny)
            y_full = repeat(reshape(y_axis, 1, ny), nx, 1)
        else
            error("NPZ serialization: coordinate axes do not match field layout; " *
                  "size(pd.x)=$(size(x_raw)), size(pd.y)=$(size(y_raw)), " *
                  "size.(pd.data)=$(size.(data))")
        end
        values = ntuple(i -> vec(fields[i]), 4)
        return vec(x_full), vec(y_full), values
    end
    # Adaptive plot objects may store flat node-wise coordinate vectors while
    # retaining matrix containers for the primitive fields.
    if length(x_raw) == n_values && length(y_raw) == n_values
        values = ntuple(i -> vec(fields[i]), 4)
        return Float64.(vec(x_raw)), Float64.(vec(y_raw)), values
    end
    error("NPZ serialization: unsupported PlotData2D layout; " *
          "size(pd.x)=$(size(x_raw)), size(pd.y)=$(size(y_raw)), " *
          "size.(pd.data)=$(size.(data))")
end
function print_physical_diagnostics!(integrator, x, y, rho, v, p,
                                     grid_fields, spike_tip)
    semi = integrator.p
    t = integrator.t
    total_mass = Trixi.integrate(diagnostic_mass, integrator.u, semi)
    total_energy = Trixi.integrate(diagnostic_total_energy, integrator.u, semi)

    if !isfinite(initial_total_mass[])
        initial_total_mass[] = total_mass
    end
    if !isfinite(initial_total_energy[])
        t == 0.0 ||
            error("Cumulative energy balance is unavailable before initialization")
        initialize_energy_balance_state!(integrator)
    end

    mass_drift = total_mass - initial_total_mass[]
    relative_mass_drift = abs(mass_drift) /
                          max(abs(initial_total_mass[]), eps(Float64))
    rho_values = vec(rho)
    v_values = vec(v)
    p_values = vec(p)
    rho_min, rho_max = extrema(rho_values)
    p_min, p_max = extrema(p_values)
    symmetry_error = mirror_symmetry_error(grid_fields)

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

    energy_change = total_energy - initial_total_energy[]
    gravity_work_integral = cumulative_gravity_work[]
    energy_residual = energy_change - gravity_work_integral
    energy_closure = abs(energy_residual) /
                     max(abs(energy_change), abs(gravity_work_integral),
                         eps(Float64))

    record_verification_error!(:mass, relative_mass_drift)
    record_verification_error!(:symmetry, symmetry_error)
    record_verification_error!(:wall, relative_wall_violation)
    record_verification_error!(:energy_balance, energy_closure)

    verification_latest[:time] = t
    verification_latest[:mass_drift] = mass_drift
    verification_latest[:relative_mass_drift] = relative_mass_drift
    verification_latest[:rho_min] = rho_min
    verification_latest[:rho_max] = rho_max
    verification_latest[:p_min] = p_min
    verification_latest[:p_max] = p_max
    verification_latest[:wall_max_abs_v] = wall_velocity
    verification_latest[:max_abs_v] = max_abs_v
    verification_latest[:energy_change] = energy_change
    verification_latest[:cumulative_gravity_work] = gravity_work_integral
    verification_latest[:energy_residual] = energy_residual
    verification_latest[:energy_closure] = energy_closure
    verification_latest[:spike_tip] = spike_tip

    println(Printf.@sprintf(
        "Diagnostics t=%.5f: dM=%.6e, rel_dM=%.6e, rho=[%.6e,%.6e], p=[%.6e,%.6e], sym=%.6e, wall_max_abs_v=%.6e, delta_E=%.6e, cumulative_gravity_work=%.6e, energy_residual=%.6e, energy_closure=%.6e, spike_tip=%.6e",
        t, mass_drift, relative_mass_drift,
        rho_min, rho_max, p_min, p_max, symmetry_error,
        wall_velocity, energy_change, gravity_work_integral,
        energy_residual, energy_closure, spike_tip))
    return nothing
end

function save_npz_file(integrator)
    semi = integrator.p
    t = integrator.t
    if isfinite(last_npz_save_time[]) &&
       abs(t - last_npz_save_time[]) <= 1.0e-10
        return nothing
    end

    save_index = round(Int, t / dt_save)
    0 <= save_index < length(archive_save_times) ||
        error("NPZ archive: save time $t is outside the scheduled save times")

    # Retain the established primitive representation used by the plotting
    # cell, and independently archive the unmodified conservative DG state.
    pd = PlotData2D(integrator.u, semi, solution_variables=cons2prim)
    x, y, primitive_fields = npz_plot_node_arrays(pd)
    rho, u, v, p = primitive_fields
    n_values = length(rho)
    if length(x) != n_values || length(y) != n_values ||
       length(u) != n_values || length(v) != n_values ||
       length(p) != n_values
        error("NPZ serialization: incompatible coordinate and field lengths")
    end

    raw_x, raw_y, raw_conserved = raw_dg_state(integrator)
    grid_fields = interpolate_conserved_grid(raw_x, raw_y, raw_conserved)
    spike_tip = archive_spike_tip_position(grid_fields[1],
                                           comparison_y_axis)
    isfinite(spike_tip) ||
        error("Spike-tip archive: no grid cell has rho >= 1.5 at t=$t")

    payload = Dict{String, Any}(
        "t" => Float64(t),
        "x" => x, "y" => y,
        "rho" => rho, "u" => u, "v" => v, "p" => p,
        "raw_x" => raw_x, "raw_y" => raw_y,
        "raw_rho" => raw_conserved[1],
        "raw_rho_u" => raw_conserved[2],
        "raw_rho_v" => raw_conserved[3],
        "raw_energy" => raw_conserved[4],
        "comparison_x" => collect(comparison_x_axis),
        "comparison_y" => collect(comparison_y_axis),
        "grid_rho" => grid_fields[1],
        "grid_rho_u" => grid_fields[2],
        "grid_rho_v" => grid_fields[3],
        "grid_energy" => grid_fields[4],
        "spike_tip" => spike_tip)
    merge!(payload, archive_metadata(save_index))

    filename = joinpath(npz_out, Printf.@sprintf("time_%08.5f.npz", t))
    npzwrite(filename, payload)
    verify_npz_file(filename)

    push!(spike_tip_times, t)
    push!(spike_tip_positions, spike_tip)
    track_filename = joinpath(npz_out, "spike_tip_track.npz")
    npzwrite(track_filename, Dict{String, Any}(
        "times" => copy(spike_tip_times),
        "spike_tip" => copy(spike_tip_positions),
        "save_times" => copy(archive_save_times)))
    track_data = npzread(track_filename)
    length(track_data["times"]) == length(spike_tip_times) &&
        length(track_data["spike_tip"]) == length(spike_tip_positions) &&
        all(isfinite, track_data["times"]) &&
        all(isfinite, track_data["spike_tip"]) ||
        error("NPZ sanity check failed for $track_filename")

    println("Saved and NPZ-verified: $(basename(filename)); " *
            Printf.@sprintf("spike_tip=%.6e", spike_tip))
    print_physical_diagnostics!(integrator, x, y, rho, v, p,
                                grid_fields, spike_tip)
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
#     Here, we set `cfl=0.3` for the polydeg-5 run while retaining CFL-based timestep control.
#
# ## 3. Observability & Interventions (`CallbackSet`)
# Aggregates the diagnostic and control agents into the main solver loop:
# * **`AnalysisCallback`**: Computes in-situ functional quantities (Entropy, Kinetic Energy, $L_2/L_\infty$ errors) to monitor physical validity.
# * **`SaveSolutionCallback`**: Performs full-state serialization (HDF5-compatible) for restarts.
# * **`AMRCallback`**: (From Cell 7) Triggers mesh adaptation.
# %%
tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)
# Return maximum mirror residuals for the projected state and for the gravity
# source evaluated at the same DG nodes. The mesh pairing supplies the element
# map; coordinate matching supplies the local-node map without assuming a
# particular p4est node orientation.
function mirror_setup_errors(integrator)
    semi_local = integrator.p
    state = Trixi.wrap_array(integrator.u, semi_local)
    coordinates = semi_local.cache.elements.node_coordinates
    partners = mirror_amr_leaf_partners(semi_local)
    projection_errors = zeros(Float64, 4)
    source_errors = zeros(Float64, 4)
    for element in axes(state, 4)
        partner = partners[element]
        for j in axes(state, 3), i in axes(state, 2)
            x = coordinates[1, i, j, element]
            y = coordinates[2, i, j, element]
            target_x = 0.25 - x
            mirror_i = 0
            mirror_j = 0
            best_distance = Inf
            for jj in axes(state, 3), ii in axes(state, 2)
                distance =
                    (coordinates[1, ii, jj, partner] - target_x)^2 +
                    (coordinates[2, ii, jj, partner] - y)^2
                if distance < best_distance
                    best_distance = distance
                    mirror_i = ii
                    mirror_j = jj
                end
            end
            best_distance <= 1.0e-24 ||
                error("Initial symmetry verification: no matching mirror " *
                      "DG node for element $element, local node ($i,$j)")
            for variable in 1:4
                value = state[variable, i, j, element]
                mirror_value =
                    state[variable, mirror_i, mirror_j, partner]
                residual = variable == 2 ? value + mirror_value :
                           value - mirror_value
                projection_errors[variable] =
                    max(projection_errors[variable], abs(residual))
            end
            local_state = SVector(
                state[1, i, j, element],
                state[2, i, j, element],
                state[3, i, j, element],
                state[4, i, j, element])
            mirror_state = SVector(
                state[1, mirror_i, mirror_j, partner],
                state[2, mirror_i, mirror_j, partner],
                state[3, mirror_i, mirror_j, partner],
                state[4, mirror_i, mirror_j, partner])
            local_source = source_terms_gravity(
                local_state, SVector(x, y), integrator.t, equations)
            mirror_source = source_terms_gravity(
                mirror_state,
                SVector(coordinates[1, mirror_i, mirror_j, partner],
                        coordinates[2, mirror_i, mirror_j, partner]),
                integrator.t, equations)
            for variable in 1:4
                residual = variable == 2 ?
                           local_source[variable] + mirror_source[variable] :
                           local_source[variable] - mirror_source[variable]
                source_errors[variable] =
                    max(source_errors[variable], abs(residual))
            end
        end
    end
    return projection_errors, source_errors
end
function verify_symmetric_setup!(integrator)
    projection_errors, source_errors = mirror_setup_errors(integrator)
    epsilon_ic = maximum(projection_errors)
    epsilon_source = maximum(source_errors)
    verification_latest[:initial_projection_symmetry] = epsilon_ic
    verification_latest[:source_symmetry] = epsilon_source
    println(Printf.@sprintf(
        "Initial-projection mirror symmetry: eps_IC=%.6e; rho=%.6e, rho_u=%.6e, rho_v=%.6e, E=%.6e",
        epsilon_ic, projection_errors[1], projection_errors[2],
        projection_errors[3], projection_errors[4]))
    println(Printf.@sprintf(
        "Gravity-source mirror symmetry: eps_S=%.6e; S_rho=%.6e, S_rho_u=%.6e, S_rho_v=%.6e, S_E=%.6e",
        epsilon_source, source_errors[1], source_errors[2],
        source_errors[3], source_errors[4]))
    if !(isfinite(epsilon_ic) && epsilon_ic <= 1.0e-12)
        @warn "Initial-projection symmetry exceeds the 1e-12 target" epsilon_ic
    end
    if !(isfinite(epsilon_source) && epsilon_source <= 1.0e-12)
        @warn "Source-term symmetry exceeds the 1e-12 target" epsilon_source
    end
    isfinite(epsilon_ic) && epsilon_ic <= 1.0e-10 ||
        error("Initial-projection mirror symmetry verification failed: " *
              "eps_IC=$epsilon_ic exceeds the 1e-10 abort threshold")
    isfinite(epsilon_source) && epsilon_source <= 1.0e-10 ||
        error("Gravity-source mirror symmetry verification failed: " *
              "eps_S=$epsilon_source exceeds the 1e-10 abort threshold")
    return nothing
end
function initialize_symmetry_verification(callback, u, t, integrator)
    # This callback is placed after amr_callback, so the check also covers the
    # L2 projection performed by adapt_initial_condition=true.
    verify_symmetric_setup!(integrator)
    return nothing
end
function finalize_symmetry_verification(callback, u, t, integrator)
    if integrator.t >= 3.0 - 8 * eps(3.0)
        raw_x, raw_y, raw_conserved = raw_dg_state(integrator)
        grid_fields = interpolate_conserved_grid(raw_x, raw_y, raw_conserved)
        final_symmetry = mirror_symmetry_error(grid_fields)
        verification_latest[:time] = integrator.t
        verification_latest[:symmetry] = final_symmetry
        verification_latest[:final_symmetry] = final_symmetry
        println(Printf.@sprintf(
            "Final mirror symmetry at t=%.5f: eps_sym=%.6e (acceptance <= 1.0e-10)",
            integrator.t, final_symmetry))
        isfinite(final_symmetry) && final_symmetry <= 1.0e-10 ||
            error("Final mirror-symmetry acceptance failed: " *
                  "eps_sym(3.0)=$final_symmetry exceeds 1e-10")
    end
    return nothing
end
symmetry_verification_callback = DiscreteCallback(
    (u, t, integrator) -> false,
    integrator -> nothing;
    initialize=initialize_symmetry_verification,
    finalize=finalize_symmetry_verification,
    save_positions=(false, false))
summary_callback = SummaryCallback()
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)
alive_callback = AliveCallback(analysis_interval=analysis_interval)
# Save full-state snapshots at the required physical-time interval.
trixi_out = joinpath(@__DIR__, "trixi_output")
save_solution = SaveSolutionCallback(dt=dt_save,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     output_directory=trixi_out,
                                     solution_variables=cons2prim)
stepsize_callback = StepsizeCallback(cfl=0.3)
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        amr_callback,
                        symmetry_verification_callback,
                        energy_balance_callback,
                        save_npz_callback,
                        stepsize_callback)
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
function final_diagnostic_value(name)
    value = get(verification_latest, name, NaN)
    isfinite(value) ||
        error("Final verification summary: missing or non-finite diagnostic $name")
    return value
end
function archived_spike_tip_at(times, positions, target_time)
    index = findfirst(
        time -> isapprox(time, target_time; atol=1.0e-10, rtol=0.0), times)
    index === nothing &&
        error("Spike-tip archive: missing save time $target_time")
    return positions[index]
end
function print_verification_summary()
    final_filename = joinpath(npz_out,
                              Printf.@sprintf("time_%08.5f.npz", 3.0))
    verify_npz_file(final_filename)
    final_data = npzread(final_filename)
    final_time = Float64(npz_scalar(final_data["t"]))
    isapprox(final_time, 3.0; atol=1.0e-10, rtol=0.0) ||
        error("Final verification summary: the final archive is not at t=3")
    # Recompute the metric from the archived common-grid conservative state.
    final_grids = (final_data["grid_rho"], final_data["grid_rho_u"],
                   final_data["grid_rho_v"], final_data["grid_energy"])
    final_symmetry = mirror_symmetry_error(final_grids)
    isfinite(final_symmetry) && final_symmetry <= 1.0e-10 ||
        error("Final mirror-symmetry acceptance failed during summary: " *
              "eps_sym(3.0)=$final_symmetry exceeds 1e-10")
    minimum_mass_error = get(verification_minima, :mass, NaN)
    isfinite(minimum_mass_error) ||
        error("Final verification summary: no finite mass diagnostic")
    final_mass_drift = final_diagnostic_value(:mass_drift)
    final_relative_mass_drift =
        final_diagnostic_value(:relative_mass_drift)
    initial_projection_error =
        final_diagnostic_value(:initial_projection_symmetry)
    source_error = final_diagnostic_value(:source_symmetry)
    wall_velocity = final_diagnostic_value(:wall_max_abs_v)
    energy_change = final_diagnostic_value(:energy_change)
    gravity_work = final_diagnostic_value(:cumulative_gravity_work)
    energy_residual = final_diagnostic_value(:energy_residual)
    energy_closure = final_diagnostic_value(:energy_closure)
    rho_min, rho_max = extrema(final_data["rho"])
    p_min, p_max = extrema(final_data["p"])
    rho_min > 0.0 && p_min > 0.0 ||
        error("Final verification summary: nonpositive density or pressure")
    # Confirm that the spike-tip track contains every common save time before
    # reporting the late-time positions needed by the next iteration.
    track_filename = joinpath(npz_out, "spike_tip_track.npz")
    isfile(track_filename) ||
        error("Final verification summary: missing spike_tip_track.npz")
    track_data = npzread(track_filename)
    haskey(track_data, "times") && haskey(track_data, "spike_tip") ||
        error("Final verification summary: incomplete spike-tip archive")
    track_times = Float64.(vec(track_data["times"]))
    track_positions = Float64.(vec(track_data["spike_tip"]))
    length(track_times) == length(archive_save_times) &&
        length(track_positions) == length(archive_save_times) ||
        error("Final verification summary: incomplete common-time archive")
    all(isfinite, track_times) && all(isfinite, track_positions) ||
        error("Final verification summary: non-finite spike-tip archive")
    for index in eachindex(archive_save_times)
        isapprox(track_times[index], archive_save_times[index];
                 atol=1.0e-10, rtol=0.0) ||
            error("Final verification summary: noncommon spike-tip save time " *
                  "$(track_times[index])")
    end
    spike_tip_2 = archived_spike_tip_at(track_times, track_positions, 2.0)
    spike_tip_3 = archived_spike_tip_at(track_times, track_positions, 3.0)
    println("Minimum achieved error: ",
            Printf.@sprintf("%.6e", minimum_mass_error),
            " (mass); ",
            Printf.@sprintf(
                "final_dM=%.6e, final_relative_dM=%.6e; ",
                final_mass_drift, final_relative_mass_drift),
            "mass-conservation diagnostic only; no exact RTI solution exists")
    println(Printf.@sprintf(
        "Symmetry: eps_sym(3.0)=%.6e (acceptance <= 1.0e-10), eps_IC=%.6e, eps_S=%.6e",
        final_symmetry, initial_projection_error, source_error))
    println(Printf.@sprintf(
        "Positivity at t=3: rho=[%.6e,%.6e], p=[%.6e,%.6e]; raw and common-grid NPZ positivity checks passed",
        rho_min, rho_max, p_min, p_max))
    println(Printf.@sprintf(
        "Wall no-penetration at t=3: max_wall_abs_v=%.6e",
        wall_velocity))
    println(Printf.@sprintf(
        "Energy balance at t=3: delta_E=%.6e, cumulative_gravity_work=%.6e, energy_residual=%.6e, energy_closure=%.6e",
        energy_change, gravity_work, energy_residual, energy_closure))
    println(Printf.@sprintf(
        "Spike-tip archive: samples=%d, spike_tip(2.0)=%.6e, spike_tip(3.0)=%.6e",
        length(track_times), spike_tip_2, spike_tip_3))
    return nothing
end
println("Done.")
print_verification_summary()
