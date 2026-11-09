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
#     * Approximates solutions using tensor-product polynomials of degree $N=6$ on **Legendre-Gauss-Lobatto (LGL)** nodes.
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
polydeg = 6
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
# Wrap only the density-Löhner indicator. The native ControllerThreeLevel
# receives mirror-paired indicator values before computing target levels and
# therefore retains its native thresholds, level bounds, and return type.
struct MirrorPairedAMRIndicator{I}
    base_indicator::I
end
# Construct the mirror partner of every leaf from its physical bounds. Keys use
# level-6 dyadic coordinates, so pairing does not depend on leaf storage order.
function mirror_amr_leaf_partners(semi_or_cache)
    coordinates = if hasproperty(semi_or_cache, :elements)
        # ControllerThreeLevel supplies semi.cache, not the semidiscretization.
        semi_or_cache.elements.node_coordinates
    else
        semi_or_cache.cache.elements.node_coordinates
    end
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
raw_dg_path_repair_enabled = Ref(false)
shock_mirror_pairing_enabled = Ref(false)
amr_projection_order_check_enabled = Ref(false)
raw_dg_path_repair_count = Ref(0)
# Return every mirror pair exactly once and in a storage-independent order.
# This ordering is used for indicator decisions only, never for solution data.
function canonical_mirror_pairs(partners)
    pairs = Tuple{Int, Int}[]
    for element in eachindex(partners)
        partner = partners[element]
        1 <= partner <= length(partners) ||
            error("Mirror-paired decisions: invalid partner index $partner")
        partners[partner] == element ||
            error("Mirror-paired decisions: the partner map is not involutive")
        if element < partner
            push!(pairs, (element, partner))
        end
    end
    sort!(pairs)
    return pairs
end
function pair_mirror_amr_indicators!(element_variables, semi)
    # Pair leaves on the mesh active before this AMR operation.
    partners = mirror_amr_leaf_partners(semi)
    n_elements = length(partners)
    pairs = canonical_mirror_pairs(partners)
    paired_indicator_count = 0
    # Sorted keys and canonical pairs make the decision path independent of
    # dictionary iteration and leaf storage order.
    for key in sort!(collect(keys(element_variables)); by=string)
        indicator_values = element_variables[key]
        if indicator_values isa AbstractVector &&
           length(indicator_values) == n_elements
            native_indicator_values = copy(indicator_values)
            for (element, partner) in pairs
                paired_value = max(native_indicator_values[element],
                                   native_indicator_values[partner])
                indicator_values[element] = paired_value
                indicator_values[partner] = paired_value
            end
            paired_indicator_count += 1
        end
    end
    paired_indicator_count == 1 ||
        error("Mirror-paired AMR: expected exactly one density-Löhner " *
              "indicator vector with one value per leaf element, but found " *
              "$paired_indicator_count")
    return element_variables
end
function Trixi.get_element_variables!(element_variables,
                                      indicator::MirrorPairedAMRIndicator,
                                      u, semi)
    Trixi.get_element_variables!(element_variables,
                                 getfield(indicator, :base_indicator),
                                 u, semi)
    return pair_mirror_amr_indicators!(element_variables, semi)
end
function (indicator::MirrorPairedAMRIndicator)(u, mesh, equations, solver, cache;
                                               t, iter)
    base_indicator = getfield(indicator, :base_indicator)
    indicator_values = base_indicator(u, mesh, equations, solver, cache;
                                      t=t, iter=iter)
    element_variables = Dict{Symbol, Any}(:indicator => indicator_values)
    pair_mirror_amr_indicators!(element_variables, cache)
    return element_variables[:indicator]
end
# Pass-through until repair_raw_dg_symmetry_paths! activates it. The base
# indicator still computes every blending coefficient with unchanged strength
struct MirrorPairedShockIndicator{I}
    base_indicator::I
end

# SaveSolutionCallback queries element variables through the volume integral.
# Forward this serialization hook to the native indicator so its established
# Trixi implementation remains responsible for the archived indicator data.
function Trixi.get_element_variables!(element_variables,
                                      indicator::MirrorPairedShockIndicator,
                                      volume_integral::VolumeIntegralShockCapturingHG)
    return Trixi.get_element_variables!(element_variables,
                                        getfield(indicator, :base_indicator),
                                        volume_integral)
end
function (indicator::MirrorPairedShockIndicator)(u, mesh, equations, solver, cache)
    base_indicator = getfield(indicator, :base_indicator)
    alpha = base_indicator(u, mesh, equations, solver, cache)
    shock_mirror_pairing_enabled[] || return alpha
    alpha isa AbstractVector ||
        error("Shock symmetry repair: expected one blending coefficient per leaf")
    partners = mirror_amr_leaf_partners(cache)
    length(alpha) == length(partners) ||
        error("Shock symmetry repair: blending and leaf counts differ")
    native_alpha = copy(alpha)
    for (element, partner) in canonical_mirror_pairs(partners)
        paired_alpha = max(native_alpha[element], native_alpha[partner])
        alpha[element] = paired_alpha
        alpha[partner] = paired_alpha
    end
    return alpha
end
# Install the inactive wrapper before the ODE is assembled. While inactive this
# is exactly the previous DGSEM operator; all fluxes and indicator parameters
# are retained.
paired_shock_indicator = MirrorPairedShockIndicator(indicator_sc)
volume_integral = VolumeIntegralShockCapturingHG(paired_shock_indicator;
                                                 volume_flux_dg=flux_ranocha,
                                                 volume_flux_fv=flux_lax_friedrichs)
dg = DGSEM(polydeg=polydeg, surface_flux=flux_hllc,
           volume_integral=volume_integral)
semi = SemidiscretizationHyperbolic(mesh, equations,
                                    initial_condition_rayleigh_taylor, dg;
                                    boundary_conditions=boundary_conditions,
                                    source_terms=source_terms_gravity)
amr_indicator = IndicatorLöhner(semi, variable=density_indicator)
paired_amr_indicator = MirrorPairedAMRIndicator(amr_indicator)
amr_controller = ControllerThreeLevel(semi, paired_amr_indicator,
                                      base_level=4,
                                      med_level=5, med_threshold=0.1,
                                      max_level=6, max_threshold=0.3)
base_amr_callback = AMRCallback(semi, amr_controller,
                                interval=20,
                                adapt_initial_condition=true)
# Called by the dual-grid diagnostic only when the raw DG representation itself
# carries the residual. Common-grid interpolation defects must use the separate
# interpolation/mirror-index repair branch instead.
function repair_raw_dg_symmetry_paths!(integrator, raw_symmetry, grid_symmetry;
                                       phase="symmetry diagnostic")
    isfinite(raw_symmetry) && isfinite(grid_symmetry) ||
        error("Raw-DG symmetry repair received a non-finite residual")
    raw_symmetry > 1.0e-10 || return nothing
    first_activation = !raw_dg_path_repair_enabled[]
    raw_dg_path_repair_enabled[] = true
    shock_mirror_pairing_enabled[] = true
    amr_projection_order_check_enabled[] = true
    raw_dg_path_repair_count[] += 1
    # Inspect the active solver rather than merely modifying unused globals.
    local_semi = integrator.p
    local_volume_integral = local_semi.solver.volume_integral
    local_indicator = if hasproperty(local_volume_integral, :indicator)
        getfield(local_volume_integral, :indicator)
    elseif hasproperty(local_volume_integral, :indicator_sc)
        getfield(local_volume_integral, :indicator_sc)
    else
        error("Raw-DG symmetry repair: cannot inspect the shock indicator")
    end
    local_indicator isa MirrorPairedShockIndicator ||
        error("Raw-DG symmetry repair: the active shock indicator is not wrapped")
    getfield(local_indicator, :base_indicator) === indicator_sc ||
        error("Raw-DG symmetry repair: the base shock indicator changed")
    paired_amr_indicator isa MirrorPairedAMRIndicator ||
        error("Raw-DG symmetry repair: the AMR indicator is not mirror-paired")
    pair_count = length(canonical_mirror_pairs(
        mirror_amr_leaf_partners(local_semi)))
    verification_latest[:raw_path_repair] = 1.0
    verification_latest[:shock_mirror_pairing] = 1.0
    verification_latest[:amr_projection_order_check] = 1.0
    verification_latest[:raw_path_repair_activations] =
        Float64(raw_dg_path_repair_count[])
    if first_activation
        println(Printf.@sprintf(
            "Raw-DG symmetry repair activated at %s: eps_raw=%.6e, eps_grid=%.6e, mirror_pairs=%d; shock/blending decisions are now mirror-paired and AMR projection/prolongation inputs use canonical ordering",
            phase, raw_symmetry, grid_symmetry, pair_count))
    end
    return nothing
end
# Native Trixi projection/prolongation remains element-local and untouched.
# These checks ensure that it receives canonical paired decisions, preserves the
# required topology, and does not introduce a non-finite projected state.
function inspect_amr_projection_prolongation!(integrator, phase)
    amr_projection_order_check_enabled[] || return nothing
    all(isfinite, integrator.u) ||
        error("AMR projection/prolongation $phase encountered a non-finite state")
    pair_count = length(canonical_mirror_pairs(
        mirror_amr_leaf_partners(integrator.p)))
    verification_latest[:amr_projection_pair_count] = Float64(pair_count)
    println("AMR projection/prolongation ", phase,
            ": canonical mirror-pair inputs=", pair_count,
            "; native element-local L2 operation ordering retained; " *
            "solution state not symmetrized")
    return nothing
end
function mirror_paired_amr_affect!(integrator)
    report_amr_symmetry!(integrator, "before AMR")
    inspect_amr_projection_prolongation!(integrator, "before AMR")
    base_amr_callback.affect!(integrator)
    mirror_amr_leaf_partners(integrator.p)
    inspect_amr_projection_prolongation!(integrator, "after AMR")
    report_amr_symmetry!(integrator, "after AMR")
    return nothing
end
function mirror_paired_amr_initialize(callback, u, t, integrator)
    report_amr_symmetry!(integrator, "before initial AMR")
    inspect_amr_projection_prolongation!(integrator, "before initial AMR")
    if base_amr_callback.initialize !== nothing
        base_amr_callback.initialize(base_amr_callback, u, t, integrator)
    end
    mirror_amr_leaf_partners(integrator.p)
    inspect_amr_projection_prolongation!(integrator, "after initial AMR")
    report_amr_symmetry!(integrator, "after initial AMR")
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
npz_out = joinpath(@__DIR__, "data_npz_polydeg$(polydeg)")
mkpath(npz_out)
dt_save = 0.1
cfl = 0.25
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
mirror_symmetry_target = 1.0e-10
comparison_grid_repair_enabled = Ref(false)
comparison_grid_repair_count = Ref(0)
# Reset all per-run mutable state without changing the base mesh or AMR policy.
# The escalation driver supplies polydeg=7 and CFL=0.20 after constructing the
# corresponding degree-7 semidiscretization.
function reset_run_state!(run_polydeg=polydeg, run_cfl=cfl)
    global polydeg = Int(run_polydeg)
    global cfl = Float64(run_cfl)
    global npz_out = joinpath(@__DIR__, "data_npz_polydeg$(polydeg)")
    mkpath(npz_out)
    initial_total_mass[] = NaN
    initial_total_energy[] = NaN
    previous_energy_time[] = NaN
    previous_gravity_work_rate[] = NaN
    cumulative_gravity_work[] = 0.0
    last_npz_save_time[] = NaN
    empty!(verification_minima)
    empty!(verification_latest)
    empty!(spike_tip_times)
    empty!(spike_tip_positions)
    comparison_grid_repair_enabled[] = false
    comparison_grid_repair_count[] = 0
    if isdefined(@__MODULE__, :raw_dg_path_repair_enabled)
        raw_dg_path_repair_enabled[] = false
        shock_mirror_pairing_enabled[] = false
        amr_projection_order_check_enabled[] = false
        raw_dg_path_repair_count[] = 0
    end
    return nothing
end
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
function raw_dg_mirror_symmetry_error(integrator)
    semi_local = integrator.p
    state = Trixi.wrap_array(integrator.u, semi_local)
    coordinates = semi_local.cache.elements.node_coordinates
    ndims(state) == 4 && size(state, 1) == 4 ||
        error("Raw symmetry diagnostic: expected a four-variable 2D DG state")
    ndims(coordinates) == 4 && size(coordinates, 1) == 2 ||
        error("Raw symmetry diagnostic: invalid node-coordinate storage")
    size(state, 2) == size(coordinates, 2) &&
        size(state, 3) == size(coordinates, 3) &&
        size(state, 4) == size(coordinates, 4) ||
        error("Raw symmetry diagnostic: inconsistent state and coordinate layouts")
    partners = mirror_amr_leaf_partners(semi_local)
    numerator = 0.0
    denominator = 0.0
    for element in axes(state, 4)
        partner = partners[element]
        for j in axes(state, 3), i in axes(state, 2)
            x = coordinates[1, i, j, element]
            y = coordinates[2, i, j, element]
            target_x = 0.25 - x
            mirror_i = 0
            mirror_j = 0
            best_distance = Inf
            # Match local nodes geometrically without assuming p4est's node
            # orientation. This only reads the unmodified conservative state.
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
                error("Raw symmetry diagnostic: no matching mirror DG node " *
                      "for element $element, local node ($i,$j)")
            for variable in 1:4
                value = state[variable, i, j, element]
                mirror_value =
                    state[variable, mirror_i, mirror_j, partner]
                # rho, rho_v, and E are even; rho_u is odd under reflection.
                residual = variable == 2 ? value + mirror_value :
                           value - mirror_value
                numerator += residual^2
                denominator += value^2
            end
        end
    end
    denominator > 0.0 || return NaN
    return sqrt(numerator / denominator)
end
function interpolate_conserved_grid(x, y, fields;
                                    semi_local=semi,
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
    # raw_dg_state uses this same active semi. Reuse its hard-asserted leaf
    # pairing so that coincident DG nodes on element faces are paired through
    # their elements rather than through an order-dependent coordinate lookup.
    coordinates = semi_local.cache.elements.node_coordinates
    ndims(coordinates) == 4 && size(coordinates, 1) == 2 ||
        error("Comparison-grid interpolation: invalid active DG coordinates")
    node_ni = size(coordinates, 2)
    node_nj = size(coordinates, 3)
    n_elements = size(coordinates, 4)
    nodes_per_element = node_ni * node_nj
    n_source == nodes_per_element * n_elements ||
        error("Comparison-grid interpolation: raw state and active mesh differ")
    element_partners = mirror_amr_leaf_partners(semi_local)
    mirror_source = zeros(Int, n_source)
    for element in 1:n_elements
        partner = element_partners[element]
        for j in 1:node_nj, i in 1:node_ni
            source = i + node_ni * (j - 1) +
                     nodes_per_element * (element - 1)
            target_x = 0.25 - coordinates[1, i, j, element]
            target_y = coordinates[2, i, j, element]
            best_distance = Inf
            mirror_i = 0
            mirror_j = 0
            # Local-node matching does not assume a p4est node orientation.
            for jj in 1:node_nj, ii in 1:node_ni
                distance =
                    (coordinates[1, ii, jj, partner] - target_x)^2 +
                    (coordinates[2, ii, jj, partner] - target_y)^2
                if distance < best_distance
                    best_distance = distance
                    mirror_i = ii
                    mirror_j = jj
                end
            end
            best_distance <= 1.0e-24 ||
                error("Comparison-grid interpolation: no matching mirror " *
                      "DG node for element $element, local node ($i,$j)")
            mirror_source[source] =
                mirror_i + node_ni * (mirror_j - 1) +
                nodes_per_element * (partner - 1)
        end
    end
    all(index -> index > 0, mirror_source) ||
        error("Comparison-grid interpolation: incomplete mirror-node map")
    for source in 1:n_source
        mirror_source[mirror_source[source]] == source ||
            error("Comparison-grid interpolation: mirror-node map is not " *
                  "one-to-one")
    end
    # Spatial buckets make exact Euclidean nearest-neighbor queries affordable.
    # For the default grid these are 64 x 256 square physical buckets.
    bucket_nx = min(nx, 64)
    bucket_ny = min(ny, 256)
    bucket_dx = 0.25 / bucket_nx
    bucket_dy = 1.0 / bucket_ny
    buckets = [Int[] for _ in 1:(bucket_nx * bucket_ny)]
    for k in 1:n_source
        bi = clamp(floor(Int, x_source[k] / bucket_dx) + 1, 1, bucket_nx)
        bj = clamp(floor(Int, y_source[k] / bucket_dy) + 1, 1, bucket_ny)
        push!(buckets[bi + bucket_nx * (bj - 1)], k)
    end
    min_bucket_width = min(bucket_dx, bucket_dy)
    max_radius = max(bucket_nx, bucket_ny)
    function nearest_source(x_target, y_target)
        bucket_i = clamp(floor(Int, x_target / bucket_dx) + 1,
                         1, bucket_nx)
        bucket_j = clamp(floor(Int, y_target / bucket_dy) + 1,
                         1, bucket_ny)
        best_source = 0
        best_distance = Inf
        radius = 0
        while radius <= max_radius
            i_min = max(1, bucket_i - radius)
            i_max = min(bucket_nx, bucket_i + radius)
            j_min = max(1, bucket_j - radius)
            j_max = min(bucket_ny, bucket_j + radius)
            for bj in j_min:j_max
                for bi in i_min:i_max
                    # Interior buckets were searched on earlier rings.
                    if radius > 0 &&
                       max(abs(bi - bucket_i), abs(bj - bucket_j)) != radius
                        continue
                    end
                    for k in buckets[bi + bucket_nx * (bj - 1)]
                        distance = (x_source[k] - x_target)^2 +
                                   (y_source[k] - y_target)^2
                        # Deterministic tie-breaking removes storage-order
                        # dependence from coincident DG visualization nodes.
                        if distance < best_distance ||
                           (distance == best_distance &&
                            (best_source == 0 || k < best_source))
                            best_distance = distance
                            best_source = k
                        end
                    end
                end
            end
            # Every unsearched bucket is at least radius*min_bucket_width
            # away. Thus no later ring can contain a closer source node.
            if best_source != 0 &&
               best_distance < (radius * min_bucket_width)^2
                return best_source
            end
            radius += 1
        end
        best_source != 0 ||
            error("Comparison-grid interpolation: nearest-neighbor search failed")
        return best_source
    end
    grids = ntuple(_ -> Matrix{Float64}(undef, ny, nx), 4)
    # Both assignments read raw DG data independently. No value is averaged,
    # copied across the mirror plane, sign-corrected, or symmetrized.
    for j in 1:ny
        y_target = (j - 0.5) * dy
        for i in 1:div(nx, 2)
            x_target = (i - 0.5) * dx
            mirror_i = nx + 1 - i
            mirror_x_target = (mirror_i - 0.5) * dx
            source = nearest_source(x_target, y_target)
            if comparison_grid_repair_enabled[]
                # Repair the stencil/mirror-index selection by querying both
                # target locations. Choose between the two admissible paired
                # stencils using their combined geometric interpolation error.
                right_source = nearest_source(mirror_x_target, y_target)
                candidate = mirror_source[right_source]
                current_partner = mirror_source[source]
                current_distance =
                    (x_source[source] - x_target)^2 +
                    (y_source[source] - y_target)^2 +
                    (x_source[current_partner] - mirror_x_target)^2 +
                    (y_source[current_partner] - y_target)^2
                candidate_distance =
                    (x_source[candidate] - x_target)^2 +
                    (y_source[candidate] - y_target)^2 +
                    (x_source[right_source] - mirror_x_target)^2 +
                    (y_source[right_source] - y_target)^2
                if candidate_distance < current_distance ||
                   (candidate_distance == current_distance &&
                    candidate < source)
                    source = candidate
                end
            end
            mirror_partner = mirror_source[source]
            for variable in 1:4
                grids[variable][j, i] = field_sources[variable][source]
                grids[variable][j, mirror_i] =
                    field_sources[variable][mirror_partner]
            end
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
function repair_comparison_grid_symmetry!(integrator, raw_symmetry,
                                          grid_symmetry, phase)
    first_activation = !comparison_grid_repair_enabled[]
    comparison_grid_repair_enabled[] = true
    comparison_grid_repair_count[] += 1
    verification_latest[:comparison_grid_repair] = 1.0
    verification_latest[:comparison_grid_repair_activations] =
        Float64(comparison_grid_repair_count[])
    if first_activation
        println(Printf.@sprintf(
            "Comparison-grid repair activated at %s: eps_raw=%.6e, eps_grid=%.6e; bidirectional mirror-stencil selection enabled without modifying the PDE state",
            phase, raw_symmetry, grid_symmetry))
    end
    return nothing
end
# Compute both representations and execute exactly the Advisor's conditional
# repair logic. The returned grid is the one subsequently archived/plotted.
function dual_grid_symmetry_diagnostics!(integrator, phase)
    raw_x, raw_y, raw_conserved = raw_dg_state(integrator)
    raw_symmetry = raw_dg_mirror_symmetry_error(integrator)
    grid_fields = interpolate_conserved_grid(
        raw_x, raw_y, raw_conserved; semi_local=integrator.p)
    grid_symmetry = mirror_symmetry_error(grid_fields)
    isfinite(raw_symmetry) && isfinite(grid_symmetry) ||
        error("Dual symmetry diagnostic: non-finite residual at t=$(integrator.t)")
    println(Printf.@sprintf(
        "Dual symmetry %s at t=%.8f: eps_raw=%.6e, eps_grid=%.6e",
        phase, integrator.t, raw_symmetry, grid_symmetry))
    raw_more_symmetric =
        grid_symmetry > mirror_symmetry_target &&
        raw_symmetry < 0.1 * grid_symmetry
    if raw_more_symmetric
        # This branch must not alter the PDE solver or the raw solution.
        repair_comparison_grid_symmetry!(
            integrator, raw_symmetry, grid_symmetry, phase)
        grid_fields = interpolate_conserved_grid(
            raw_x, raw_y, raw_conserved; semi_local=integrator.p)
        grid_symmetry = mirror_symmetry_error(grid_fields)
        isfinite(grid_symmetry) ||
            error("Comparison-grid repair produced a non-finite residual")
        println(Printf.@sprintf(
            "Comparison-grid repair result at %s: eps_raw=%.6e, eps_grid=%.6e",
            phase, raw_symmetry, grid_symmetry))
    elseif raw_symmetry > mirror_symmetry_target
        # Comparable residuals indicate a raw-DG/path issue rather than an
        # interpolation-only issue. Cell 7 performs the conditional inspection
        # and activates mirror-paired shock/AMR decision paths.
        repair_raw_dg_symmetry_paths!(
            integrator, raw_symmetry, grid_symmetry; phase=phase)
    end
    verification_latest[:symmetry] = grid_symmetry
    verification_latest[:raw_symmetry] = raw_symmetry
    verification_latest[:grid_symmetry] = grid_symmetry
    verification_latest[:time] = integrator.t
    return raw_x, raw_y, raw_conserved, grid_fields,
           raw_symmetry, grid_symmetry
end
function report_amr_symmetry!(integrator, phase)
    diagnostics = dual_grid_symmetry_diagnostics!(integrator, phase)
    raw_symmetry = diagnostics[5]
    grid_symmetry = diagnostics[6]
    verification_latest[:amr_raw_symmetry] = raw_symmetry
    verification_latest[:amr_grid_symmetry] = grid_symmetry
    return nothing
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
function archive_metadata(save_index, actual_save_times)
    return Dict{String, Any}(
        "polydeg" => polydeg,
        "mesh" => metadata_bytes(
            "P4estMesh [0,0.25]x[0,1]; initial_refinement_level=3; " *
            "periodic_x=true; slip_walls_y=true"),
        "amr_policy" => metadata_bytes(
            "mirror-paired IndicatorLoehner(rho); " *
            "ControllerThreeLevel(base=4,med=5,max=6," *
            "med_threshold=0.1,max_threshold=0.3); interval=20"),
        "cfl" => cfl,
        "dt_save" => dt_save,
        "save_times" => copy(actual_save_times),
        "scheduled_save_times" => copy(archive_save_times),
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
    isfile(filename) || error("NPZ sanity check: missing file $filename")
    data = npzread(filename)
    required_keys = (
        "t", "x", "y", "rho", "u", "v", "p",
        "raw_x", "raw_y", "raw_rho", "raw_rho_u", "raw_rho_v",
        "raw_energy", "comparison_x", "comparison_y",
        "grid_rho", "grid_rho_u", "grid_rho_v", "grid_energy",
        "spike_tip", "symmetry_raw", "symmetry_grid",
        "polydeg", "mesh", "amr_policy", "cfl",
        "save_times", "scheduled_save_times", "dt_save",
        "comparison_nx", "comparison_ny", "save_index")
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
    scheduled_times = Float64.(vec(data["scheduled_save_times"]))
    schedule_valid =
        length(scheduled_times) == length(archive_save_times) &&
        all(index -> isapprox(scheduled_times[index],
                              archive_save_times[index];
                              atol=1.0e-12, rtol=0.0),
            eachindex(archive_save_times))
    save_index = Int(npz_scalar(data["save_index"]))
    save_index_valid = 0 <= save_index < length(archive_save_times)
    save_times_valid = save_index_valid &&
        length(saved_times) == save_index + 1 &&
        all(index -> isapprox(saved_times[index], archive_save_times[index];
                              atol=1.0e-10, rtol=0.0),
            eachindex(saved_times))
    metadata_valid =
        Int(npz_scalar(data["polydeg"])) == polydeg &&
        Float64(npz_scalar(data["cfl"])) == cfl &&
        Float64(npz_scalar(data["dt_save"])) == dt_save &&
        Int(npz_scalar(data["comparison_nx"])) == comparison_nx &&
        Int(npz_scalar(data["comparison_ny"])) == comparison_ny &&
        save_index_valid &&
        schedule_valid &&
        save_times_valid &&
        length(data["mesh"]) > 0 && length(data["amr_policy"]) > 0
    spike_tip = Float64(npz_scalar(data["spike_tip"]))
    spike_valid = isfinite(spike_tip) && 0.0 <= spike_tip <= 1.0
    symmetry_valid =
        all(key -> isfinite(Float64(npz_scalar(data[key]))),
            ("symmetry_raw", "symmetry_grid")) &&
        all(key -> Float64(npz_scalar(data[key])) >= 0.0,
            ("symmetry_raw", "symmetry_grid"))
    file_time = Float64(npz_scalar(data["t"]))
    time_valid = save_index_valid && isfinite(file_time) &&
        isapprox(file_time, archive_save_times[save_index + 1];
                 atol=1.0e-10, rtol=0.0)
    if !(primitive_coverage && raw_coverage && grid_coverage &&
         finite_state && finite_coordinates && positive_state &&
         symmetry_valid && metadata_valid && spike_valid && time_valid)
        error("NPZ sanity check failed for $filename: " *
              "primitive_coverage=$primitive_coverage, " *
              "raw_coverage=$raw_coverage, grid_coverage=$grid_coverage, " *
              "finite_state=$finite_state, " *
              "finite_coordinates=$finite_coordinates, " *
              "positive_state=$positive_state, symmetry=$symmetry_valid, " *
              "metadata=$metadata_valid, spike_tip=$spike_valid, " *
              "time=$time_valid")
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
                                     grid_fields, spike_tip, raw_symmetry)
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
    grid_symmetry = mirror_symmetry_error(grid_fields)
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
    record_verification_error!(:symmetry, grid_symmetry)
    record_verification_error!(:raw_symmetry, raw_symmetry)
    record_verification_error!(:grid_symmetry, grid_symmetry)
    record_verification_error!(:wall, relative_wall_violation)
    record_verification_error!(:energy_balance, energy_closure)
    verification_latest[:symmetry] = grid_symmetry
    verification_latest[:raw_symmetry] = raw_symmetry
    verification_latest[:grid_symmetry] = grid_symmetry
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
        "Diagnostics t=%.5f: dM=%.6e, rel_dM=%.6e, rho=[%.6e,%.6e], p=[%.6e,%.6e], sym_raw=%.6e, sym_grid=%.6e, wall_max_abs_v=%.6e, delta_E=%.6e, cumulative_gravity_work=%.6e, energy_residual=%.6e, energy_closure=%.6e, spike_tip=%.6e",
        t, mass_drift, relative_mass_drift,
        rho_min, rho_max, p_min, p_max, raw_symmetry, grid_symmetry,
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
    scheduled_time = archive_save_times[save_index + 1]
    isapprox(t, scheduled_time; atol=1.0e-10, rtol=0.0) ||
        error("NPZ archive: integrator time $t is not the common save time " *
              "$scheduled_time; the solver must stop at archive_save_times")
    save_index == length(spike_tip_times) ||
        error("NPZ archive: noncontiguous common-time save; expected save " *
              "index $(length(spike_tip_times)) but received $save_index")
    actual_save_times = copy(spike_tip_times)
    push!(actual_save_times, t)
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
    raw_x, raw_y, raw_conserved, grid_fields, raw_symmetry, grid_symmetry =
        dual_grid_symmetry_diagnostics!(integrator, "save")
    isfinite(raw_symmetry) && isfinite(grid_symmetry) ||
        error("Symmetry archive: non-finite residual at t=$t")
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
        "spike_tip" => spike_tip,
        "symmetry_raw" => raw_symmetry,
        "symmetry_grid" => grid_symmetry)
    merge!(payload, archive_metadata(save_index, actual_save_times))
    filename = joinpath(npz_out,
                        Printf.@sprintf("time_%08.5f.npz", scheduled_time))
    npzwrite(filename, payload)
    verify_npz_file(filename)
    push!(spike_tip_times, t)
    push!(spike_tip_positions, spike_tip)
    track_filename = joinpath(npz_out, "spike_tip_track.npz")
    npzwrite(track_filename, Dict{String, Any}(
        "times" => copy(spike_tip_times),
        "spike_tip" => copy(spike_tip_positions),
        "save_times" => copy(spike_tip_times),
        "scheduled_save_times" => copy(archive_save_times)))
    track_data = npzread(track_filename)
    length(track_data["times"]) == length(spike_tip_times) &&
        length(track_data["spike_tip"]) == length(spike_tip_positions) &&
        all(isfinite, track_data["times"]) &&
        all(isfinite, track_data["spike_tip"]) ||
        error("NPZ sanity check failed for $track_filename")
    println("Saved and NPZ-verified: $(basename(filename)); " *
            Printf.@sprintf("spike_tip=%.6e", spike_tip))
    print_physical_diagnostics!(integrator, x, y, rho, v, p,
                                grid_fields, spike_tip, raw_symmetry)
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
#     Here, we set `cfl=0.25` for the polydeg-6 confirmation run while retaining CFL-based timestep control.
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
    # Initial L2 projection and AMR accumulate several floating-point
    # operations, so scale the roundoff checks with the state magnitude rather
    # than rejecting roundoff-scale residuals using a fixed absolute cutoff.
    state_scale = max(1.0, maximum(abs, integrator.u))
    roundoff_scale = eps(Float64) * state_scale
    log_threshold = max(1.0e-15, 2 * roundoff_scale)
    abort_threshold = 64 * roundoff_scale
    verification_latest[:initial_projection_symmetry] = epsilon_ic
    verification_latest[:source_symmetry] = epsilon_source
    verification_latest[:initial_symmetry_log_threshold] = log_threshold
    verification_latest[:initial_symmetry_abort_threshold] = abort_threshold
    println(Printf.@sprintf(
        "Initial-projection mirror symmetry: eps_IC=%.6e; rho=%.6e, rho_u=%.6e, rho_v=%.6e, E=%.6e",
        epsilon_ic, projection_errors[1], projection_errors[2],
        projection_errors[3], projection_errors[4]))
    println(Printf.@sprintf(
        "Gravity-source mirror symmetry: eps_S=%.6e; S_rho=%.6e, S_rho_u=%.6e, S_rho_v=%.6e, S_E=%.6e",
        epsilon_source, source_errors[1], source_errors[2],
        source_errors[3], source_errors[4]))
    isfinite(epsilon_ic) ||
        error("Initial-projection mirror symmetry verification failed: " *
              "eps_IC=$epsilon_ic is non-finite")
    isfinite(epsilon_source) ||
        error("Gravity-source mirror symmetry verification failed: " *
              "eps_S=$epsilon_source is non-finite")
    if epsilon_ic > log_threshold
        @warn "Initial-projection symmetry exceeds the roundoff logging threshold" epsilon_ic log_threshold
    end
    if epsilon_source > log_threshold
        @warn "Source-term symmetry exceeds the roundoff logging threshold" epsilon_source log_threshold
    end
    epsilon_ic <= abort_threshold ||
        error("Initial-projection mirror symmetry verification failed: " *
              "eps_IC=$epsilon_ic exceeds the roundoff abort threshold " *
              "$abort_threshold")
    epsilon_source <= abort_threshold ||
        error("Gravity-source mirror symmetry verification failed: " *
              "eps_S=$epsilon_source exceeds the roundoff abort threshold " *
              "$abort_threshold")
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
        # Use the same diagnostics and conditional repair branches as every
        # save and AMR report. This reads, but never symmetrizes, the state.
        diagnostics = dual_grid_symmetry_diagnostics!(integrator, "final")
        final_raw_symmetry = diagnostics[5]
        final_grid_symmetry = diagnostics[6]
        verification_latest[:time] = integrator.t
        # Retain :symmetry as the common-grid value for compatibility.
        verification_latest[:symmetry] = final_grid_symmetry
        verification_latest[:raw_symmetry] = final_raw_symmetry
        verification_latest[:grid_symmetry] = final_grid_symmetry
        verification_latest[:final_symmetry] = final_grid_symmetry
        verification_latest[:final_raw_symmetry] = final_raw_symmetry
        verification_latest[:final_grid_symmetry] = final_grid_symmetry
        symmetry_target = mirror_symmetry_target
        verification_latest[:symmetry_target] = symmetry_target
        println(Printf.@sprintf(
            "Final mirror symmetry at t=%.5f: eps_sym_raw=%.6e, eps_sym_grid=%.6e (diagnostic target <= %.1e)",
            integrator.t, final_raw_symmetry, final_grid_symmetry,
            symmetry_target))
        # Report both diagnostics without aborting a completed simulation.
        for (representation, residual) in
            (("raw DG", final_raw_symmetry),
             ("common grid", final_grid_symmetry))
            if !isfinite(residual)
                @warn "Final mirror-symmetry diagnostic is non-finite; continuing to finalization" representation=representation eps_sym=residual
            elseif residual > symmetry_target
                @warn "Final mirror-symmetry diagnostic exceeds the target; continuing to archiving and plotting" representation=representation eps_sym=residual symmetry_target=symmetry_target
            end
        end
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
stepsize_callback = StepsizeCallback(cfl=cfl)
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
function execute_rti_run!(ode_problem, run_callbacks)
    return solve(ode_problem, RDPK3SpFSAL49();
                 dt=1.0,
                 adaptive=false,
                 callback=run_callbacks,
                 saveat=0.01,
                 tstops=archive_save_times[2:end],
                 maxiters=1e7)
end

# Degree-6 confirmation run. Cell 12 conditionally invokes this same execution
# path with the degree-7 ODE and callbacks if degree agreement is not demonstrated.
sol = execute_rti_run!(ode, callbacks)
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
function generate_summary_plots()
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
        linewidth = 0,
        margin = 2Plots.mm
    )
    # Rows are density and vertical velocity; columns are the four save times.
    for (field, field_label) in (("rho", "ρ"), ("v", "v"))
        for t_plot in plot_times
            filename = joinpath(npz_out,
                                Printf.@sprintf("time_%08.5f.npz", t_plot))
            println(filename)
            verify_npz_file(filename)
            data = npzread(filename)
            # Automatic color scaling; no fixed clims or colormap.
            field_grid = rasterize_data(data["x"], data["y"], data[field],
                                        vis_nx, vis_ny)
            t_val = npz_scalar(data["t"])
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
    return nothing
end
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
function compare_degree5_archive(final_data)
    reference_directory = joinpath(@__DIR__, "data_npz_polydeg5")
    target_times = (2.0, 3.0)
    reference_filenames = [
        joinpath(reference_directory,
                 Printf.@sprintf("time_%08.5f.npz", target_time))
        for target_time in target_times
    ]
    missing_references = [
        filename for filename in reference_filenames if !isfile(filename)
    ]
    verification_latest[:degree5_cross_validation_available] = 0.0
    if !isempty(missing_references)
        println("Degree-5 cross-validation unavailable: missing reference archive(s) ",
                join(missing_references, ", "))
        println("Degree-agreement certification remains incomplete; missing " *
                "reference data will not trigger degree-7 escalation.")
        return nothing
    end

    verification_latest[:degree5_cross_validation_available] = 1.0
    comparisons = NamedTuple[]
    for (target_time, reference_filename) in
        zip(target_times, reference_filenames)
        if target_time == 3.0
            current_data = final_data
        else
            current_filename = joinpath(
                npz_out, Printf.@sprintf("time_%08.5f.npz", target_time))
            verify_npz_file(current_filename)
            current_data = npzread(current_filename)
        end
        reference_data = npzread(reference_filename)
        required_keys = ("t", "polydeg", "grid_rho", "spike_tip",
                         "comparison_x", "comparison_y")
        all(key -> haskey(reference_data, key), required_keys) ||
            error("Degree-5 cross-validation: incomplete archive " *
                  "$reference_filename")
        Int(npz_scalar(reference_data["polydeg"])) == 5 ||
            error("Degree-5 cross-validation: reference is not degree 5")
        isapprox(Float64(npz_scalar(reference_data["t"])), target_time;
                 atol=1.0e-10, rtol=0.0) ||
            error("Degree-5 cross-validation: noncommon reference time")
        current_rho = Float64.(current_data["grid_rho"])
        reference_rho = Float64.(reference_data["grid_rho"])
        size(current_rho) == (comparison_ny, comparison_nx) &&
            size(reference_rho) == size(current_rho) ||
            error("Degree-5 cross-validation: common-grid density layouts differ")
        length(reference_data["comparison_x"]) == comparison_nx &&
            length(reference_data["comparison_y"]) == comparison_ny ||
            error("Degree-5 cross-validation: inconsistent comparison axes")
        all(isfinite, current_rho) && all(isfinite, reference_rho) ||
            error("Degree-5 cross-validation: non-finite density data")
        denominator = sum(abs2, reference_rho)
        denominator > 0.0 ||
            error("Degree-5 cross-validation: zero reference density norm")
        relative_l2 = sqrt(sum(abs2, current_rho .- reference_rho) /
                           denominator)
        current_tip = Float64(npz_scalar(current_data["spike_tip"]))
        reference_tip = Float64(npz_scalar(reference_data["spike_tip"]))
        isfinite(relative_l2) && isfinite(current_tip) &&
            isfinite(reference_tip) ||
            error("Degree-5 cross-validation: non-finite comparison metric")
        spike_delta = current_tip - reference_tip
        push!(comparisons,
              (time=target_time, relative_l2=relative_l2,
               spike_delta=spike_delta))
        println(Printf.@sprintf(
            "Degree-%d/degree-5 comparison at t=%.1f: relative_L2_rho=%.6e, spike_tip_current=%.6e, spike_tip_degree5=%.6e, delta_spike_tip=%.6e, abs_delta_spike_tip=%.6e",
            polydeg, target_time, relative_l2, current_tip, reference_tip,
            spike_delta, abs(spike_delta)))
    end
    # Screening tolerances for the otherwise unquantified escalation rule.
    # They affect certification only, never the PDE solution. The spike
    # tolerance corresponds to two common-grid rows.
    relative_l2_tolerance = 1.0e-2
    spike_tip_tolerance = 2.0 / comparison_ny
    degree_agreement = all(
        comparison -> comparison.relative_l2 <= relative_l2_tolerance &&
                      abs(comparison.spike_delta) <= spike_tip_tolerance,
        comparisons)
    verification_latest[:degree_agreement] = degree_agreement ? 1.0 : 0.0
    println(Printf.@sprintf(
        "Degree-agreement decision: %s; both times require relative_L2_rho <= %.1e and abs_delta_spike_tip <= %.6e",
        degree_agreement ? "demonstrated" : "not demonstrated",
        relative_l2_tolerance, spike_tip_tolerance))
    return degree_agreement
end
function execute_degree7_escalation!()
    polydeg == 6 ||
        error("Degree escalation: the confirmation run must be degree 6")
    escalation_polydeg = 7
    escalation_cfl = 0.20
    println(Printf.@sprintf(
        "Executing degree escalation now: polydeg=%d, CFL=%.2f; identical base mesh and AMR policy, with no aggressive mesh refinement",
        escalation_polydeg, escalation_cfl))
    # This updates polydeg, cfl, and npz_out and clears every per-run
    # diagnostic, archive-position, energy-balance, and repair state.
    reset_run_state!(escalation_polydeg, escalation_cfl)
    verification_latest[:escalation_executed] = 1.0
    verification_latest[:escalation_polydeg] = Float64(escalation_polydeg)
    verification_latest[:escalation_cfl] = escalation_cfl
    # Rebuild the complete spatial operator at degree 7. Fluxes, shock strength,
    # indicator variables, and smoothing settings remain unchanged.
    global basis = LobattoLegendreBasis(polydeg)
    global indicator_sc = IndicatorHennemannGassner(
        equations, basis,
        alpha_max=0.2,
        alpha_min=1.0e-3,
        alpha_smooth=true,
        variable=density_indicator)
    global paired_shock_indicator = MirrorPairedShockIndicator(indicator_sc)
    global volume_integral = VolumeIntegralShockCapturingHG(
        paired_shock_indicator;
        volume_flux_dg=flux_ranocha,
        volume_flux_fv=flux_lax_friedrichs)
    global dg = DGSEM(polydeg=polydeg, surface_flux=flux_hllc,
                      volume_integral=volume_integral)
    # Same root connectivity, physical domain, initial refinement, and
    # periodic/wall topology as the degree-5 archive and degree-6 run.
    global mesh = P4estMesh((1, 1), polydeg=polydeg,
                            coordinates_min=coordinates_min,
                            coordinates_max=coordinates_max,
                            initial_refinement_level=3,
                            periodicity=(true, false))
    global semi = SemidiscretizationHyperbolic(
        mesh, equations, initial_condition_rayleigh_taylor, dg;
        boundary_conditions=boundary_conditions,
        source_terms=source_terms_gravity)
    # Recreate the identical mirror-paired density-AMR policy.
    global amr_indicator = IndicatorLöhner(semi, variable=density_indicator)
    global paired_amr_indicator = MirrorPairedAMRIndicator(amr_indicator)
    global amr_controller = ControllerThreeLevel(
        semi, paired_amr_indicator,
        base_level=4,
        med_level=5, med_threshold=0.1,
        max_level=6, max_threshold=0.3)
    global base_amr_callback = AMRCallback(
        semi, amr_controller,
        interval=20,
        adapt_initial_condition=true)
    global amr_callback = DiscreteCallback(
        base_amr_callback.condition,
        mirror_paired_amr_affect!;
        initialize=mirror_paired_amr_initialize,
        save_positions=base_amr_callback.save_positions)
    global ode = semidiscretize(semi, tspan)
    # Construct fresh callbacks so no callback retains degree-6 run state.
    global summary_callback = SummaryCallback()
    global analysis_callback = AnalysisCallback(
        semi, interval=analysis_interval)
    global alive_callback = AliveCallback(analysis_interval=analysis_interval)
    global trixi_out = joinpath(@__DIR__, "trixi_output_polydeg$(polydeg)")
    global save_solution = SaveSolutionCallback(
        dt=dt_save,
        save_initial_solution=true,
        save_final_solution=true,
        output_directory=trixi_out,
        solution_variables=cons2prim)
    global symmetry_verification_callback = DiscreteCallback(
        (u, t, integrator) -> false,
        integrator -> nothing;
        initialize=initialize_symmetry_verification,
        finalize=finalize_symmetry_verification,
        save_positions=(false, false))
    global energy_balance_callback = DiscreteCallback(
        energy_balance_condition, accumulate_gravity_work!;
        initialize=initialize_energy_balance_callback,
        save_positions=(false, false))
    global save_npz_callback = PeriodicCallback(
        save_npz_file, dt_save;
        initial_affect=true,
        save_positions=(false, false),
        finalize=finalize_npz_save)
    global stepsize_callback = StepsizeCallback(cfl=cfl)
    global callbacks = CallbackSet(
        summary_callback,
        analysis_callback,
        alive_callback,
        save_solution,
        amr_callback,
        symmetry_verification_callback,
        energy_balance_callback,
        save_npz_callback,
        stepsize_callback)
    # Execute the mandated run rather than merely recording escalation metadata.
    global sol = execute_rti_run!(ode, callbacks)
    final_filename = joinpath(
        npz_out, Printf.@sprintf("time_%08.5f.npz", 3.0))
    verify_npz_file(final_filename)
    println(Printf.@sprintf(
        "Degree-7 escalation completed and archived: polydeg=%d, CFL=%.2f, output=%s",
        polydeg, cfl, npz_out))
    return sol
end
function print_verification_summary(; allow_escalation=true)
    final_filename = joinpath(npz_out,
                              Printf.@sprintf("time_%08.5f.npz", 3.0))
    verify_npz_file(final_filename)
    final_data = npzread(final_filename)
    final_time = Float64(npz_scalar(final_data["t"]))
    isapprox(final_time, 3.0; atol=1.0e-10, rtol=0.0) ||
        error("Final verification summary: the final archive is not at t=3")
    # Recompute the common-grid residual and retrieve the residual archived
    # directly from the unmodified raw DG representation.
    final_grids = (final_data["grid_rho"], final_data["grid_rho_u"],
                   final_data["grid_rho_v"], final_data["grid_energy"])
    final_grid_symmetry = mirror_symmetry_error(final_grids)
    final_raw_symmetry =
        Float64(npz_scalar(final_data["symmetry_raw"]))
    archived_grid_symmetry =
        Float64(npz_scalar(final_data["symmetry_grid"]))
    isfinite(final_raw_symmetry) && final_raw_symmetry >= 0.0 &&
        isfinite(final_grid_symmetry) && final_grid_symmetry >= 0.0 ||
        error("Final verification summary: invalid dual-grid symmetry residual")
    isapprox(final_grid_symmetry, archived_grid_symmetry;
             atol=1.0e-14, rtol=1.0e-10) ||
        error("Final verification summary: archived and recomputed " *
              "common-grid symmetry residuals differ")
    symmetry_target = 1.0e-10
    raw_status = final_raw_symmetry <= symmetry_target ?
                 "target met" : "target not met"
    grid_status = final_grid_symmetry <= symmetry_target ?
                  "target met" : "target not met"
    for (representation, residual) in
        (("raw DG", final_raw_symmetry),
         ("common grid", final_grid_symmetry))
        if residual > symmetry_target
            @warn "Final mirror-symmetry diagnostic exceeds the target; continuing verification summary" representation=representation eps_sym=residual symmetry_target=symmetry_target
        end
    end
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
        "Symmetry: eps_sym_raw(3.0)=%.6e (%s), eps_sym_grid(3.0)=%.6e (%s), diagnostic target <= %.1e, eps_IC=%.6e, eps_S=%.6e",
        final_raw_symmetry, raw_status,
        final_grid_symmetry, grid_status, symmetry_target,
        initial_projection_error, source_error))
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
    degree_agreement = compare_degree5_archive(final_data)
    symmetry_certified = final_raw_symmetry <= symmetry_target &&
                         final_grid_symmetry <= symmetry_target
    if polydeg == 6 && degree_agreement === false && allow_escalation
        escalation_polydeg = 7
        escalation_cfl = 0.20
        verification_latest[:escalation_required] = 1.0
        verification_latest[:escalation_polydeg] = escalation_polydeg
        verification_latest[:escalation_cfl] = escalation_cfl
        @warn "Degree-independent agreement was not demonstrated; executing the degree-7 confirmation run" polydeg=escalation_polydeg cfl=escalation_cfl
        println(Printf.@sprintf(
            "Escalation required: executing polydeg=%d, CFL=%.2f with the same base mesh and AMR policy; no aggressive mesh refinement",
            escalation_polydeg, escalation_cfl))

        execute_degree7_escalation!()

        # Certify the newly active degree-7 archive and diagnostics. The final
        # plotting call below then uses data_npz_polydeg7 automatically.
        return print_verification_summary(allow_escalation=false)
    elseif polydeg == 6 && degree_agreement === false
        error("Degree-7 escalation was required but was not executed")
    else
        # Missing reference data is not evidence of degree disagreement and
        # therefore does not trigger another expensive PDE solve.
        verification_latest[:escalation_required] = 0.0
        if polydeg == 7
            verification_latest[:escalation_executed] = 1.0
            println("Degree escalation: polydeg-7, CFL=0.20 run executed; " *
                    "its certification metrics are reported above.")
        elseif degree_agreement === nothing
            println("Degree escalation: not triggered because the degree-5 " *
                    "reference archive is unavailable; degree-independent " *
                    "agreement remains uncertified.")
        elseif degree_agreement === true
            println("Degree escalation: not required by the degree-agreement criteria.")
        end
    end
    degree_agreement_status = if degree_agreement === nothing
        "unavailable (missing degree-5 reference archive)"
    else
        string(degree_agreement)
    end
    println("Certification status: symmetry_target_met=$symmetry_certified, " *
            "degree_agreement=$degree_agreement_status; integrity diagnostics are reported above")
    return nothing
end
print_verification_summary()
# Use degree 6 when agreement is demonstrated; otherwise use the completed
# degree-7 escalation archive. Both cases produce the required summary_all.png.
generate_summary_plots()
println("Done.")
