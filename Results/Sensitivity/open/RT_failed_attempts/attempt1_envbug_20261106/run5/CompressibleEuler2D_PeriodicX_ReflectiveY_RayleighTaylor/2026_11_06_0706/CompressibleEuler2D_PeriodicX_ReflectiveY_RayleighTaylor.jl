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
using Pkg

# Instantiate the externally restored, known-good Project.toml/Manifest.toml.
# Do not update versions: the manifest must preserve the compatible
# Julia/Trixi/LoopVectorization/VectorizationBase dependency stack.
Pkg.instantiate()
Pkg.precompile()
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
# Defines the 2D compressible Euler equations for conservation of mass,
# momentum, and total energy, using the ideal-gas equation of state with
# γ = 1.4. Gravity is supplied separately as a source term.

# %%
equations = CompressibleEulerEquations2D(1.4)
# %% [markdown]

# ## Cell 3: Initial Conditions and Source Terms (Lid-Driven Cavity)
# ## Problem Setup: Rayleigh–Taylor Instability
# Initializes a heavy fluid below a light fluid with a smoothed interface,
# piecewise hydrostatic pressure, and a single-mode velocity perturbation.
# A unit gravitational body force acts in the positive y-direction.
# %%
@inline function initial_condition_rayleigh_taylor(x, t,
        equations::CompressibleEulerEquations2D)
    y = x[2]
    # Smooth transition from rho = 2 below the interface to rho = 1 above it.
    rho = 2.0 - 0.5 * (1.0 + tanh(50.0 * (y - 0.5)))
    # Piecewise hydrostatic pressure, continuous at y = 0.5.
    p = y < 0.5 ? 2.0 * y + 1.0 : y + 1.5
    u = 0.0
    sound_speed = sqrt(equations.gamma * p / rho)
    v = -0.025 * sound_speed * cos(8.0 * pi * x[1]) * sin(pi * y)^6
    # prim2cons forms E = p/(gamma - 1) + rho*(u^2 + v^2)/2.
    return prim2cons(SVector(rho, u, v, p), equations)
end
# Unperturbed hydrostatic state used only for the well-balanced verification run.
@inline function initial_condition_hydrostatic(x, t,
        equations::CompressibleEulerEquations2D)
    y = x[2]

    # Use exactly the same density and pressure as the perturbed RTI state.
    rho = 2.0 - 0.5 * (1.0 + tanh(50.0 * (y - 0.5)))
    p = y < 0.5 ? 2.0 * y + 1.0 : y + 1.5

    # Zero velocity gives E = p/(gamma - 1).
    return prim2cons(SVector(rho, 0.0, 0.0, p), equations)
end

@inline function source_terms_gravity(u, x, t,
        equations::CompressibleEulerEquations2D)
    g = 1.0
    rho, rho_v1, rho_v2, rho_e = u

    # Vertical momentum source: rho*g; energy source: rho*g*v = g*rho_v2.
    return SVector(0.0, 0.0, g * rho, g * rho_v2)
end

# Density variable used by shock capturing and AMR.
@inline density_variable(u, equations) = u[1]
# %% [markdown]

# ## Cell 4: Spatial Discretization & Hybrid Shock Capturing
#
# ## Methodology: Nodal Discontinuous Galerkin (DGSEM)
# Constructs the spatial operator $\mathcal{L}(\mathbf{U})$ using the **Discontinuous Galerkin Spectral Element Method (DGSEM)**.
#
# 1.  **Function Space (`LobattoLegendreBasis`)**:
#     * Approximates solutions using tensor-product polynomials of degree $N=4$ on **Legendre-Gauss-Lobatto (LGL)** nodes.
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
#     * **Riemann Solver**: Harten-Lax-van Leer with Contact (HLLC).
#     * **Role**: Computes interface fluxes while explicitly restoring the contact wave.
#     * **Physics**: Improves density-contact resolution while retaining robust shock treatment.
#     * **Physics**: An approximate Riemann solver that estimates the fastest signal velocities (wave speeds). It is robust and positivity-preserving, making it ideal for compressible flows with strong shocks.
#
# 4.  **Indicator Function (`IndicatorHennemannGassner`)**:
#     * Dynamically computes the blending coefficient $\alpha \in [0, 1]$ element-wise.
#      * **Mechanism**: Monitors the spectral decay of energy in the highest modes of the polynomial expansion. If energy in high modes is large (indicating a discontinuity), $\alpha \to 1$ (activates FV sub-cells).
#
# %%
polydeg = 4
basis = LobattoLegendreBasis(polydeg)

# Construct identical DGSEM discretizations for the baseline run and
# conditional blending or polynomial-degree reruns.
function build_rti_dgsem(alpha_max; polynomial_degree=polydeg)
    basis_local = LobattoLegendreBasis(polynomial_degree)
    indicator_sc = IndicatorHennemannGassner(equations, basis_local,
                                             alpha_max=alpha_max,
                                             alpha_min=0.001,
                                             alpha_smooth=true,
                                             variable=density_variable)
    volume_integral = VolumeIntegralShockCapturingHG(
        indicator_sc;
        volume_flux_dg=flux_ranocha,
        volume_flux_fv=flux_lax_friedrichs)

    return DGSEM(polydeg=polynomial_degree, surface_flux=flux_hllc,
                 volume_integral=volume_integral)
end

# Baseline degree-4 configuration; conditional reruns can specify
# polynomial_degree=3 and/or alpha_max=0.5 without changing the fluxes.
dg = build_rti_dgsem(1.0; polynomial_degree=polydeg)
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
# Retain a single root tree and the template initial refinement level.
mesh = P4estMesh((1, 1), polydeg=polydeg,
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max,
                 initial_refinement_level=3,
                 periodicity=(true, false)) # Periodic in x; physical walls in y
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
# Reflective/free-slip walls at y = 0 and y = 1.
# The x-direction is periodic, so only the physical y-boundaries are specified.
boundary_conditions = (y_neg = boundary_condition_slip_wall,
                       y_pos = boundary_condition_slip_wall)
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
function build_rti_amr_callback(semi_local; max_level=6)
    indicator = IndicatorLöhner(semi_local, variable=density_variable)
    controller = ControllerThreeLevel(semi_local, indicator,
                                      base_level=4,
                                      med_level=5, med_threshold=0.1,
                                      max_level=max_level,
                                      max_threshold=0.3)

    return AMRCallback(semi_local, controller,
                       interval=20,
                       adapt_initial_condition=true)
end

# Baseline AMR; the conditional rerun uses build_rti_amr_callback(...; max_level=7).
amr_callback = build_rti_amr_callback(semi)
# %% [markdown]

# ## Cell 8: Data I/O & Surrogate Interoperability
#
# ## Strategy: Cross-Language Serialization (.npz)
# Implements a custom callback to export solution snapshots in **NumPy** format. This bridges the high-performance Julia solver with Python-based analysis tools or ML surrogate models (e.g., the ATHENA Advisor).
#
# 1.  **State Transformation ($\mathbf{u} \to \mathbf{w}$)**:
#     * The solver evolves **Conservative Variables** $\mathbf{u} = [\rho, \rho v_1, \rho v_2, \rho E]^T$.
#     * For analysis, we map these to **Primitive Variables** $\mathbf{w} = [\rho, v_1, v_2, p]^T$ using the `cons2prim` mapping during extraction.
#
# 2.  **Spatial Reconstruction (`PlotData2D`)**:
#     * Projects the solution from the high-order Discontinuous Galerkin polynomial basis (on adaptive quadtrees) onto a structured 2D Cartesian grid suitable for visualization and tensor-based processing.
#
# 3.  **Temporal Sampling**:
#     * **Trigger**: `DiscreteCallback` executes every $N=40$ timesteps or at $t=T_{end}$.
#     * **Output**: Saves fields $(\rho, u, v, p)$ along with grid coordinates $(x, y)$ to `data_npz/`.
#
# %%
npz_out = joinpath(@__DIR__, "data_npz")
mkpath(npz_out)
dt_save = 0.1
npz_visualization_times = (0.75, 1.50, 2.25, 3.00)
last_npz_save_time = Ref(-Inf)
npz_save_times = Ref(Float64[])
next_npz_save_index = Ref(1)
# Retained blending diagnostics drive the conditional capped-blending rerun.
blending_history = NamedTuple[]
blending_cap_requested = Ref(false)
blending_cap_alpha_max = 0.5
# Allow a conditional rerun to write to a separate, explicit output directory.
function set_npz_output_directory!(directory)
    global npz_out = directory
    mkpath(npz_out)
    return npz_out
end
# State used by the conservation and energy--work diagnostics.
initial_mass = Ref(NaN)
initial_total_energy = Ref(NaN)
last_diagnostic_time = Ref(NaN)
last_work_rate = Ref(NaN)
cumulative_gravity_work = Ref(0.0)
# Integrate the conservative state on the current (possibly adapted) mesh.
# Since g = 1, rho*g*v is exactly the third conserved component rho_v2.
function state_integrals(u_ode, semi)
    u = Trixi.wrap_array(u_ode, semi)
    mass = Trixi.integrate((u_local, equations) -> u_local[1], u, semi)
    total_energy = Trixi.integrate((u_local, equations) -> u_local[4], u, semi)
    work_rate = Trixi.integrate((u_local, equations) -> u_local[3], u, semi)
    return mass, total_energy, work_rate
end
# Return active fraction, maximum alpha, and the spatial extent of active elements.
function blending_statistics(semi)
    alpha = vec(semi.cache.alpha)
    active_elements = findall(value -> value > 0, alpha)
    number_of_elements = length(alpha)
    active_fraction = number_of_elements == 0 ? 0.0 :
                      length(active_elements) / number_of_elements
    maximum_alpha = number_of_elements == 0 ? 0.0 : maximum(alpha)
    if isempty(active_elements)
        return active_fraction, maximum_alpha, NaN, NaN, NaN, NaN
    end
    coordinates = semi.cache.elements.node_coordinates
    x_minimum = Inf
    x_maximum = -Inf
    y_minimum = Inf
    y_maximum = -Inf
    for element in active_elements
        x_minimum = min(x_minimum,
                        minimum(view(coordinates, 1, :, :, element)))
        x_maximum = max(x_maximum,
                        maximum(view(coordinates, 1, :, :, element)))
        y_minimum = min(y_minimum,
                        minimum(view(coordinates, 2, :, :, element)))
        y_maximum = max(y_maximum,
                        maximum(view(coordinates, 2, :, :, element)))
    end
    return active_fraction, maximum_alpha,
           x_minimum, x_maximum, y_minimum, y_maximum
end
function log_verification_diagnostics(integrator, rho, rho_v1, rho_v2,
                                      rho_e, y)
    semi = integrator.p
    t = integrator.t
    rho_values = vec(rho)
    rho_v1_values = vec(rho_v1)
    rho_v2_values = vec(rho_v2)
    rho_e_values = vec(rho_e)
    y_values = vec(y)
    pressure = (equations.gamma - 1) .* (
        rho_e_values .-
        0.5 .* (rho_v1_values.^2 .+ rho_v2_values.^2) ./ rho_values)
    vertical_velocity = rho_v2_values ./ rho_values
    finite_state = all(isfinite, rho_values) &&
                   all(isfinite, rho_v1_values) &&
                   all(isfinite, rho_v2_values) &&
                   all(isfinite, rho_e_values) &&
                   all(isfinite, pressure)
    positive_state = all(value -> value > 0, rho_values) &&
                     all(value -> value > 0, pressure)
    # Check the nodes adjacent to y=0 and y=1. PlotData2D normally includes
    # the wall nodes; the fallback selects the closest available node layer.
    wall_mask = (y_values .<= 1e-8) .| (y_values .>= 1.0 - 1e-8)
    if !any(wall_mask)
        wall_distance = min.(abs.(y_values), abs.(1.0 .- y_values))
        wall_mask = wall_distance .<= minimum(wall_distance) + 1e-12
    end
    maximum_wall_velocity = maximum(abs.(vertical_velocity[wall_mask]))
    mass, total_energy, work_rate = state_integrals(integrator.u, semi)
    if !isfinite(last_diagnostic_time[])
        initial_mass[] = mass
        initial_total_energy[] = total_energy
        cumulative_gravity_work[] = 0.0
    else
        elapsed_time = t - last_diagnostic_time[]
        cumulative_gravity_work[] +=
            0.5 * (last_work_rate[] + work_rate) * elapsed_time
    end
    mass_drift = mass - initial_mass[]
    energy_change = total_energy - initial_total_energy[]
    energy_work_residual = energy_change - cumulative_gravity_work[]
    active_fraction, maximum_alpha,
    alpha_x_minimum, alpha_x_maximum,
    alpha_y_minimum, alpha_y_maximum = blending_statistics(semi)
    push!(blending_history,
          (time=Float64(t),
           active_fraction=active_fraction,
           maximum_alpha=maximum_alpha,
           x_minimum=alpha_x_minimum,
           x_maximum=alpha_x_maximum,
           y_minimum=alpha_y_minimum,
           y_maximum=alpha_y_maximum))
    # Operational spreading test: strong blending is active on a substantial
    # fraction of elements and extends broadly in both physical directions.
    # This requests a post-run cap; it never disables shock capturing or
    # modifies the baseline discretization during the current solve.
    extents_are_finite = all(isfinite,
                             (alpha_x_minimum, alpha_x_maximum,
                              alpha_y_minimum, alpha_y_maximum))
    if !blending_cap_requested[] && extents_are_finite &&
       maximum_alpha > blending_cap_alpha_max && active_fraction >= 0.25
        x_coverage = (alpha_x_maximum - alpha_x_minimum) /
                     (coordinates_max[1] - coordinates_min[1])
        y_coverage = (alpha_y_maximum - alpha_y_minimum) /
                     (coordinates_max[2] - coordinates_min[2])
        if x_coverage >= 0.75 && y_coverage >= 0.25
            blending_cap_requested[] = true
            Printf.@printf(
                "BLENDING_CAP_REQUEST t=%.5f cap_alpha_max=%.2f active_fraction=%.6e x_coverage=%.6e y_coverage=%.6e\n",
                t, blending_cap_alpha_max, active_fraction,
                x_coverage, y_coverage)
        end
    end
    Printf.@printf(
        "VERIFY t=%.5f rho=[%.6e,%.6e] p=[%.6e,%.6e] finite=%s positive=%s wall_max_abs_v=%.6e\n",
        t, minimum(rho_values), maximum(rho_values),
        minimum(pressure), maximum(pressure),
        string(finite_state), string(positive_state), maximum_wall_velocity)
    Printf.@printf(
        "VERIFY t=%.5f mass=%.16e mass_drift=%.6e energy=%.16e delta_energy=%.6e work_rate=%.6e cumulative_work=%.6e delta_energy_minus_work=%.6e\n",
        t, mass, mass_drift, total_energy, energy_change, work_rate,
        cumulative_gravity_work[], energy_work_residual)
    Printf.@printf(
        "VERIFY t=%.5f alpha_active_fraction=%.6e alpha_max=%.6e alpha_x_extent=[%.6e,%.6e] alpha_y_extent=[%.6e,%.6e]\n",
        t, active_fraction, maximum_alpha,
        alpha_x_minimum, alpha_x_maximum,
        alpha_y_minimum, alpha_y_maximum)
    last_diagnostic_time[] = t
    last_work_rate[] = work_rate
    return nothing
end
# Identity mapping: PlotData2D returns the conserved variables directly.
function save_npz_file(integrator)
    semi = integrator.p
    t = integrator.t
    pd = PlotData2D(integrator.u, semi,
                    solution_variables=cons2cons)
    # PlotData2D stores one SVector of conserved variables per plotted point.
    x = collect(pd.x)
    y = collect(pd.y)
    rho = collect(getindex.(pd.data, 1))
    rho_v1 = collect(getindex.(pd.data, 2))
    rho_v2 = collect(getindex.(pd.data, 3))
    rho_e = collect(getindex.(pd.data, 4))
    log_verification_diagnostics(integrator, rho, rho_v1, rho_v2, rho_e, y)
    filename = joinpath(npz_out, Printf.@sprintf("time_%08.5f.npz", t))
    npzwrite(filename, Dict("t" => Float64(t), "x" => x, "y" => y,
                            "rho" => rho,
                            "rho_v1" => rho_v1,
                            "rho_v2" => rho_v2,
                            "rho_e" => rho_e))
    last_npz_save_time[] = t
    # Consume every scheduled time reached by this callback. Normally this
    # advances exactly one entry because every scheduled time is a tstop.
    schedule = npz_save_times[]
    while next_npz_save_index[] <= length(schedule) &&
          schedule[next_npz_save_index[]] <= t + 1e-8
        next_npz_save_index[] += 1
    end
    println("Saved: $(basename(filename))")
    return nothing
end
function build_npz_save_schedule(t_start, t_end)
    regular_save_times = collect(t_start:dt_save:t_end)
    if isempty(regular_save_times) ||
       regular_save_times[end] < t_end - 1e-12
        push!(regular_save_times, t_end)
    else
        regular_save_times[end] = t_end
    end
    # Include the exact visualization times in addition to, not instead of,
    # the complete dt_save ground-truth schedule.
    requested_times = vcat(regular_save_times,
                           collect(npz_visualization_times), [t_end])
    filter!(time -> t_start - 1e-12 <= time <= t_end + 1e-12,
            requested_times)
    sort!(requested_times)
    schedule = Float64[]
    for time in requested_times
        save_time = clamp(time, t_start, t_end)
        if isempty(schedule) || abs(save_time - schedule[end]) > 1e-10
            push!(schedule, save_time)
        end
    end
    if isempty(schedule) || schedule[1] > t_start + 1e-12
        pushfirst!(schedule, t_start)
    else
        schedule[1] = t_start
    end
    schedule[end] = t_end
    return schedule
end
# Save the initial state and install physical-time stops for every later save.
function init_save_npz(cb, u, t, integrator)
    last_npz_save_time[] = -Inf
    initial_mass[] = NaN
    initial_total_energy[] = NaN
    last_diagnostic_time[] = NaN
    last_work_rate[] = NaN
    cumulative_gravity_work[] = 0.0
    empty!(blending_history)
    blending_cap_requested[] = false
    t_end = integrator.sol.prob.tspan[2]
    npz_save_times[] = build_npz_save_schedule(t, t_end)
    # The initial state is written explicitly below, so the first callback
    # target is the first scheduled time strictly after the initial time.
    next_npz_save_index[] = 1
    while next_npz_save_index[] <= length(npz_save_times[]) &&
          npz_save_times[][next_npz_save_index[]] <= t + 1e-10
        next_npz_save_index[] += 1
    end
    for save_time in npz_save_times[]
        if save_time > t + 1e-10
            add_tstop!(integrator, save_time)
        end
    end
    save_npz_file(integrator)
end
# The tstops make 0.75, 2.25, and every other requested snapshot exact.
function condition_save_npz(u, t, integrator)
    schedule = npz_save_times[]
    index = next_npz_save_index[]
    due = index <= length(schedule) && t >= schedule[index] - 1e-10
    t_end = integrator.sol.prob.tspan[2]
    is_finished = t >= t_end - 1e-10
    final_save_needed = is_finished &&
                        last_npz_save_time[] < t_end - 1e-10
    return due || final_save_needed
end
save_npz_callback = DiscreteCallback(condition_save_npz,
                                     save_npz_file,
                                     save_positions=(false, false),
                                     initialize=init_save_npz)
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
# Save the complete conservative state at the required physical-time cadence.
trixi_out = joinpath(@__DIR__, "trixi_output")

save_solution = SaveSolutionCallback(dt=dt_save,
                                     save_initial_solution=true,
                                     save_final_solution=true,
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
            dt=1.0,
            adaptive=false,
            callback=callbacks,
            saveat = 0.01,
            maxiters = 1e7);
# Configuration of the run currently exposed to post-processing.
active_alpha_max = 1.0
active_max_level = 6
active_polydeg = polydeg
active_source_terms = source_terms_gravity
active_source_name = "standard_gravity"
active_run_directory = npz_out

# Reusable driver for conditional blending, source-term, polynomial-degree,
# and AMR-escalation reruns.
function run_rti_case(; alpha_max=1.0, max_level=6,
                      polynomial_degree=active_polydeg,
                      source_terms=active_source_terms,
                      output_directory=joinpath(@__DIR__, "data_npz"),
                      label="rerun")
    set_npz_output_directory!(output_directory)
    dg_local = build_rti_dgsem(
        alpha_max; polynomial_degree=polynomial_degree)
    mesh_local = P4estMesh((1, 1), polydeg=polynomial_degree,
                           coordinates_min=coordinates_min,
                           coordinates_max=coordinates_max,
                           initial_refinement_level=3,
                           periodicity=(true, false))
    semi_local = SemidiscretizationHyperbolic(
        mesh_local, equations, initial_condition_rayleigh_taylor, dg_local;
        boundary_conditions=boundary_conditions,
        source_terms=source_terms)
    amr_local = build_rti_amr_callback(semi_local; max_level=max_level)
    ode_local = semidiscretize(semi_local, tspan)
    callbacks_local = CallbackSet(
        SummaryCallback(),
        AnalysisCallback(semi_local, interval=analysis_interval),
        AliveCallback(analysis_interval=analysis_interval),
        SaveSolutionCallback(
            dt=dt_save,
            save_initial_solution=true,
            save_final_solution=true,
            output_directory=joinpath(@__DIR__, "trixi_output_" * label),
            solution_variables=cons2cons),
        amr_local,
        save_npz_callback,
        StepsizeCallback(cfl=0.5))
    sol_local = solve(ode_local, RDPK3SpFSAL49();
                      dt=1.0,
                      adaptive=false,
                      callback=callbacks_local,
                      saveat=0.01,
                      maxiters=1e7)
    return (sol=sol_local, semi=semi_local, dg=dg_local, ode=ode_local,
            callbacks=callbacks_local, alpha_max=alpha_max,
            max_level=max_level, polynomial_degree=polynomial_degree,
            source_terms=source_terms, output_directory=output_directory)
end

# Enact the modest blending cap when the Cell-8 diagnostics request it.
if blending_cap_requested[]
    println("BLENDING_CAP_RERUN: rerunning with alpha_max = ",
            blending_cap_alpha_max)
    capped_run = run_rti_case(
        alpha_max=blending_cap_alpha_max,
        max_level=active_max_level,
        polynomial_degree=active_polydeg,
        source_terms=active_source_terms,
        output_directory=joinpath(@__DIR__, "data_npz_alpha_max_0p5"),
        label="alpha_max_0p5")
    sol = capped_run.sol
    semi = capped_run.semi
    dg = capped_run.dg
    ode = capped_run.ode
    callbacks = capped_run.callbacks
    active_alpha_max = capped_run.alpha_max
    active_max_level = capped_run.max_level
    active_polydeg = capped_run.polynomial_degree
    active_source_terms = capped_run.source_terms
    active_run_directory = capped_run.output_directory
    npz_out = active_run_directory
end
# Separate hydrostatic-balance verification: identical rho_0 and p_0, but v_0 = 0.
hydrostatic_velocity_history = Tuple{Float64, Float64, Float64}[]
hydrostatic_last_log_time = Ref(-Inf)
function hydrostatic_velocity_extrema(integrator)
    semi_local = integrator.p
    u = Trixi.wrap_array(integrator.u, semi_local)
    maximum_u = 0.0
    maximum_v = 0.0
    # Inspect every DG node directly, without visualization interpolation.
    for element in axes(u, 4), j in axes(u, 3), i in axes(u, 2)
        rho = u[1, i, j, element]
        if !(rho > 0.0) || !isfinite(rho)
            return Inf, Inf
        end
        velocity_u = u[2, i, j, element] / rho
        velocity_v = u[3, i, j, element] / rho
        if !isfinite(velocity_u) || !isfinite(velocity_v)
            return Inf, Inf
        end
        maximum_u = max(maximum_u, abs(velocity_u))
        maximum_v = max(maximum_v, abs(velocity_v))
    end
    return maximum_u, maximum_v
end
function log_hydrostatic_balance(integrator)
    t = Float64(integrator.t)
    maximum_u, maximum_v = hydrostatic_velocity_extrema(integrator)
    previous_maximum = isempty(hydrostatic_velocity_history) ? 0.0 :
                       max(hydrostatic_velocity_history[end][2],
                           hydrostatic_velocity_history[end][3])
    current_maximum = max(maximum_u, maximum_v)
    growth_since_previous_log = current_maximum - previous_maximum
    push!(hydrostatic_velocity_history, (t, maximum_u, maximum_v))
    hydrostatic_last_log_time[] = t
    Printf.@printf(
        "HYDROSTATIC t=%.5f max_abs_u=%.6e max_abs_v=%.6e max_velocity_growth_since_previous_log=%.6e\n",
        t, maximum_u, maximum_v, growth_since_previous_log)
    return nothing
end
function initialize_hydrostatic_balance_log(cb, u, t, integrator)
    empty!(hydrostatic_velocity_history)
    hydrostatic_last_log_time[] = -Inf
    t_end = integrator.sol.prob.tspan[2]
    log_times = collect(0.1:0.1:t_end)
    if isempty(log_times) || log_times[end] < t_end - 1e-12
        push!(log_times, t_end)
    else
        log_times[end] = t_end
    end
    for log_time in log_times
        add_tstop!(integrator, log_time)
    end
    log_hydrostatic_balance(integrator)
end
function condition_hydrostatic_balance_log(u, t, integrator)
    t_end = integrator.sol.prob.tspan[2]
    next_log_time = hydrostatic_last_log_time[] + 0.1
    due = isfinite(next_log_time) && t >= next_log_time - 1e-12
    final_log_needed = t >= t_end - 1e-12 &&
                       hydrostatic_last_log_time[] < t_end - 1e-12
    return due || final_log_needed
end
hydrostatic_log_callback = DiscreteCallback(
    condition_hydrostatic_balance_log,
    log_hydrostatic_balance,
    save_positions=(false, false),
    initialize=initialize_hydrostatic_balance_log)
# Use an independent mesh so this verification run cannot mutate the main AMR mesh.
mesh_hydrostatic = P4estMesh((1, 1), polydeg=polydeg,
                             coordinates_min=coordinates_min,
                             coordinates_max=coordinates_max,
                             initial_refinement_level=3,
                             periodicity=(true, false))
semi_hydrostatic = SemidiscretizationHyperbolic(
    mesh_hydrostatic, equations, initial_condition_hydrostatic, dg;
    boundary_conditions=boundary_conditions,
    source_terms=source_terms_gravity)
amr_indicator_hydrostatic = IndicatorLöhner(semi_hydrostatic,
                                            variable=density_variable)
amr_controller_hydrostatic = ControllerThreeLevel(
    semi_hydrostatic, amr_indicator_hydrostatic,
    base_level=4,
    med_level=5, med_threshold=0.1,
    max_level=6, max_threshold=0.3)
amr_callback_hydrostatic = AMRCallback(semi_hydrostatic,
                                       amr_controller_hydrostatic,
                                       interval=20,
                                       adapt_initial_condition=true)
ode_hydrostatic = semidiscretize(semi_hydrostatic, (0.0, 1.0))
stepsize_callback_hydrostatic = StepsizeCallback(cfl=0.5)
callbacks_hydrostatic = CallbackSet(amr_callback_hydrostatic,
                                    hydrostatic_log_callback,
                                    stepsize_callback_hydrostatic)
sol_hydrostatic = solve(ode_hydrostatic, RDPK3SpFSAL49();
                        dt=1.0,
                        adaptive=false,
                        callback=callbacks_hydrostatic,
                        save_everystep=false,
                        maxiters=1e7)
function summarize_hydrostatic_balance(mode)
    @assert !isempty(hydrostatic_velocity_history)
    final_log = hydrostatic_velocity_history[end]
    final_maximum_velocity = max(final_log[2], final_log[3])
    peak_maximum_velocity = maximum(
        max(record[2], record[3]) for record in hydrostatic_velocity_history)
    previous_maximum_velocity = length(hydrostatic_velocity_history) >= 2 ?
        max(hydrostatic_velocity_history[end - 1][2],
            hydrostatic_velocity_history[end - 1][3]) :
        final_maximum_velocity
    last_interval_growth =
        final_maximum_velocity - previous_maximum_velocity
    Printf.@printf(
        "HYDROSTATIC_SUMMARY mode=%s t=%.5f final_max_abs_u=%.6e final_max_abs_v=%.6e final_max_velocity=%.6e peak_max_velocity=%.6e perturbation_reference=%.6e final_over_reference=%.6e last_interval_growth=%.6e\n",
        mode, final_log[1], final_log[2], final_log[3],
        final_maximum_velocity, peak_maximum_velocity, 0.03,
        final_maximum_velocity / 0.03, last_interval_growth)
    return (final_maximum_velocity=final_maximum_velocity,
            last_interval_growth=last_interval_growth)
end
# Background-balanced gravity source for the fallback verification run.
# At the unperturbed state, the momentum source equals the analytic
# hydrostatic pressure gradient. The correction vanishes away from the
# smoothed interface wherever dp0/dy = rho0*g.
@inline function source_terms_gravity_wellbalanced(u, x, t,
        equations::CompressibleEulerEquations2D)
    rho, rho_v1, rho_v2, rho_e = u
    y = x[2]
    rho_background = 2.0 - 0.5 * (1.0 + tanh(50.0 * (y - 0.5)))
    dpdy_background = y < 0.5 ? 2.0 : 1.0
    # g = 1; retain the same conservative source structure.
    momentum_source = rho + (dpdy_background - rho_background)
    vertical_velocity = rho_v2 / rho
    return SVector(0.0, 0.0, momentum_source,
                   momentum_source * vertical_velocity)
end
hydrostatic_summary = summarize_hydrostatic_balance("standard_gravity")
hydrostatic_violation =
    hydrostatic_summary.final_maximum_velocity >= 0.03 ||
    hydrostatic_summary.last_interval_growth > 1e-8
if hydrostatic_violation
    println("HYDROSTATIC_FALLBACK: residual velocities are too large or " *
            "still growing; rerunning with the background-balanced " *
            "gravity/source discretization.")
    mesh_hydrostatic_wb = P4estMesh(
        (1, 1), polydeg=active_polydeg,
        coordinates_min=coordinates_min,
        coordinates_max=coordinates_max,
        initial_refinement_level=3,
        periodicity=(true, false))
    semi_hydrostatic_wb = SemidiscretizationHyperbolic(
        mesh_hydrostatic_wb, equations, initial_condition_hydrostatic, dg;
        boundary_conditions=boundary_conditions,
        source_terms=source_terms_gravity_wellbalanced)
    amr_callback_hydrostatic_wb =
        build_rti_amr_callback(semi_hydrostatic_wb)
    ode_hydrostatic_wb = semidiscretize(semi_hydrostatic_wb, (0.0, 1.0))
    callbacks_hydrostatic_wb = CallbackSet(
        amr_callback_hydrostatic_wb,
        hydrostatic_log_callback,
        StepsizeCallback(cfl=0.5))
    sol_hydrostatic_wb = solve(ode_hydrostatic_wb, RDPK3SpFSAL49();
                               dt=1.0,
                               adaptive=false,
                               callback=callbacks_hydrostatic_wb,
                               save_everystep=false,
                               maxiters=1e7)
    hydrostatic_summary =
        summarize_hydrostatic_balance("well_balanced_gravity")
    if hydrostatic_summary.final_maximum_velocity < 0.03 &&
       hydrostatic_summary.last_interval_growth <= 1e-8
        println("HYDROSTATIC_FALLBACK: acceptance criteria satisfied.")
    else
        println("HYDROSTATIC_FALLBACK WARNING: the well-balanced run " *
                "still does not satisfy the acceptance criteria.")
    end

    # Adopt the well-balanced source for the production ground-truth run,
    # preserving the currently selected blending, AMR, and polynomial degree.
    println("HYDROSTATIC_FALLBACK: rerunning the main RTI production case " *
            "with the well-balanced gravity/source discretization.")
    wellbalanced_run = run_rti_case(
        alpha_max=active_alpha_max,
        max_level=active_max_level,
        polynomial_degree=active_polydeg,
        source_terms=source_terms_gravity_wellbalanced,
        output_directory=joinpath(@__DIR__, "data_npz_wellbalanced"),
        label="wellbalanced")
    sol = wellbalanced_run.sol
    semi = wellbalanced_run.semi
    dg = wellbalanced_run.dg
    ode = wellbalanced_run.ode
    callbacks = wellbalanced_run.callbacks
    active_alpha_max = wellbalanced_run.alpha_max
    active_max_level = wellbalanced_run.max_level
    active_polydeg = wellbalanced_run.polynomial_degree
    active_source_terms = wellbalanced_run.source_terms
    active_source_name = "well_balanced_gravity"
    active_run_directory = wellbalanced_run.output_directory
    npz_out = active_run_directory
    println("HYDROSTATIC_FALLBACK: production snapshots now use ",
            active_run_directory)
end
# %% [markdown]


# ## Cell 11: Data Regularization (Rasterization)
# ## Methodology: Gapless Uniform-Grid Interpolation
# Interpolates the scattered AMR/DG visualization samples onto one uniform
# 512 x 2048 grid covering [0, 0.25] x [0, 1]. A spatial bin structure limits
# each interpolation to nearby samples. Positive inverse-distance weights avoid
# extrapolation overshoots, while every pixel is filled without NaN gaps.
# This interpolation is used only for rendering; the conserved NPZ data remain
# unchanged.
function rasterize_data(x, y, z, nx=512, ny=2048; neighbors=6)
    @assert nx >= 2 && ny >= 2
    @assert neighbors >= 1
    @assert length(x) == length(y)
    @assert length(x) == length(z)

    x_min, x_max = 0.0, 0.25
    y_min, y_max = 0.0, 1.0

    # Retain only finite samples inside the physical domain. Roundoff-sized
    # coordinate excursions are clamped back onto the boundary.
    source_x = Float64[]
    source_y = Float64[]
    source_z = Float64[]
    sizehint!(source_x, length(x))
    sizehint!(source_y, length(x))
    sizehint!(source_z, length(x))

    for k in 1:length(x)
        xk = Float64(x[k])
        yk = Float64(y[k])
        zk = Float64(z[k])
        if isfinite(xk) && isfinite(yk) && isfinite(zk) &&
           x_min - 1e-12 <= xk <= x_max + 1e-12 &&
           y_min - 1e-12 <= yk <= y_max + 1e-12
            push!(source_x, clamp(xk, x_min, x_max))
            push!(source_y, clamp(yk, y_min, y_max))
            push!(source_z, zk)
        end
    end

    number_of_sources = length(source_x)
    number_of_sources > 0 ||
        throw(ArgumentError("No finite source samples are available for interpolation."))

    target_dx = (x_max - x_min) / (nx - 1)
    target_dy = (y_max - y_min) / (ny - 1)

    # Adapt the source-bin size to the sample density, with a lower bound
    # related to the target pixel size.
    mean_source_spacing = sqrt((x_max - x_min) * (y_max - y_min) /
                               number_of_sources)
    bin_size = max(2.0 * mean_source_spacing,
                   4.0 * target_dx, 4.0 * target_dy)

    number_of_bins_x = max(1, ceil(Int, (x_max - x_min) / bin_size))
    number_of_bins_y = max(1, ceil(Int, (y_max - y_min) / bin_size))
    bins = Dict{Int, Vector{Int}}()

    for source_index in 1:number_of_sources
        bin_x = clamp(floor(Int, source_x[source_index] / bin_size) + 1,
                      1, number_of_bins_x)
        bin_y = clamp(floor(Int, source_y[source_index] / bin_size) + 1,
                      1, number_of_bins_y)
        key = bin_x + (bin_y - 1) * number_of_bins_x
        push!(get!(() -> Int[], bins, key), source_index)
    end

    grid = Matrix{Float64}(undef, ny, nx)

    # Rows are independent, so interpolation can use the available Julia threads.
    Threads.@threads for j in 1:ny
        best_distances = Vector{Float64}(undef, neighbors)
        best_indices = Vector{Int}(undef, neighbors)

        y_target = clamp(y_min + (j - 1) * target_dy, y_min, y_max)
        target_bin_y = clamp(floor(Int, y_target / bin_size) + 1,
                             1, number_of_bins_y)

        for i in 1:nx
            x_target = clamp(x_min + (i - 1) * target_dx, x_min, x_max)
            target_bin_x = clamp(floor(Int, x_target / bin_size) + 1,
                                 1, number_of_bins_x)

            fill!(best_distances, Inf)
            fill!(best_indices, 0)

            # Expand through neighboring source bins. Once the kth nearest
            # distance is closer than any subsequently reachable bin, the local
            # interpolation stencil is complete.
            radius = 0
            maximum_radius = max(number_of_bins_x, number_of_bins_y)

            while radius <= maximum_radius
                minimum_bin_x = max(1, target_bin_x - radius)
                maximum_bin_x = min(number_of_bins_x, target_bin_x + radius)
                minimum_bin_y = max(1, target_bin_y - radius)
                maximum_bin_y = min(number_of_bins_y, target_bin_y + radius)

                for bin_y in minimum_bin_y:maximum_bin_y
                    for bin_x in minimum_bin_x:maximum_bin_x
                        # Visit only the perimeter of the current ring.
                        if max(abs(bin_x - target_bin_x),
                               abs(bin_y - target_bin_y)) != radius
                            continue
                        end

                        key = bin_x + (bin_y - 1) * number_of_bins_x
                        haskey(bins, key) || continue

                        for source_index in bins[key]
                            delta_x = source_x[source_index] - x_target
                            delta_y = source_y[source_index] - y_target
                            distance_squared = delta_x^2 + delta_y^2

                            if distance_squared < best_distances[end]
                                position = neighbors
                                while position > 1 &&
                                      best_distances[position - 1] > distance_squared
                                    best_distances[position] =
                                        best_distances[position - 1]
                                    best_indices[position] =
                                        best_indices[position - 1]
                                    position -= 1
                                end
                                best_distances[position] = distance_squared
                                best_indices[position] = source_index
                            end
                        end
                    end
                end

                if best_indices[end] != 0 &&
                   best_distances[end] <= (radius * bin_size)^2
                    break
                end
                radius += 1
            end

            best_indices[1] != 0 ||
                error("Failed to construct an interpolation stencil at " *
                      "($x_target, $y_target).")

            # Coincident visualization nodes can occur at element interfaces.
            # Average coincident samples instead of dividing by zero distance.
            if best_distances[1] <= 1e-24
                exact_sum = 0.0
                exact_count = 0
                for neighbor in 1:neighbors
                    if best_indices[neighbor] != 0 &&
                       best_distances[neighbor] <= 1e-24
                        exact_sum += source_z[best_indices[neighbor]]
                        exact_count += 1
                    end
                end
                grid[j, i] = exact_sum / exact_count
            else
                weighted_sum = 0.0
                weight_sum = 0.0
                for neighbor in 1:neighbors
                    source_index = best_indices[neighbor]
                    source_index == 0 && continue

                    # Shepard interpolation with distance exponent two.
                    weight = 1.0 / best_distances[neighbor]
                    weighted_sum += weight * source_z[source_index]
                    weight_sum += weight
                end
                grid[j, i] = weighted_sum / weight_sum
            end
        end
    end

    return grid
end
# %% [markdown]

# ## Cell 12: Post-Processing & Temporal Diagnostics
# ## Methodology: Uniform-Grid Heatmaps and Focused Diagnostics
# Render density and vertical velocity on the same gapless 512 x 2048 grid.
# All four density panels share one color range, and all velocity panels share
# one symmetric range. The main figure has no colorbars; a separate diagnostic
# figure includes them. Interface metrics use the rho = 1.5 contour envelope.
println("Generating density and vertical-velocity summary...")
target_times = [0.75, 1.50, 2.25, 3.00]
files = [joinpath(npz_out, file) for file in readdir(npz_out)
         if endswith(file, ".npz")]
@assert !isempty(files) "No NPZ solution snapshots were found."
snapshot_records = map(files) do file
    matched = match(r"time_([0-9]+\.[0-9]+)\.npz$", basename(file))
    matched === nothing &&
        error("Cannot parse snapshot time from $(basename(file))")
    (file=file, time=parse(Float64, matched.captures[1]))
end
sort!(snapshot_records, by=record -> record.time)
function exact_snapshot_index(records, target_time)
    # Filenames have five decimal places, so allow only their rounding error.
    index = findfirst(records) do record
        isapprox(record.time, target_time; atol=5.1e-6, rtol=0.0)
    end
    index === nothing && error(
        "Missing exact NPZ snapshot at t=$target_time. " *
        "Nearest-time substitution is prohibited. Add all visualization " *
        "times, especially t=0.75, to the NPZ tstop/save schedule.")
    return index
end
indices = [exact_snapshot_index(snapshot_records, target_time)
           for target_time in target_times]
@assert length(unique(indices)) == length(target_times)
snapshots = [npzread(snapshot_records[index].file) for index in indices]
visualization_nx = 512
visualization_ny = 2048
visualization_x = collect(range(0.0, 0.25, length=visualization_nx))
visualization_y = collect(range(0.0, 1.0, length=visualization_ny))
rendered_snapshots = Any[]
for (snapshot_index, data) in enumerate(snapshots)
    x = Float64.(vec(data["x"]))
    y = Float64.(vec(data["y"]))
    rho = Float64.(vec(data["rho"]))
    rho_v2 = Float64.(vec(data["rho_v2"]))
    all(isfinite, rho) || error("Non-finite density in visualization snapshot.")
    all(value -> value > 0.0, rho) ||
        error("Non-positive density in visualization snapshot.")
    vertical_velocity = rho_v2 ./ rho
    all(isfinite, vertical_velocity) ||
        error("Non-finite vertical velocity in visualization snapshot.")
    rho_grid = rasterize_data(x, y, rho,
                              visualization_nx, visualization_ny)
    velocity_grid = rasterize_data(x, y, vertical_velocity,
                                   visualization_nx, visualization_ny)
    all(isfinite, rho_grid) || error("Density rendering grid contains gaps.")
    all(isfinite, velocity_grid) ||
        error("Velocity rendering grid contains gaps.")
    push!(rendered_snapshots,
          (time=target_times[snapshot_index],
           rho=rho_grid,
           v=velocity_grid))
end
# Reload and rasterize an exact four-snapshot set after a conditional rerun.
function render_exact_snapshot_set(directory)
    local_files = [joinpath(directory, file) for file in readdir(directory)
                   if endswith(file, ".npz")]
    @assert !isempty(local_files) "No NPZ snapshots in the selected production run."
    records = map(local_files) do file
        matched = match(r"time_([0-9]+\.[0-9]+)\.npz$", basename(file))
        matched === nothing &&
            error("Cannot parse snapshot time from $(basename(file))")
        (file=file, time=parse(Float64, matched.captures[1]))
    end
    sort!(records, by=record -> record.time)
    local_indices = [exact_snapshot_index(records, target_time)
                     for target_time in target_times]
    @assert length(unique(local_indices)) == length(target_times)
    rendered = Any[]
    for (snapshot_index, index) in enumerate(local_indices)
        data = npzread(records[index].file)
        x = Float64.(vec(data["x"]))
        y = Float64.(vec(data["y"]))
        rho = Float64.(vec(data["rho"]))
        rho_v2 = Float64.(vec(data["rho_v2"]))
        all(isfinite, rho) && all(value -> value > 0.0, rho) ||
            error("Invalid density in a production visualization snapshot.")
        vertical_velocity = rho_v2 ./ rho
        all(isfinite, vertical_velocity) ||
            error("Invalid velocity in a production visualization snapshot.")
        rho_grid = rasterize_data(x, y, rho,
                                  visualization_nx, visualization_ny)
        velocity_grid = rasterize_data(x, y, vertical_velocity,
                                       visualization_nx, visualization_ny)
        all(isfinite, rho_grid) && all(isfinite, velocity_grid) ||
            error("Production rendering grid contains gaps.")
        push!(rendered,
              (time=target_times[snapshot_index],
               rho=rho_grid,
               v=velocity_grid))
    end
    return rendered
end

# Operational contact-ringing detector. A monotone contact has one persistent
# density-slope sign. Ringing requires repeated slope reversals and an
# overshoot beyond the local window's endpoint densities. Small slope changes
# are ignored to avoid mistaking interpolation roundoff for oscillations.
function contact_ringing_metrics(rho_grid)
    number_of_y = size(rho_grid, 1)
    contact_columns = 0
    ringing_columns = 0
    maximum_ringing_overshoot = 0.0
    maximum_slope_reversals = 0
    half_window = 12

    for i in axes(rho_grid, 2)
        column = view(rho_grid, :, i)
        crossings = Int[]
        for j in 1:(number_of_y - 1)
            rho_lower = column[j]
            rho_upper = column[j + 1]
            if (rho_lower <= 1.5 <= rho_upper) ||
               (rho_upper <= 1.5 <= rho_lower)
                push!(crossings, j)
            end
        end
        isempty(crossings) && continue
        contact_columns += 1

        column_has_ringing = false
        column_maximum_overshoot = 0.0
        column_maximum_reversals = 0
        for crossing in crossings
            first_index = max(2, crossing - half_window)
            last_index = min(number_of_y - 1,
                             crossing + half_window + 1)
            last_index - first_index + 1 >= 5 || continue

            values = view(column, first_index:last_index)
            left_density = first(values)
            right_density = last(values)
            transition = abs(right_density - left_density)
            transition >= 0.25 || continue

            local_span = maximum(values) - minimum(values)
            slope_tolerance = 0.02 * max(local_span, transition)
            slope_signs = Int[]
            for j in first_index:(last_index - 1)
                slope = column[j + 1] - column[j]
                slope_sign = slope > slope_tolerance ? 1 :
                             slope < -slope_tolerance ? -1 : 0
                if slope_sign != 0 &&
                   (isempty(slope_signs) ||
                    slope_signs[end] != slope_sign)
                    push!(slope_signs, slope_sign)
                end
            end
            slope_reversals = max(0, length(slope_signs) - 1)
            overshoot = max(
                0.0,
                maximum(values) - max(left_density, right_density),
                min(left_density, right_density) - minimum(values))

            if slope_reversals >= 2 && overshoot >= 0.02
                column_has_ringing = true
                column_maximum_overshoot =
                    max(column_maximum_overshoot, overshoot)
                column_maximum_reversals =
                    max(column_maximum_reversals, slope_reversals)
            end
        end

        if column_has_ringing
            ringing_columns += 1
            maximum_ringing_overshoot =
                max(maximum_ringing_overshoot, column_maximum_overshoot)
            maximum_slope_reversals =
                max(maximum_slope_reversals, column_maximum_reversals)
        end
    end

    ringing_fraction = contact_columns == 0 ? 0.0 :
                       ringing_columns / contact_columns
    return (contact_columns=contact_columns,
            ringing_columns=ringing_columns,
            ringing_fraction=ringing_fraction,
            maximum_ringing_overshoot=maximum_ringing_overshoot,
            maximum_slope_reversals=maximum_slope_reversals)
end

polydeg3_rerun_requested = Ref(false)
for snapshot in rendered_snapshots
    ringing = contact_ringing_metrics(snapshot.rho)
    ringing_detected =
        ringing.ringing_columns >= 2 &&
        ringing.ringing_fraction >= 0.01 &&
        ringing.maximum_ringing_overshoot >= 0.02
    if active_polydeg == 4 && ringing_detected
        polydeg3_rerun_requested[] = true
    end
    Printf.@printf(
        "CONTACT_RINGING polydeg=%d t=%.2f contact_columns=%d ringing_columns=%d ringing_fraction=%.6e max_overshoot=%.6e max_slope_reversals=%d detected=%s\n",
        active_polydeg, snapshot.time, ringing.contact_columns,
        ringing.ringing_columns, ringing.ringing_fraction,
        ringing.maximum_ringing_overshoot,
        ringing.maximum_slope_reversals, string(ringing_detected))
end

# Advisor-mandated degree-4 contingency. Retain the currently selected
# HLLC/DG-FV blending, local density-based AMR, and gravity source.
if polydeg3_rerun_requested[] && active_polydeg == 4
    println("POLYDEG_CONTINGENCY: degree-4 contact ringing detected; " *
            "rerunning the production case with polynomial_degree=3. " *
            "No global dissipation is added.")
    degree3_run = run_rti_case(
        alpha_max=active_alpha_max,
        max_level=active_max_level,
        polynomial_degree=3,
        source_terms=active_source_terms,
        output_directory=joinpath(@__DIR__, "data_npz_polydeg_3"),
        label="polydeg_3")
    sol = degree3_run.sol
    semi = degree3_run.semi
    dg = degree3_run.dg
    ode = degree3_run.ode
    callbacks = degree3_run.callbacks
    active_alpha_max = degree3_run.alpha_max
    active_max_level = degree3_run.max_level
    active_polydeg = degree3_run.polynomial_degree
    active_source_terms = degree3_run.source_terms
    active_run_directory = degree3_run.output_directory
    npz_out = active_run_directory

    # All subsequent scales, metrics, resolution comparisons, and figures use
    # the degree-3 production state rather than the rejected degree-4 state.
    rendered_snapshots = render_exact_snapshot_set(npz_out)
    for snapshot in rendered_snapshots
        ringing = contact_ringing_metrics(snapshot.rho)
        Printf.@printf(
            "CONTACT_RINGING_POLYDEG3 t=%.2f contact_columns=%d ringing_columns=%d ringing_fraction=%.6e max_overshoot=%.6e max_slope_reversals=%d\n",
            snapshot.time, ringing.contact_columns,
            ringing.ringing_columns, ringing.ringing_fraction,
            ringing.maximum_ringing_overshoot,
            ringing.maximum_slope_reversals)
    end
    Printf.@printf(
        "POLYDEG_CONTINGENCY_COMPLETE polydeg=%d max_level=%d alpha_max=%.2f snapshot_directory=%s\n",
        active_polydeg, active_max_level, active_alpha_max,
        active_run_directory)
end

# Auto-derived common scales over all four requested snapshots.
rho_minimum = minimum(minimum(snapshot.rho)
                      for snapshot in rendered_snapshots)
rho_maximum = maximum(maximum(snapshot.rho)
                      for snapshot in rendered_snapshots)
if rho_maximum <= rho_minimum
    padding = max(1e-12, 1e-6 * abs(rho_minimum))
    rho_minimum -= padding
    rho_maximum += padding
end
velocity_limit = maximum(maximum(abs.(snapshot.v))
                         for snapshot in rendered_snapshots)
isfinite(velocity_limit) || error("Non-finite common velocity scale.")
velocity_limit = max(velocity_limit, 1e-12)
function make_summary_panels(show_colorbar)
    panels = Any[]
    # First row: density with one common scale.
    for snapshot in rendered_snapshots
        panel_title = Printf.@sprintf(
            "ρ  t = %.2f\n[%.5f, %.5f]",
            snapshot.time, rho_minimum, rho_maximum)
        push!(panels,
              heatmap(visualization_x, visualization_y, snapshot.rho;
                      color=:viridis,
                      clims=(rho_minimum, rho_maximum),
                      colorbar=show_colorbar,
                      title=panel_title,
                      xlabel="x",
                      ylabel="y",
                      xlims=(0.0, 0.25),
                      ylims=(0.0, 1.0),
                      aspect_ratio=:equal,
                      yflip=false,
                      grid=false,
                      legend=false,
                      framestyle=:box,
                      titlefontsize=10,
                      guidefontsize=9,
                      tickfontsize=8,
                      margin=2Plots.mm))
    end
    # Second row: vertical velocity with one common symmetric scale.
    for snapshot in rendered_snapshots
        panel_title = Printf.@sprintf(
            "v  t = %.2f\n[%.5f, %.5f]",
            snapshot.time, -velocity_limit, velocity_limit)
        push!(panels,
              heatmap(visualization_x, visualization_y, snapshot.v;
                      color=:RdBu,
                      clims=(-velocity_limit, velocity_limit),
                      colorbar=show_colorbar,
                      title=panel_title,
                      xlabel="x",
                      ylabel="y",
                      xlims=(0.0, 0.25),
                      ylims=(0.0, 1.0),
                      aspect_ratio=:equal,
                      yflip=false,
                      grid=false,
                      legend=false,
                      framestyle=:box,
                      titlefontsize=10,
                      guidefontsize=9,
                      tickfontsize=8,
                      margin=2Plots.mm))
    end
    return panels
end
println("Saving summary_all.png...")
# show_colorbar=false enforces colorbar=false for every panel in summary_all.
summary_plot = plot(make_summary_panels(false)...,
                    layout=(2, 4),
                    size=(2400, 1800),
                    margin=5Plots.mm)
savefig(summary_plot, joinpath(@__DIR__, "summary_all.png"))
println("Saving summary_diagnostic.png...")
diagnostic_plot = plot(make_summary_panels(true)...,
                       layout=(2, 4),
                       size=(2700, 1900),
                       margin=5Plots.mm)
savefig(diagnostic_plot, joinpath(@__DIR__, "summary_diagnostic.png"))
# Return the rho=1.5 contour envelope. Spike and bubble amplitudes are measured
# relative to the initial interface y=0.5. The contour-envelope width is used
# consistently as the mixing-zone-width metric for both polynomial degrees.
function interface_metrics(rho_grid)
    interface_level = 1.5
    interface_y_minimum = Inf
    interface_y_maximum = -Inf
    for i in axes(rho_grid, 2)
        for j in 1:(size(rho_grid, 1) - 1)
            rho_lower = rho_grid[j, i]
            rho_upper = rho_grid[j + 1, i]
            crosses_level =
                (rho_lower <= interface_level <= rho_upper) ||
                (rho_upper <= interface_level <= rho_lower)
            if crosses_level && rho_lower != rho_upper
                fraction = (interface_level - rho_lower) /
                           (rho_upper - rho_lower)
                crossing_y = visualization_y[j] +
                             fraction * (visualization_y[j + 1] -
                                         visualization_y[j])
                interface_y_minimum = min(interface_y_minimum, crossing_y)
                interface_y_maximum = max(interface_y_maximum, crossing_y)
            end
        end
    end
    if !isfinite(interface_y_minimum) || !isfinite(interface_y_maximum)
        return (interface_y_minimum=NaN,
                interface_y_maximum=NaN,
                spike_amplitude=NaN,
                bubble_amplitude=NaN,
                mixing_zone_width=NaN)
    end
    return (interface_y_minimum=interface_y_minimum,
            interface_y_maximum=interface_y_maximum,
            spike_amplitude=max(0.0, interface_y_maximum - 0.5),
            bubble_amplitude=max(0.0, 0.5 - interface_y_minimum),
            mixing_zone_width=interface_y_maximum - interface_y_minimum)
end
study_times = [1.50, 2.25, 3.00]
current_metrics = Dict{Float64, Any}()
current_contact_metrics = Dict{Float64, Any}()
amr_escalation_requested = Ref(false)
amr_escalation_reasons = String[]
# Estimate the 25--75% density-transition thickness from the local slope at
# every rho=1.5 crossing. Larger values indicate a more smeared contact.
function contact_transition_metrics(rho_grid)
    thicknesses = Float64[]
    grid_dy = visualization_y[2] - visualization_y[1]
    for i in axes(rho_grid, 2)
        for j in 1:(size(rho_grid, 1) - 1)
            rho_lower = rho_grid[j, i]
            rho_upper = rho_grid[j + 1, i]
            crosses_interface =
                (rho_lower <= 1.5 <= rho_upper) ||
                (rho_upper <= 1.5 <= rho_lower)
            if crosses_interface && rho_lower != rho_upper
                # The density interval [1.25, 1.75] has width 0.5.
                push!(thicknesses,
                      0.5 * grid_dy / abs(rho_upper - rho_lower))
            end
        end
    end
    if isempty(thicknesses)
        return (median_thickness=NaN,
                p90_thickness=NaN,
                crossing_count=0)
    end
    return (median_thickness=median(thicknesses),
            p90_thickness=quantile(thicknesses, 0.9),
            crossing_count=length(thicknesses))
end
for snapshot in rendered_snapshots
    if snapshot.time in study_times
        metrics = interface_metrics(snapshot.rho)
        contact_metrics = contact_transition_metrics(snapshot.rho)
        current_metrics[snapshot.time] = metrics
        current_contact_metrics[snapshot.time] = contact_metrics
        Printf.@printf(
            "RESOLUTION_CURRENT polydeg=%d t=%.2f interface_y=[%.6e,%.6e] spike_amplitude=%.6e bubble_amplitude=%.6e mixing_zone_width=%.6e\n",
            active_polydeg, snapshot.time,
            metrics.interface_y_minimum, metrics.interface_y_maximum,
            metrics.spike_amplitude, metrics.bubble_amplitude,
            metrics.mixing_zone_width)
    end
end
function read_exact_npz_snapshot(directory, target_time)
    for filename in sort(readdir(directory))
        endswith(filename, ".npz") || continue
        matched = match(r"time_([0-9]+\.[0-9]+)\.npz$", filename)
        matched === nothing && continue
        snapshot_time = parse(Float64, matched.captures[1])
        if isapprox(snapshot_time, target_time;
                    atol=5.1e-6, rtol=0.0)
            return npzread(joinpath(directory, filename))
        end
    end
    return nothing
end
# No archive path was supplied in the scientific specification. Support an
# explicit path and conventional archive locations without inventing data.
run9_candidates = String[
    get(ENV, "RUN9_NPZ_DIR", ""),
    joinpath(@__DIR__, "run9_data_npz"),
    normpath(joinpath(@__DIR__, "..", "run9", "data_npz")),
    normpath(joinpath(@__DIR__, "..", "run_9", "data_npz"))
]
run9_directory = nothing
for candidate in run9_candidates
    if !isempty(candidate) && isdir(candidate)
        run9_directory = candidate
        break
    end
end
if run9_directory === nothing
    println("RESOLUTION_STUDY: archived Run-9 NPZ data were not found. " *
            "Set ENV[\"RUN9_NPZ_DIR\"] to the archive directory. " *
            "Current-run metrics were logged; no baseline comparison or " *
            "quantitative error was fabricated.")
else
    println("RESOLUTION_STUDY: using archived Run-9 data from ",
            run9_directory)
    for target_time in study_times
        baseline_data = read_exact_npz_snapshot(run9_directory, target_time)
        if baseline_data === nothing
            println("RESOLUTION_STUDY: no exact archived Run-9 snapshot at ",
                    "t=", target_time, "; comparison skipped.")
            continue
        end
        baseline_x = Float64.(vec(baseline_data["x"]))
        baseline_y = Float64.(vec(baseline_data["y"]))
        baseline_rho = Float64.(vec(baseline_data["rho"]))
        baseline_grid = rasterize_data(baseline_x, baseline_y, baseline_rho,
                                       visualization_nx, visualization_ny)
        baseline_metrics = interface_metrics(baseline_grid)
        baseline_contact = contact_transition_metrics(baseline_grid)
        metrics = current_metrics[target_time]
        contact = current_contact_metrics[target_time]
        Printf.@printf(
            "RESOLUTION_STUDY t=%.2f current_interface=[%.6e,%.6e] run9_interface=[%.6e,%.6e] current_spike=%.6e run9_spike=%.6e delta_spike=%.6e current_bubble=%.6e run9_bubble=%.6e delta_bubble=%.6e current_width=%.6e run9_width=%.6e delta_width=%.6e\n",
            target_time,
            metrics.interface_y_minimum, metrics.interface_y_maximum,
            baseline_metrics.interface_y_minimum,
            baseline_metrics.interface_y_maximum,
            metrics.spike_amplitude, baseline_metrics.spike_amplitude,
            metrics.spike_amplitude - baseline_metrics.spike_amplitude,
            metrics.bubble_amplitude, baseline_metrics.bubble_amplitude,
            metrics.bubble_amplitude - baseline_metrics.bubble_amplitude,
            metrics.mixing_zone_width, baseline_metrics.mixing_zone_width,
            metrics.mixing_zone_width - baseline_metrics.mixing_zone_width)
        grid_dy = visualization_y[2] - visualization_y[1]
        median_smeared =
            isfinite(contact.median_thickness) &&
            isfinite(baseline_contact.median_thickness) &&
            contact.median_thickness >
                1.10 * baseline_contact.median_thickness + grid_dy
        tail_smeared =
            isfinite(contact.p90_thickness) &&
            isfinite(baseline_contact.p90_thickness) &&
            contact.p90_thickness >
                1.10 * baseline_contact.p90_thickness + grid_dy
        interface_lost =
            !isfinite(metrics.mixing_zone_width) &&
            isfinite(baseline_metrics.mixing_zone_width)
        contacts_underresolved =
            target_time >= 2.25 &&
            (median_smeared || tail_smeared || interface_lost)
        Printf.@printf(
            "RESOLUTION_CONTACT t=%.2f current_median_thickness=%.6e run9_median_thickness=%.6e current_p90_thickness=%.6e run9_p90_thickness=%.6e current_crossings=%d run9_crossings=%d underresolved=%s\n",
            target_time, contact.median_thickness,
            baseline_contact.median_thickness, contact.p90_thickness,
            baseline_contact.p90_thickness, contact.crossing_count,
            baseline_contact.crossing_count,
            string(contacts_underresolved))
        if contacts_underresolved
            amr_escalation_requested[] = true
            push!(amr_escalation_reasons,
                  Printf.@sprintf(
                      "t=%.2f: median_smeared=%s tail_smeared=%s interface_lost=%s",
                      target_time, string(median_smeared),
                      string(tail_smeared), string(interface_lost)))
        end
    end
end
# render_exact_snapshot_set is defined before the polynomial-degree
# contingency so every conditional rerun uses the same exact-time renderer.
if amr_escalation_requested[] && active_max_level < 7
    println("AMR_ESCALATION: late-time contacts remain under-resolved; " *
            "rerunning with local density-based max_level=7.")
    for reason in amr_escalation_reasons
        println("AMR_ESCALATION_REASON: ", reason)
    end
    escalated_run = run_rti_case(
        alpha_max=active_alpha_max,
        max_level=7,
        polynomial_degree=active_polydeg,
        source_terms=active_source_terms,
        output_directory=joinpath(@__DIR__, "data_npz_max_level_7"),
        label="max_level_7")
    sol = escalated_run.sol
    semi = escalated_run.semi
    dg = escalated_run.dg
    ode = escalated_run.ode
    callbacks = escalated_run.callbacks
    active_alpha_max = escalated_run.alpha_max
    active_max_level = escalated_run.max_level
    active_polydeg = escalated_run.polynomial_degree
    active_source_terms = escalated_run.source_terms
    active_run_directory = escalated_run.output_directory
    npz_out = active_run_directory
    # Replace all rendering data with the escalated run before final output.
    rendered_snapshots = render_exact_snapshot_set(npz_out)
    rho_minimum = minimum(minimum(snapshot.rho)
                          for snapshot in rendered_snapshots)
    rho_maximum = maximum(maximum(snapshot.rho)
                          for snapshot in rendered_snapshots)
    if rho_maximum <= rho_minimum
        padding = max(1e-12, 1e-6 * abs(rho_minimum))
        rho_minimum -= padding
        rho_maximum += padding
    end
    velocity_limit = maximum(maximum(abs.(snapshot.v))
                             for snapshot in rendered_snapshots)
    isfinite(velocity_limit) ||
        error("Non-finite escalated velocity scale.")
    velocity_limit = max(velocity_limit, 1e-12)
    current_metrics = Dict{Float64, Any}()
    current_contact_metrics = Dict{Float64, Any}()
    for snapshot in rendered_snapshots
        if snapshot.time in study_times
            metrics = interface_metrics(snapshot.rho)
            contact = contact_transition_metrics(snapshot.rho)
            current_metrics[snapshot.time] = metrics
            current_contact_metrics[snapshot.time] = contact
            Printf.@printf(
                "RESOLUTION_ESCALATED polydeg=%d max_level=%d t=%.2f interface_y=[%.6e,%.6e] spike_amplitude=%.6e bubble_amplitude=%.6e mixing_zone_width=%.6e median_contact_thickness=%.6e\n",
                active_polydeg, active_max_level, snapshot.time,
                metrics.interface_y_minimum, metrics.interface_y_maximum,
                metrics.spike_amplitude, metrics.bubble_amplitude,
                metrics.mixing_zone_width, contact.median_thickness)
            if run9_directory !== nothing
                baseline_data =
                    read_exact_npz_snapshot(run9_directory, snapshot.time)
                if baseline_data !== nothing
                    baseline_grid = rasterize_data(
                        Float64.(vec(baseline_data["x"])),
                        Float64.(vec(baseline_data["y"])),
                        Float64.(vec(baseline_data["rho"])),
                        visualization_nx, visualization_ny)
                    baseline_metrics = interface_metrics(baseline_grid)
                    Printf.@printf(
                        "RESOLUTION_STUDY_ESCALATED t=%.2f delta_spike=%.6e delta_bubble=%.6e delta_width=%.6e\n",
                        snapshot.time,
                        metrics.spike_amplitude -
                            baseline_metrics.spike_amplitude,
                        metrics.bubble_amplitude -
                            baseline_metrics.bubble_amplitude,
                        metrics.mixing_zone_width -
                            baseline_metrics.mixing_zone_width)
                end
            end
        end
    end
    println("Regenerating final summaries from the max_level=7 run...")
    summary_plot = plot(make_summary_panels(false)...,
                        layout=(2, 4),
                        size=(2400, 1800),
                        margin=5Plots.mm)
    savefig(summary_plot, joinpath(@__DIR__, "summary_all.png"))
    diagnostic_plot = plot(make_summary_panels(true)...,
                           layout=(2, 4),
                           size=(2700, 1900),
                           margin=5Plots.mm)
    savefig(diagnostic_plot,
            joinpath(@__DIR__, "summary_diagnostic.png"))
    Printf.@printf(
        "AMR_ESCALATION_COMPLETE polydeg=%d max_level=%d alpha_max=%.2f snapshot_directory=%s\n",
        active_polydeg, active_max_level, active_alpha_max,
        active_run_directory)
elseif amr_escalation_requested[]
    println("AMR_ESCALATION: max_level=7 is already active; no repeated rerun.")
end
println("Done.")
