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
#
# ## Governing Equations: Hyperbolic-Parabolic System
# Defines the mixed hyperbolic-parabolic system using a modular composition of flux terms:
# $$
# \frac{\partial \mathbf{u}}{\partial t} + \nabla \cdot \left( \mathbf{f}_{inv}(\mathbf{u}) - \mathbf{f}_{visc}(\mathbf{u}, \nabla \mathbf{u}) \right) = \mathbf{0}
# $$
#
# 1.  **Hyperbolic Core (`CompressibleEulerEquations2D`)**:
#     * Defines the inviscid flux $\mathbf{f}_{inv}(\mathbf{u})$ (Conservation of Mass, Momentum, Energy).
#     * **Equation of State**: Ideal Gas Law with $\gamma = 1.4$.
#
# 2.  **Parabolic Extension (`CompressibleNavierStokesDiffusion2D`)**:
#     * Augments the core with viscous flux $\mathbf{f}_{visc}$.
#     * **Constitutive Relations**: Newtonian fluid ($\mu = 0.001$) and Fourier heat conduction ($Pr = 0.72$).
#
#     * **Viscous Numerical Scheme: Bassi-Rebay 1 (BR1)**
#         * *Implicit Default*: By initializing this struct without `gradient_variables`, Trixi defaults to the **BR1** formulation.
#         * **Methodology**: A Local Discontinuous Galerkin (LDG) approach. It lifts the gradients using central fluxes for both the auxiliary variable ($\nabla \mathbf{u}$) and the viscous flux.
#         * **Alternatives**: To use entropy-stable gradients (Ultraweak/chain-rule compliant), one would pass `gradient_variables=GradientVariablesEntropy()` to this constructor.
# %%
equations = CompressibleEulerEquations2D(1.4)
# %% [markdown]

# ## Cell 3: Initial Conditions and Source Terms (Lid-Driven Cavity)
#
# ## Problem Setup: Low-Mach Number Benchmark
# Configures the standard **Lid-Driven Cavity** benchmark. Since the underlying solver is compressible, we initialize the domain in the **low-Mach number limit** ($Ma = 0.1$) to approximate incompressible flow behavior (pseudo-compressibility regime).
#
# ### 1. Initial State $\mathbf{U}(\mathbf{x}, 0)$
# The domain is initialized at rest with constant density. Pressure is scaled to satisfy the Mach number constraint via the speed of sound relation $c = \sqrt{\gamma p / \rho}$:
# $$
# p = \frac{1}{\gamma Ma^2}, \quad \mathbf{v} = \mathbf{0}, \quad \rho = 1.0
# $$
# * **Transformation**: The primitive state $(\rho, \mathbf{v}, p)$ is immediately mapped to the conservative state vector $\mathbf{u} = [\rho, \rho v_1, \rho v_2, \rho E]^T$ required by the hyperbolic solver.
#
# ### 2. Source Terms $\mathbf{S}(\mathbf{u})$
# Defines the RHS source vector for the balance law $\partial_t \mathbf{u} + \nabla \cdot \mathbf{f} = \mathbf{S}$.
# * **Body Forces**: Set to zero ($\mathbf{S} = \mathbf{0}$) for this benchmark, as the flow is driven purely by the moving lid boundary condition (shear-driven), not by external fields like gravity.
#
# %%
@inline function initial_condition_rayleigh_taylor(x, t,
                                                   equations::CompressibleEulerEquations2D)
    x1, x2 = x

    rho = 2.0 + (1.0 - 2.0) * 0.5 * (1.0 + tanh(50.0 * (x2 - 0.5)))
    p = x2 < 0.5 ? 2.0 * x2 + 1.0 : x2 + 1.5

    u = 0.0
    c = sqrt(equations.gamma * p / rho)
    v = -0.025 * c * cos(8.0 * pi * x1) * sin(pi * x2)^6

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
#         \mathbf{f}_{numerical} = (1 - \alpha) \mathbf{f}_{DG} + \alpha \mathbf{f}_{FV}
#         $$
#     * **Numerical Fluxes**: The DG volume flux, FV volume flux, and DGSEM surface flux use their constructor defaults without explicit overrides.
#
# 3.  **Indicator Function (`IndicatorHennemannGassner`)**:
#     * Dynamically computes the blending coefficient $\alpha \in [0, 1]$ element-wise using the default shock-capturing strength.
#     * **Mechanism**: Monitors the spectral decay of energy in the highest modes of the polynomial expansion. Increased high-mode energy activates additional FV blending near discontinuities.
#     * **Sensed Variable**: Density, matching the AMR indicator.
#
# %%
# Polynomial degree N = 3, as documented above.
polydeg = 3
basis = LobattoLegendreBasis(polydeg)

# Preserve the default shock-capturing strength and numerical fluxes.
# Only the required density indicator variable is selected explicitly.
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         variable=density_indicator)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc)
dg = DGSEM(polydeg=polydeg, volume_integral=volume_integral)
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
                 periodicity=(true, false)) # Periodic in x; reflective walls in y
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
# %%
# Reflective free-slip walls at y = 0 and y = 1.
# The x-boundaries are periodic through the mesh topology.
boundary_conditions = Dict(
    :y_pos => boundary_condition_slip_wall,
    :y_neg => boundary_condition_slip_wall
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
                                      med_threshold=0.1,
                                      max_threshold=0.3)
# 3. Callback
amr_callback = AMRCallback(semi, amr_controller,
                           interval=20,
                           adapt_initial_condition=true)
# %% [markdown]

# ## Cell 8: serializes each element contiguously with i as the fastest index.
# Full-state NPZ export. The Cartesian interpolation helper is completed in Cell 11.
import DiffEqBase

dt_save = 0.1
npz_out = joinpath(@__DIR__, "npz_output")
mkpath(npz_out)

# Retain the Ground Truth cadence and add the two visualization times that are
# not integer multiples of dt_save. Duplicates such as t=1.5 and t=3.0 are removed.
npz_save_times = sort(unique(vcat(collect(0.0:dt_save:3.0),
                                  [0.75, 1.5, 2.25, 3.0])))

function _save_npz_snapshot(integrator)
    semi = integrator.p
    _, local_equations, _, cache = Trixi.mesh_equations_solver_cache(semi)
    state = Trixi.wrap_array(integrator.u, semi)

    ndims(state) == 4 && size(state, 1) == 4 ||
        error("NPZ export expects a four-variable, two-dimensional DGSEM state.")

    n_nodes = size(state, 2)
    n_elements = size(state, 4)
    size(state, 3) == n_nodes ||
        error("NPZ export expects a tensor-product DGSEM state.")

    n_points = n_nodes^2 * n_elements
    n_points > 0 || error("Cannot save an empty DG snapshot.")

    # These containers are rebuilt/resized by Trixi after AMR. Extracting them
    # anew at every callback avoids any assumption of a static mesh.
    coordinates = cache.elements.node_coordinates
    coordinate_dimensions_valid =
        ndims(coordinates) == 4 &&
        size(coordinates, 1) == 2 &&
        size(coordinates, 2) == n_nodes &&
        size(coordinates, 3) == n_nodes &&
        size(coordinates, 4) == n_elements
    coordinate_dimensions_valid ||
        error("The AMR solution and node-coordinate containers are inconsistent.")

    x = Vector{Float64}(undef, n_points)
    y = Vector{Float64}(undef, n_points)
    rho = Vector{Float64}(undef, n_points)
    rho_u = Vector{Float64}(undef, n_points)
    rho_v = Vector{Float64}(undef, n_points)
    total_energy = Vector{Float64}(undef, n_points)

    # Serialize each element contiguously with the first DG node index fastest.
    # This is the ordering consumed by interpolate_dg_to_cartesian in Cell 11.
    for element in 1:n_elements
        offset = (element - 1) * n_nodes^2
        for j in 1:n_nodes, i in 1:n_nodes
            point = offset + (j - 1) * n_nodes + i
            x[point] = coordinates[1, i, j, element]
            y[point] = coordinates[2, i, j, element]
            rho[point] = state[1, i, j, element]
            rho_u[point] = state[2, i, j, element]
            rho_v[point] = state[3, i, j, element]
            total_energy[point] = state[4, i, j, element]
        end
    end

    velocity_u = rho_u ./ rho
    velocity_v = rho_v ./ rho
    pressure = (local_equations.gamma - 1) .*
               (total_energy .-
                0.5 .* (rho_u.^2 .+ rho_v.^2) ./ rho)

    t = Float64(integrator.t)
    arrays = (x, y, rho, rho_u, rho_v, total_energy,
              velocity_u, velocity_v, pressure)

    dimensions_valid = all(array -> length(array) == n_points, arrays)
    all_finite = isfinite(t) &&
                 all(array -> all(isfinite, array), arrays)

    x_range = extrema(x)
    y_range = extrema(y)
    rho_range = extrema(rho)
    v_range = extrema(velocity_v)

    coordinate_tolerance = 1e-10
    coordinates_valid =
        x_range[1] < x_range[2] &&
        y_range[1] < y_range[2] &&
        x_range[1] >= -coordinate_tolerance &&
        x_range[2] <= 0.25 + coordinate_tolerance &&
        y_range[1] >= -coordinate_tolerance &&
        y_range[2] <= 1.0 + coordinate_tolerance

    rho_positive = all(rho .> 0.0)
    p_positive = all(pressure .> 0.0)

    println("NPZ snapshot validation at t=", t)
    println("  state dimensions: ", size(state),
            "; points per exported array: ", n_points)
    println("  coordinate ranges: x=", x_range, ", y=", y_range)
    println("  extrema: rho=", rho_range, ", v=", v_range)
    println("  all values finite=", all_finite)
    println("  rho > 0=", rho_positive, ", p > 0=", p_positive)
    println("  coordinates inside domain=", coordinates_valid)

    if !(dimensions_valid && all_finite && coordinates_valid &&
         rho_positive && p_positive)
        error("Invalid saved snapshot at t=$t: " *
              "dimensions_valid=$dimensions_valid, all_finite=$all_finite, " *
              "coordinates_valid=$coordinates_valid, " *
              "rho_positive=$rho_positive, p_positive=$p_positive. " *
              "Aborting at the first invalid saved state; investigate positivity, " *
              "wall fluxes, and the gravity-source implementation before changing CFL.")
    end

    filename = joinpath(npz_out,
                        Printf.@sprintf("time_%08.5f.npz", t))

    # rho, rho_u, rho_v, and E constitute the complete conservative state.
    # Primitive fields are included for validation and visualization.
    payload = Dict{String, Any}(
        "t" => t,
        "x" => x,
        "y" => y,
        "rho" => rho,
        "rho_u" => rho_u,
        "rho_v" => rho_v,
        "E" => total_energy,
        "u" => velocity_u,
        "v" => velocity_v,
        "p" => pressure
    )

    npzwrite(filename, payload)
    (isfile(filename) && filesize(filename) > 0) ||
        error("NPZ snapshot was not written successfully: $filename")
    println("Saved NPZ snapshot: ", filename)

    return nothing
end

function _make_save_npz_callback(save_times)
    next_save_index = Ref(1)

    function condition(u, t, integrator)
        index = next_save_index[]
        index <= length(save_times) || return false
        tolerance = 1e-12 * max(1.0, abs(t))
        return t >= save_times[index] - tolerance
    end

    function affect!(integrator)
        index = next_save_index[]
        index <= length(save_times) || return nothing

        target_time = save_times[index]
        tolerance = 1e-12 * max(1.0, abs(integrator.t))
        isapprox(integrator.t, target_time; atol=tolerance, rtol=0.0) ||
            error("NPZ export missed scheduled save time $target_time; " *
                  "the integrator stopped at $(integrator.t).")

        _save_npz_snapshot(integrator)
        next_save_index[] = index + 1
        return nothing
    end

    function initialize(callback, u, t, integrator)
        next_save_index[] = 1
        tolerance = 1e-12 * max(1.0, abs(t))

        while next_save_index[] <= length(save_times) &&
              save_times[next_save_index[]] < t - tolerance
            next_save_index[] += 1
        end

        # Explicitly export the initial state, then schedule every remaining save.
        if next_save_index[] <= length(save_times) &&
           abs(t - save_times[next_save_index[]]) <= tolerance
            _save_npz_snapshot(integrator)
            next_save_index[] += 1
        end

        first_future_index = next_save_index[]
        for index in first_future_index:length(save_times)
            DiffEqBase.add_tstop!(integrator, save_times[index])
        end

        return nothing
    end

    return DiffEqBase.DiscreteCallback(condition, affect!;
                                       initialize=initialize,
                                       save_positions=(false, false))
end

save_npz_callback = _make_save_npz_callback(npz_save_times)


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
# We save fewer solution files to save disk space, but analysis runs often
trixi_out = joinpath(@__DIR__, "trixi_output")
# Time-based solution output at the Ground Truth saving cadence
save_solution = SaveSolutionCallback(dt=dt_save,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     output_directory=trixi_out)
stepsize_callback = StepsizeCallback()
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        amr_callback,
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
            adaptive=false,
            dt=1.0,
            callback=callbacks,
            saveat=0.01,
            maxiters=10_000_000);
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
function _lagrange_basis_1d(q, nodes)
    values = Vector{Float64}(undef, length(nodes))
    for i in eachindex(nodes)
        value = 1.0
        for k in eachindex(nodes)
            if k != i
                value *= (q - nodes[k]) / (nodes[i] - nodes[k])
            end
        end
        values[i] = value
    end
    return values
end
function interpolate_dg_to_cartesian(x, y, z;
                                     nx=201, ny=801,
                                     n_nodes=polydeg + 1)
    n_points = length(x)
    n_points > 0 || error("Cannot interpolate an empty DG snapshot.")
    length(y) == n_points && length(z) == n_points ||
        error("DG interpolation requires equally sized x, y, and field arrays.")
    nx >= 2 && ny >= 2 ||
        error("The Cartesian interpolation grid needs at least two points per direction.")
    n_nodes >= 2 || error("Invalid DG node count: $n_nodes")
    all(isfinite, x) && all(isfinite, y) && all(isfinite, z) ||
        error("DG interpolation received nonfinite coordinates or field values.")
    # The element count is obtained from each snapshot, so AMR may change the
    # mesh without requiring any change to this interpolation routine.
    points_per_element = n_nodes^2
    n_points % points_per_element == 0 ||
        error("The snapshot does not contain complete $(n_nodes)×$(n_nodes) DG elements.")
    n_elements = div(n_points, points_per_element)

    x_grid = collect(range(0.0, 0.25; length=nx))
    y_grid = collect(range(0.0, 1.0; length=ny))
    grid_sum = zeros(Float64, ny, nx)
    grid_count = zeros(Int, ny, nx)

    for element in 1:n_elements
        offset = (element - 1) * points_per_element
        element_indices = (offset + 1):(offset + points_per_element)
        x_min, x_max = extrema(view(x, element_indices))
        y_min, y_max = extrema(view(y, element_indices))

        # Recover the one-dimensional LGL nodes of this rectangular element.
        # The exporter serializes i fastest, followed by j.
        x_element = [x[offset + i] for i in 1:n_nodes]
        y_element = [y[offset + (j - 1) * n_nodes] for j in 1:n_nodes]

        x_tol = 32 * eps(Float64) * max(1.0, abs(x_min), abs(x_max))
        y_tol = 32 * eps(Float64) * max(1.0, abs(y_min), abs(y_max))

        # Restrict evaluation to Cartesian points covered by this element.
        i_lo = searchsortedfirst(x_grid, x_min - x_tol)
        i_hi = searchsortedlast(x_grid, x_max + x_tol)
        j_lo = searchsortedfirst(y_grid, y_min - y_tol)
        j_hi = searchsortedlast(y_grid, y_max + y_tol)

        if i_lo <= i_hi && j_lo <= j_hi
            lx_values = [
                _lagrange_basis_1d(clamp(x_grid[i], x_min, x_max), x_element)
                for i in i_lo:i_hi
            ]
            ly_values = [
                _lagrange_basis_1d(clamp(y_grid[j], y_min, y_max), y_element)
                for j in j_lo:j_hi
            ]

            for j_local in eachindex(ly_values)
                j_grid = j_lo + j_local - 1
                ly = ly_values[j_local]

                for i_local in eachindex(lx_values)
                    i_grid = i_lo + i_local - 1
                    lx = lx_values[i_local]
                    value = 0.0

                    for j in 1:n_nodes, i in 1:n_nodes
                        node_index = offset + (j - 1) * n_nodes + i
                        value += z[node_index] * lx[i] * ly[j]
                    end

                    # Adjacent DG elements contribute independently at shared
                    # interfaces; averaging avoids order-dependent selection.
                    grid_sum[j_grid, i_grid] += value
                    grid_count[j_grid, i_grid] += 1
                end
            end
        end
    end

    grid = Matrix{Float64}(undef, ny, nx)
    missing_count = 0

    for j in 1:ny, i in 1:nx
        if grid_count[j, i] == 0
            grid[j, i] = NaN
            missing_count += 1
        else
            grid[j, i] = grid_sum[j, i] / grid_count[j, i]
        end
    end

    missing_count == 0 ||
        error("DG-to-Cartesian interpolation left $missing_count uncovered grid points; " *
              "refusing to generate a partially blank heatmap.")
    all(isfinite, grid) ||
        error("DG-to-Cartesian interpolation produced nonfinite grid values.")
    size(grid) == (length(y_grid), length(x_grid)) ||
        error("Incorrect grid orientation: expected rows=y and columns=x.")

    return x_grid, y_grid, grid
end


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
# %%
println("Generating Rayleigh-Taylor summary plots...")
plot_times = (0.75, 1.5, 2.25, 3.0)
function validate_plot_snapshot(t_val, data, x, y, rho, u, v, p, E)
    n_points = length(x)
    dimensions_valid = n_points > 0 &&
                       length(y) == n_points &&
                       length(rho) == n_points &&
                       length(u) == n_points &&
                       length(v) == n_points &&
                       length(p) == n_points &&
                       length(E) == n_points
    println("Pre-plot validation at t=", t_val)
    println("  array dimensions: x=", size(data["x"]),
            ", y=", size(data["y"]),
            ", rho=", size(data["rho"]),
            ", u=", size(data["u"]),
            ", v=", size(data["v"]),
            ", p=", size(data["p"]),
            ", E=", size(data["E"]))
    dimensions_valid ||
        error("Invalid saved snapshot at t=$t_val: empty or inconsistent array dimensions.")
    x_range = extrema(x)
    y_range = extrema(y)
    rho_range = extrema(rho)
    v_range = extrema(v)
    p_range = extrema(p)
    all_finite = isfinite(t_val) &&
                 all(isfinite, x) && all(isfinite, y) &&
                 all(isfinite, rho) && all(isfinite, u) &&
                 all(isfinite, v) && all(isfinite, p) &&
                 all(isfinite, E)
    rho_positive = all(rho .> 0.0)
    p_positive = all(p .> 0.0)
    coordinate_tolerance = 1e-10
    coordinates_valid = x_range[1] < x_range[2] &&
                        y_range[1] < y_range[2] &&
                        x_range[1] >= -coordinate_tolerance &&
                        x_range[2] <= 0.25 + coordinate_tolerance &&
                        y_range[1] >= -coordinate_tolerance &&
                        y_range[2] <= 1.0 + coordinate_tolerance
    println("  coordinate ranges: x=", x_range, ", y=", y_range)
    println("  extrema: rho=", rho_range, ", v=", v_range,
            ", p=", p_range)
    println("  all values finite=", all_finite)
    println("  rho > 0=", rho_positive, ", p > 0=", p_positive)
    println("  coordinates inside domain=", coordinates_valid)
    if !(all_finite && coordinates_valid && rho_positive && p_positive)
        error("Invalid saved snapshot at t=$t_val: all_finite=$all_finite, " *
              "coordinates_valid=$coordinates_valid, " *
              "rho_positive=$rho_positive, p_positive=$p_positive. " *
              "Aborting at the first invalid plotting state; investigate positivity, " *
              "wall fluxes, and the gravity-source implementation before changing CFL.")
    end
    return nothing
end
function finite_color_limits(field)
    all(isfinite, field) ||
        error("Cannot derive color limits from nonfinite field data.")
    lower, upper = extrema(field)
    # Keep a constant field renderable without introducing a hardcoded color range.
    if lower == upper
        delta = max(abs(lower), 1.0) * 1e-12
        return (lower - delta, upper + delta)
    end
    return (lower, upper)
end
function load_npz_snapshot(target_time)
    filename = joinpath(npz_out,
                        Printf.@sprintf("time_%08.5f.npz", target_time))
    isfile(filename) ||
        error("Required solution snapshot is missing: $filename")

    data = npzread(filename)
    raw_time = data["t"]
    t_val = raw_time isa Number ? Float64(raw_time) : Float64(only(raw_time))

    isapprox(t_val, target_time; atol=1e-8, rtol=0.0) ||
        error("Snapshot time mismatch in $filename: expected $target_time, " *
              "but the file contains $t_val.")

    return (
        data=data,
        t=t_val,
        x=vec(data["x"]),
        y=vec(data["y"]),
        rho=vec(data["rho"]),
        u=vec(data["u"]),
        v=vec(data["v"]),
        p=vec(data["p"]),
        E=vec(data["E"])
    )
end

# Validate every saved state in chronological order before creating any figure.
# Thus, the first invalid saved state is reported immediately.
println("Validating all saved NPZ snapshots before plotting...")
for target_time in npz_save_times
    saved = load_npz_snapshot(target_time)
    validate_plot_snapshot(saved.t, saved.data,
                           saved.x, saved.y,
                           saved.rho, saved.u, saved.v,
                           saved.p, saved.E)
end
println("Validated ", length(npz_save_times), " saved NPZ snapshots.")

snapshots = Vector{Any}(undef, length(plot_times))

for (snapshot_index, target_time) in enumerate(plot_times)
    saved = load_npz_snapshot(target_time)
    t_val = saved.t
    x = saved.x
    y = saved.y
    rho = saved.rho
    v = saved.v

    x_grid, y_grid, rho_grid =
        interpolate_dg_to_cartesian(x, y, rho)
    x_grid_v, y_grid_v, v_grid =
        interpolate_dg_to_cartesian(x, y, v)

    x_grid_v == x_grid && y_grid_v == y_grid ||
        error("Density and velocity were interpolated onto different grids.")

    orientation_valid =
        size(rho_grid) == (length(y_grid), length(x_grid)) &&
        size(v_grid) == (length(y_grid), length(x_grid))
    orientation_valid ||
        error("Incorrect heatmap orientation at t=$t_val: rows must correspond " *
              "to y and columns to x.")

    all(isfinite, rho_grid) && all(isfinite, v_grid) ||
        error("Nonfinite interpolated plotting data at t=$t_val.")

    rho_clims = finite_color_limits(rho_grid)
    v_clims = finite_color_limits(v_grid)

    println("  fixed-grid dimensions: rho=", size(rho_grid),
            ", v=", size(v_grid))
    println("  fixed-grid coordinate ranges: x=", extrema(x_grid),
            ", y=", extrema(y_grid))
    println("  fixed-grid extrema: rho=", extrema(rho_grid),
            ", v=", extrema(v_grid))

    snapshots[snapshot_index] = (
        t=t_val,
        x=x_grid,
        y=y_grid,
        rho=rho_grid,
        v=v_grid,
        rho_clims=rho_clims,
        v_clims=v_clims
    )
end
# Render and verify the standalone final-time density plot before constructing
# the composite figure.
debug_snapshot = snapshots[end]
isapprox(debug_snapshot.t, 3.0; atol=1e-8, rtol=0.0) ||
    error("The debug snapshot is not the required t=3.0 state.")
debug_title = Printf.@sprintf("ρ, t=%.2f", debug_snapshot.t)
debug_rho_plot = heatmap(debug_snapshot.x, debug_snapshot.y,
                         debug_snapshot.rho;
                         title=debug_title,
                         xlabel="x", ylabel="y",
                         xlims=(0.0, 0.25), ylims=(0.0, 1.0),
                         clims=debug_snapshot.rho_clims,
                         color=:viridis,
                         colorbar=false,
                         legend=false,
                         aspect_ratio=:equal,
                         framestyle=:box,
                         size=(800, 1000))
debug_path = joinpath(@__DIR__, "debug_rho_t3.png")
savefig(debug_rho_plot, debug_path)
(isfile(debug_path) && filesize(debug_path) > 0) ||
    error("debug_rho_t3.png was not written successfully")
println("Saved: $debug_path")
# Explicitly populate density in subplots 1-4 (row 1) and vertical velocity
# in subplots 5-8 (row 2).
println("Saving summary_all.png...")
final_plot = plot(layout=(2, 4), size=(2000, 1200),
                  margin=4Plots.mm)
for column in 1:length(plot_times)
    snapshot = snapshots[column]
    time_label = Printf.@sprintf("%.2f", snapshot.t)
    heatmap!(final_plot, snapshot.x, snapshot.y, snapshot.rho;
             subplot=column,
             title="ρ, t=$time_label",
             xlabel="x", ylabel="y",
             xlims=(0.0, 0.25), ylims=(0.0, 1.0),
             clims=snapshot.rho_clims,
             color=:viridis,
             colorbar=false,
             legend=false,
             aspect_ratio=:equal,
             framestyle=:box)
    heatmap!(final_plot, snapshot.x, snapshot.y, snapshot.v;
             subplot=length(plot_times) + column,
             title="v, t=$time_label",
             xlabel="x", ylabel="y",
             xlims=(0.0, 0.25), ylims=(0.0, 1.0),
             clims=snapshot.v_clims,
             color=:plasma,
             colorbar=false,
             legend=false,
             aspect_ratio=:equal,
             framestyle=:box)
end
length(final_plot.subplots) == 8 ||
    error("The populated summary figure does not contain eight subplots.")
summary_path = joinpath(@__DIR__, "summary_all.png")
savefig(final_plot, summary_path)
(isfile(summary_path) && filesize(summary_path) > 0) ||
    error("summary_all.png was not written successfully")
println("Saved: $summary_path")
println("Done.")
