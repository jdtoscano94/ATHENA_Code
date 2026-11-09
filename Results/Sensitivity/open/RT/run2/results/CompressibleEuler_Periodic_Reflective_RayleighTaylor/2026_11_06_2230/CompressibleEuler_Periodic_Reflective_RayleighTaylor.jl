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
# The RTI model uses the 2D compressible Euler equations with an ideal-gas
# equation of state (gamma = 1.4). Gravity is added as a source term in Cell 3.
equations = CompressibleEulerEquations2D(1.4)

# ## Cell 3: Initial Conditions and Source Terms (Lid-Driven Cavity)
# ## Problem Setup: Rayleigh–Taylor Instability
#
# A smoothed heavy-to-light density interface is initialized at y = 0.5.
# The piecewise pressure is hydrostatic away from the interface, and a
# single-mode vertical-velocity perturbation seeds the instability.
# Gravity acts in the positive y-direction.
#
# %%
@inline function initial_condition_rti(x, t, equations::CompressibleEulerEquations2D)
    x_coord, y = x

    # Heavy fluid (rho = 2) below the interface; light fluid (rho = 1) above.
    rho = 2.0 + (1.0 - 2.0) * 0.5 *
          (1.0 + tanh(50.0 * (y - 0.5)))

    p = y < 0.5 ? 2.0 * y + 1.0 : y + 1.5

    u = 0.0
    sound_speed = sqrt(equations.gamma * p / rho)
    v = -0.025 * sound_speed * cos(8.0 * pi * x_coord) *
        sin(pi * y)^6

    # prim2cons gives E = p/(gamma - 1) + rho*(u^2 + v^2)/2.
    return prim2cons(SVector(rho, u, v, p), equations)
end

@inline function source_terms_gravity(u, x, t,
                                      equations::CompressibleEulerEquations2D)
    g = 1.0
    rho, rho_v1, rho_v2, rho_e = u

    # rho_v2 = rho*v, so the final component is g*rho*v.
    return SVector(0.0, 0.0, g * rho, g * rho_v2)
end

# Conservative-state density used by the shock-capturing and AMR indicators.
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
#     * **Role**: Computes the numerical flux across element interfaces while explicitly resolving the contact wave.
#     * **Physics**: Reduces numerical smearing of the density interfaces that dominate the RTI evolution.
#
# 4.  **Indicator Function (`IndicatorHennemannGassner`)**:
#     * Dynamically computes the blending coefficient $\alpha \in [0, 1]$ element-wise.
#      * **Mechanism**: Monitors the spectral decay of energy in the highest modes of the polynomial expansion. If energy in high modes is large (indicating a discontinuity), $\alpha$ increases toward `alpha_max` (activating FV blending).
#
# %%
polydeg = 5
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.25,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_indicator)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=flux_ranocha,
                                                 volume_flux_fv=flux_lax_friedrichs)
dg = DGSEM(polydeg=polydeg, surface_flux=flux_hllc, volume_integral=volume_integral)
# %% [markdown]


# ## Cell 5: Computational Domain & Adaptive Mesh Topology
#
# ## Methodology: Forest-of-Octrees (`p4est`)
# Defines the spatial domain $\Omega$ and the underlying mesh topology using the **`p4est`** library. This framework represents the mesh as a forest of dynamic quadtrees (in 2D), enabling efficient, scalable parallel adaptive mesh refinement (AMR).
# 1. Geometric Domain:
#     * Physical domain Omega = [0, 0.25] x [0, 1].
#     * A single root quadtree covers the rectangular domain.
#
# 2. Discretization Topology (`P4estMesh`):
#     * Supports dynamic h-refinement through AMR.
#     * Retains the template initial refinement level and polynomial degree.
#
# 3. Boundary Topology:
#     * x-direction: periodic.
#     * y-direction: nonperiodic, with reflective walls imposed in Cell 6.

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
# Reflective/free-slip walls at y = 0 and y = 1.
# The periodic x-boundaries require no explicit boundary-condition entries.
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
                                      med_level=5, med_threshold=0.05,
                                      max_level=6, max_threshold=0.15)
# 3. Callback
amr_callback = AMRCallback(semi, amr_controller,
                           interval=20,
                           adapt_initial_condition=true)
# %% [markdown]


# ## Cell 8: Data I/O & Surrogate Interoperability
# ## Strategy: Full-State Cross-Language Serialization (.npz)
#
# Each snapshot contains the complete conservative state
# (rho, rho*u, rho*v, rho*E), together with primitive fields used for
# visualization. Files are written to a fresh run-local directory so that
# post-processing cannot accidentally consume results from an earlier run.
#
# %%
run_dir = mktempdir(@__DIR__; prefix="rti_run_", cleanup=false)
println("Fresh standalone run directory: ", run_dir)
mkpath(run_dir)
npz_out = joinpath(run_dir, "data_npz")
mkpath(npz_out)
dt_save = 0.1
visualization_times = [0.75, 1.5, 2.25, 3.0]
# Ground-truth cadence plus the additional times required for visualization.
# The time-integration cell uses these times as tstops.
npz_save_times = sort(unique(round.(vcat(collect(0.0:dt_save:3.0),
                                        visualization_times), digits=10)))
next_npz_save_index = Ref(1)
initial_mass = Ref{Float64}(NaN)
initial_modified_energy = Ref{Float64}(NaN)
function print_runtime_diagnostics(integrator)
    semi = integrator.p
    _, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    u = Trixi.wrap_array(integrator.u, semi)

    weights = solver.basis.weights
    node_coordinates = cache.elements.node_coordinates
    inverse_jacobian = cache.elements.inverse_jacobian

    state_finite = all(isfinite, integrator.u)
    derived_fields_finite = true

    rho_min, rho_max = Inf, -Inf
    p_min, p_max = Inf, -Inf
    mass = 0.0
    modified_energy = 0.0
    max_abs_v_y_neg = 0.0
    max_abs_v_y_pos = 0.0
    max_abs_rho_v_y_neg = 0.0
    max_abs_rho_v_y_pos = 0.0

    # LGL nodes include element boundaries, so the physical wall nodes are
    # available directly. Quadrature weights include each element's Jacobian.
    for element in axes(u, 4), j in axes(u, 3), i in axes(u, 2)
        u_node = Trixi.get_node_vars(u, equations, solver, i, j, element)
        primitive = cons2prim(u_node, equations)
        rho = primitive[1]
        v = primitive[3]
        p = primitive[4]
        rho_v = u_node[3]
        y = node_coordinates[2, i, j, element]

        derived_fields_finite = derived_fields_finite &&
                                all(isfinite, primitive) &&
                                isfinite(u_node[4])

        rho_min = min(rho_min, rho)
        rho_max = max(rho_max, rho)
        p_min = min(p_min, p)
        p_max = max(p_max, p)

        quadrature_weight = weights[i] * weights[j] /
                            inverse_jacobian[element]
        mass += quadrature_weight * rho

        # u_node[4] is E = rho*e. With g=1, E-rho*g*y includes
        # internal, kinetic, and gravitational potential energy.
        modified_energy += quadrature_weight * (u_node[4] - rho * y)

        if abs(y) <= 1.0e-12
            max_abs_v_y_neg = max(max_abs_v_y_neg, abs(v))
            max_abs_rho_v_y_neg = max(max_abs_rho_v_y_neg, abs(rho_v))
        elseif abs(y - 1.0) <= 1.0e-12
            max_abs_v_y_pos = max(max_abs_v_y_pos, abs(v))
            max_abs_rho_v_y_pos = max(max_abs_rho_v_y_pos, abs(rho_v))
        end
    end

    if !isfinite(initial_mass[])
        initial_mass[] = mass
        initial_modified_energy[] = modified_energy
    end

    mass_drift = mass - initial_mass[]
    energy_drift = modified_energy - initial_modified_energy[]
    relative_mass_drift = mass_drift /
                          max(abs(initial_mass[]), eps(Float64))
    relative_energy_drift = energy_drift /
                            max(abs(initial_modified_energy[]), eps(Float64))
    all_fields_finite = state_finite && derived_fields_finite &&
                        isfinite(mass) && isfinite(modified_energy)
    positivity = rho_min > 0.0 && p_min > 0.0

    println(Printf.@sprintf(
        "Diagnostics t=%.5f: finite=%s, positivity=%s, rho=[%.8e, %.8e], p=[%.8e, %.8e]",
        integrator.t, string(all_fields_finite), string(positivity),
        rho_min, rho_max, p_min, p_max))
    println(Printf.@sprintf(
        "  mass=%.16e, mass_drift=%.3e, relative_mass_drift=%.3e; modified_energy=%.16e, energy_drift=%.3e, relative_energy_drift=%.3e",
        mass, mass_drift, relative_mass_drift, modified_energy, energy_drift,
        relative_energy_drift))
    println(Printf.@sprintf(
        "  max|v| at y=0: %.3e; max|v| at y=1: %.3e; max|rho*v| at y=0: %.3e; max|rho*v| at y=1: %.3e",
        max_abs_v_y_neg, max_abs_v_y_pos,
        max_abs_rho_v_y_neg, max_abs_rho_v_y_pos))

    # Make the autoscale visualization quantitatively auditable at exactly
    # the four requested plotting times.
    if any(target_time -> abs(integrator.t - target_time) <= 1.0e-8,
           visualization_times)
        println(Printf.@sprintf(
            "Plot-time extrema t=%.5f: rho_min=%.16e, rho_max=%.16e, p_min=%.16e, p_max=%.16e",
            integrator.t, rho_min, rho_max, p_min, p_max))
    end

    # The final scheduled NPZ save is at t=3, so this table is generated from
    # the final state of the current fresh run.
    if abs(integrator.t - npz_save_times[end]) <= 1.0e-8
        println("Final verification table")
        println("Metric | Value")
        println("--- | ---")
        println("all_conserved_variables_finite | $(state_finite)")
        println("all_derived_fields_finite | $(derived_fields_finite)")
        println("positivity_rho_and_p | $(positivity)")
        println(Printf.@sprintf("min(rho) | %.16e", rho_min))
        println(Printf.@sprintf("min(p) | %.16e", p_min))
        println(Printf.@sprintf("relative_mass_drift | %.16e",
                                relative_mass_drift))
        println(Printf.@sprintf("relative_modified_energy_drift | %.16e",
                                relative_energy_drift))
        println(Printf.@sprintf("max_wall_abs_v_y0 | %.16e",
                                max_abs_v_y_neg))
        println(Printf.@sprintf("max_wall_abs_v_y1 | %.16e",
                                max_abs_v_y_pos))
        println(Printf.@sprintf("max_wall_normal_mass_flux_abs_rho_v_y0 | %.16e",
                                max_abs_rho_v_y_neg))
        println(Printf.@sprintf("max_wall_normal_mass_flux_abs_rho_v_y1 | %.16e",
                                max_abs_rho_v_y_pos))
    end

    return nothing
end
function save_npz_file(integrator)
    semi = integrator.p
    t = integrator.t
    print_runtime_diagnostics(integrator)
    # Reconstruct all conservative variables on the visualization grid.
    pd_cons = PlotData2D(integrator.u, semi,
                         solution_variables=cons2cons)
    pd_prim = PlotData2D(integrator.u, semi,
                         solution_variables=cons2prim)
    rho   = getindex.(pd_cons.data, 1)
    rho_u = getindex.(pd_cons.data, 2)
    rho_v = getindex.(pd_cons.data, 3)
    rho_e = getindex.(pd_cons.data, 4)
    # Primitive fields retained for direct plotting and interoperability.
    u = getindex.(pd_prim.data, 2)
    v = getindex.(pd_prim.data, 3)
    p = getindex.(pd_prim.data, 4)
    filename = joinpath(npz_out,
                        Printf.@sprintf("time_%08.5f.npz", t))
    npzwrite(filename, Dict("t"=>t,
                            "x"=>pd_cons.x, "y"=>pd_cons.y,
                            "rho"=>rho,
                            "rho_u"=>rho_u,
                            "rho_v"=>rho_v,
                            "rho_e"=>rho_e,
                            "u"=>u, "v"=>v, "p"=>p))
    println("Saved: $(basename(filename))")
    return nothing
end
function condition_save_npz(u, t, integrator)
    index = next_npz_save_index[]
    return index <= length(npz_save_times) &&
           t >= npz_save_times[index] - 1.0e-10
end
function affect_save_npz(integrator)
    save_npz_file(integrator)
    # Advance past every schedule entry reached at this integrator time.
    while next_npz_save_index[] <= length(npz_save_times) &&
          integrator.t >= npz_save_times[next_npz_save_index[]] - 1.0e-10
        next_npz_save_index[] += 1
    end
    return nothing
end
function init_save_npz(cb, u, t, integrator)
    next_npz_save_index[] = 1
    initial_mass[] = NaN
    initial_modified_energy[] = NaN
    affect_save_npz(integrator) # Save and diagnose the initial state at t = 0.
end
save_npz_callback = DiscreteCallback(condition_save_npz,
                                     affect_save_npz,
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
#     Here, we set `cfl=0.3` to ensure the method remains within the stability region of the Runge-Kutta scheme.
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
trixi_out = joinpath(run_dir, "trixi_output")
# Time-based full-state output for ground truth and restart interoperability.
save_solution = SaveSolutionCallback(dt=dt_save,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     output_directory=trixi_out,
                                     solution_variables=cons2cons)
stepsize_callback = StepsizeCallback(cfl=0.3)
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
            dt=1.0,
            adaptive=false,
            callback=callbacks,
            saveat=dt_save,
            tstops=npz_save_times,
            maxiters=1e7);
# %% [markdown]

# ## Cell 11: Data Regularization (Rasterization)
# ## Methodology: AMR-Aware Element-Patch Interpolation
# `PlotData2D` evaluates the DG polynomial on visualization nodes within
# each AMR element. Element views are determined from the actual array
# dimensions and validated before tensor-product interpolation. A scattered
# fallback handles layout-free data without leaving unfilled grid cells.

function _rti_barycentric_weights(nodes)
    weights = ones(Float64, length(nodes))
    for i in eachindex(nodes), j in eachindex(nodes)
        if i != j
            weights[i] /= nodes[i] - nodes[j]
        end
    end
    return weights
end

function _rti_barycentric_basis!(basis, nodes, weights, q)
    fill!(basis, 0.0)

    for k in eachindex(nodes)
        if q == nodes[k]
            basis[k] = 1.0
            return nothing
        end
    end

    denominator = 0.0
    for k in eachindex(nodes)
        basis[k] = weights[k] / (q - nodes[k])
        denominator += basis[k]
    end
    for k in eachindex(basis)
        basis[k] /= denominator
    end
    return nothing
end

# Return one element without flattening the complete PlotData2D arrays.
function _rti_patch_view(a, layout, nvis, element)
    if layout === :elements_last
        return view(a, :, :, element)
    elseif layout === :elements_first
        return view(a, element, :, :)
    elseif layout === :elements_middle
        return view(a, :, element, :)
    elseif layout === :element_columns
        return view(a, :, element)
    elseif layout === :element_rows
        return view(a, element, :)
    elseif layout === :columns_contiguous
        columns = ((element - 1) * nvis + 1):(element * nvis)
        return view(a, :, columns)
    elseif layout === :rows_contiguous
        rows = ((element - 1) * nvis + 1):(element * nvis)
        return view(a, rows, :)
    elseif layout === :columns_interleaved
        nelements = size(a, 2) ÷ nvis
        columns = element:nelements:(nvis * nelements)
        return view(a, :, columns)
    elseif layout === :rows_interleaved
        nelements = size(a, 1) ÷ nvis
        rows = element:nelements:(nvis * nelements)
        return view(a, rows, :)
    end
    error("Unknown PlotData2D element layout: $layout")
end

function _rti_layout_candidates(a)
    candidates = Tuple{Symbol, Int, Int}[]

    if ndims(a) == 3
        s = size(a)
        if s[1] == s[2] && s[1] > 1
            push!(candidates, (:elements_last, s[1], s[3]))
        end
        if s[2] == s[3] && s[2] > 1
            push!(candidates, (:elements_first, s[2], s[1]))
        end
        if s[1] == s[3] && s[1] > 1
            push!(candidates, (:elements_middle, s[1], s[2]))
        end
    elseif ndims(a) == 2
        s1, s2 = size(a)

        # Horizontally concatenated nvis-by-nvis element patches.
        if s1 >= 2 && s1 <= 64 && s2 % s1 == 0
            nelements = s2 ÷ s1
            push!(candidates, (:columns_contiguous, s1, nelements))
            push!(candidates, (:columns_interleaved, s1, nelements))
        end

        # Vertically concatenated nvis-by-nvis element patches.
        if s2 >= 2 && s2 <= 64 && s1 % s2 == 0
            nelements = s1 ÷ s2
            push!(candidates, (:rows_contiguous, s2, nelements))
            push!(candidates, (:rows_interleaved, s2, nelements))
        end

        # One flattened element per column or per row.
        nvis = isqrt(s1)
        if nvis > 1 && nvis * nvis == s1
            push!(candidates, (:element_columns, nvis, s2))
        end
        nvis = isqrt(s2)
        if nvis > 1 && nvis * nvis == s2
            push!(candidates, (:element_rows, nvis, s1))
        end
    end

    return unique(candidates)
end

# Convert one element view to sorted tensor-product nodes and a [y, x] patch.
# Returning `nothing` rejects an incorrect element slicing without throwing.
function _rti_tensor_patch(xb, yb, zb, nvis)
    xv = vec(Float64.(xb))
    yv = vec(Float64.(yb))
    zv = vec(Float64.(zb))

    length(xv) == nvis * nvis || return nothing
    length(yv) == length(xv) && length(zv) == length(xv) || return nothing

    xkeys = round.(xv, digits=14)
    ykeys = round.(yv, digits=14)
    xs = sort(unique(xkeys))
    ys = sort(unique(ykeys))

    length(xs) == nvis && length(ys) == nvis || return nothing

    patch = fill(NaN, nvis, nvis)
    for k in eachindex(xv)
        ix = searchsortedfirst(xs, xkeys[k])
        iy = searchsortedfirst(ys, ykeys[k])
        ix <= nvis && iy <= nvis || return nothing
        patch[iy, ix] = zv[k]
    end

    # Missing tensor-product combinations remain NaN. This also rejects
    # duplicate coordinate assignments and non-finite solution values.
    all(isfinite, patch) || return nothing
    return xs, ys, patch
end

function _rti_layout_is_valid(x, y, z, layout, nvis, nelements)
    for element in 1:nelements
        xb = _rti_patch_view(x, layout, nvis, element)
        yb = _rti_patch_view(y, layout, nvis, element)
        zb = _rti_patch_view(z, layout, nvis, element)
        if _rti_tensor_patch(xb, yb, zb, nvis) === nothing
            return false
        end
    end
    return true
end

# Layout-free fallback for scattered/triangulated PlotData2D output. Spatial
# bins avoid an O(number_of_pixels * number_of_samples) nearest-neighbor scan.
function _rti_scattered_raster(xv, yv, zv, x_grid, y_grid)
    nx = length(x_grid)
    ny = length(y_grid)
    x_min, x_max = extrema(x_grid)
    y_min, y_max = extrema(y_grid)

    bins = [Int[] for j in 1:ny, i in 1:nx]
    for k in eachindex(xv)
        i = clamp(floor(Int, (xv[k] - x_min) / (x_max - x_min) * nx) + 1,
                  1, nx)
        j = clamp(floor(Int, (yv[k] - y_min) / (y_max - y_min) * ny) + 1,
                  1, ny)
        push!(bins[j, i], k)
    end

    grid = Matrix{Float64}(undef, ny, nx)
    neighbors = Int[]
    max_radius = max(nx, ny)

    for j in 1:ny, i in 1:nx
        empty!(neighbors)

        # Add only the new square ring at each radius, so samples are never
        # duplicated. Twelve samples give a local interpolation stencil.
        for radius in 0:max_radius
            j_lo, j_hi = max(1, j - radius), min(ny, j + radius)
            i_lo, i_hi = max(1, i - radius), min(nx, i + radius)
            for jj in j_lo:j_hi, ii in i_lo:i_hi
                if max(abs(jj - j), abs(ii - i)) == radius
                    append!(neighbors, bins[jj, ii])
                end
            end
            if length(neighbors) >= 12
                break
            end
        end

        isempty(neighbors) && error("No scattered samples available for interpolation.")

        weighted_sum = 0.0
        weight_sum = 0.0
        exact_sum = 0.0
        exact_count = 0

        for k in neighbors
            dx = (xv[k] - x_grid[i]) / (x_max - x_min)
            dy = (yv[k] - y_grid[j]) / (y_max - y_min)
            distance_squared = dx * dx + dy * dy

            if distance_squared <= 1.0e-30
                # Average duplicate DG values at shared element boundaries.
                exact_sum += zv[k]
                exact_count += 1
            else
                weight = 1.0 / distance_squared
                weighted_sum += weight * zv[k]
                weight_sum += weight
            end
        end

        if exact_count > 0
            grid[j, i] = exact_sum / exact_count
        else
            weight_sum > 0.0 || error("Degenerate scattered interpolation stencil.")
            grid[j, i] = weighted_sum / weight_sum
        end
    end

    return grid
end

function rasterize_data(x, y, z, nx=256, ny=1024)
    if !(size(x) == size(y) && size(x) == size(z))
        error("PlotData2D coordinate and field arrays must have identical dimensions.")
    end
    if nx < 2 || ny < 2
        error("The rasterization grid must have at least two nodes per direction.")
    end

    xv = vec(Float64.(x))
    yv = vec(Float64.(y))
    zv = vec(Float64.(z))

    isempty(xv) && error("Cannot rasterize an empty PlotData2D array.")
    length(xv) == length(yv) && length(xv) == length(zv) ||
        error("PlotData2D coordinate and field lengths do not match.")
    if !(all(isfinite, xv) && all(isfinite, yv) && all(isfinite, zv))
        error("Non-finite coordinate or field value in PlotData2D input.")
    end

    x_min, x_max = 0.0, 0.25
    y_min, y_max = 0.0, 1.0
    x_grid = collect(range(x_min, x_max; length=nx))
    y_grid = collect(range(y_min, y_max; length=ny))

    # A candidate is accepted only after every element has been verified as
    # a complete tensor-product patch. No consecutive-block layout is assumed.
    selected = nothing
    for candidate in _rti_layout_candidates(x)
        layout, nvis, nelements = candidate
        if _rti_layout_is_valid(x, y, z, layout, nvis, nelements)
            selected = candidate
            break
        end
    end

    if selected === nothing
        grid = _rti_scattered_raster(xv, yv, zv, x_grid, y_grid)
        all(isfinite, grid) || error("Non-finite value in scattered plot grid.")
        return grid
    end

    layout, nvis, nelements = selected
    values_sum = zeros(Float64, ny, nx)
    values_count = zeros(Int, ny, nx)

    for element in 1:nelements
        xb = _rti_patch_view(x, layout, nvis, element)
        yb = _rti_patch_view(y, layout, nvis, element)
        zb = _rti_patch_view(z, layout, nvis, element)
        patch_data = _rti_tensor_patch(xb, yb, zb, nvis)
        patch_data === nothing &&
            error("PlotData2D element layout failed after validation.")
        xs, ys, patch = patch_data

        weights_x = _rti_barycentric_weights(xs)
        weights_y = _rti_barycentric_weights(ys)
        basis_x = Vector{Float64}(undef, nvis)
        basis_y = Vector{Float64}(undef, nvis)

        i_first = max(1, searchsortedfirst(x_grid, xs[1]))
        i_last = min(nx, searchsortedlast(x_grid, xs[end]))
        j_first = max(1, searchsortedfirst(y_grid, ys[1]))
        j_last = min(ny, searchsortedlast(y_grid, ys[end]))

        if i_first <= i_last && j_first <= j_last
            for j in j_first:j_last, i in i_first:i_last
                _rti_barycentric_basis!(basis_x, xs, weights_x, x_grid[i])
                _rti_barycentric_basis!(basis_y, ys, weights_y, y_grid[j])

                value = 0.0
                for jj in 1:nvis, ii in 1:nvis
                    value += basis_y[jj] * patch[jj, ii] * basis_x[ii]
                end

                # Shared element faces and corners are averaged, avoiding
                # visible seams without drawing any element-edge lines.
                values_sum[j, i] += value
                values_count[j, i] += 1
            end
        end
    end

    grid = Matrix{Float64}(undef, ny, nx)
    for j in 1:ny, i in 1:nx
        if values_count[j, i] > 0
            grid[j, i] = values_sum[j, i] / values_count[j, i]
        else
            # Roundoff/boundary fallback: every pixel must receive a value.
            best_index = 1
            best_distance = Inf
            for k in eachindex(xv)
                distance = ((xv[k] - x_grid[i]) / (x_max - x_min))^2 +
                           ((yv[k] - y_grid[j]) / (y_max - y_min))^2
                if distance < best_distance
                    best_distance = distance
                    best_index = k
                end
            end
            grid[j, i] = zv[best_index]
        end
    end

    all(isfinite, grid) || error("Non-finite value in interpolated plot grid.")
    return grid
end
# %% [markdown]

# ## Cell 12: Post-Processing & Temporal Diagnostics
# ## RTI Visualization
#
# Density (rho) and vertical velocity (v) are plotted at the four requested
# times. Each panel uses automatic color scaling, with no colorbar or fixed
# colormap. The final layout has two rows (rho, v) and four columns (time).
#
# %%
println("Generating RTI density and vertical-velocity summary...")
plots_array = []
visualization_times = (0.75, 1.5, 2.25, 3.0)
style = (
    aspect_ratio = :equal,
    xlims = (0.0, 0.25),
    ylims = (0.0, 1.0),
    yflip = false, # y increases upward: y=0 at the bottom, y=1 at the top.
    axis = nothing,
    framestyle = :none,
    legend = false,
    colorbar = false,
    margin = 2Plots.mm
)
nx_plot, ny_plot = 256, 1024
x_plot = collect(range(0.0, 0.25; length=nx_plot))
y_plot = collect(range(0.0, 1.0; length=ny_plot))
# Field-major ordering gives all density panels in row 1 and all vertical-
# velocity panels in row 2. Each heatmap receives a complete ny-by-nx array,
# so no scatter points, element edges, transparent rows, or mesh bands appear.
for (variable, label) in (("rho", "ρ"), ("v", "v"))
    for target_time in visualization_times
        filename = joinpath(
            npz_out,
            Printf.@sprintf("time_%08.5f.npz", target_time)
        )
        isfile(filename) ||
            error("Missing requested snapshot: $(basename(filename))")
        println(filename)
        data = npzread(filename)
        field_grid = rasterize_data(data["x"], data["y"], data[variable],
                                    nx_plot, ny_plot)
        size(field_grid) == (ny_plot, nx_plot) ||
            error("Interpolated field has incorrect orientation or dimensions.")
        push!(plots_array,
              heatmap(x_plot, y_plot, field_grid;
                      title="$(label), t=$(target_time)",
                      style...))
    end
end
println("Saving summary_all.png...")
final_plot = plot(plots_array...,
                  layout=(2, 4),
                  size=(1600, 2200),
                  margin=5Plots.mm)
savefig(final_plot, joinpath(run_dir, "summary_all.png"))
println("Done.")
