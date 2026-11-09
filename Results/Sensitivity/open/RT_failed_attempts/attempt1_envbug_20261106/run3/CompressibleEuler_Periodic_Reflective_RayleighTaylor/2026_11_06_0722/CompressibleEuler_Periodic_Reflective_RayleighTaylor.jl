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
import Pkg

# LoopVectorization 0.12.171 expects VectorizationBase.transposeshuffle.
# Pin a compatible pair before Trixi loads its vectorized kernels; newer
# incompatible VectorizationBase versions cause the failure reported in Cell 6.
Pkg.add([
    Pkg.PackageSpec(name="LoopVectorization", version="0.12.171"),
    Pkg.PackageSpec(name="VectorizationBase", version="0.21.70"),
])
Pkg.pin("LoopVectorization")
Pkg.pin("VectorizationBase")

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
# Defines the hyperbolic balance law
# $$
# \frac{\partial \mathbf{u}}{\partial t}
# + \nabla \cdot \mathbf{f}(\mathbf{u}) = \mathbf{S}(\mathbf{u}),
# $$
# where $\mathbf{f}$ contains the inviscid mass, momentum, and energy fluxes.
# The gravitational source $\mathbf{S}$ is defined separately with the RTI setup.
#
# * **Equations (`CompressibleEulerEquations2D`)**: Conservation of mass,
#   momentum, and total energy in two spatial dimensions.
# * **Equation of State**: Ideal gas with $\gamma = 1.4$.
# * **Inviscid model**: No viscous or heat-conduction operator is required.

# %%
equations = CompressibleEulerEquations2D(1.4)
# %% [markdown]

# ## Cell 3: Initial Conditions and Source Terms (Lid-Driven Cavity)
# ## Problem Setup: Single-Mode Rayleigh–Taylor Instability
#
# Initializes a heavy fluid below a light fluid across a smoothed interface at
# $y = 0.5$. The piecewise pressure satisfies $\partial_y p = \rho g$ in the
# heavy and light regions, with gravity $g = 1$ acting in the $+y$ direction.
# A single-wavelength vertical-velocity perturbation seeds the instability.
#
# The conservative state is
# $\mathbf{u} = [\rho, \rho u, \rho v, E]^T$, where
# $E = p/(\gamma-1) + \frac{1}{2}\rho v^2$ because the initial horizontal
# velocity is zero.
# %%
@inline function initial_condition_rti(x, t, equations::CompressibleEulerEquations2D)
    x_coord, y = x
    gamma = equations.gamma
    # Smoothed material interface: heavy fluid below, light fluid above.
    rho = 2.0 + (1.0 - 2.0) * 0.5 * (1.0 + tanh(50.0 * (y - 0.5)))
    # Piecewise hydrostatic pressure.
    p = y < 0.5 ? 2.0 * y + 1.0 : y + 1.5
    # Single-mode perturbation with exactly one wavelength in x.
    u = 0.0
    sound_speed = sqrt(gamma * p / rho)
    v = -0.025 * sound_speed * cos(8.0 * pi * x_coord) * sin(pi * y)^6
    total_energy = p / (gamma - 1.0) + 0.5 * rho * v^2
    return SVector(rho, rho * u, rho * v, total_energy)
end
@inline function initial_condition_rti_control(x, t,
                                               equations::CompressibleEulerEquations2D)
    _, y = x
    gamma = equations.gamma

    # Identical material interface and pressure background as the main run.
    rho = 2.0 + (1.0 - 2.0) * 0.5 * (1.0 + tanh(50.0 * (y - 0.5)))
    p = y < 0.5 ? 2.0 * y + 1.0 : y + 1.5

    # Zero perturbation implies u = v = 0 and purely internal energy.
    total_energy = p / (gamma - 1.0)

    return SVector(rho, 0.0, 0.0, total_energy)
end
@inline function source_terms_gravity(u, x, t,
                                      equations::CompressibleEulerEquations2D)
    g = 1.0
    rho = u[1]
    rho_v2 = u[3]
    # Energy source: rho * g * v = g * (rho * v).
    return SVector(0.0, 0.0, g * rho, g * rho_v2)
end
# Density is the feature indicator for both shock capturing and AMR.
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
#     * **Riemann Solver**: Harten-Lax-van Leer-Contact (HLLC).
#     * **Role**: Computes numerical fluxes across element interfaces while explicitly restoring the contact wave.
#     * **Physics**: Reduces dissipation at the density material interface and better preserves rolled filaments than HLL.
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
#
# Defines the RTI domain $\Omega = [0, 0.25] \times [0, 1]$ using a forest of
# dynamic quadtrees. Four root trees are used in the $y$-direction so that each
# root tree covers a square $0.25 \times 0.25$ region.
#
# * **Initial Resolution**: `initial_refinement_level=3` is retained from the
#   template; subsequent local refinement is controlled by the AMR callback.
# * **Boundary Topology**: `periodicity=(true, false)` makes the $x$-boundaries
#   periodic. The boundaries at $y=0$ and $y=1$ remain physical boundaries for
#   the reflective free-slip wall conditions.

# %%
coordinates_min = (0.0, 0.0)
coordinates_max = (0.25, 1.0)

# Four root trees in y match the 1:4 domain aspect ratio.
mesh = P4estMesh((1, 4), polydeg=polydeg,
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max,
                 initial_refinement_level=3,
                 periodicity=(true, false))
# %% [markdown]

# ## Cell 6: General Boundary Condition Specification & System Assembly
# ## Boundary Conditions and Hyperbolic System Assembly
#
# The $x$-boundaries are periodic through the mesh topology and therefore do not
# require entries in the boundary-condition dictionary. At $y=0$ and $y=1$,
# `boundary_condition_slip_wall` imposes a reflective free-slip wall: the normal
# velocity is mirrored, while density, tangential momentum, and pressure are
# preserved in the reflected state.
#
# The Euler equations, RTI initial state, DGSEM solver, and gravitational source
# are combined into a purely hyperbolic semidiscretization.
# %%
boundary_conditions = (
    y_neg = boundary_condition_slip_wall,
    y_pos = boundary_condition_slip_wall,
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
#     * **Mechanism**: It estimates the local smoothness of the solution by computing a normalized second derivative (curvature) of the **density**.
#     * **Role**: Acts as a "feature detector," identifying regions with high gradients (shear layers) or discontinuities that require higher resolution to resolve.
#
# 2.  **The Controller: Hysteresis Logic (`ControllerThreeLevel`)**
#     * Maps the continuous indicator value $\epsilon$ to discrete mesh operations (Refine/Coarsen/Keep) using a **Three-Level Hysteresis** approach to prevent "chattering" (rapid toggling between levels).
#     * **Refine Condition ($h \to h/2$)**: If error $\epsilon > 0.3$ (`max_threshold`). Splits the element into 4 children (quadtree).
#     * **Coarsen Condition ($h \to 2h$)**: If error $\epsilon < 0.1$ (`med_threshold`). Merges 4 children into 1 parent.
#     * **Deadband**: If $0.1 \le \epsilon \le 0.3$, the mesh is frozen to ensure stability.
#     * **Bounds**: Enforces hard constraints on resolution levels ($L_{min}=4, L_{max}=7$) to guarantee memory limits.
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
                                      max_level=7, max_threshold=0.3)
# 3. Callback
amr_callback = AMRCallback(semi, amr_controller,
                           interval=20,
                           adapt_initial_condition=true)
# %% [markdown]


# ## Cell 8: Data I/O & Surrogate Interoperability
# ## Strategy: Fixed-Time Cross-Language Serialization (.npz)
#
# Saves the complete conservative state
# $(\rho, \rho u, \rho v, E)$ every `dt_save = 0.1` time units for ground truth.
# Additional snapshots at the requested visualization times ensure that the
# off-grid times $t=0.75$ and $t=2.25$ are available without interpolation.
# The vertical velocity $v$ is also stored for the summary figure.
# %%
using DiffEqCallbacks: PresetTimeCallback
npz_out = joinpath(@__DIR__, "data_npz")
control_npz_out = joinpath(@__DIR__, "data_npz_control")
mkpath(npz_out)
dt_save = 0.1
visualization_times = [0.75, 1.5, 2.25, 3.0]
npz_save_times = sort(unique(vcat(collect(0.0:dt_save:3.0),
                                  visualization_times)))
# Integrate a coordinate-dependent quantity using the native DG quadrature.
# This is used only for diagnostics and does not modify the solver state.
function integrate_rti_quantity(func, u_ode, semi)
    _, eq, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    u = Trixi.wrap_array(u_ode, semi)
    weights = solver.basis.weights
    inverse_jacobian = cache.elements.inverse_jacobian
    integral = 0.0
    for element in Trixi.eachelement(solver, cache)
        for j in Trixi.eachnode(solver), i in Trixi.eachnode(solver)
            u_local = Trixi.get_node_vars(u, eq, solver, i, j, element)
            x_local = Trixi.get_node_coords(cache.elements.node_coordinates,
                                             eq, solver, i, j, element)
            inv_jacobian = if ndims(inverse_jacobian) == 1
                inverse_jacobian[element]
            else
                inverse_jacobian[i, j, element]
            end
            integral += func(u_local, x_local, eq) * weights[i] * weights[j] /
                        inv_jacobian
        end
    end
    return integral
end
# Create an independent saver and diagnostic history for each simulation case.
function make_save_npz_callback(output_directory, case_label)
    mkpath(output_directory)
    diagnostics = NamedTuple[]
    reference_mass = Ref(NaN)
    reference_energy_budget = Ref(NaN)
    function save_npz_file(integrator)
        semi = integrator.p
        t = integrator.t
        # Extract the full conservative state on the visualization grid.
        pd = PlotData2D(integrator.u, semi, solution_variables=cons2cons)
        rho   = getindex.(pd.data, 1)
        rho_u = getindex.(pd.data, 2)
        rho_v = getindex.(pd.data, 3)
        rho_e = getindex.(pd.data, 4)
        # Primitive fields used only for diagnostics and visualization.
        v = rho_v ./ rho
        p = (equations.gamma - 1.0) .*
            (rho_e .- 0.5 .* (rho_u.^2 .+ rho_v.^2) ./ rho)
        # Conservation and energy-budget diagnostics using DG quadrature.
        # With gravity in +y, potential energy is -rho*g*y and the conserved
        # budget is integral(E - rho*g*y) dA.
        mass = integrate_rti_quantity((u, x, eq) -> u[1],
                                      integrator.u, semi)
        total_energy = integrate_rti_quantity((u, x, eq) -> u[4],
                                              integrator.u, semi)
        g = 1.0
        gravitational_potential_energy = integrate_rti_quantity(
            (u, x, eq) -> -g * u[1] * x[2], integrator.u, semi)
        energy_budget = total_energy + gravitational_potential_energy
        energy_budget = total_energy + gravitational_potential_energy
        if !isfinite(reference_mass[])
            reference_mass[] = mass
            reference_energy_budget[] = energy_budget
        end
        mass_drift = mass - reference_mass[]
        relative_mass_drift = mass_drift / reference_mass[]
        energy_budget_drift = energy_budget - reference_energy_budget[]
        rho_flat = vec(rho)
        p_flat = vec(p)
        v_flat = vec(v)
        y_flat = vec(pd.y)
        min_rho = minimum(rho_flat)
        min_pressure = minimum(p_flat)
        max_abs_v = maximum(abs, v_flat)
        finite_state = all(isfinite, integrator.u) && all(isfinite, rho_flat) &&
                       all(isfinite, p_flat) && all(isfinite, v_flat)
        # RMS vertical deformation of the density transition layer about y=0.5.
        # The weight peaks at rho=1.5 and vanishes in either pure fluid.
        heavy_fraction = clamp.(rho_flat .- 1.0, 0.0, 1.0)
        light_fraction = clamp.(2.0 .- rho_flat, 0.0, 1.0)
        interface_weights = 4.0 .* heavy_fraction .* light_fraction
        interface_weight_sum = sum(interface_weights)
        interface_deformation = if interface_weight_sum > 0.0
            sqrt(sum(interface_weights .* (y_flat .- 0.5).^2) /
                 interface_weight_sum)
        else
            NaN
        end
        filename = joinpath(output_directory,
                            Printf.@sprintf("time_%08.5f.npz", t))
        npzwrite(filename, Dict("t" => t, "x" => pd.x, "y" => pd.y,
                                "rho" => rho, "rho_u" => rho_u,
                                "rho_v" => rho_v, "rho_e" => rho_e,
                                "v" => v,
                                "mass" => mass,
                                "mass_drift" => mass_drift,
                                "relative_mass_drift" => relative_mass_drift,
                                "total_energy" => total_energy,
                                "gravitational_potential_energy" =>
                                    gravitational_potential_energy,
                                "energy_budget" => energy_budget,
                                "energy_budget_drift" => energy_budget_drift,
                                "min_rho" => min_rho,
                                "min_pressure" => min_pressure,
                                "finite_state" => (finite_state ? 1 : 0),
                                "max_abs_v" => max_abs_v,
                                "interface_deformation" => interface_deformation))
        push!(diagnostics,
              (t=t, mass=mass, mass_drift=mass_drift,
               relative_mass_drift=relative_mass_drift,
               total_energy=total_energy,
               gravitational_potential_energy=gravitational_potential_energy,
               energy_budget=energy_budget,
               energy_budget_drift=energy_budget_drift,
               min_rho=min_rho, min_pressure=min_pressure,
               finite_state=finite_state, max_abs_v=max_abs_v,
               interface_deformation=interface_deformation))
        Printf.@printf("[%s] t=%.5f mass=%.8e mass_drift=%.3e energy_budget=%.8e energy_drift=%.3e\n",
                       case_label, t, mass, mass_drift, energy_budget,
                       energy_budget_drift)
        Printf.@printf("[%s] min_rho=%.3e min_p=%.3e max|v|=%.3e interface_deformation=%.3e finite=%s\n",
                       case_label, min_rho, min_pressure, max_abs_v,
                       interface_deformation, string(finite_state))
        if !finite_state || min_rho <= 0.0 || min_pressure <= 0.0
            @warn "Non-finite or non-positive state detected" case=case_label t=t \
                  min_rho=min_rho min_pressure=min_pressure finite_state=finite_state
        end
        println("Saved: $(basename(filename))")
        return nothing
    end
    callback = PresetTimeCallback(npz_save_times, save_npz_file;
                                  save_positions=(false, false))
    return callback, diagnostics
end
# The main run retains the original output path and callback variable.
save_npz_callback, main_diagnostics = make_save_npz_callback(npz_out, "main")
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
#     Here, we set `cfl=0.35` to accommodate the more restrictive time-step requirement of the degree-5 discretization.
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
trixi_out = joinpath(@__DIR__, "trixi_output")
# Save the complete conservative state at fixed physical-time intervals.
save_solution = SaveSolutionCallback(dt=dt_save,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     output_directory=trixi_out,
                                     solution_variables=cons2cons)
stepsize_callback = StepsizeCallback(cfl=0.35)
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
            saveat=0.01,
            maxiters=1e7);
# Mandatory unperturbed control case: identical discretization and background,
# but with the velocity-perturbation amplitude set to zero.
println("Running unperturbed background control case...")
control_mesh = P4estMesh((1, 4), polydeg=polydeg,
                         coordinates_min=coordinates_min,
                         coordinates_max=coordinates_max,
                         initial_refinement_level=3,
                         periodicity=(true, false))
control_semi = SemidiscretizationHyperbolic(control_mesh, equations,
                                            initial_condition_rti_control, dg;
                                            boundary_conditions=boundary_conditions,
                                            source_terms=source_terms_gravity)
control_amr_indicator = IndicatorLöhner(control_semi,
                                        variable=density_indicator)
control_amr_controller = ControllerThreeLevel(control_semi,
                                              control_amr_indicator,
                                              base_level=4,
                                              med_level=5, med_threshold=0.1,
                                              max_level=7, max_threshold=0.3)
control_amr_callback = AMRCallback(control_semi, control_amr_controller,
                                   interval=20,
                                   adapt_initial_condition=true)
control_save_npz_callback, control_diagnostics =
    make_save_npz_callback(control_npz_out, "control")
control_ode = semidiscretize(control_semi, tspan)
control_summary_callback = SummaryCallback()
control_analysis_callback = AnalysisCallback(control_semi,
                                             interval=analysis_interval)
control_alive_callback = AliveCallback(analysis_interval=analysis_interval)
control_trixi_out = joinpath(@__DIR__, "trixi_output_control")
control_save_solution = SaveSolutionCallback(dt=dt_save,
                                             save_initial_solution=true,
                                             save_final_solution=true,
                                             output_directory=control_trixi_out,
                                             solution_variables=cons2cons)
control_stepsize_callback = StepsizeCallback(cfl=0.35)
control_callbacks = CallbackSet(control_summary_callback,
                                control_analysis_callback,
                                control_alive_callback,
                                control_save_solution,
                                control_amr_callback,
                                control_save_npz_callback,
                                control_stepsize_callback)
control_sol = solve(control_ode, RDPK3SpFSAL49();
                    dt=1.0,
                    adaptive=false,
                    callback=control_callbacks,
                    saveat=0.01,
                    maxiters=1e7);
# Background-motion report at every fixed-time NPZ save.
println("Unperturbed control: background motion versus time")
for diagnostic in control_diagnostics
    Printf.@printf("t=%.5f max|v|=%.8e interface_deformation=%.8e\n",
                   diagnostic.t, diagnostic.max_abs_v,
                   diagnostic.interface_deformation)
end

npzwrite(joinpath(control_npz_out, "background_diagnostics.npz"),
         Dict("t" => getproperty.(control_diagnostics, :t),
              "max_abs_v" => getproperty.(control_diagnostics, :max_abs_v),
              "interface_deformation" =>
                  getproperty.(control_diagnostics, :interface_deformation)))

# Final auditable conservation and positivity summary.
println("Final conservation, energy-budget, and positivity diagnostics")
println("Energy sign convention: g = +1 in y; potential energy = -rho*g*y; " *
        "conserved budget = integral(E - rho*g*y) dA.")
for (case_label, diagnostics) in (("main", main_diagnostics),
                                  ("control", control_diagnostics))
    diagnostic = diagnostics[end]
    positive_state = diagnostic.min_rho > 0.0 && diagnostic.min_pressure > 0.0

    Printf.@printf("[%s] final t=%.5f mass=%.8e mass_drift=%.8e relative_mass_drift=%.8e\n",
                   case_label, diagnostic.t, diagnostic.mass,
                   diagnostic.mass_drift, diagnostic.relative_mass_drift)
    Printf.@printf("[%s] total_energy=%.8e potential_energy=%.8e energy_budget=%.8e energy_budget_drift=%.8e\n",
                   case_label, diagnostic.total_energy,
                   diagnostic.gravitational_potential_energy,
                   diagnostic.energy_budget, diagnostic.energy_budget_drift)
    Printf.@printf("[%s] min_rho=%.8e min_p=%.8e finite_state=%s positive_state=%s\n",
                   case_label, diagnostic.min_rho, diagnostic.min_pressure,
                   string(diagnostic.finite_state), string(positive_state))
end

# Resolution verification against the previous control's reported ranges.
# Strictly below both lower endpoints guarantees a decrease despite not having
# the previous run's exact time series; otherwise, report an inconclusive check.
previous_control_max_abs_v_range = (1.3e-3, 1.6e-3)
previous_control_interface_deformation_range = (0.02, 0.03)
control_final = control_diagnostics[end]

control_velocity_decreased =
    control_final.max_abs_v < previous_control_max_abs_v_range[1]
control_deformation_decreased =
    control_final.interface_deformation <
    previous_control_interface_deformation_range[1]
control_convergence_verified =
    control_velocity_decreased && control_deformation_decreased

Printf.@printf("Control resolution check: previous max|v| range=[%.3e, %.3e], elevated final max|v|=%.8e, strict_decrease=%s\n",
               previous_control_max_abs_v_range[1],
               previous_control_max_abs_v_range[2],
               control_final.max_abs_v, string(control_velocity_decreased))
Printf.@printf("Control resolution check: previous deformation range=[%.3e, %.3e], elevated final deformation=%.8e, strict_decrease=%s\n",
               previous_control_interface_deformation_range[1],
               previous_control_interface_deformation_range[2],
               control_final.interface_deformation,
               string(control_deformation_decreased))

if control_convergence_verified
    println("Zero-perturbation control convergence verified: both background diagnostics decreased below the previous reported ranges.")
else
    println("Zero-perturbation control convergence not confirmed: at least one final diagnostic remains within or above the previous reported range.")
end
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
function rasterize_data(x, y, z, nx=200, ny=200)
    # 1. Create a grid
    grid = fill(NaN, ny, nx) # NaN makes empty space transparent
    # 2. Determine bounds
    x_min, x_max = 0.0, 0.25
    y_min, y_max = 0.0, 1.0
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    # 3. Map every point to a grid cell
    # (Simple "Nearest Point" binning - very fast)
    for k in 1:length(x)
        # Find i (x index) and j (y index)
        i = floor(Int, (x[k] - x_min) / dx) + 1
        j = floor(Int, (y[k] - y_min) / dy) + 1
        # Bounds check
        if 1 <= i <= nx && 1 <= j <= ny
            grid[j, i] = z[k]
        end
    end
    return grid
end
# %% [markdown]

# ## Cell 12: Post-Processing & Temporal Diagnostics
# ## Rayleigh–Taylor Instability: Temporal Diagnostics
#
# Visualizes density $\rho$ and vertical velocity $v$ at
# $t = 0.75, 1.5, 2.25, 3.0$. Rows correspond to $[\rho, v]$ and columns to
# the four requested times. Color scales adjust automatically for each field,
# and colorbars are disabled.
# %%
println("Generating RTI summary plots...")
plot_times = [0.75, 1.5, 2.25, 3.0]
# Load the exact snapshots written by the fixed-time NPZ callback.
snapshot_files = [
    joinpath(npz_out, Printf.@sprintf("time_%08.5f.npz", t_plot))
    for t_plot in plot_times
]
snapshots = [npzread(filename) for filename in snapshot_files]
style = (
    seriestype = :scatter,
    markersize = 2.5,
    markerstrokewidth = 0,
    aspect_ratio = :equal,
    xlims = (0.0, 0.25),
    ylims = (0.0, 1.0),
    axis = nothing,
    framestyle = :none,
    legend = false,
    colorbar = false,
    margin = 2Plots.mm
)
plots_array = []
# Field-major ordering gives density in row 1 and vertical velocity in row 2.
for (variable, field_label) in (("rho", "ρ"), ("v", "v"))
    for (t_plot, data) in zip(plot_times, snapshots)
        x = vec(data["x"])
        y = vec(data["y"])
        values = vec(data[variable])
        push!(plots_array,
              plot(x, y;
                   zcolor=values,
                   title="$(field_label), t=$(round(t_plot, digits=2))",
                   style...))
    end
end
println("Saving summary_all.png...")
final_plot = plot(plots_array...;
                  layout=(2, 4),
                  size=(1600, 1800),
                  margin=5Plots.mm)
savefig(final_plot, joinpath(@__DIR__, "summary_all.png"))
# Interface diagnostics for the current max_level=7 run and retained Run 6.
# Thresholds define consistent diagnostic metrics, not exact error norms.
function rti_interface_diagnostics(data)
    rho = vec(data["rho"])
    y = vec(data["y"])
    # Uppermost heavy-fluid location represents the spike-tip diagnostic.
    spike_nodes = rho .>= 1.5
    spike_tip = any(spike_nodes) ? maximum(y[spike_nodes]) : NaN
    # Intermediate densities define the measured mixing-zone extent.
    mixing_nodes = (rho .> 1.05) .& (rho .< 1.95)
    if any(mixing_nodes)
        mixing_zone_bottom = minimum(y[mixing_nodes])
        mixing_zone_top = maximum(y[mixing_nodes])
    else
        mixing_zone_bottom = NaN
        mixing_zone_top = NaN
    end
    return (spike_tip=spike_tip,
            mixing_zone_bottom=mixing_zone_bottom,
            mixing_zone_top=mixing_zone_top,
            mixing_zone_extent=mixing_zone_top - mixing_zone_bottom)
end
# Retained Run 6 snapshots are optional because no external reference data
# are supplied. Current-run diagnostics are always printed.
run6_npz_out = joinpath(@__DIR__, "data_npz_run6")
run6_files = [
    joinpath(run6_npz_out, Printf.@sprintf("time_%08.5f.npz", t_plot))
    for t_plot in plot_times
]
current_diagnostics = [rti_interface_diagnostics(data) for data in snapshots]

println("Current max_level=7 interface diagnostics (not error norms):")
for (t_plot, current) in zip(plot_times, current_diagnostics)
    Printf.@printf("t=%.2f spike_tip=%.6f mixing_zone_extent=%.6f\n",
                   t_plot, current.spike_tip, current.mixing_zone_extent)
end

if all(isfile, run6_files)
    println("Comparing current max_level=7 interface diagnostics with Run 6...")
    run6_diagnostics = [
        rti_interface_diagnostics(npzread(filename))
        for filename in run6_files
    ]

    for (t_plot, run6, current) in zip(plot_times, run6_diagnostics,
                                       current_diagnostics)
        Printf.@printf("t=%.2f spike_tip: Run6=%.6f current=%.6f delta=%.3e; mixing_extent: Run6=%.6f current=%.6f delta=%.3e\n",
                       t_plot, run6.spike_tip, current.spike_tip,
                       current.spike_tip - run6.spike_tip,
                       run6.mixing_zone_extent, current.mixing_zone_extent,
                       current.mixing_zone_extent - run6.mixing_zone_extent)
    end

    npzwrite(joinpath(npz_out, "run6_interface_comparison.npz"),
             Dict("t" => plot_times,
                  "spike_tip_run6" =>
                      getproperty.(run6_diagnostics, :spike_tip),
                  "spike_tip_current" =>
                      getproperty.(current_diagnostics, :spike_tip),
                  "mixing_zone_extent_run6" =>
                      getproperty.(run6_diagnostics, :mixing_zone_extent),
                  "mixing_zone_extent_current" =>
                      getproperty.(current_diagnostics, :mixing_zone_extent)))

    println("Visually compare the thinnest density filaments at t=2.25 and t=3.0 in summary_all.png with the retained Run 6 summary; sharper current filaments indicate resolution-dependent softening.")
else
    println("Run 6 snapshots unavailable in $(run6_npz_out); spike-tip, mixing-zone, and filament comparisons cannot be completed.")
    println("Current max_level=7 diagnostics were printed above; retain these snapshots for the next resolution comparison.")
end
println("Done.")
