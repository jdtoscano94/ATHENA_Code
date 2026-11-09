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
# Defines the two-dimensional compressible Euler equations for conservation
# of mass, momentum, and total energy:
# $$
# \frac{\partial \mathbf{u}}{\partial t}
# + \nabla \cdot \mathbf{f}_{inv}(\mathbf{u}) = \mathbf{S}(\mathbf{u}).
# $$
#
# * **Equation of State**: Ideal gas with $\gamma = 1.4$.
# * **Physics Regime**: Inviscid and non-conducting ($\mu = 0$, $k = 0$),
#   so no parabolic diffusion model is required.
# * **Source Term**: The gravitational body force for the Rayleigh–Taylor
#   instability is supplied separately to the semidiscretization.

# %%
equations = CompressibleEulerEquations2D(1.4)
# %% [markdown]


# ## Cell 3: Initial Conditions and Source Terms (Lid-Driven Cavity)
# ## Problem Setup: Compressible Rayleigh–Taylor Instability
#
# Initializes a heavy fluid below a light fluid, with the material interface
# smoothed around $y = 0.5$. A single-mode vertical-velocity perturbation
# seeds the instability. Gravity acts in the positive $y$ direction.
#
# The primitive state $(\rho, u, v, p)$ is converted to the conserved state
# $(\rho, \rho u, \rho v, E)$ required by the Euler solver.

# %%
@inline function initial_condition_rti(x, t, equations::CompressibleEulerEquations2D)
    x_coord, y = x

    # Heavy fluid ($\rho = 2$) below the interface; light fluid ($\rho = 1$) above.
    rho = 2.0 - 0.5 * (1.0 + tanh(50.0 * (y - 0.5)))

    # Continuous piecewise pressure profile.
    p = y < 0.5 ? 2.0 * y + 1.0 : y + 1.5

    u = 0.0
    sound_speed = sqrt(equations.gamma * p / rho)
    v = -0.025 * sound_speed * cos(8.0 * pi * x_coord) * sin(pi * y)^6

    # prim2cons forms E = p / (gamma - 1) + rho * (u^2 + v^2) / 2.
    return prim2cons(SVector(rho, u, v, p), equations)
end

@inline function source_terms_gravity(u, x, t,
                                      equations::CompressibleEulerEquations2D)
    g = 1.0
    rho = u[1]
    rho_v = u[3]

    # Since rho_v = rho * v, the final component is rho * g * v.
    return SVector(0.0, 0.0, g * rho, g * rho_v)
end

# Density sensor shared by shock capturing and adaptive mesh refinement.
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
#
# 3.  **Interface Flux (`surface_flux=flux_hll`)**:
#     * **Riemann Solver**: Harten-Lax-van Leer (HLL).
#     * **Role**: Computes the unique numerical flux across element interfaces where the solution is discontinuous.
#     * **Physics**: An approximate Riemann solver that estimates the fastest signal velocities (wave speeds). It is robust and positivity-preserving, making it ideal for compressible flows with strong shocks.
#
# 4.  **Indicator Function (`IndicatorHennemannGassner`)**:
#     * Dynamically computes the blending coefficient $\alpha \in [0, 1]$ element-wise.
#      * **Mechanism**: Monitors the spectral decay of energy in the highest modes of the polynomial expansion. If energy in high modes is large (indicating a discontinuity), $\alpha \to 1$ (activates FV sub-cells).
#
# %%
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=1.0,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_indicator)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=flux_ranocha,
                                                 volume_flux_fv=flux_lax_friedrichs)
dg = DGSEM(polydeg=polydeg, surface_flux=flux_hll, volume_integral=volume_integral)
# %% [markdown]


# ## Cell 5: Computational Domain & Adaptive Mesh Topology
# ## Methodology: Forest-of-Octrees (`p4est`)
# Defines the RTI domain using a dynamic quadtree mesh that supports AMR.
#
# * **Domain**: $\Omega = [0, 0.25] \times [0, 1]$.
# * **Topology**: One root tree with uniform initial refinement level 3.
# * **Boundaries**: Periodic in $x$; nonperiodic in $y$, where reflective/
#   free-slip wall conditions will be applied.

# %%
coordinates_min = (0.0, 0.0)
coordinates_max = (0.25, 1.0)

mesh = P4estMesh((1, 1), polydeg=polydeg,
                 coordinates_min=coordinates_min,
                 coordinates_max=coordinates_max,
                 initial_refinement_level=3,
                 periodicity=(true, false))


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
# ## Boundary Conditions and Hyperbolic System Assembly
#
# The $x$-boundaries are periodic through the mesh topology. At $y=0$ and
# $y=1$, `boundary_condition_slip_wall` imposes reflective/free-slip Euler
# wall conditions: zero normal velocity with mirrored density, tangential
# momentum, and energy.
#
# The inviscid spatial operator is assembled together with the RTI initial
# condition and gravitational source term. No parabolic operator is required.

# %%
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
                                      med_level=5, med_threshold=0.1,
                                      max_level=6, max_threshold=0.3)
# 3. Callback
amr_callback = AMRCallback(semi, amr_controller,
                           interval=20,
                           adapt_initial_condition=true)
# %% [markdown]


# ## Cell 8: Data I/O & Surrogate Interoperability
# ## Strategy: Cross-Language Serialization (.npz)
# Exports the complete conserved state on a structured PlotData2D grid.
# Ground-truth snapshots are written every 0.1 time units, with additional
# snapshots at the requested visualization times.

# %%
using DiffEqCallbacks: PresetTimeCallback

npz_out = joinpath(@__DIR__, "data_npz")
mkpath(npz_out)

dt_save = 0.1
visualization_times = [0.75, 1.5, 2.25, 3.0]
npz_save_times = sort(unique(vcat(collect(0.0:dt_save:3.0),
                                  visualization_times)))

function save_npz_file(integrator)
    semi = integrator.p
    t = integrator.t

    # Keep the conserved variables: rho, rho*u, rho*v, and E.
    pd = PlotData2D(integrator.u, semi; solution_variables=cons2cons)

    rho = getindex.(pd.data, 1)
    rho_u = getindex.(pd.data, 2)
    rho_v = getindex.(pd.data, 3)
    total_energy = getindex.(pd.data, 4)

    filename = joinpath(npz_out, Printf.@sprintf("time_%08.5f.npz", t))

    npzwrite(filename, Dict("t" => t, "x" => pd.x, "y" => pd.y,
                            "rho" => rho, "rho_u" => rho_u,
                            "rho_v" => rho_v, "E" => total_energy))
    println("Saved: $(basename(filename))")
    return nothing
end

# The callback inserts the requested times into the integrator's time stops.
save_npz_callback = PresetTimeCallback(npz_save_times, save_npz_file;
                                       save_positions=(false, false))
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
# Fixed physical-time interval for output
dt_save = 0.1

# Save the complete conserved state at fixed physical-time intervals.
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
    x_min, x_max = 0.0, 1.0
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
# ## Post-Processing: Rayleigh–Taylor Summary
# Visualizes density and vertical velocity at the four requested times.
# Rows correspond to fields and columns correspond to time. Color scaling
# is automatic for every panel, and colorbars are disabled.

# %%
println("Generating RTI summary plots...")

# Load the exact visualization snapshots created by the NPZ callback.
snapshots = []
for t_target in visualization_times
    filename = joinpath(npz_out,
                        Printf.@sprintf("time_%08.5f.npz", t_target))
    isfile(filename) || error("Missing solution snapshot: $filename")
    push!(snapshots, npzread(filename))
end

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

# Field-major ordering gives density in the first row and vertical velocity
# in the second row, with time increasing across the columns.
for field in (:rho, :v)
    for (t_target, data) in zip(visualization_times, snapshots)
        x = vec(data["x"])
        y = vec(data["y"])

        if field === :rho
            values = vec(data["rho"])
            field_label = "ρ"
        else
            # Vertical velocity from the saved conserved variables.
            values = vec(data["rho_v"] ./ data["rho"])
            field_label = "v"
        end

        push!(plots_array,
              plot(x, y, zcolor=values,
                   title="$field_label, t=$(round(t_target, digits=2))";
                   style...))
    end
end

println("Saving summary_all.png...")
final_plot = plot(plots_array..., layout=(2, 4),
                  size=(2000, 1200), margin=5Plots.mm)
savefig(final_plot, joinpath(@__DIR__, "summary_all.png"))
println("Done.")
