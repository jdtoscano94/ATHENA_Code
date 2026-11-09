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
semi = SemidiscretizationHyperbolic(mesh, equations,
                                    initial_condition_rti, dg;
                                    boundary_conditions=Dict(
                                        :y_neg => boundary_condition_slip_wall,
                                        :y_pos => boundary_condition_slip_wall
                                    ),
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
mkpath(npz_out)

dt_save = 0.1
visualization_times = [0.75, 1.5, 2.25, 3.0]
npz_save_times = sort(unique(vcat(collect(0.0:dt_save:3.0),
                                  visualization_times)))

function save_npz_file(integrator)
    semi = integrator.p
    t = integrator.t

    # Extract the full conservative state on the visualization grid.
    pd = PlotData2D(integrator.u, semi, solution_variables=cons2cons)

    rho   = getindex.(pd.data, 1)
    rho_u = getindex.(pd.data, 2)
    rho_v = getindex.(pd.data, 3)
    rho_e = getindex.(pd.data, 4)

    # Primitive vertical velocity for the requested RTI visualization.
    v = rho_v ./ rho

    filename = joinpath(npz_out, Printf.@sprintf("time_%08.5f.npz", t))

    npzwrite(filename, Dict("t" => t, "x" => pd.x, "y" => pd.y,
                            "rho" => rho, "rho_u" => rho_u,
                            "rho_v" => rho_v, "rho_e" => rho_e,
                            "v" => v))
    println("Saved: $(basename(filename))")
    return nothing
end

# Force exact output times, including the non-multiples of dt_save.
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
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

trixi_out = joinpath(@__DIR__, "trixi_output")

# Save the complete conservative state at fixed physical-time intervals.
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
println("Done.")