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

# %% [markdown]

# ## Cell 2: Physics Model Definition (Compressible Euler)
#
# ## Governing Equations: Hyperbolic System
# Defines the hyperbolic system for the compressible Euler equations:
# $$
# \frac{\partial \mathbf{u}}{\partial t} + \nabla \cdot \mathbf{f}_{inv}(\mathbf{u}) = \mathbf{0}
# $$
#
# 1.  **Hyperbolic Core (`CompressibleEulerEquations2D`)**:
#     * Defines the inviscid flux $\mathbf{f}_{inv}(\mathbf{u})$ (Conservation of Mass, Momentum, Energy).
#     * **Equation of State**: Ideal Gas Law with $\gamma = 1.4$, as specified by the problem.
#
# %%
# The Kelvin-Helmholtz instability is an inviscid phenomenon.
# We only need the compressible Euler equations.
 equations = CompressibleEulerEquations2D(1.4)

# ## Cell 3: Initial Conditions (Kelvin-Helmholtz Instability)
#
# ## Problem Setup: Shear Layer with Perturbation
# Configures the initial state for the Kelvin-Helmholtz instability. The setup consists of a smooth, stratified shear layer in the y-direction, which is perturbed by a single-mode sinusoidal velocity field to trigger the instability.
#
# ### 1. Initial State $\mathbf{U}(\mathbf{x}, 0)$
# The primitive variables are defined as follows:
# * **Density ($\rho$):** A smooth transition from $\rho=2.0$ at the bottom to $\rho=1.0$ at the top.
#   $$ \rho(y) = 1.5 - 0.5 \cdot \tanh\left(\frac{y - 0.5}{0.03}\right) $$
# * **Pressure ($p$):** Uniform pressure throughout the domain.
#   $$ p = 2.5 $$
# * **Velocity ($\mathbf{v}$):** A horizontal shear layer with a vertical perturbation.
#   $$ u(y) = 0.5 \cdot \tanh\left(\frac{y - 0.5}{0.03}\right) $$
#   $$ v(x,y) = 0.01 \cdot \sin(4\pi x) \cdot \exp\left(-\frac{(y - 0.5)^2}{0.01}\right) $$
#
# * **Transformation**: These primitive variables are converted to the conservative state vector $\mathbf{u} = [\rho, \rho u, \rho v, \rho E]^T$ required by the solver.
# %%
@inline function initial_condition_kelvin_helmholtz(x, t, equations::CompressibleEulerEquations2D)
    # Unpack coordinates
    x_coord, y_coord = x
    # Density profile
    rho = 1.5 - 0.5 * tanh((y_coord - 0.5) / 0.03)
    # Constant pressure
    p = 2.5
    # Horizontal velocity (shear layer)
    u = 0.5 * tanh((y_coord - 0.5) / 0.03)
    # Vertical velocity (perturbation)
    v = 0.01 * sin(4 * pi * x_coord) * exp(-(y_coord - 0.5)^2 / 0.01)
    # Convert primitives to conservatives
    return prim2cons(SVector(rho, u, v, p), equations)
end
@inline function density(u, equations)
    return u[1]
end


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
polydeg = 5
basis = LobattoLegendreBasis(polydeg)

# For this smooth, subsonic flow, shock capturing introduces excessive dissipation
# and is disabled as per the strategy. We use the standard weak form DG operator.
volume_integral = VolumeIntegralWeakForm()

dg = DGSEM(polydeg=polydeg, surface_flux=flux_hll, volume_integral=volume_integral)
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
#     * `periodicity=(true, false)`: Enables periodic wrapping in the x-direction, while treating the y-direction boundaries as physical walls requiring slip boundary conditions.
#
# %%
coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)

# trees_per_dimension (1, 1) for square aspect ratio
mesh = P4estMesh((1, 1), polydeg=polydeg,
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max,
                 initial_refinement_level=3, # Start coarser, let AMR refine
                 periodicity=(true, false)) # Periodic in x, walls in y
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
# Simplified boundary conditions for inviscid simulation
boundary_conditions = Dict(
    :y_pos => boundary_condition_slip_wall,
    :y_neg => boundary_condition_slip_wall,
)

# Construct the semidiscretization for the inviscid problem
semi = SemidiscretizationHyperbolic(mesh, equations,
                                    initial_condition_kelvin_helmholtz, dg;
                                    boundary_conditions=boundary_conditions)
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
amr_indicator = IndicatorLöhner(semi, variable=density)
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

# 1. The Saver Function
function save_npz_file(integrator)
    semi = integrator.p
    t    = integrator.t

    # Extract data
    pd = PlotData2D(integrator.u, semi, solution_variables=cons2prim)
    
    rho = getindex.(pd.data, 1)
    u   = getindex.(pd.data, 2)
    v   = getindex.(pd.data, 3)
    p   = getindex.(pd.data, 4)
    
    filename = joinpath(npz_out, Printf.@sprintf("time_%08.5f.npz", t))
    
    npzwrite(filename, Dict("t"=>t, "x"=>pd.x, "y"=>pd.y, 
                            "rho"=>rho, "u"=>u, "v"=>v, "p"=>p))
    println("Saved: $(basename(filename))")
    return nothing
end

# 2. Initialization: Force save at t=0
function init_save_npz(cb, u, t, integrator)
    save_npz_file(integrator)
end

# 3. Condition: Every 40 steps OR at the end
function condition_save_npz(u, t, integrator)
    t_end = integrator.sol.prob.tspan[2]
    is_finished = t >= (t_end - 1e-5)
    
    return integrator.iter % 40 == 0 || is_finished
end

# 4. Create Callback
save_npz_callback = DiscreteCallback(condition_save_npz, 
                                     save_npz_file, 
                                     save_positions=(false,false),
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
# * **Time Domain**: `tspan = (0.0, 3.5)` defines the simulation window $t \in [0, T_{end}]$.
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
tspan = (0.0, 2.5)
ode = semidiscretize(semi, tspan)
summary_callback = SummaryCallback()
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
alive_callback = AliveCallback(analysis_interval = analysis_interval)
# We save fewer solution files to save disk space, but analysis runs often
trixi_out = joinpath(@__DIR__, "trixi_output")
# Interval=20 ensures we have enough frames for the 4x4 plot
save_solution = SaveSolutionCallback(interval = 20,
                                     save_initial_solution = true,
                                     output_directory = trixi_out,
                                     solution_variables = cons2prim)
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
println("Generating clean heatmaps...")
plots_array = []
# Find and sort the NPZ files
files = sort(filter(f -> endswith(f, ".npz"), readdir(npz_out, join=true)))

# Check if enough files were generated for the plot
if length(files) < 4
    error("Not enough snapshots generated for a 4x4 plot. Found $(length(files)) files in '$npz_out'.")
end

# Select four evenly spaced snapshots (first, 1/3, 2/3, last)
indices = round.(Int, range(1, stop=length(files), length=4))
for idx in indices
    println(files[idx])
    data = npzread(files[idx])
    t_val = data["t"]
    # 1. READ & FLATTEN (Critical for Plots.jl performance)
    x = vec(data["x"])
    y = vec(data["y"])
    # 2. STYLE
    style = (
        seriestype = :scatter,
        markersize = 2.5,
        markerstrokewidth = 0,
        c = :coolwarm,
        aspect_ratio = :equal,
        xlims = (0, 1),
        ylims = (0, 1),
        axis = nothing,
        framestyle = :none,
        legend = false,
        colorbar = true,
        margin = 2Plots.mm
    )
    # Density: Fix around 1.0 to hide machine epsilon noise
    push!(plots_array, plot(x, y, zcolor=vec(data["rho"]),
                            #clims=(0.99, 1.01),
                            title="ρ, t=$(round(t_val, digits=2))"; style...))
    # Velocity: Fix between -1 and 1 (Lid speed is 1.0)
    push!(plots_array, plot(x, y, zcolor=vec(data["u"]),
                            #clims=(-1.0, 1.0),
                            title="u, t=$(round(t_val, digits=2))"; style...))
    push!(plots_array, plot(x, y, zcolor=vec(data["v"]),
                            #clims=(-1.0, 1.0),
                            title="v, t=$(round(t_val, digits=2))"; style...))
    # Pressure: Let it auto-scale (or fix if you prefer)
    push!(plots_array, plot(x, y, zcolor=vec(data["p"]),
                            title="p, t=$(round(t_val, digits=2))"; style...))
end
# Combine and Save
println("Saving summary_all.png...")
final_plot = plot(plots_array..., layout=(4,4), size=(2000, 1200), margin=5Plots.mm)
savefig(final_plot, joinpath(@__DIR__, "summary_all.png"))
println("Done.")