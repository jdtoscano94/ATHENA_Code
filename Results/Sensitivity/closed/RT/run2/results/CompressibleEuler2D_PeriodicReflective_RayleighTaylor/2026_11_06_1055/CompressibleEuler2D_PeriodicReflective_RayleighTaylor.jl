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
@inline function initial_condition_rt(x, t, equations::CompressibleEulerEquations2D)
    x1, x2 = x[1], x[2]
    
    rho = 2.0 - 0.5 * (1.0 + tanh(50.0 * (x2 - 0.5)))
    
    if x2 < 0.5
        p = 2.0 * x2 + 1.0
    else
        p = x2 + 1.5
    end
    
    c = sqrt(equations.gamma * p / rho)
    
    u = 0.0
    v = -0.025 * c * cos(8.0 * pi * x1) * sin(pi * x2)^6
    
    return prim2cons(SVector(rho, u, v, p), equations)
end

@inline function source_terms_rt(u, x, t, equations::CompressibleEulerEquations2D)
    g = 1.0
    rho, rho_v1, rho_v2, rho_e = u
    return SVector(0.0, 0.0, g * rho, g * rho_v2)
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
                                         variable=Trixi.density)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=flux_ranocha,
                                                 volume_flux_fv=flux_lax_friedrichs)
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
#     * `periodicity=(false, false)`: Disables periodic wrapping. This treats the domain boundaries $\partial \Omega$ as physical walls, necessitating explicit boundary conditions (Dirichlet/Neumann) for the Navier-Stokes operator.
#
# %%
coordinates_min = (0.0, 0.0)
coordinates_max = (0.25, 1.0)

# trees_per_dimension (1, 4) for 1:4 aspect ratio
mesh = P4estMesh((1, 4), polydeg=polydeg,
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
# 2. Map to Mesh Boundaries
boundary_conditions = Dict(
    :y_pos => boundary_condition_slip_wall,
    :y_neg => boundary_condition_slip_wall
)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_rt, dg;
                                    boundary_conditions = boundary_conditions,
                                    source_terms = source_terms_rt)
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
amr_indicator = IndicatorLöhner(semi, variable=Trixi.density)
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
    
    rho_v = rho .* v
    
    filename = joinpath(npz_out, Printf.@sprintf("time_%08.5f.npz", t))
    
    npzwrite(filename, Dict("t"=>t, "x"=>pd.x, "y"=>pd.y, 
                            "rho"=>rho, "rho_v"=>rho_v, "u"=>u, "v"=>v, "p"=>p))
    println("Saved: $(basename(filename))")
    return nothing
end

# 2. Initialization: Force save at t=0
function init_save_npz(cb, u, t, integrator)
    save_npz_file(integrator)
end

# 3. Condition: Save every dt = 0.1
function condition_save_npz(u, t, integrator)
    t_end = integrator.sol.prob.tspan[2]
    is_finished = t >= (t_end - 1e-5)
    
    # Save if we crossed a 0.1 interval
    dt_save = 0.1
    t_prev = t - integrator.dt
    crossed_interval = floor(t / dt_save) > floor(t_prev / dt_save)
    
    return crossed_interval || is_finished
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
tspan = (0.0, 3.0) #
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

files = sort([joinpath(npz_out, f) for f in readdir(npz_out) if endswith(f, ".npz")])
target_times = [0.0, 1.0, 2.0, 3.0]
selected_files = String[]

for tt in target_times
    closest_file = files[1]
    min_diff = Inf
    for f in files
        t_str = replace(replace(basename(f), "time_" => ""), ".npz" => "")
        t_val = parse(Float64, t_str)
        if abs(t_val - tt) < min_diff
            min_diff = abs(t_val - tt)
            closest_file = f
        end
    end
    push!(selected_files, closest_file)
end

plots_rho = []
plots_rhov = []

for f in selected_files
    println(f)
    data = npzread(f)
    t_val = data["t"]
    
    # 1. READ & FLATTEN
    x = vec(data["x"])
    y = vec(data["y"])
    
    # 2. STYLE
    style = (
        seriestype = :scatter,
        markersize = 2.5,     
        markerstrokewidth = 0, 
        c = :coolwarm,
        aspect_ratio = :equal,
        xlims = (0, 0.25),        
        ylims = (0, 1.0),
        axis = nothing,           
        framestyle = :none,    
        legend = false,   
        colorbar = false,
        margin = 2Plots.mm
    )
    
    push!(plots_rho, plot(x, y, zcolor=vec(data["rho"]), title="ρ, t=$(round(t_val, digits=2))"; style...))
    push!(plots_rhov, plot(x, y, zcolor=vec(data["rho_v"]), title="ρv, t=$(round(t_val, digits=2))"; style...))
end

# Combine and Save
plots_array = vcat(plots_rho, plots_rhov)
println("Saving summary_all.png...")
final_plot = plot(plots_array..., layout=(2,4), size=(1600, 800), margin=5Plots.mm)
savefig(final_plot, joinpath(@__DIR__, "summary_all.png"))
println("Done.")