# %% [markdown]
# # Preamble: Solver and Discretization Initialization
# 
# ## Framework: Method of Lines (MOL) for Hyperbolic Conservation Laws
# Initializes the core components to solve conservation laws of the form $\partial_t u + \partial_x f(u) = 0$.
# 
# 1.  **Spatial Discretization (`Trixi`)**:
#     * Provides the spatial operator $\mathcal{L}(u)$ for the semi-discrete form $\frac{du}{dt} = \mathcal{L}(u)$.
#     * Handles high-order Discontinuous Galerkin (DGSEM) operators, `TreeMesh` (adaptive Cartesian mesh), and hybrid shock capturing (HG) strategies.
# 
# 2.  **Time Integration (`OrdinaryDiffEq`)**:
#     * Implements the ecosystem for solving ordinary differential equations.
#     * Provides **Strong Stability Preserving (SSP)** Runge-Kutta schemes (e.g., `SSPRK43`), which are optimized for hyperbolic problems where maintaining Total Variation Diminishing (TVD) properties is critical.
# 
# 3.  **Diagnostics (`Plots`, `NPZ`)**:
#     * **`Plots`**: Utilities for in-situ visualization of the solution $u(x,t)$ during the simulation.
#     * **`NPZ`**: specialized bridge for exporting solution tensors to `.npz` format, facilitating interoperability with Python-based analysis tools (like ATHENA's diagnostic agents).
# 

# %%
using Trixi
using OrdinaryDiffEq
using Plots
using NPZ

# %% [markdown]
# # Cell 1: Problem Setup and Spectral Discretization
# 
# ## 1. Initial Condition: Sinusoidal Shock Test
# We define the specific problem instance using a smooth initial condition that is guaranteed to develop a discontinuity (shock) over time due to the non-linear advection term $u \partial_x u$.
# $$
# u(x,0) = -\sin(2\pi x), \quad x \in [-1, 1]
# $$
# * **Scientific Purpose**: The wave compresses as it propagates, forming a sharp gradient (shock) around $t \approx 1/\pi$. This serves as a "stress test" for the shock-capturing capabilities.
# 
# ## 2. Spectral Basis & Discretization
# * **`polydeg = 3`**: Specifies the polynomial degree $N$. The solution is approximated by polynomials of degree $N$ inside each element.
# * **`basis = LobattoLegendreBasis(polydeg)`**:
#     * Constructs the 1D spectral element basis using **Lagrange polynomials** on **Gauss-Lobatto-Legendre (GLL)** nodes.
#     * **Why this matters**: Using GLL nodes allows for a **collocation** approach where solution nodes and quadrature nodes coincide. This leads to a diagonal mass matrix, making the explicit time-stepping highly efficient.
# 
# ## 3. Computational Domain (TreeMesh)
# * **`TreeMesh`**: Constructs a 1D hierarchical Cartesian mesh on $[-1.0, 1.0]$.
# * **Resolution**: Initialized with **128 elements** ($2^7$).
# * **`periodicity=true`**: Enforces periodic boundary conditions, allowing the wave to recirculate without boundary artifacts.

# %%
function initial_condition_sine(x, t, equations::InviscidBurgersEquation1D)
    return SVector(-sin(2*pi * x[1]))
end

polydeg = 3
num_elements = 128
basis = LobattoLegendreBasis(polydeg)

coordinates_min = -1.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=Int(log2(num_elements)),
                n_cells_max=10_000,
                periodicity=true)

# %% [markdown]
# # Cell 2: Physics, Fluxes, and Shock Capturing
# 
# ## 1. Physics: Inviscid Burgers' Equation
# * **`equations = InviscidBurgersEquation1D()`**: Instantiates the conservation law $\partial_t u + \partial_x(u^2/2) = 0$.
# * **Role**: Acts as the fundamental governing equation. Its non-linearity ($u^2$) drives the formation of discontinuities even from smooth initial data.
# 
# ## 2. Numerical Fluxes (The DG-FV Split)
# We employ a split-flux formulation to balance accuracy and stability:
# * **`volume_flux = flux_central`**: The **high-order DG flux**. It is energy-conserving and non-dissipative, ensuring spectral accuracy in smooth regions.
# * **`surface_flux = flux_lax_friedrichs`**: The **dissipative interface flux**. It acts as a Riemann solver at element boundaries, providing the necessary upwinding/viscosity to stabilize the method.
# 
# ## 3. Hybrid Shock Capturing (Hennemann-Gassner)
# Hyperbolic problems with shocks require limiting to prevent Gibbs oscillations. We use a **Hybrid DG-FV** approach rather than a simple slope limiter.
# * **`IndicatorHennemannGassner`**:
#     * Acts as the "trouble detector." It computes a blending factor $\alpha \in [0, 1]$ for each element based on the energy in the highest spectral modes.
#     * **High Modes = Oscillation Risk** $\rightarrow$ Increase $\alpha$.
# * **`VolumeIntegralShockCapturingHG`**:
#     * **Mechanism**: Blends the two fluxes: $F_{total} = (1-\alpha)F_{DG} + \alpha F_{FV}$.
#     * **Result**: The method automatically switches to a robust Finite Volume scheme ($F_{FV}$) locally near shocks (where $\alpha \to 1$) while retaining high-order DG accuracy ($F_{DG}$) in smooth regions (where $\alpha \to 0$).
# """

# %%
equations = InviscidBurgersEquation1D()

volume_flux = flux_central
surface_flux = flux_lax_friedrichs


indicator_sc = IndicatorHennemannGassner(equations, basis,
                                          alpha_max=1.0,
                                          alpha_min=0.0001,
                                          alpha_smooth=false,
                                          variable=first)

volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                   volume_flux_dg=volume_flux,
                                                   volume_flux_fv=surface_flux)

# %% [markdown]
# # Cell 3: Solver Assembly and Temporal Logic
# 
# ## 1. Solver Assembly (DGSEM)
# * **`solver = DGSEM(...)`**: aggregates the spectral basis, interface fluxes, and shock-capturing integrals into a unified spatial operator.
# * **Role**: This object defines exactly *how* the spatial derivatives $\nabla \cdot f(u)$ are approximated on the mesh.
# 
# ## 2. Semi-Discretization (Method of Lines)
# * **`SemidiscretizationHyperbolic`**: Binds the physical model (Equations/ICs) to the numerical domain (Mesh/Solver).
# * **Mathematical Form**: Reduces the PDE to a large system of Ordinary Differential Equations (ODEs):
# $$
# \frac{d\mathbf{U}}{dt} = \mathcal{L}(\mathbf{U}, t)
# $$
# where $\mathbf{U}$ contains the degrees of freedom at all GLL nodes, and $\mathcal{L}$ is the spatial operator defined by the DGSEM.
# 
# ## 3. Adaptive Time-Stepping & Callbacks
# Explicit high-order methods are conditionally stable (CFL condition). We use callbacks to manage this automatically.
# * **`StepsizeCallback(cfl=0.3)`**:
#     * **Function**: dynamically adjusts $\Delta t$ at every step such that $\Delta t \leq \text{CFL} \cdot \frac{\Delta x_{\min}}{\lambda_{\max}}$.
#     * **Constraint**: `cfl=0.3` is chosen conservatively. While pure DG supports CFL $\approx 1/(N+1)$, the shock-capturing limiting often requires a smaller timestep to ensure positivity and stability.
# * **`AnalysisCallback`**: Computes and logs integral norms (errors, entropy) to monitor simulation health.

# %%

solver = DGSEM(basis, surface_flux, volume_integral)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_sine, solver)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

analysis_callback = AnalysisCallback(semi, interval=100)
alive_callback = AliveCallback(analysis_interval=100)
stepsize_callback = StepsizeCallback(cfl=0.3)

callbacks = CallbackSet(analysis_callback, alive_callback, stepsize_callback)

# %% [markdown]
# # Cell 4: Simulation Execution
# 
# ## 1. Time Integration Strategy
# * **`sol = solve(ode, SSPRK43(); ...)`**: Executes the time evolution using the **Strong Stability Preserving Runge-Kutta (SSPRK) 4(3)** method.
# * **Why SSPRK?**: Standard Runge-Kutta methods (like RK4) can introduce oscillations that violate the TVD property, causing the simulation to blow up when shocks form. SSP methods guarantee that the solution remains bounded in the chosen norm (usually TVD) under a specific CFL condition.
# 
# ## 2. Time-Stepping Control
# * **`adaptive=false`**: We disable the ODE solver's built-in *error-based* adaptivity (PID control).
# * **CFL Control**: Instead, we rely on the `StepsizeCallback` (defined in the previous cell) to adjust $\Delta t$ based on the **Courant-Friedrichs-Lewy (CFL)** condition. This is the standard stability criterion for hyperbolic PDEs: information cannot propagate across more than one element per step.
# 
# ## 3. Data Persistence (`saveat`)
# * **`n_save = 200`**: We verify the evolution by saving 200 snapshots.
# * **`saveat=LinRange(...)`**: Forces the solver to interpolate and store the solution at exact, equidistant time intervals. This is critical for creating smooth animations and space-time heatmaps (visual observations for the diagnostic agents) without the jitter of variable time steps.

# %%
n_save = 200
sol = solve(ode, SSPRK43(); 
            dt=1.0,
            adaptive=false,
            callback=callbacks,
            saveat=LinRange(tspan[1], tspan[2], n_save))

# %% [markdown]
# # Cell 5: Diagnostics and Observation Generation
# 
# ## 1. Observation Strategy: The Diagnostic Dashboard
# Instead of analyzing isolated files, we construct a unified **1x2 Diagnostic Dashboard** to serve as the primary observation ($O_n$) for the Advisor Agent.
# * **Left Panel (Heatmap)**: Visualizes the global space-time evolution ($x, t$).
# * **Right Panel (Slice)**: Visualizes the local final state $u(x, T_{final})$.
# 
# ## 2. Scientific Interpretation
# * **Heatmap**: Used to verify the **characteristic curves**. The agent looks for characteristics converging into a sharp line (shock formation) and checks for global stability (absence of checkerboard patterns).
# * **Final Slice**: Used to quantify **shock resolution**. The agent inspects the steepness of the gradient to assess numerical diffusion and checks the post-shock region for Gibbs oscillations (numerical dispersion).
# 
# ## 3. Artifact Generation
# * **`summary_all.png`**: The combined image file.
# * **Constraint**: The Vision-Language Model (VLM) is trained to ingest this specific aspect ratio. Keeping the heatmap and slice side-by-side allows the agent to correlate a wiggle in the heatmap directly with its amplitude in the slice.

# %%
# ----- Extract solution on uniform grid for heatmap ------------
pd = PlotData1D(sol.u[1], semi)
x_uniform = pd.x
nx = length(x_uniform)

usol = zeros(nx, n_save)
for (i, u) in enumerate(sol.u)
    pd = PlotData1D(u, semi)
    usol[:, i] = pd.data
end

# ----- 1. Create Heatmap (Left Panel) -------------------------
p_heatmap = heatmap(sol.t, x_uniform, usol,
    xlabel="Time (t)",
    ylabel="Space (x)",
    title="Evolution",
    color=:jet)

# ----- 2. Create Final Slice (Right Panel) --------------------
p_slice = plot(sol.u[end], semi, 
     label="t=$(tspan[2])", 
     xlabel="x", 
     ylabel="u",
     linewidth=2,
     legend=:topright,
     title="Final Solution")

# ----- 3. Combine into Dashboard (1x2) ------------------------
p_combined = plot(p_heatmap, p_slice, 
    layout=(1, 2), 
    size=(1000, 400),
    margin=5Plots.mm,
    )

savefig(p_combined, joinpath(@__DIR__, "summary_all.png"))
display(p_combined)

println("Dashboard saved to summary_all.png")

# %% [markdown]
# # Cell 6: Data Persistence and Verification

# %%
usol_save = Float64.(usol')  # Transpose to (time, space) format
t_array = collect(Float64.(sol.t))
x_array = collect(Float64.(x_uniform))

npz_path = joinpath(@__DIR__, "burgers.npz")
npzwrite(npz_path, Dict(
    "t" => t_array,
    "x" => x_array,
    "usol" => usol_save
))

println("Data saved as NPZ to burgers.npz")

# ----- Verify the saved file ----------------------------------------
println("\n--- File Verification ---")
file_size = filesize(npz_path)
println("File size: $(file_size) bytes ($(round(file_size/1024/1024, digits=2)) MB)")

data_check = npzread(npz_path)
println("Keys in file: ", keys(data_check))
println("Shape of usol: ", size(data_check["usol"]))
println("Shape of t: ", size(data_check["t"]))
println("Shape of x: ", size(data_check["x"]))


