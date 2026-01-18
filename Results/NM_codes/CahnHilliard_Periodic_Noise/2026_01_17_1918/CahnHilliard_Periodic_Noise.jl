# %% [markdown]
# # Preamble and Environment Initialization
#
# ## Framework: Fourier Spectral Method of Lines
# Initializes the computational environment to solve PDEs on periodic domains using the spectral method of lines: $\frac{d\hat{u}}{dt} = \mathcal{L}(\hat{u}) + \mathcal{N}(\hat{u})$.
#
# 1.  **Spatial Discretization (`FFTW`, `LinearAlgebra`)**:
#     * **`FFTW`**: Provides the Fast Fourier Transform engine. We compute spatial derivatives in frequency space ($\partial_x \rightarrow ik, \partial_{xx} \rightarrow -k^2$) to achieve **exponential convergence** (spectral accuracy).
#     * **`LinearAlgebra`**: Used for efficient matrix-vector operations in the spectral domain.
#
# 2.  **Time Integration (`DifferentialEquations`, `SciMLOperators`)**:
#     * **`DifferentialEquations`**: The core solver ecosystem. It enables the **`SplitODEProblem`** interface, which is critical for treating the stiff linear part $\mathcal{L}$ implicitly (stable) and the non-stiff nonlinear part $\mathcal{N}$ explicitly (fast).
#     * **`SciMLOperators`**: Wraps the linear operator (e.g., the Laplacian matrix) into a form that the ODE solver can efficiently invert without forming dense matrices unnecessarily.
#
# %%
using FFTW
using DifferentialEquations
using LinearAlgebra
using SciMLOperators
using Plots
using NPZ
using DiffEqCallbacks
using Random
using Statistics
using Printf
# %% [markdown]

# ## Cell 1: Spectral Operator Library
#
# ## 1. The `SpectralOperators` Struct
# * **Purpose**: Encapsulates the spatial discretization machinery. Instead of passing raw arrays ($x, k$) around, we bundle the physical grid and the spectral derivative operators into a single "Toolbox" object (`ops`).
# * **Efficiency**: We pre-compute common derivatives ($\partial_x, \partial_{xx}, \dots$) during initialization. This prevents "Allocation Hell" by ensuring we don't re-calculate these vectors at every time step.
#
# ## 2. Frequency Domain Construction (`fftfreq`)
# * **`k = 2π .* fftfreq(...)`**: Generates the standard FFT wavenumber grid.
# * **Constraint**: We must use `fftfreq` (not a simple range) because `FFTW` expects a specific ordering of frequencies: $[0, 1, \dots, N/2, -N/2, \dots, -1]$. Incorrect ordering is a common failure mode for manual implementations.
#
# ## 3. Derivative Logic (The "Legos")
# * **Mechanism**: In Fourier space, differentiation is defined algebraically as multiplication: $\frac{\partial^n u}{\partial x^n} \xrightarrow{\mathcal{F}} (ik)^n \hat{u}$.
# * **Precomputed Fields (`ops.∂²x`)**: Provide instant access to standard operators for diffusion and advection.
# * **Extensibility (`derivative(ops, n)`)**:
#     * A helper function allowing the Agent to request arbitrary derivative orders on the fly.
#     * **Example**: If the Strategy Agent discovers a "Fifth-Order KdV" equation, the Implementation Agent can simply call `derivative(ops, 5)` without needing to re-write the struct.
# %%
struct SpectralOperators
    nx::Int
    ny::Int
    Lx::Float64
    Ly::Float64
    x::Matrix{Float64}
    y::Matrix{Float64}
    k2::Matrix{Float64}      # Squared wavenumber: kx^2 + ky^2
    mask::Matrix{Float64}    # De-aliasing mask
end

function SpectralOperators(nx::Int, ny::Int, Lx::Float64, Ly::Float64)
    dx = Lx / nx
    dy = Ly / ny

    # Physical grids (meshgrid style)
    x_vec = range(0, Lx - dx, length=nx)
    y_vec = range(0, Ly - dy, length=ny)
    x = [xi for xi in x_vec, yi in y_vec]
    y = [yi for xi in x_vec, yi in y_vec]

    # Frequency grids
    kx_vec = 2π .* fftfreq(nx, nx/Lx)
    ky_vec = 2π .* fftfreq(ny, ny/Ly)
    kx = [kxi for kxi in kx_vec, kyi in ky_vec]
    ky = [kyi for kxi in kx_vec, kyi in ky_vec]

    # Laplacian operator in Fourier space (-k^2)
    k2 = kx.^2 .+ ky.^2

    # De-aliasing Mask (1/2 Rule: Keep lowest 50% of spectrum)
    mask = ones(nx, ny)
    kmax_x = (nx/2) * (2π/Lx)
    kmax_y = (ny/2) * (2π/Ly)

    for j in 1:ny
        for i in 1:nx
            if abs(kx[i,j]) >= 0.5 * kmax_x || abs(ky[i,j]) >= 0.5 * kmax_y
                mask[i,j] = 0.0
            end
        end
    end

    return SpectralOperators(nx, ny, Lx, Ly, x, y, k2, mask)
end
# %% [markdown]

# ## Cell 3: Problem Instantiation and Initial State
#
# ## 1. Discretization Parameters
# * **`nn = 511`**: Spatial resolution ($N$).
#     * **Why Odd?**: We use an odd number of grid points to ensure a unique mapping between positive and negative frequencies, preventing the "Nyquist ambiguity" where the highest frequency mode is its own complex conjugate.
# * **`domain`**: The physical periodic box $[-1.0, 1.0]$.
# * **`ops`**: Instantiates the spectral operator toolbox (grid $x$ and derivatives $ik$).
#
# ## 2. Initial Condition (The Spectral Transform)
# * **`u0 = ...`**: Defines the initial profile $u(x,0)$ in **Physical Space**.
# * **`u0_hat = fft(u0)`**:
#     * **Critical Step**: The solver evolves the **Fourier Coefficients** $\hat{u}(t)$, not the physical values.
#     * **Action**: We immediately transform the initial condition into frequency space using the Fast Fourier Transform (FFT).
#     * **Note**: Since the input is real, `u0_hat` will be Hermitian symmetric (conceptually), though we store the full complex vector.
#
# ## 3. Time Integration Setup
# * **`t_span = (0.0, 1.0)`**: The integration window $[T_{start}, T_{end}]$.
# * **`t_save`**:
#     * Defines the specific time points for saving snapshots (Observations).
#     * **Decoupled**: This is separate from the internal `dt` of the solver (which is adaptive). It ensures the Advisor Agent receives uniformly spaced observations for easier analysis (e.g., making animations).
# """
# %%
nx = 128
ny = 128
Lx = 2π
Ly = 2π
# Create spectral operators
ops = SpectralOperators(nx, ny, Lx, Ly)
# Initial Condition: Random noise
Random.seed!(1234) # Ensure reproducibility
u0 = 0.05 .* (2.0 .* rand(nx, ny) .- 1.0)
# Transform to spectral space and flatten (required for Diagonal operator)
u0_hat = vec(fft(u0))

# Apply de-aliasing mask (flattened)
u0_hat .*= vec(ops.mask)
# ----- Time Setup -----
t_span = (0.0, 1.0)
dt_save = 0.1
t_save = range(t_span[1], t_span[2], step=dt_save)
# %% [markdown]


# ## Cell 4: Physics Definition (SplitODE Formulation)
#
# ## 1. Methodology: Implicit-Explicit (IMEX) Splitting
# We formulate the PDE as a **Split ODE Problem** to handle stiffness efficiently:
# $$ \frac{d\hat{u}}{dt} = \mathcal{L}\hat{u} + \mathcal{N}(\hat{u}) $$
# * **$\mathcal{L}$ (Stiff)**: The linear diffusion/reaction terms. Solved **Implicitly** (stable).
# * **$\mathcal{N}$ (Non-Stiff)**: The nonlinear advection/reaction terms. Solved **Explicitly** (fast).
#
# ## 2. The Linear Operator ($\mathcal{L}$)
# * **`linear_coeffs = 5.0 .+ 0.0001 .* ops.∂²x`**:
#     * Represents the operator $5I + 0.0001 \nabla^2$.
#     * **Spectral Efficiency**: In Fourier space, $\nabla^2$ is simply multiplication by $-k^2$. The operator becomes a **Diagonal Matrix**, which is trivial to invert ($O(N)$) compared to a dense finite difference matrix ($O(N^3)$).
# * **`MatrixOperator`**: Wraps the diagonal vector to interface with `DifferentialEquations.jl`'s linear algebra system.
#
# ## 3. The Nonlinear Operator ($\mathcal{N}$)
# * **`nonlinear_operator!`**: Implementation of the **Pseudo-Spectral Method**:
#     1.  **`ifft`**: Transform state $\hat{u} \to u$ (Physical Space).
#     2.  **Compute**: Evaluate $-5u^3$ point-wise (Cheap in physical space).
#     3.  **`fft`**: Transform result back $\to \hat{N}$ (Spectral Space).
#     * **Aliasing Note**: We perform this on the collocated grid. For strict de-aliasing, one would use the 3/2-rule (padding), but for this blueprint, we accept minor aliasing errors for code simplicity.
#
# ## 4. Problem Assembly
# * **`SplitODEProblem`**: Defines the differential equation without solving it yet. It binds the stiff/non-stiff terms and the initial condition `u0_hat` into a solvable object.
# %%
println("\nMethod: SplitODEProblem with ETDRK4 (Cahn-Hilliard)")
# Parameters
epsilon = 0.01
# Linear operator: L(u) = -∇²u - ε∇⁴u
# In Fourier space: L_hat = k² - εk⁴
# Note: ∇² -> -k², so -∇² -> k²
#       ∇⁴ -> k⁴, so -ε∇⁴ -> -εk⁴
linear_coeffs = ops.k2 .- epsilon .* (ops.k2.^2)
L_matrix = Diagonal(vec(linear_coeffs)) # Flatten for Diagonal wrapper
L_op = MatrixOperator(L_matrix)
# Nonlinear operator: N(u) = ∇²(u³)
# In Fourier space: N_hat = -k² * FFT(u³)
function nonlinear_operator!(dN, u_hat, p, t)
    ops = p
    nx, ny = ops.nx, ops.ny

    # 1. Reshape input to 2D for spectral operations
    u_hat_2d = reshape(u_hat, nx, ny)

    # 2. Transform to physical space
    u = real(ifft(u_hat_2d))

    # 3. Compute cubic term
    u3 = u.^3

    # 4. Transform back to spectral space
    u3_hat = fft(u3)

    # 5. Apply de-aliasing mask (Strict 1/2 rule)
    u3_hat .*= ops.mask

    # 6. Apply Laplacian (-k²) and flatten result
    dN .= vec(-ops.k2 .* u3_hat)
end
# Create the problem
prob_split = SplitODEProblem(L_op, nonlinear_operator!,
                             u0_hat, t_span, ops)
# %% [markdown]


# ## Cell 5: Simulation Execution and Reconstruction
#
# ## 1. The Solver: ETDRK4
# * **`solve(prob, ETDRK4(), ...)`**:
#     * We use **Exponential Time Differencing Runge-Kutta (4th Order)**.
#     * **Why**: Standard Runge-Kutta (RK4) integrates the stiff term $e^{\mathcal{L}t}$ numerically, which requires tiny timesteps. **ETD** methods use the *exact* analytic solution for the linear part ($e^{\mathcal{L}h}$) and only numerically integrate the nonlinear deviation. This is the "Gold Standard" for spectral PDEs.
# * **`saveat=t_save`**: Forces the solver to interpolate output exactly at the requested frames, ensuring smooth animations regardless of the internal adaptive time-stepping.
#
# ## 2. Spectral Reconstruction (The "Inverse" Step)
# The solver output `sol_split.u` contains the history of **Fourier Coefficients** ($\hat{u}$), which are complex numbers and human-unreadable.
# * **`usol_fourier`**: We flatten the solution into a matrix `(N_modes, N_time)`.
# * **`ifft` (Inverse Fast Fourier Transform)**:
#     * We loop through time to transform $\hat{u}(t) \to u(x,t)$.
#     * **`real(...)`**: The FFT of real data often contains tiny imaginary errors (machine epsilon $\sim 10^{-16}$) due to floating-point arithmetic. We strictly discard these to ensure the final result is physically valid.
# %%
println("Solving...")
function step_callback(integrator)
    @printf("t = %.4f / %.1f (%.1f%%)  \r",
            integrator.t, integrator.sol.prob.tspan[2],
            100*integrator.t/integrator.sol.prob.tspan[2])
    flush(stdout)
end
cb = DiscreteCallback((u,t,integrator) -> true,
                      step_callback,
                      save_positions=(false,false))
@time sol_split = solve(prob_split, ETDRK4(),
                        dt=1e-4,
                        saveat=t_save,
                        callback=cb,
                        progress=true,
                        progress_steps=10)
println("\n✓ Time integration completed: $(length(sol_split.t)) steps")
# ----- Process Solution ------
# sol_split.u is a Vector of Matrices (Time -> [Nx, Ny])
# We stack them into a 3D array: (Nx, Ny, Time)
n_time = length(sol_split.t)
usol = zeros(nx, ny, n_time)
for i in 1:n_time
    # Reshape 1D state vector back to 2D for IFFT
    u_hat_2d = reshape(sol_split.u[i], nx, ny)
    usol[:, :, i] = real(ifft(u_hat_2d))
end
# ----- Mass Conservation Check ------
# Mass = Integral(u) dA approx Sum(u) * dx * dy
dA = (ops.Lx / ops.nx) * (ops.Ly / ops.ny)
mass_0 = sum(usol[:, :, 1]) * dA
mass_end = sum(usol[:, :, end]) * dA
mass_diff = abs(mass_end - mass_0)
println("\n--- Conservation Metrics ---")
@printf("Total Mass (t=0.0): %.6e\n", mass_0)
@printf("Total Mass (t=1.0): %.6e\n", mass_end)
@printf("Mass Error:         %.6e\n", mass_diff)
if mass_diff > 1e-10
    @warn "Mass conservation violation detected!"
else
    println("✓ Mass conservation verified.")
end
# ----- Visualization ------
# Strategy: 1 Row x 4 Columns (Snapshots)
# Select 4 indices roughly equidistant
indices = round.(Int, range(1, n_time, length=4))
times = sol_split.t[indices]
plots_list = []
for (k, idx) in enumerate(indices)
    u_snap = usol[:, :, idx]
    t_val = sol_split.t[idx]
    p = heatmap(ops.x[:,1], ops.y[1,:], u_snap',
        title = @sprintf("t = %.2f", t_val),
        aspect_ratio = :equal,
        colorbar = false, # CRITICAL REQUIREMENT
        c = :viridis,
        xlims = (0, ops.Lx),
        ylims = (0, ops.Ly),
        axis = false,
        grid = false
    )
    push!(plots_list, p)
end
p_combined = plot(plots_list...,
    layout = (1, 4),
    size = (1200, 300),
    margin = 2Plots.mm
)
display(p_combined)
savefig(p_combined, joinpath(@__DIR__, "summary_all.png"))
println("✓ Dashboard saved to summary_all.png")
# %% [markdown]


# ## Cell 6: Data Export
#
# %%
# Prepare data for export
# usol is (Nx, Ny, Time). Permute to (Time, Nx, Ny) for standard video format
usol_save = permutedims(usol, (3, 1, 2))
t_array = collect(Float64.(sol_split.t))
x_array = ops.x[:, 1] # Extract 1D axis from meshgrid
y_array = ops.y[1, :] # Extract 1D axis from meshgrid

npz_path = joinpath(@__DIR__, "cahn_hilliard_spectral.npz")

npzwrite(npz_path, Dict(
    "t" => t_array,
    "x" => x_array,
    "y" => y_array,
    "usol" => usol_save
))

println("✓ Data saved as NPZ to cahn_hilliard_spectral.npz")

# ----- Verify the saved file ----------------------------------------
println("\n--- File Verification ---")
file_size = filesize(npz_path)
println("File size: $(file_size) bytes ($(round(file_size/1024/1024, digits=2)) MB)")

data_check = npzread(npz_path)
println("Keys in file: ", keys(data_check))
println("Shape of usol: ", size(data_check["usol"]))
println("Shape of t: ", size(data_check["t"]))
println("Shape of x: ", size(data_check["x"]))
println("Shape of y: ", size(data_check["y"]))