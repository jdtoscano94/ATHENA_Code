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
struct SpectralOperators2D
    nx::Int
    ny::Int
    Lx::Float64
    Ly::Float64
    x::Vector{Float64}
    y::Vector{Float64}
    
    # Precomputed operators
    k2::Matrix{Float64}      # |k|^2
    dealias_mask::Matrix{Float64} # 2/3 rule mask
end

function SpectralOperators2D(nx::Int, ny::Int, domain_x::Vector{Float64}, domain_y::Vector{Float64})
    Lx = domain_x[2] - domain_x[1]
    Ly = domain_y[2] - domain_y[1]
    
    dx = Lx / nx
    dy = Ly / ny
    
    x = range(domain_x[1], domain_x[2]-dx, length=nx)
    y = range(domain_y[1], domain_y[2]-dy, length=ny)
    
    # Frequency grids
    kx = 2π .* fftfreq(nx, nx/Lx)
    ky = 2π .* fftfreq(ny, ny/Ly)
    
    # Create 2D grids of wavenumbers
    KX = reshape(kx, (nx, 1))
    KY = reshape(ky, (1, ny))
    
    k2 = (KX.^2) .+ (KY.^2)
    
    # De-aliasing Mask (2/3 Rule)
    kmax_x = maximum(abs, kx)
    kmax_y = maximum(abs, ky)
    
    cutoff_x = (2/3) * kmax_x
    cutoff_y = (2/3) * kmax_y
    
    mask = ones(Float64, nx, ny)
    for j in 1:ny
        for i in 1:nx
            if abs(KX[i]) >= cutoff_x || abs(KY[j]) >= cutoff_y
                mask[i, j] = 0.0
            end
        end
    end
    
    return SpectralOperators2D(nx, ny, Lx, Ly, collect(x), collect(y), k2, mask)
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
domain_x = [0.0, 2π]
domain_y = [0.0, 2π]
# Create spectral operators
ops = SpectralOperators2D(nx, ny, domain_x, domain_y)
X = repeat(ops.x, 1, ny)
Y = repeat(ops.y', nx, 1)
# Initial Condition: Random Noise
# Seed for reproducibility
Random.seed!(1234)
noise = 2.0 .* rand(nx, ny) .- 1.0 # Uniform [-1, 1]
u0 = 0.0 .+ 0.05 .* noise
u0_hat = vec(fft(u0))
# ----- Time Setup ---------------------------------
t_span = (0.0, 2.0)
t_save = [0.0, 0.1, 0.5, 2.0]
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
epsilon = 0.001
# Linear operator: L = k^2 - epsilon*k^4
# In Fourier space, Laplacian Δ -> -k^2.
# The linear part of CH is Δ(u^3 - u) - εΔ²u -> (-k^2) - ε(k^4) = -k^2 - εk^4?
# Wait, the equation is ∂u/∂t = Δ(u^3 - u) - εΔ²u
# Rearranging: ∂u/∂t = Δ(u^3) - Δu - εΔ²u
# Linear part L(u) = -Δu - εΔ²u
# In Fourier: L(u_hat) = -(-k^2)u_hat - ε(k^4)u_hat = (k^2 - εk^4)u_hat
linear_coeffs = ops.k2 .- epsilon .* (ops.k2.^2)
# Use DiagonalOperator for element-wise multiplication on the flattened state vector
L_op = DiagonalOperator(vec(linear_coeffs))

# Nonlinear operator: N(u) = Δ(u^3) -> -k^2 * fft(u^3)
function nonlinear_operator!(dN, u_hat, p, t)
    ops = p
    nx, ny = ops.nx, ops.ny

    # 1. Reshape flattened state to 2D for FFT
    u_hat_2d = reshape(u_hat, (nx, ny))

    # 2. Transform to physical space
    u = real(ifft(u_hat_2d))

    # 3. Compute nonlinear term u^3
    u3 = u.^3

    # 4. Transform back to spectral space
    u3_hat = fft(u3)

    # 5. Apply De-aliasing (2/3 Rule)
    u3_hat .*= ops.dealias_mask

    # 6. Apply Laplacian (-k^2) and flatten result
    # N = -k^2 * u3_hat
    dN .= vec(-ops.k2 .* u3_hat)
end
# Create the problem
prob_split = SplitODEProblem(L_op, nonlinear_operator!, u0_hat, t_span, ops)
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
@time sol_split = solve(prob_split, ETDRK4(),
                        dt=1e-4,
                        saveat=t_save,
                        progress=true,
                        progress_steps=100)
println("✓ Time integration completed: $(length(sol_split.t)) steps")
# ----- Process Solution -----------------------------------------
# sol_split.u is a vector of flattened arrays (nx*ny) due to DiagonalOperator usage
n_saves = length(sol_split.t)
usol = zeros(nx, ny, n_saves)

for i in 1:n_saves
    # Reshape flattened vector back to 2D for IFFT
    u_hat_2d = reshape(sol_split.u[i], (nx, ny))
    usol[:, :, i] = real(ifft(u_hat_2d))
end
# ----- Metrics: Mass Conservation -------------------------------
# Mass = Integral(u) dA approx Sum(u) * dx * dy
dx = ops.Lx / nx
dy = ops.Ly / ny
dA = dx * dy
mass_0 = sum(usol[:, :, 1]) * dA
mass_final = sum(usol[:, :, end]) * dA
mass_error = abs(mass_final - mass_0)
@printf("\n--- Conservation Metrics ---\n")
@printf("Mass (t=0.0): %.6f\n", mass_0)
@printf("Mass (t=%.1f): %.6f\n", t_span[2], mass_final)
@printf("Absolute Error: %.6e\n", mass_error)
# ----- Visualization --------------------------------------------
# Strategy: 1 Row x 4 Columns (Time), Heatmap, No Colorbar
plots_list = []
for i in 1:n_saves
    t_val = sol_split.t[i]
    p = heatmap(ops.x, ops.y, usol[:, :, i]',
        title="t = $(t_val)",
        xlabel="x", ylabel="y",
        aspect_ratio=:equal,
        colorbar=false,
        c=:viridis,
        xlims=(0, 2π), ylims=(0, 2π)
    )
    push!(plots_list, p)
end
p_combined = plot(plots_list...,
    layout=(1, 4),
    size=(1200, 350),
    margin=3Plots.mm
)
display(p_combined)
savefig(p_combined, joinpath(@__DIR__, "summary_all.png"))
println("✓ Dashboard saved to summary_all.png")
# %% [markdown]


# ## Cell 6: Data Export
#
# %%
# Transpose to (time, x, y) format for Python agents
# usol is currently (nx, ny, time)
usol_save = permutedims(Float64.(usol), (3, 1, 2))
t_array = collect(Float64.(t_save))
x_array = collect(Float64.(ops.x))
y_array = collect(Float64.(ops.y))

npz_path = joinpath(@__DIR__, "cahn_hilliard_diffeq.npz")

npzwrite(npz_path, Dict(
    "t" => t_array,
    "x" => x_array,
    "y" => y_array,
    "usol" => usol_save
))

println("✓ Data saved as NPZ to cahn_hilliard_diffeq.npz")

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