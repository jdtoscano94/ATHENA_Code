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
    nn::Int           # Number of points
    L::Float64        # Domain length
    x::Vector{Float64}  # Spatial grid
    
    # Precomputed derivative operators
    ∂x::Vector{ComplexF64}   # First derivative
    ∂²x::Vector{ComplexF64}  # Second derivative
    ∂³x::Vector{ComplexF64}  # Third derivative
    ∂⁴x::Vector{ComplexF64}  # Fourth derivative
end

function SpectralOperators(nn::Int, domain::Vector{Float64})
    L = domain[2] - domain[1]
    dx = L / nn
    x = range(domain[1], domain[2]-dx, length=nn)
    
    # Frequency grid
    k = 2π .* fftfreq(nn, nn/L)
    ik = im .* k
    
    # Precompute all common derivative operators
    ∂x = ik
    ∂²x = ik.^2
    ∂³x = ik.^3
    ∂⁴x = ik.^4
    
    return SpectralOperators(nn, L, collect(x), ∂x, ∂²x, ∂³x, ∂⁴x)
end

# Custom derivative of any order
derivative(ops::SpectralOperators, order::Int) = (im .* 2π .* fftfreq(ops.nn, ops.nn/ops.L)).^order

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
nn = 512

domain = [-1.0, 1.0]

# Create spectral operators
ops = SpectralOperators(nn, domain)
x = ops.x

# Initial Condition: u(x,0) = -sin(π x)
u0 = -sin.(π .* x)
u0_hat = fft(u0)

# ----- Time Setup -----------------------------------------------
t_span = (0.0, 1.0)
# Save at frequent intervals (dt=0.01) to ensure we capture the required snapshots
t_save = 0.0:0.01:1.0
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
println("\nMethod: SplitODEProblem with ETDRK4 (Viscous Burgers)")

# Viscosity parameter
ν = 1.0 / (100 * π)

# Linear operator: ν * ∂²x
# In Fourier space: ν * (-k²)
linear_coeffs = ν .* ops.∂²x
L_matrix = Diagonal(linear_coeffs)
L_op = MatrixOperator(L_matrix)

# Nonlinear operator: -u * ∂x u = -0.5 * ∂x(u^2) (Conservative form)
function nonlinear_operator!(dN, u_hat, p, t)
    ops = p
    u = real(ifft(u_hat))
    
    # Compute flux in physical space: f(u) = 0.5 * u^2
    flux = 0.5 .* u.^2
    
    # Transform flux to spectral space
    flux_hat = fft(flux)
    
    # Compute derivative in spectral space: -ik * flux_hat
    dN .= -ops.∂x .* flux_hat
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


# ----- Process Solution -----------------------------------------
usol_fourier = hcat(sol_split.u...)
 n_time = length(sol_split.t)
usol = zeros(nn, n_time)

for i in 1:n_time
    usol[:, i] = real(ifft(usol_fourier[:, i]))
end

# ----- Visualization --------------------------------------------
x_plot = range(domain[1], domain[2], length=nn+1)
# Append the first point to the end to close the periodic loop for plotting
usol_plot = vcat(usol, usol[1:1, :]) 

# Strategy: Snapshots at t = 0, 0.25, 0.5, 1.0
target_times = [0.0, 0.25, 0.5, 1.0]
# Find indices in the saved solution closest to target times
indices = [argmin(abs.(sol_split.t .- t)) for t in target_times]

plots_list = []
for (i, idx) in enumerate(indices)
    t_val = sol_split.t[idx]
    u_val = usol_plot[:, idx]
    
    p = plot(x_plot, u_val,
        title="t = $(round(t_val, digits=2))",
        xlabel="x",
        ylabel= (i==1) ? "u(x,t)" : "",
        linewidth=2,
        legend=false,
        color=:blue,
        ylim=(-1.1, 1.1), # Bounded by initial condition magnitude
        grid=true,
        colorbar=false
    )
    push!(plots_list, p)
end

# Combine into 1x4 Dashboard
p_combined = plot(plots_list..., 
    layout=(1, 4), 
    size=(1200, 350),
    margin=5Plots.mm,
)
display(p_combined)

savefig(p_combined, joinpath(@__DIR__, "summary_all.png"))
println("✓ Dashboard saved to summary_all.png")
# %% [markdown]


# ## Cell 6: Data Export
#
# %%
# Transpose to (time, space) format for Python agents
usol_save = Float64.(usol_plot')
t_array = collect(Float64.(t_save))
x_array = collect(Float64.(x_plot))

npz_path = joinpath(@__DIR__, "burgers_spectral.npz")

npzwrite(npz_path, Dict(
    "t" => t_array,
    "x" => x_array,
    "usol" => usol_save
))

println("✓ Data saved as NPZ to burgers_spectral.npz")

# ----- Verify the saved file ----------------------------------------
println("\n--- File Verification ---")
file_size = filesize(npz_path)
println("File size: $(file_size) bytes ($(round(file_size/1024/1024, digits=2)) MB)")

data_check = npzread(npz_path)
println("Keys in file: ", keys(data_check))
println("Shape of usol: ", size(data_check["usol"]))
println("Shape of t: ", size(data_check["t"]))
println("Shape of x: ", size(data_check["x"]))
