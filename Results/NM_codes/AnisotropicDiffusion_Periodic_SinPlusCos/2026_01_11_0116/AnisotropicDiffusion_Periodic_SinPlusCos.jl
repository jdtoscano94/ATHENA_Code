# %% [markdown]
# # Preamble: Solver and Analytical Framework Initialization
# 
# ## Framework: Method of Characteristics (MoC) for Hyperbolic Conservation Laws
# Initializes the components to generate the **Exact Entropy Solution** for scalar conservation laws of the form $\partial_t u + \partial_x f(u) = 0$.
# 
# 1.  **Analytical Core (`Roots`, `LinearAlgebra`)**:
#     * Implements the **Lagrangian** perspective, tracing characteristic curves $x(t) = x_0 + f'(u_0)t$.
#     * Solves the implicit characteristic equation $x - x_0 - u(x_0)t = 0$ using efficient root-finding algorithms (Secant Method).
#     * **Shock Fitting**: Applies the **Rankine-Hugoniot condition** (via the Equal Area Rule) to geometrically construct the vertical shock interface where characteristics cross, ensuring the unique physical (entropy) solution.
# 
# 2.  **Diagnostics (`Plots`, `NPZ`)**:
#     * **`Plots`**: Visualizes the exact solution $u(x,t)$ to verify shock sharpness and absence of numerical oscillations (Gibbs phenomenon).
#     * **`NPZ`**: Exports the high-fidelity "Ground Truth" tensor to `.npz` format, serving as the absolute reference for PINN training and Trixi verification.
# 

# %%
using Plots
using Roots
using LinearAlgebra
using NPZ

# %% [markdown]

# ## Cell 1: Core Analytical Solver Definition
# ## Algorithm: Exact Analytical Evaluation (Fourier Modes)
# The function `solve_diffusion_analytical` computes the exact solution for the
# linear anisotropic diffusion equation on a periodic square.
#
# ### Physics:
# Equation: u_t = 0.25 * (u_xx + 2u_xy + u_yy)
# Domain: [0, 2π] x [0, 2π]
# IC: u(x,y,0) = sin(x+y) + cos(x-y)
#
# ### Solution:
# By diagonalizing the operator in rotated coordinates (ξ=x+y, η=x-y),
# the solution is derived as:
# u(x,y,t) = exp(-1.0 * t) * sin(x + y) + cos(x - y)
"""
    solve_diffusion_analytical(x, y, t)
Evaluates the exact solution for the 2D anisotropic diffusion problem.
Inputs x and y can be scalars or arrays (broadcasting is used).
"""
function solve_diffusion_analytical(x, y, t)
    return @. exp(-1.0 * t) * sin(x + y) + cos(x - y)
end
# %% [markdown]


# ## Cell 2: Problem Configuration and Coordinate Mapping
# ## Physical Problem
# * **Domain:** (x,y) ∈ [0, 2π]²
# * **Time:** t ∈ [0, 2]
# * **IC:** u₀(x,y) = sin(x+y) + cos(x-y)

# --- 1. Define Physical Grid (2D) ---
x_min, x_max = 0.0, 2π
y_min, y_max = 0.0, 2π
t_start, t_final = 0.0, 2.0

N_samples = 256 
N_time_steps = 200

x_vals = range(x_min, x_max, length=N_samples)
y_vals = range(y_min, y_max, length=N_samples)
t_array = range(t_start, t_final, length=N_time_steps)

println("Problem configured.")
println("Domain: [0, 2π] x [0, 2π]")
println("Grid: $N_samples x $N_samples spatial points.")
println("Time: [0, 2] with $N_time_steps snapshots.")
# %% [markdown]

# ## Cell 3: Time Evolution Loop
#
# ## Batch Processing
# Unlike time-stepping solvers, the Method of Characteristics allows independent evaluation. We loop through the requested time steps to construct the full solution tensor `usol` for visualization and PINN training data.
#
# * **Complexity:** $O(N_x \times N_t)$ independent root-finding operations.
# * **Precision:** Valid at $t=0$ and $t > t_{shock}$ equally.
# %%
# Initialize solution matrix (Space x Time) to match PlotData1D conventions usually used in Trixi
# However, heatmap usually expects (x, y, z). We will store as (Space, Time) locally.
# Initialize solution tensor (X x Y x Time)
usol = zeros(N_samples, N_samples, N_time_steps)

println("Starting time evolution loop...")

for (i, t_current) in enumerate(t_array)
    # Evaluate exact solution on the 2D grid
    # Broadcasting x_vals (vector) and y_vals' (row vector) creates the 2D grid evaluation
    u_slice = solve_diffusion_analytical(x_vals, y_vals', t_current)
    
    # Store in tensor
    usol[:, :, i] = u_slice
end

println("Evolution complete.")
# %% [markdown]

# ## Cell 4: Visualization Dashboard
#
# ## Dual-Panel Analysis
# 1.  **Space-Time Heatmap (Left):** Visualizes the characteristic convergence and shock formation over the full duration.
# 2.  **Final Time Slice (Right):** Provides a precise view of the shock profile at $t=1.0$ to verify the "Equal Area" cut.
# %%
# ----- Visualization: 2D Snapshots ----------------
# Strategy: Plot u(x,y,t) at four distinct time instants
target_times = [0.0, 2.0/3.0, 4.0/3.0, 2.0]
plot_list = []

println("Generating snapshots...")

for t_target in target_times
    # Find closest time index
    idx = argmin(abs.(t_array .- t_target))
    t_actual = t_array[idx]
    
    # Extract field (x, y) at time t
    # usol is (N_x, N_y, N_t). heatmap(x, y, z) maps z[i,j] to x[i], y[j]
    u_field = usol[:, :, idx]
    
    p = heatmap(x_vals, y_vals, u_field,
        title = "t ≈ $(round(t_actual, digits=2))",
        xlabel = "x", ylabel = "y",
        colorbar = false,
        color = :viridis,
        aspect_ratio = :equal,
        xlims = (0, 2π), ylims = (0, 2π)
    )
    push!(plot_list, p)
end

# ----- Combine and Save ----------------
 p_combined = plot(plot_list..., 
    layout = (1, 4),
    size = (1600, 380),
    margin = 5Plots.mm
)

savefig(p_combined, joinpath(@__DIR__, "summary_all.png"))
display(p_combined)

println("Dashboard saved to summary_all.png")
# %% [markdown]

# ## Cell 5: Data Persistence and Verification
# ## Artifact Generation
# Save the high-fidelity exact solution to `anisotropic_diffusion.npz`.
# Structure: (Time, X, Y)

# Permute to (Time, X, Y) for Python/Trixi compatibility
usol_save = permutedims(Float64.(usol), (3, 1, 2))
t_save = collect(Float64.(t_array))
x_save = collect(Float64.(x_vals))
y_save = collect(Float64.(y_vals))

npz_path = joinpath(@__DIR__, "anisotropic_diffusion.npz")
npzwrite(npz_path, Dict(
    "t" => t_save,
    "x" => x_save,
    "y" => y_save,
    "usol" => usol_save
))

println("Data saved as NPZ to anisotropic_diffusion.npz")

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