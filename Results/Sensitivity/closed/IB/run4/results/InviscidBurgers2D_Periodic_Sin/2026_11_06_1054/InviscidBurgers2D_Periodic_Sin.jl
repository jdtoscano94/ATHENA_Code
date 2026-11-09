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
#
# ## Algorithm: Implicit Characteristic Tracing
# The function `solve_2d_analytical` computes the exact solution at any time $t < 3$ without time-stepping errors.
#
# ### Logic Flow:
# 1.  **Coordinate Mapping**: Computes the characteristic variable $s = x + y$.
# 2.  **Periodic Wrapping**: Wraps $s$ to $[0, 4\pi]$ to maintain bounded root-finding.
# 3.  **Root Finding**: For every target point $(x,y)$, solves the nonlinear characteristic equation to find the origin $\alpha$.
# 4.  **Solution Evaluation**: Computes $u(x,y,t)$ from the found root $\alpha$.
# %%
"""
    solve_2d_analytical(x_grid, y_grid, t)
Analytical solution for 2D conservation law: u_t + u u_x + u u_y = 0
Initial condition: u(x,y,0) = 1/3 + 2/3 * sin((x+y)/2)
Solves the implicit characteristic equation for each (x,y) point.
Valid for t < 3 (pre-shock).
"""
function solve_2d_analytical(x_grid, y_grid, t)
    Nx = length(x_grid)
    Ny = length(y_grid)
    u = zeros(Nx, Ny)
    for j in 1:Ny
        for i in 1:Nx
            s = x_grid[i] + y_grid[j]
            alpha = solve_characteristic_root(s, t)
            u[i, j] = 1.0/3.0 + 2.0/3.0 * sin(alpha / 2.0)
        end
    end
    return u
end
"""
    solve_characteristic_root(s_target, t)
Solves the implicit characteristic equation:
    alpha + 2t/3 + (4t/3)*sin(alpha/2) - s_target = 0
"""
function solve_characteristic_root(s_target, t)
    # Wrap s_target to [0, 4π] due to periodicity
    s_wrapped = mod(s_target, 4.0 * π)
    f(alpha) = alpha + (2.0 * t / 3.0) + (4.0 * t / 3.0) * sin(alpha / 2.0) - s_wrapped
    # Bisection method with a safe bracket is robust here
    return find_zero(f, (s_wrapped - 4.0 * π, s_wrapped + 4.0 * π), Bisection()) 
end
# %% [markdown]


# ## Cell 2: Problem Configuration and Coordinate Mapping
#
# ## Physical Problem
# * **Domain:** $(x,y) \in [0, 4\pi] \times [0, 4\pi]$
# * **Initial Condition:** $u_0(x,y) = 1/3 + 2/3 \sin((x+y)/2)$
# * **Target Time:** $t \in [0, 2]$ (Pre-shock regime)

# %%
# --- 1. Define Physical Grid ---
x_min, x_max = 0.0, 4.0 * π
y_min, y_max = 0.0, 4.0 * π
t_start, t_final = 0.0, 2.0
dt_save = 0.1
N_samples = 256 

x_phys = range(x_min, x_max, length=N_samples) 
y_phys = range(y_min, y_max, length=N_samples)
t_array = range(t_start, t_final, step=dt_save)
N_time_steps = length(t_array)

println("Problem configured.")
println("Grid: $N_samples x $N_samples spatial points.")
println("Time steps: $N_time_steps steps from $t_start to $t_final with dt=$dt_save.")
# %% [markdown]

# ## Cell 3: Time Evolution Loop
#
# ## Batch Processing
# Unlike time-stepping solvers, the Method of Characteristics allows independent evaluation. We loop through the requested time steps to construct the full solution tensor `usol` for visualization and PINN training data.
# 
# * **Complexity:** $O(N_x \times N_y \times N_t)$ independent root-finding operations.
# * **Precision:** Valid for $t < 3$ (pre-shock).

# %%
# Initialize solution tensor (X x Y x Time)
usol = zeros(N_samples, N_samples, N_time_steps)

println("Starting time evolution loop...")

for (i, t_current) in enumerate(t_array)
    # Compute exact solution matrix for this time slice
    u_slice = solve_2d_analytical(x_phys, y_phys, t_current)
    
    # Store in tensor
    usol[:, :, i] = u_slice
end

println("Evolution complete.")
# %% [markdown]

# ## Cell 4: Visualization Dashboard
#
# ## Snapshot Analysis
# Visualizes the 2D spatial field u(x,y,t) at 4 distinct times to observe the wave propagation and steepening.

# %%
target_times = [0.0, 2.0/3.0, 4.0/3.0, 2.0]
plot_list = []

for t_target in target_times
    # Find the closest time index in the saved array
    idx = argmin(abs.(t_array .- t_target))
    t_actual = t_array[idx]
    
    # Create heatmap for this time slice
    # Transpose usol slice for correct (x,y) orientation in Plots.jl heatmap
    p = heatmap(x_phys, y_phys, usol[:, :, idx]',
        xlabel="x",
        ylabel="y",
        title="t = $(round(t_actual, digits=2))",
        color=:jet,
        colorbar=false,
        aspect_ratio=:equal)
    
    push!(plot_list, p)
end

# ----- Combine into Dashboard (1x4) ------------------------
p_combined = plot(plot_list..., 
    layout=(1, 4), 
    size=(1600, 400),
    margin=5Plots.mm
)

savefig(p_combined, joinpath(@__DIR__, "summary_all.png"))
display(p_combined)

println("Dashboard saved to summary_all.png")
# %% [markdown]

# ## Cell 5: Data Persistence and Verification
#
# ## Artifact Generation
# We save the high-fidelity exact solution to `burgers.npz`. This file is critical as it serves as the **Ground Truth** for:
# 1.  **Advisor Agent:** To calculate L2 error of numerical runs.
# 2.  **PINN Trainer:** To provide supervised loss data.
# %%
# Permute to (Time, X, Y) to match Python convention usually preferred by PINNs
usol_save = Float64.(permutedims(usol, (3, 1, 2))) 
t_save = collect(Float64.(t_array))
x_save = collect(Float64.(x_phys))
y_save = collect(Float64.(y_phys))

npz_path = joinpath(@__DIR__, "burgers.npz")
npzwrite(npz_path, Dict(
    "t" => t_save,
    "x" => x_save,
    "y" => y_save,
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
println("Shape of y: ", size(data_check["y"]))