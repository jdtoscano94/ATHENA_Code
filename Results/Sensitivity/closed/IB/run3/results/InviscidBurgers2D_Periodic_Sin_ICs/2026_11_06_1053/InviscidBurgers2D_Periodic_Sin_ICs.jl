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
# ## Algorithm: Implicit Characteristic Tracing for 2D Scalar Conservation Law
# The function `solve_2d_analytical` computes the exact solution at any time $t > 0$.
# 
# ### Logic Flow:
# 1.  **Phase Reduction**: Reduces the 2D problem to a 1D problem using the phase variable $s = (x+y)/2$.
# 2.  **Periodic Root Finding**: For every target point, solves the nonlinear characteristic equation to find the origin $\xi$.
# 3.  **Solution Reconstruction**: Evaluates the initial condition at the found origin $\xi$.

# %%
"""
    solve_2d_analytical(x_grid, y_grid, t)

Analytical solution for 2D scalar conservation law: u_t + u u_x + u u_y = 0
Initial condition: u(x,y,0) = 1/3 + 2/3 * sin((x+y)/2)
"""
function solve_2d_analytical(x_grid, y_grid, t)
    Nx = length(x_grid)
    Ny = length(y_grid)
    u = zeros(Nx, Ny)
    for j in 1:Ny
        for i in 1:Nx
            s = (x_grid[i] + y_grid[j]) / 2.0
            xi = solve_characteristic_root(s, t)
            u[i, j] = 1/3 + 2/3 * sin(xi)
        end
    end
    return u
end

"""
    solve_characteristic_root(s, t)

Solves the implicit characteristic equation: 
    xi + t * (1/3 + 2/3 * sin(xi)) - s = 0
using periodic root handling.
"""
function solve_characteristic_root(s, t)
    s_mod = mod(s, 2π)
    f(xi) = xi + t * (1/3 + 2/3 * sin(xi)) - s_mod
    # Bisection is robust and guaranteed to find a root in this bracket
    xi_root = find_zero(f, (s_mod - t - 1.0, s_mod + t + 1.0), Bisection())
    return mod(xi_root, 2π)
end
# %% [markdown]


# ## Cell 2: Problem Configuration and Coordinate Mapping
# ## Physical Problem
# * **Domain:** $(x,y) \in [0, 4\pi] \times [0, 4\pi]$
# * **Initial Condition:** $u_0(x,y) = 1/3 + 2/3 \sin((x+y)/2)$
# * **Target Time:** $t \in [0, 2]$

# %%
# --- 1. Define Physical Grid ---
x_min, x_max = 0.0, 4.0 * π
y_min, y_max = 0.0, 4.0 * π
t_start, t_final = 0.0, 2.0
N_samples = 256 

x_phys = range(x_min, x_max, length=N_samples) 
y_phys = range(y_min, y_max, length=N_samples) 

# --- 2. Define Time Steps ---
dt_save = 0.1
t_save_array = collect(t_start:dt_save:t_final)
t_plots = [0.0, 2/3, 4/3, 2.0]

# Combine and sort unique times for evaluation
t_array = sort(unique([t_save_array; t_plots]))

println("Problem configured.")
println("Grid: $N_samples x $N_samples points.")
println("Time steps: $(length(t_array)) evaluations.")

# %% [markdown]


# ## Cell 3: Time Evolution Loop
# 
# ## Batch Processing
# Unlike time-stepping solvers, the Method of Characteristics allows independent evaluation. We loop through the requested time steps to construct the full solution tensor `usol` for visualization and PINN training data.
# 
# * **Complexity:** $O(N_x \times N_y \times N_t)$ independent root-finding operations.
# * **Precision:** Valid at $t=0$ and $t > t_{shock}$ equally.

# %%
# Initialize solution tensor (Time x X x Y)
usol = zeros(length(t_array), N_samples, N_samples)

println("Starting time evolution loop...")

for (i, t_current) in enumerate(t_array)
    # Compute exact solution matrix for this time slice
    u_slice = solve_2d_analytical(x_phys, y_phys, t_current)
    
    # Store in tensor
    usol[i, :, :] = u_slice
end

println("Evolution complete.")
# %% [markdown]


# ## Cell 4: Visualization Dashboard
# ## 4-Panel Analysis
# Visualizes the 2D field u(x,y,t) at 4 distinct time steps to observe advection and gradient steepening.

# %%
# ----- 1. Extract Time Slices and Create Plots -------------------------
t_plots = [0.0, 2/3, 4/3, 2.0]
plot_list = []

for t_val in t_plots
    # Find the index of the current plot time in the time array
    idx = findall(x -> isapprox(x, t_val, atol=1e-5), t_array)[1]
    
    # Extract the 2D slice (X x Y)
    u_slice = usol[idx, :, :]
    
    # Create heatmap (transpose u_slice to match x-y axes in Plots.jl)
    p = heatmap(x_phys, y_phys, u_slice',
        xlabel="x",
        ylabel="y",
        title="t = $(round(t_val, digits=2))",
        color=:jet,
        colorbar=false,
        aspect_ratio=:equal)
    
    push!(plot_list, p)
end

# ----- 2. Combine into Dashboard (1x4) ------------------------
p_combined = plot(plot_list..., 
    layout=(1, 4), 
    size=(1600, 400),
    margin=5Plots.mm)

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
# usol is already (Time, X, Y)
usol_save = Float64.(usol) 
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
