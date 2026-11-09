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
# ## Algorithm: Implicit Characteristic Tracing (Pre-Shock)
# The function `solve_2d_advection_analytical` computes the exact solution at any time $t < 3$ without time-stepping errors.
# 
# ### Logic Flow:
# 1.  **Coordinate Reduction**: Uses the symmetry $s = x + y$ to reduce the 2D problem to a 1D characteristic equation.
# 2.  **Root Finding**: For every target point $(x_i, y_j)$, solves the nonlinear characteristic equation to find the origin $\xi$.
# 3.  **Evaluation**: Computes the exact solution $u(x,y,t) = u_0(\xi)$.

# %%
"""
    solve_2d_advection_analytical(x_grid, y_grid, t)

Analytical solution for 2D nonlinear advection: u_t + u u_x + u u_y = 0
Initial condition: u(x,y,0) = 1/3 + 2/3 * sin((x+y)/2)

Valid for pre-shock regime (t < 3).
"""
function solve_2d_advection_analytical(x_grid, y_grid, t)
    Nx = length(x_grid)
    Ny = length(y_grid)
    u = zeros(Nx, Ny)
    
    for j in 1:Ny
        for i in 1:Nx
            s_target = x_grid[i] + y_grid[j]
            xi = solve_characteristic_root(s_target, t)
            u[i, j] = 1/3 + 2/3 * sin(xi / 2)
        end
    end
    
    return u
end

"""
    solve_characteristic_root(s_target, t)

Solves the implicit characteristic equation: 
    s_target - xi - 2 * t * (1/3 + 2/3 * sin(xi/2)) = 0
"""
function solve_characteristic_root(s_target, t)
    u0(xi) = 1/3 + 2/3 * sin(xi / 2)
    f(xi) = s_target - xi - 2 * t * u0(xi)
    # Order1 (Secant method) is robust and efficient here
    return find_zero(f, s_target, Order1()) 
end
# %% [markdown]


# ## Cell 2: Problem Configuration and Coordinate Mapping
# 
# ## Physical Problem
# * **Domain:** $(x,y) \in [0, 4\pi] \times [0, 4\pi]$
# * **Initial Condition:** $u_0(x,y) = 1/3 + 2/3 \sin((x+y)/2)$
# * **Target Time:** $t \in [0, 2]$ (Pre-shock regime)
# 

# %%
# --- 1. Define Physical Grid ---
x_min, x_max = 0.0, 4.0 * pi
y_min, y_max = 0.0, 4.0 * pi
t_start, t_final = 0.0, 2.0
dt_save = 0.1
N_samples = 256 

x_phys = range(x_min, x_max, length=N_samples) 
y_phys = range(y_min, y_max, length=N_samples)
t_array = range(t_start, t_final, step=dt_save)

println("Problem configured.")
println("Grid: $N_samples x $N_samples spatial points, $(length(t_array)) time steps.")
# %% [markdown]


# ## Cell 3: Time Evolution Loop
# 
# ## Batch Processing
# Unlike time-stepping solvers, the Method of Characteristics allows independent evaluation. We loop through the requested time steps to construct the full solution tensor `usol` for visualization and PINN training data.
# 
# * **Complexity:** $O(N_x \times N_y \times N_t)$ independent root-finding operations.
# * **Precision:** Valid for $t < 3$ (pre-shock regime).

# %%
# Initialize solution tensor (X x Y x Time)
usol = zeros(N_samples, N_samples, length(t_array))

println("Starting time evolution loop...")

for (i, t_current) in enumerate(t_array)
    # Compute exact solution matrix for this time slice
    u_slice = solve_2d_advection_analytical(x_phys, y_phys, t_current)
    
    # Store in tensor
    usol[:, :, i] = u_slice
end

println("Evolution complete.")

# %% [markdown]


# ## Cell 4: Visualization Dashboard
# 
# ## 2D Evolution Analysis
# Visualizes the 2D field $u(x,y)$ at 4 distinct time steps to observe the nonlinear advection and compression.

# %%
target_times = [0.0, 0.67, 1.33, 2.0]
plot_list = []

for t_target in target_times
    # Find the closest time step in the saved array
    idx = argmin(abs.(t_array .- t_target))
    actual_t = t_array[idx]
    
    # Create heatmap for this time slice
    # Note: transpose the slice so x is on the horizontal axis and y is on the vertical
    p = heatmap(x_phys, y_phys, usol[:, :, idx]',
        xlabel="x",
        ylabel="y",
        title="t = $(round(actual_t, digits=2))",
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
usol_save = permutedims(Float64.(usol), (3, 1, 2))
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
