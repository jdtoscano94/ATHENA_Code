# %% [markdown]
# # Preamble: Solver and Analytical Framework Initialization
#
# ## Framework: Method of Characteristics (MoC) for 2D Hyperbolic Conservation Laws
# Initializes the components to generate the **Exact Solution** for the 2D scalar conservation law $\partial_t u + u \partial_x u + u \partial_y u = 0$.
#
# 1.  **Analytical Core (`Roots`, `LinearAlgebra`)**:
#     * Implements the **Lagrangian** perspective, tracing characteristic curves where $u$ is constant.
#     * The characteristic equations are $dx/dt = u$, $dy/dt = u$, which implies that the solution depends on the Lagrangian coordinate $s = x+y$.
#     * Solves the implicit characteristic equation $s - s_0 - 2u_0(s_0)t = 0$ using efficient root-finding algorithms to find the initial coordinate $s_0$.
#
# 2.  **Diagnostics (`Plots`, `NPZ`)`:
#     * **`Plots`**: Visualizes the exact solution $u(x,y,t)$ at different time steps to show wave steepening and shock formation.
#     * **`NPZ`**: Exports the high-fidelity "Ground Truth" tensor to `.npz` format, serving as a reference for other numerical methods.
#
# %%
using Plots
using Roots
using LinearAlgebra
using NPZ
# %% [markdown]


# ## Cell 1: Core Analytical Solver Definition
#
# ## Algorithm: Implicit Characteristic Tracing for 2D Burgers' Equation
# The function `solve_burgers_2d_analytical` computes the exact solution at any time $t \ge 0$ by tracing characteristics back to their origin.
#
# ### Logic Flow:
# 1.  **Coordinate Transformation**: For each target point $(x, y)$ at time $t$, compute the Lagrangian coordinate $s = x + y$.
# 2.  **Root Finding**: Solve the implicit characteristic equation $s - s_0 - 2u_0(s_0)t = 0$ to find the initial Lagrangian coordinate $s_0$.
# 3.  **Solution Evaluation**: The solution $u(x, y, t)$ is constant along characteristics, so it is given by the initial condition evaluated at $s_0$, i.e., $u(x,y,t) = u_0(s_0)$.
"""
    u0_func(s0)
Defines the initial condition for the 2D Burgers' problem based on the
Lagrangian coordinate s0 = x0 + y0.
IC: u(x,y,0) = 1/3 + 2/3 * sin((x+y)/2)
"""
u0_func(s0) = 1/3 + 2/3 * sin(s0 / 2)
"""
    solve_s0_root(s_target, t)
Solves the implicit characteristic equation for the 2D Burgers' problem:
s_target - s0 - 2 * u0_func(s0) * t = 0
"""
function solve_s0_root(s_target, t)
    # Define the function whose root we want to find
    f(s0) = s_target - s0 - 2 * u0_func(s0) * t
    
    # For t=0, the solution is trivial: s0 = s_target
    if t == 0.0
        return s_target
    end
    
    # Use a robust bracketing method to fix convergence failures. The displacement
    # |s_target - s0| is bounded by 2 * max|u0| * t. With max|u0|=1 and max(t)=2,
    # the max displacement is 4. A search interval of +/- 5 is safe.
    search_interval = (s_target - 5.0, s_target + 5.0)
    return find_zero(f, search_interval)
end
"""
    solve_burgers_2d_analytical(xx, yy, t)
Computes the analytical solution for the 2D Inviscid Burgers' equation on a grid.
"""
function solve_burgers_2d_analytical(xx, yy, t)
    # Ensure the input grids have the same dimensions
    @assert size(xx) == size(yy) "Input grids xx and yy must have the same size."
    # Initialize the solution grid
    u_sol = similar(xx, Float64)
    # Iterate over each point in the grid
    for I in CartesianIndices(xx)
        s_target = xx[I] + yy[I]
        s0 = solve_s0_root(s_target, t)
        u_sol[I] = u0_func(s0)
    end
    return u_sol
end
# %% [markdown]


# ## Cell 2: Problem Configuration
#
# ## Physical Problem
# * **Domain:** $(x,y) \in [0, 4\pi]^2$
# * **Initial Condition:** $u_0(x,y) = \frac{1}{3} + \frac{2}{3}\sin\left(\frac{x+y}{2}\right)$
# * **Target Times:** $t \in [0, 2]$
# %%
# --- 1. Define Physical Grid and Time Points ---
x_min, x_max = 0.0, 4.0 * pi
y_min, y_max = 0.0, 4.0 * pi
t_final = 2.0
N_samples = 128 # Resolution for each spatial dimension
# Define the 1D coordinate arrays
x_coords = range(x_min, x_max, length=N_samples)
y_coords = range(y_min, y_max, length=N_samples)
# Create 2D meshgrid
xx = [x for x in x_coords, y in y_coords]
yy = [y for x in x_coords, y in y_coords]
# Define specific time points for generating snapshots
t_array = [0.0, 0.5, 1.0, 1.5, 2.0]
t_array=[ 0.0
0.4000000000000002
0.8000000000000005
1.2000000000000008
1.6000000000000012
2.0000000000000013]
N_time_steps = length(t_array)
println("Problem configured for 2D Inviscid Burgers' equation.")
println("Grid: $N_samples x $N_samples points.")
println("Time snapshots: ", t_array)


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
usol = zeros(N_samples, N_samples, N_time_steps)
println("Starting time evolution loop...")
for (i, t_current) in enumerate(t_array)
    # Compute exact 2D solution slice for this time
    u_slice = solve_burgers_2d_analytical(xx, yy, t_current)
    # Store the 2D slice in the 3D tensor
    usol[:, :, i] = u_slice
    println("Computed solution for t = $t_current")
end
println("Evolution complete.")
# %% [markdown]


# ## Cell 4: Visualization Dashboard
#
# ## Multi-Panel Analysis
# A series of 2D heatmaps are generated to visualize the solution $u(x,y,t)$ at the specified time snapshots. This allows for a clear view of the wave steepening process in two dimensions.
# %%
# --- 1. Determine consistent color limits for all plots ---
clim_min = minimum(usol)
clim_max = maximum(usol)
clims = (clim_min, clim_max)
# --- 2. Create a plot for each time snapshot ---
plot_list = []
for i in 1:N_time_steps
    t_current = t_array[i]
    p = heatmap(x_coords, y_coords, usol[:, :, i]', # Transpose for correct orientation
        xlabel="x",
        ylabel="y",
        title="Solution at t = $(round(t_current, digits=2))",
        color=:jet,
        aspect_ratio=:equal,
        clims=clims)
    push!(plot_list, p)
end
# --- 3. Combine into a single dashboard ---
p_combined = plot(plot_list...,
    layout=(2, 3),
    size=(1200, 700),
    margin=5Plots.mm)
savefig(p_combined, joinpath(@__DIR__, "summary_all.png"))
display(p_combined)
println("Dashboard saved to summary_all.png")


# ## Cell 5: Data Persistence and Verification
#
# ## Artifact Generation
# We save the high-fidelity exact solution to `burgers.npz`. This file is critical as it serves as the **Ground Truth** for:
# 1.  **Advisor Agent:** To calculate L2 error of numerical runs.
# 2.  **PINN Trainer:** To provide supervised loss data.
# %%
# Permute dimensions from (X, Y, Time) to (Time, X, Y) for Python/PINN convention
usol_save = Float64.(permutedims(usol, (3, 1, 2)))
# Collect coordinate vectors
t_save = collect(Float64.(t_array))
x_save = collect(Float64.(x_coords))
y_save = collect(Float64.(y_coords))
npz_path = joinpath(@__DIR__, "burgers.npz")
npzwrite(npz_path, Dict(
    "t" => t_save,
    "x" => x_save,
    "y" => y_save,
    "usol" => usol_save
))
println("Data saved as NPZ to burgers.npz")
# ----- Verify the saved file -----------------------------------------------
println("\n--- File Verification ---")
file_size = filesize(npz_path)
println("File size: $(file_size) bytes ($(round(file_size/1024/1024, digits=2)) MB)")
data_check = npzread(npz_path)
println("Keys in file: ", keys(data_check))
println("Shape of usol: ", size(data_check["usol"]))
println("Shape of t: ", size(data_check["t"]))
println("Shape of x: ", size(data_check["x"]))
println("Shape of y: ", size(data_check["y"]))
