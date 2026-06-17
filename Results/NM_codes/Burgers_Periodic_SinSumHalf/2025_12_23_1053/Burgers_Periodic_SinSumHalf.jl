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
# ## Algorithm: Implicit Characteristic Tracing with Shock Fitting
# The function `solve_burgers_analytical` computes the exact solution at any time $t > 0$ without time-stepping errors.
# 
# ### Logic Flow:
# 1.  **Shock Tracking**: Calculates the exact position $x_s(t) = \pi + \alpha t$ where the shock forms (for a sine wave).
# 2.  **Domain Decomposition**: Splits the domain into **Pre-shock** (Expansion) and **Post-shock** (Compression) regions based on $x_s(t)$.
# 3.  **Root Finding**: For every target point $x_i$, solves the nonlinear characteristic equation to find the origin $\xi$.
# 4.  **Entropy Enforcement**: Uses geometric reflection (`reverse` logic) to handle the multi-valued fold and strictly enforce the Equal Area Rule.

# %%
"""
    solve_burgers_analytical(x_in, t, alpha, beta)

Analytical solution for 1D Burgers' equation: u_t + u u_x = 0
Initial condition: u(x,0) = alpha + beta * sin(x)

Handles shock formation and Rankine-Hugoniot conditions using the 
Equal Area Rule logic.
"""
function solve_burgers_analytical(x_in, t, alpha, beta)
     # Work on a copy to avoid modifying the input array
     x = collect(Float64, x_in)
     pi_val = π
 
     # 1. Calculate Shock Location
     # The shock forms at x = π (for sine) and moves with speed 'alpha'
     d2 = pi_val + alpha * t
 
     # 2. Handle Periodicity
     # If the shock has moved past 2π, shift x coordinates to track it
     if (pi_val + alpha * t > 2.0 * pi_val)
         shift_count = floor((pi_val + alpha * t) / (2.0 * pi_val))
         shift_val = 2.0 * pi_val * shift_count
         x .+= shift_val
     end
 
     # 3. Split Domain relative to the Shock
     idx1 = x .<= d2
     idx2 = x .> d2
 
     x1 = x[idx1]
     x2 = x[idx2]
 
     # Prepare output array
     vv = similar(x)
 
     # 4. Solve for Left Side (Pre-shock / Expansion)
     if !isempty(x1)
         xn = x1 .- alpha * t
         u_vals = map(xi -> solve_characteristic_root(xi, t, beta), xn)
         v = alpha .+ beta .* sin.(u_vals)
         vv[idx1] = v
     end
 
     # 5. Solve for Right Side (Post-shock / Compression)
     if !isempty(x2)
         x2_flipped = reverse(2.0 * d2 .- x2)
         xn2 = x2_flipped .- alpha * t
         u2_vals = map(xi -> solve_characteristic_root(xi, t, beta), xn2)
         v2 = alpha .+ beta .* sin.(u2_vals)
         v2_final = reverse(2.0 * alpha .- v2)
         vv[idx2] = v2_final
     end
 
     return vv
 end
 
 """
     solve_characteristic_root(x_target, t, beta)
 
 Solves the implicit characteristic equation: 
     x_target - x0 - beta * sin(x0) * t = 0
 """
 function solve_characteristic_root(x_target, t, beta)
     f(x0) = x_target - x0 - beta * t * sin(x0)
     # Order1 (Secant method) is robust and efficient here
     return find_zero(f, 0.0, Order1()) 
 end

# %% [markdown]

# ## Cell 2: Problem Configuration and Coordinate Mapping
# ## Physical Problem
# * **Domain:** $(x,y) \in [0, 4\pi] \times [0, 4\pi]$
# * **Initial Condition:** $u_0(x,y) = \frac{1}{3} + \frac{2}{3}\sin(\frac{x+y}{2})$
# * **Time:** $t \in [0, 2]$

# ## Characteristic Reduction
# The 2D problem reduces to a 1D Burgers' equation along the coordinate $s = (x+y)/2$.
# We solve for $s \pmod{2\pi}$ using the analytical kernel.

# %%
# --- 1. Define Physical Grid (2D) ---
x_min, x_max = 0.0, 4.0 * pi
y_min, y_max = 0.0, 4.0 * pi
t_start, t_final = 0.0, 2.0

N_samples = 256 
N_time_steps = 200

x_vals = range(x_min, x_max, length=N_samples)
y_vals = range(y_min, y_max, length=N_samples)
t_array = range(t_start, t_final, length=N_time_steps)

println("Problem configured (2D Scalar Burgers).")
println("Domain: [0, 4π] x [0, 4π]")
println("Grid: $N_samples x $N_samples points, $N_time_steps time steps.")
# %% [markdown]


# ## Cell 3: Time Evolution Loop
#
# ## Batch Processing
# Unlike time-stepping solvers, the Method of Characteristics allows independent evaluation. We loop through the requested time steps to construct the full solution tensor `usol` for visualization and PINN training data.
#
# * **Complexity:** $O(N_x \times N_t)$ independent root-finding operations.
# * **Precision:** Valid at $t=0$ and $t > t_{shock}$ equally.
# %%
# Initialize 3D solution array (x, y, t)
usol = zeros(N_samples, N_samples, N_time_steps)

println("Starting 2D time evolution loop...")

# Precompute spatial grid for s = (x+y)/2
# We use broadcasting to create the 2D grid of s-values
# x is dim 1, y is dim 2
s_grid = [(x + y) / 2.0 for x in x_vals, y in y_vals]

# Map s to [0, 2π) for the solver.
# The IC u0(s) = 1/3 + 2/3 sin(s) is 2π-periodic in s.
# The template solver handles a single period [0, 2π] with one shock.
s_canonical = mod.(s_grid, 2 * pi)

for (i, t_current) in enumerate(t_array)
    # Parameters for u_0 = 1/3 + 2/3 sin(s)
    alpha = 1.0 / 3.0
    beta = 2.0 / 3.0

    # Solve 1D Burgers on the s-coordinate
    # The solver handles the shock fitting logic for the 1D profile
    # We flatten s_canonical to pass to the 1D solver, then reshape
    u_flat = solve_burgers_analytical(vec(s_canonical), t_current, alpha, beta)

    # Reshape back to 2D grid and store
    usol[:, :, i] = reshape(u_flat, N_samples, N_samples)
end

println("Evolution complete.")
# %% [markdown]


# ## Cell 4: Visualization Dashboard
# ----- Visualization: 2D Snapshots at Key Times ----------------
target_times = [0.0, 0.5, 1.0, 2.0]
plot_list = []

println("Generating 2D plots for times: $target_times")

for t_req in target_times
    # Find closest time index
    idx = argmin(abs.(t_array .- t_req))
    t_act = t_array[idx]
    
    # Extract field (transpose for correct x-y orientation in heatmap)
    # usol is [x, y, t]. heatmap(x, y, z) typically maps rows of z to y, cols to x.
    # Transposing usol[:,:,idx] gives shape (Ny, Nx) where rows~y, cols~x.
    u_field = usol[:, :, idx]'
    
    p = heatmap(x_vals, y_vals, u_field,
        title = "t ≈ $(round(t_act, digits=2))",
        xlabel = "x", ylabel = "y",
        color = :jet,
        colorbar = false,
        aspect_ratio = :equal,
        clims = :auto # Allow dynamic scaling per panel as shocks develop
    )
    push!(plot_list, p)
end

# ----- Combine and Save -------------------------------
p_combined = plot(plot_list..., 
    layout = (1, 4), 
    size = (1600, 400),
    margin = 5Plots.mm
)

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
# Rearrange to (Time, X, Y) for compatibility with standard ML/Data tools
usol_save = permutedims(Float64.(usol), (3, 1, 2))
t_save = collect(Float64.(t_array))
x_save = collect(Float64.(x_vals))
y_save = collect(Float64.(y_vals))

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
