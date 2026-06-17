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
    solve_moc_2d_scalar(xi_in, t)

Exact solution for the 2D scalar conservation law: u_t + u(u_x + u_y) = 0
Initial condition: u(x,y,0) = 1/3 + 2/3 * sin((x+y)/2)

Uses Method of Characteristics with Entropy-Consistent shock fitting (Equal Area Rule).
Operates on the characteristic coordinate ξ = x + y.
"""
function solve_moc_2d_scalar(xi_in, t)
    # Physics parameters derived in Strategy
    # u = alpha + beta * sin(xi/2)
    alpha = 1.0 / 3.0
    beta  = 2.0 / 3.0
    
    # Map to phase space theta = xi / 2
    # The problem is 4π-periodic in ξ, so 2π-periodic in θ.
    theta = collect(Float64, xi_in) ./ 2.0
    pi_val = π

    # 1. Calculate Shock Location in Phase Space
    # Shock forms at θ = π (inflection of sin) and moves with mean speed alpha
    theta_s = pi_val + alpha * t

    # 2. Handle Periodicity relative to Shock
    # Wrap theta to be within [theta_s - π, theta_s + π)
    # This ensures we correctly classify points as "Pre-shock" (Left) or "Post-shock" (Right)
    # regardless of how many times the domain has wrapped.
    theta_wrapped = @. mod(theta - theta_s + pi_val, 2.0 * pi_val) - pi_val + theta_s

    # 3. Split Domain relative to the Shock
    idx1 = theta_wrapped .<= theta_s # Left / Pre-shock / Expansion
    idx2 = theta_wrapped .> theta_s  # Right / Post-shock / Compression

    theta1 = theta_wrapped[idx1]
    theta2 = theta_wrapped[idx2]

    # Prepare output array
    vv = similar(theta)

    # 4. Solve for Left Side (Expansion Branch)
    if !isempty(theta1)
        # Transform to 0-mean frame
        theta_n = theta1 .- alpha * t
        # Solve implicit characteristic equation
        u_vals = map(x -> solve_characteristic_root(x, t, beta), theta_n)
        v = alpha .+ beta .* sin.(u_vals)
        vv[idx1] = v
    end

    # 5. Solve for Right Side (Compression Branch with Entropy Reflection)
    if !isempty(theta2)
        # Reflect coordinate across the shock to the expansion side
        theta2_flipped = reverse(2.0 * theta_s .- theta2)
        theta_n = theta2_flipped .- alpha * t
        
        # Solve root on the single-valued side
        u2_vals = map(x -> solve_characteristic_root(x, t, beta), theta_n)
        v2 = alpha .+ beta .* sin.(u2_vals)
        
        # Reflect value back: u - alpha = -(u_flipped - alpha)
        v2_final = reverse(2.0 * alpha .- v2)
        vv[idx2] = v2_final
    end

    return vv
end

"""
    solve_characteristic_root(x_target, t, beta)

Solves the implicit characteristic equation for the phase variable:
    x_target - x0 - beta * t * sin(x0) = 0

Uses Bisection method with guaranteed bounds to ensure convergence.
"""
function solve_characteristic_root(x_target, t, beta)
    # Define the residual function
    f(x0) = x_target - x0 - beta * t * sin(x0)
    
    # Determine bounds for the root x0
    # From x0 = x_target - beta * t * sin(x0), and sin(x0) in [-1, 1]:
    # x0 must be in [x_target - beta*t, x_target + beta*t]
    # We add a small epsilon to ensure the sign change occurs strictly inside or at bounds
    delta = abs(beta * t) + 1e-8
    bracket = (x_target - delta, x_target + delta)
    
    # Use Bisection for guaranteed convergence
    return find_zero(f, bracket, Bisection()) 
end
# %% [markdown]


# ## Cell 2: Problem Configuration and Coordinate Mapping
# ## Cell 2: Problem Configuration and Coordinate Mapping
#
# ## Physical Problem
# * **Domain:** $[0, 4\pi] \times [0, 4\pi]$
# * **Initial Condition:** $u(x,y,0) = \frac{1}{3} + \frac{2}{3}\sin\left(\frac{x+y}{2}\right)$
# * **Time Interval:** $t \in [0, 2]$
#
# ## Coordinate Transformation
# The Method of Characteristics solves the problem in the characteristic coordinate:
# $$ \xi = x + y $$
# The solution is periodic in $\xi$ and constant along characteristic lines in the $(\xi, t)$ plane.
# %%
# --- 1. Define Physical Grid (2D) ---
x_min, x_max = 0.0, 4.0 * pi
y_min, y_max = 0.0, 4.0 * pi
t_start, t_final = 0.0, 2.0
N_x, N_y = 128, 128
N_time_steps = 100
x_vals = range(x_min, x_max, length=N_x)
y_vals = range(y_min, y_max, length=N_y)
t_array = range(t_start, t_final, length=N_time_steps)
# Create 2D Meshgrid (y varies down rows, x varies across columns)
x_grid = repeat(x_vals', N_y, 1)
y_grid = repeat(y_vals, 1, N_x)
# --- 2. Characteristic Coordinate ---
# The solution depends on xi = x + y
xi_grid = x_grid .+ y_grid
println("Problem configured.")
println("Domain: [0, 4π] x [0, 4π]")
println("Grid: $N_x x $N_y spatial points, $N_time_steps time steps.")
println("Coordinate mapping xi = x + y prepared.")
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
# Initialize solution tensor (x, y, t)
usol = zeros(N_x, N_y, N_time_steps)

println("Starting time evolution loop...")

for (i, t_current) in enumerate(t_array)
    # Compute exact solution grid for this time slice
    # The solver operates on the characteristic coordinate xi = x + y
    u_slice = solve_moc_2d_scalar(xi_grid, t_current)
    
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
# Define snapshot times
snap_times = [0.0, 0.5, 1.0, 2.0]
plots_list = []

for t_snap in snap_times
    # Find closest index
    idx = argmin(abs.(t_array .- t_snap))
    actual_t = t_array[idx]
    
    # Extract 2D slice
    u_snap = usol[:, :, idx]
    
    p = heatmap(x_vals, y_vals, u_snap',
        title="t = $(round(actual_t, digits=2))",
        xlabel="x", ylabel="y",
        aspect_ratio=:equal,
        colorbar=false,
        clims=(0.0, 1.0), # Fix limits for visual consistency if desired, or remove for auto
        color=:viridis
    )
    push!(plots_list, p)
end

# Combine into 1x4 layout
p_combined = plot(plots_list..., layout=(1, 4), size=(1600, 350), margin=3Plots.mm)

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
# Transpose to (Time, Space) to match Trixi/Python convention usually preferred by PINNs
# Permute to (Time, x, y) for standard ML/Python compatibility
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