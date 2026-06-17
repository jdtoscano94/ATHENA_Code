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

Computes the exact entropy solution for the 1D Burgers equation 
u_t + u u_x = 0 with IC u(x,0) = alpha + beta * sin(x).

- Adapts to 2D grids by processing inputs element-wise.
- Enforces the entropy condition (Lax) by selecting the correct characteristic 
  root relative to the shock location at x_s = pi + alpha * t.
"""
function solve_burgers_analytical(x_in, t, alpha, beta)
    # Broadcast the point‑wise solver over the input array (handles 1D, 2D, etc.)
    return solve_burgers_point.(x_in, t, alpha, beta)
end

"""
    solve_burgers_point(x, t, alpha, beta)

Solves for a single point x at time t.
Logic:
1. Transform to Lagrangian frame X = x - alpha * t.
2. Map X to [0, 2π) to handle periodicity.
3. Identify shock location in X‑space (at X = π).
4. Bracket the root finding:
   - If X < π, the characteristic origin ξ must be in [0, π].
   - If X > π, the characteristic origin ξ must be in [π, 2π].
"""
function solve_burgers_point(x, t, alpha, beta)
    # 1. Shift to frame moving with the mean velocity alpha
    # The characteristic equation is: x = ξ + (alpha + beta*sin(ξ))*t
    # Rearranging: x - alpha*t = ξ + beta*t*sin(ξ)
    X_raw = x - alpha * t

    # 2. Map to fundamental period [0, 2π)
    X = mod(X_raw, 2π)

    # 3. Define the implicit relation for ξ: f(ξ) = ξ + beta*t*sin(ξ) - X = 0
    f(ξ) = ξ + beta * t * sin(ξ) - X

    # 4. Entropy Selection via Bracketing
    if X <= π
        ξ_root = find_zero(f, (0.0, π), Roots.Bisection())
    else
        ξ_root = find_zero(f, (π, 2π), Roots.Bisection())
    end

    # 5. Reconstruct solution u = alpha + beta * sin(ξ)
    return alpha + beta * sin(ξ_root)
end

# Helper for compatibility if needed, though logic is now inline
function solve_characteristic_root(x_target, t, beta)
    f(x0) = x_target - x0 - beta * t * sin(x0)
    return find_zero(f, 0.0, Order1())
end
# %% [markdown]


# ## Cell 2: Problem Configuration and Coordinate Mapping
#
# ## Physical Problem
# * **Domain:** $x \in [-1, 1]$
# * **Initial Condition:** $u_0(x) = -\sin(\pi x)$
# * **Target Time:** $t = 1.0$ (Post-shock regime)
#
# ## Canonical Transformation
# The analytical solver assumes a canonical domain $x \in [0, 2\pi]$ and $u \sim \sin(x)$. We map the physical coordinates:
#
# 1.  **Space:** $x_{can} = \pi(x_{phys} + 1)$
# 2.  **Time:** $t_{can} = \pi t_{phys}$ (Time scaling to preserve wave speed)
# 3.  **Phase/Amplitude:** The negative sign and amplitude are handled by setting parameters $\alpha=0, \beta=1$ and relying on the coordinate shift.
# %%
# --- 1. Define Physical Grid (2D) ---
x_min, x_max = 0.0, 4.0 * π
y_min, y_max = 0.0, 4.0 * π
t_start, t_final = 0.0, 2.0

# Resolution
Nx, Ny = 128, 128
x_phys = range(x_min, x_max, length=Nx)
y_phys = range(y_min, y_max, length=Ny)

# Time snapshots for visualization (0, 0.67, 1.33, 2.0)
t_snapshots = [0.0, 0.67, 1.33, 2.0]

# --- 2. Precompute Diagonal Coordinate ---
# The problem reduces to 1D along s = (x+y)/2
# We create a meshgrid for vectorized evaluation later
X_grid = repeat(x_phys', Ny, 1)
Y_grid = repeat(y_phys, 1, Nx)
S_grid = @. (X_grid + Y_grid) / 2.0

println("2D Problem configured: [0, 4π]x[0, 4π]")
println("Grid: $Nx x $Ny points.")
println("Snapshots: $t_snapshots")
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
# Initialize solution tensor (Space X, Space Y, Time)
num_snaps = length(t_snapshots)
usol_snapshots = zeros(Float64, Nx, Ny, num_snaps)

println("Starting time evolution for $num_snaps snapshots...")

# Physical parameters from Strategy
alpha_val = 1.0/3.0
beta_val = 2.0/3.0

for (i, t) in enumerate(t_snapshots)
    # The 2D problem reduces to 1D Burgers along the diagonal s = (x+y)/2.
    # We pass the precomputed S_grid (Nx x Ny) to the element-wise solver.
    # The solver handles the implicit relation: u = alpha + beta * sin(s - u*t)
    # and enforces the entropy condition.
    u_2d = solve_burgers_analytical(S_grid, t, alpha_val, beta_val)

    # Store in 3D tensor
    usol_snapshots[:, :, i] = u_2d
end

println("Evolution complete.")
# %% [markdown]


# ## Cell 4: Visualization Dashboard
#
# ## Dual-Panel Analysis
# 1.  **Space-Time Heatmap (Left):** Visualizes the characteristic convergence and shock formation over the full duration.
# 2.  **Final Time Slice (Right):** Provides a precise view of the shock profile at $t=1.0$ to verify the "Equal Area" cut.
# %%
# ----- Create 1x4 Dashboard of 2D Snapshots -------------------
plots_list = []
for i in 1:num_snaps
    t_val = t_snapshots[i]
    # Extract 2D slice for this time
    u_field = usol_snapshots[:, :, i]
    # Create heatmap
    # Note: heatmap expects (x, y, z). We transpose if necessary depending on backend,
    # but standard Plots.jl heatmap(x, y, z) works with z[i,j] corresponding to x[i], y[j].
    p = heatmap(x_phys, y_phys, u_field',
        title="t = $(round(t_val, digits=2))",
        xlabel=(i==1 ? "x" : ""), # Only label axes on first plot to save space
        ylabel=(i==1 ? "y" : ""),
        aspect_ratio=:equal,
        colorbar=false,
        color=:viridis,
        clims=(minimum(u_field), maximum(u_field)) # Auto-scale per plot
    )
    push!(plots_list, p)
end
# Combine into 1 row, 4 columns
p_combined = plot(plots_list...,
    layout=(1, 4),
    size=(1200, 350),
    margin=3Plots.mm
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
# Prepare data for export
# Convert to Float64 and collect
t_save = collect(Float64.(t_snapshots))
x_save = collect(Float64.(x_phys))
y_save = collect(Float64.(y_phys))

# Permute usol to (Time, X, Y) convention for easier Python/ML loading
usol_save = permutedims(Float64.(usol_snapshots), (3, 1, 2))

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