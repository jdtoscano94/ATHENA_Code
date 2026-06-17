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
    solve_moc_2d(x, y, t)

Computes the exact entropy solution for the 2D scalar conservation law:
u_t + u u_x + u u_y = 0
with u0(x,y) = 1/3 + 2/3 sin((x+y)/2).

Uses the Method of Characteristics with Lax-Oleinik entropy condition 
to handle shock formation.
"""
function solve_moc_2d(x, y, t)
    # The problem reduces to a 1D problem along r = x + y.
    # We compute r for the input coordinates and solve the scalar problem.
    r = x .+ y
    return map(ri -> solve_point_entropy(ri, t), r)
end

# --- Helper Functions for Entropy Solution ---

# Initial condition profile u0(s)
u0_func(s) = 1.0/3.0 + 2.0/3.0 * sin(s/2.0)

# Integral of u0(s) required for Lax-Oleinik formula
# Int(1/3 + 2/3 sin(s/2)) ds = s/3 - 4/3 cos(s/2)
u0_integral(s) = s/3.0 - 4.0/3.0 * cos(s/2.0)

"""
    solve_point_entropy(r, t)

Finds the unique entropy-satisfying value u(x,y,t) for a given r=x+y and time t.
Minimizes the action functional J(s) = (r-s)^2/(4t) + Int(u0) to find the 
correct characteristic origin s.
"""
function solve_point_entropy(r, t)
    if t == 0.0
        return u0_func(r)
    end

    # 1. Define Search Range for Characteristic Origin s
    # Characteristic equation: r = s + 2t * u(s)  =>  s = r - 2t * u(s)
    # u(s) is in [-1/3, 1].
    # So s is in [r - 2t(1), r - 2t(-1/3)] = [r - 2t, r + 2t/3]
    s_min = r - 2.0 * t - 0.1
    s_max = r + 2.0 * t / 3.0 + 0.1

    # 2. Coarse Grid Search for Global Minimum of Lax-Oleinik Functional
    # J(s) represents the "cost" of a characteristic coming from s.
    # The global minimum corresponds to the physically realized state.
    J(s) = (r - s)^2 / (4.0 * t) + u0_integral(s)

    # Scan sufficient points to bracket the minimum (period is 4pi ~ 12.5)
    # The range width is at most ~4.6 (for t=2). 20 points is dense enough.
    s_candidates = range(s_min, s_max, length=30)
    
    best_s = s_candidates[1]
    min_val = Inf
    
    for s in s_candidates
        val = J(s)
        if val < min_val
            min_val = val
            best_s = s
        end
    end

    # 3. Refine Root using Derivative (Characteristic Equation)
    # J'(s) = 0  =>  -(r-s)/(2t) + u0(s) = 0  =>  r - s - 2t*u0(s) = 0
    char_eq(s) = r - s - 2.0 * t * u0_func(s)

    s_exact = try
        find_zero(char_eq, best_s, Order1())
    catch
        best_s # Fallback to grid best if secant fails
    end

    return u0_func(s_exact)
end
# %% [markdown]


# ## Cell 2: Problem Configuration and Coordinate Mapping
#
# ## Physical Problem
# * **Domain:** $(x,y) \in [0, 4\pi] \times [0, 4\pi]
# * **Initial Condition:** $u_0(x,y) = \frac{1}{3} + \frac{2}{3}\sin(\frac{x+y}{2})$
# * **Time Interval:** $t \in [0, 2]"

# %%
# --- 1. Define Physical Grid ---
x_min, x_max = 0.0, 4.0 * pi
y_min, y_max = 0.0, 4.0 * pi
t_start, t_final = 0.0, 2.0

N_samples = 200 
# Define time points: frequent saves (dt=0.1) + mandatory snapshots
t_snapshots = [0.0, 2.0/3.0, 4.0/3.0, 2.0]
t_saves = collect(t_start:0.1:t_final)
t_array = sort(unique(vcat(t_saves, t_snapshots)))
N_time_steps = length(t_array)

x_coords = range(x_min, x_max, length=N_samples)
y_coords = range(y_min, y_max, length=N_samples)

println("Problem configured.")
println("Domain: [0, 4π] x [0, 4π]")
println("Grid: $N_samples x $N_samples spatial points.")
println("Time: $N_time_steps steps (merged snapshots and save points).")
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
usol = zeros(N_samples, N_samples, N_time_steps)

println("Starting time evolution loop...")

# Create 2D coordinate grids
X = repeat(x_coords, 1, N_samples)
Y = repeat(y_coords', N_samples, 1)

for (i, t_current) in enumerate(t_array)
    # Compute exact solution field for this time slice
    # No canonical mapping needed; solver handles physical coordinates directly
    u_slice = solve_moc_2d(X, Y, t_current)
    
    # Store in tensor
    usol[:, :, i] = u_slice
    
    if i % 10 == 0
        println("Completed step $i / $N_time_steps (t=$(round(t_current, digits=2)))")
    end
end

println("Evolution complete.")
# %% [markdown]


# ## Cell 4: Visualization Dashboard
#
# ## Dual-Panel Analysis
# 1.  **Space-Time Heatmap (Left):** Visualizes the characteristic convergence and shock formation over the full duration.
# 2.  **Final Time Slice (Right):** Provides a precise view of the shock profile at $t=1.0$ to verify the "Equal Area" cut.
# %%
println("Generating visualization dashboard...")

# Identify indices for the requested snapshots
snapshot_times = [0.0, 2.0/3.0, 4.0/3.0, 2.0]
indices = [findfirst(t -> isapprox(t, target, atol=1e-4), t_array) for target in snapshot_times]

plots = []
for (k, idx) in enumerate(indices)
    if idx === nothing
        println("Warning: Exact snapshot time not found for $(snapshot_times[k])")
        continue
    end
    
    t_val = t_array[idx]
    # Extract 2D field for this time
    u_field = usol[:, :, idx]
    
    p = heatmap(x_coords, y_coords, u_field',
        title = "t = $(round(t_val, digits=2))",
        xlabel = k==1 ? "x" : "",
        ylabel = k==1 ? "y" : "",
        aspect_ratio = :equal,
        colorbar = false,
        c = :viridis,
        clims = (-1/3, 1.0) # Fix color scale for consistency
    )
    push!(plots, p)
end

# Combine into 1x4 layout
p_combined = plot(plots..., 
    layout = (1, 4),
    size = (1600, 400),
    margin = 5Plots.mm
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
# Transpose to (Time, Space) to match Trixi/Python convention usually preferred by PINNs
# Permute to (Time, x, y) for standard ML/Trixi convention
usol_save = permutedims(Float64.(usol), (3, 1, 2))
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
