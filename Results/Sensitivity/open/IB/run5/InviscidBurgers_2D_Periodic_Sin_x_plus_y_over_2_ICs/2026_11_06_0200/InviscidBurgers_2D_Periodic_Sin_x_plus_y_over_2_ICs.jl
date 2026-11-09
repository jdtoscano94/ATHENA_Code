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
# ## Algorithm: Periodic Characteristic Tracing with Equal-Area Shock Fitting
# The reduced initial profile is U0(xi) = alpha + beta*sin(xi), with shock
# formation time 1/beta. Before formation, a safeguarded Newton iteration
# inverts the characteristic map. After formation, all roots are enumerated
# on the monotone branches of that map. The equal-area shock then determines
# whether the leftmost or rightmost root supplies the entropy solution.
#
# %%
@inline _burgers_u0(x0, alpha, beta) = alpha + beta * sin(x0)

"""
Construct the shock in the base period using symmetric characteristic feet
pi-s and pi+s. Their common characteristic location requires

    s = t*beta*sin(s).

For these symmetric feet, the equal-area integrand is odd about pi, so the
equal-area integral vanishes. The shock position is pi + alpha*t. Moreover,
the Rankine-Hugoniot speed is alpha and, for s > 0,
U_left > alpha > U_right, which is the Lax entropy condition.
"""
function _equal_area_shock(t, alpha, beta)
    pi_val = Float64(pi)
    position = pi_val + alpha * t
    r = t * beta

    if r <= 1.0
        return (
            position=position,
            half_width=0.0,
            left_foot=pi_val,
            right_foot=pi_val,
            u_left=alpha,
            u_right=alpha,
        )
    end

    # Dividing by s removes the trivial zero at s=0. The limiting value at
    # s=0 is supplied explicitly, giving a reliable bracket for the nonzero
    # shock-foot solution.
    function foot_equation(s)
        if s == 0.0
            return r - 1.0
        end
        return r * sin(s) / s - 1.0
    end

    s = find_zero(foot_equation, (0.0, pi_val), Bisection())
    left_foot = pi_val - s
    right_foot = pi_val + s

    return (
        position=position,
        half_width=s,
        left_foot=left_foot,
        right_foot=right_foot,
        u_left=_burgers_u0(left_foot, alpha, beta),
        u_right=_burgers_u0(right_foot, alpha, beta),
    )
end

"""
    solve_characteristic_root(x_target, t, alpha, beta)

Invert x_target = x0 + t*U0(x0) on its unique pre-shock branch.
The physical characteristic-foot window supplies a permanent bracket;
Newton steps are replaced by bisection steps whenever they leave it.
"""
function solve_characteristic_root(x_target, t, alpha, beta)
    t == 0.0 && return x_target

    # U0 lies in [alpha-beta, alpha+beta], hence every characteristic foot
    # lies in this window. For the required parameters it is
    # [x_target-t, x_target+t/3].
    lower = x_target - t * (alpha + beta)
    upper = x_target - t * (alpha - beta)

    f(x0) = x0 + t * _burgers_u0(x0, alpha, beta) - x_target
    df(x0) = 1.0 + t * beta * cos(x0)

    f_lower = f(lower)
    f_upper = f(upper)
    abs(f_lower) <= 1.0e-14 && return lower
    abs(f_upper) <= 1.0e-14 && return upper

    lo, hi = lower, upper
    initial_guess = x_target - t * _burgers_u0(x_target, alpha, beta)
    x0 = clamp(initial_guess, lo, hi)

    for _ in 1:40
        fx = f(x0)
        abs(fx) <= 1.0e-12 && return x0

        # Maintain f(lo) <= 0 <= f(hi).
        if fx > 0.0
            hi = x0
        else
            lo = x0
        end

        derivative = df(x0)
        candidate = abs(derivative) > 1.0e-12 ? x0 - fx / derivative : NaN

        # This safeguard also handles the zero derivative at gradient
        # catastrophe, t = 1/beta.
        if !isfinite(candidate) || candidate <= lo || candidate >= hi
            candidate = 0.5 * (lo + hi)
        end

        if abs(candidate - x0) <= 1.0e-13 * (1.0 + abs(x0))
            return candidate
        end
        x0 = candidate
    end

    return find_zero(f, (lo, hi), Bisection())
end

"""
Enumerate every characteristic root in the physical foot window.
Critical points of the characteristic map partition the window into monotone
branches, so no roots can be skipped between sampled points.
"""
function _characteristic_roots(x_target, t, alpha, beta)
    period = 2.0 * Float64(pi)
    lower = x_target - t * (alpha + beta)
    upper = x_target - t * (alpha - beta)
    f(x0) = x0 + t * _burgers_u0(x0, alpha, beta) - x_target

    nodes = Float64[lower, upper]
    r = t * beta

    if r > 1.0
        # Solve 1 + t*beta*cos(x0) = 0. These are the fold locations.
        critical = acos(-1.0 / r)

        for base in (critical, period - critical)
            n_first = ceil(Int, (lower - base) / period)
            n_last = floor(Int, (upper - base) / period)

            for n in n_first:n_last
                node = base + n * period
                if lower < node < upper
                    push!(nodes, node)
                end
            end
        end
    end

    sort!(nodes)
    roots = Float64[]

    # Include roots located at window endpoints or exactly at a fold.
    for node in nodes
        if abs(f(node)) <= 1.0e-12
            push!(roots, node)
        end
    end

    # Each interval is monotone and therefore contains at most one root.
    for j in 1:(length(nodes) - 1)
        left = nodes[j]
        right = nodes[j + 1]
        f_left = f(left)
        f_right = f(right)

        if f_left * f_right < 0.0
            push!(roots, find_zero(f, (left, right), Bisection()))
        end
    end

    sort!(roots)

    # Remove duplicates introduced when a root coincides with a node.
    unique_roots = Float64[]
    for root in roots
        if isempty(unique_roots) ||
           abs(root - unique_roots[end]) > 1.0e-9 * (1.0 + abs(root))
            push!(unique_roots, root)
        end
    end

    return unique_roots
end

"""
    solve_burgers_analytical(x_in, t, alpha, beta)

Exact entropy solution of the diagonally reduced Burgers problem.
x_in contains values of xi=(x+y)/2; the returned array has the same shape.
For the required problem, use alpha=1/3 and beta=2/3. The result is
2pi-periodic in xi and remains valid through shock formation.
"""
function solve_burgers_analytical(x_in, t, alpha, beta)
    t >= 0.0 || throw(ArgumentError("t must be nonnegative"))
    beta > 0.0 || throw(ArgumentError("beta must be positive"))

    x = Float64.(x_in)

    if t == 0.0
        return alpha .+ beta .* sin.(x)
    end

    values = similar(x)
    period = 2.0 * Float64(pi)
    shock_time = 1.0 / beta

    if t <= shock_time
        # The characteristic map is still single-valued. At t=shock_time it
        # has a zero derivative at the catastrophe point but remains invertible.
        for i in eachindex(x)
            xi = mod(x[i], period)
            foot = solve_characteristic_root(xi, t, alpha, beta)
            values[i] = _burgers_u0(foot, alpha, beta)
        end
        return values
    end

    shock = _equal_area_shock(t, alpha, beta)
    shock_position = mod(shock.position, period)

    for i in eachindex(x)
        xi = mod(x[i], period)
        roots = _characteristic_roots(xi, t, alpha, beta)
        isempty(roots) && error("No characteristic root found in the foot window")

        # In a three-root fold, the left state of the shock belongs to the
        # leftmost branch and the right state to the rightmost branch. Away
        # from the fold there is only one root and no selection is necessary.
        foot = if length(roots) == 1
            roots[1]
        elseif xi <= shock_position
            first(roots)
        else
            last(roots)
        end

        values[i] = _burgers_u0(foot, alpha, beta)
    end

    return values
end
# %% [markdown]


# ## Cell 2: Problem Configuration and Coordinate Mapping
# ## Physical Problem
# * **Domain:** (x,y) in [0,4π] × [0,4π], periodic in both directions
# * **Initial Condition:** u0(x,y) = 1/3 + (2/3)*sin((x+y)/2)
# * **Time Interval:** 0 ≤ t ≤ 2, with ground-truth output every 0.1
#
# The exact diagonal reduction uses xi = (x+y)/2. The grid below stores
# xi_grid[i,j] = (x_grid[i] + y_grid[j])/2, so subsequent solution arrays
# use the spatial index order (x, y).
#
# %%
# --- 1. Physical parameters and output schedule ---
alpha = 1.0 / 3.0
beta = 2.0 / 3.0
shock_time = 1.5

x_min, x_max = 0.0, 4.0 * pi
y_min, y_max = x_min, x_max
t_start, t_final = 0.0, 2.0

N_samples = 256
dt_save = 0.1
t_array = collect(t_start:dt_save:t_final)
N_time_steps = length(t_array)

# Times required by the four-panel summary. The additional t=0.75 snapshot
# can be evaluated directly without changing the uniform ground-truth saves.
snapshot_times = [0.0, 0.75, 1.5, 2.0]

# --- 2. Periodic 2D grid ---
# Omit x=4π and y=4π because they duplicate the periodic origin.
x_grid = collect(range(x_min, x_max; length=N_samples + 1))[1:N_samples]
y_grid = collect(range(y_min, y_max; length=N_samples + 1))[1:N_samples]

# Reduced characteristic coordinate on the full 2D computational grid.
xi_grid = [(x + y) / 2.0 for x in x_grid, y in y_grid]

println("Problem configured.")
println("Grid: $N_samples × $N_samples periodic spatial points.")
println("Ground truth: $N_time_steps save times with dt_save = $dt_save.")
println("Summary times: ", snapshot_times)
println("Diagonal reduction: xi = (x+y)/2; shock formation at t = $shock_time.")
# %% [markdown]


# ## Cell 3: Time Evolution Loop
# ## Batch Evaluation of the Exact Solution
# The Method of Characteristics requires no numerical time stepping. Each
# save time is evaluated independently on the full 2D grid through the
# reduced coordinate xi_grid = (x+y)/2.
#
# * **Complexity:** O(N_x × N_y × N_t) characteristic inversions.
# * **Validity:** The same solver handles smooth times, shock formation,
#   and the post-shock entropy solution.
#
# %%
# Full ground-truth tensor with dimensions (x, y, time).
usol = zeros(Float64, N_samples, N_samples, N_time_steps)

println("Starting exact-solution evaluation...")

for (i, t_current) in enumerate(t_array)
    # t_current is already the physical time; no coordinate or time rescaling
    # is required. The returned slice has dimensions (N_samples, N_samples).
    u_slice = solve_burgers_analytical(xi_grid, t_current, alpha, beta)
    usol[:, :, i] = u_slice
end

println("Evolution complete.")
println("Ground-truth tensor size: ", size(usol))


# ## Cell 4: Visualization Dashboard
# ## Four-Time Exact-Solution Dashboard
# Each panel shows only the primary field u(x,y,t). The snapshots depict
# the initial data, smooth steepening, gradient catastrophe, and the
# developed post-shock entropy solution.
#
# %%
snapshot_panels = []

for t_snapshot in snapshot_times
    # Use the stored ground truth when this time belongs to the uniform
    # save schedule. t=0.75 is evaluated independently by the exact solver.
    save_index = findfirst(
        t -> isapprox(t, t_snapshot; atol=1.0e-12, rtol=0.0),
        t_array,
    )

    u_snapshot = if save_index === nothing
        solve_burgers_analytical(xi_grid, t_snapshot, alpha, beta)
    else
        usol[:, :, save_index]
    end

    # usol is indexed as (x,y); Plots expects the heatmap matrix as (y,x).
    panel = heatmap(
        x_grid,
        y_grid,
        permutedims(u_snapshot),
        xlabel="x",
        ylabel="y",
        title="u, t = $(t_snapshot)",
        aspect_ratio=:equal,
        colorbar=false,
    )

    push!(snapshot_panels, panel)
end

# One row (field u) and four columns (requested times). No fixed colormap
# or color limits are supplied, so each panel scales automatically.
p_combined = plot(
    snapshot_panels...,
    layout=(1, 4),
    size=(1800, 420),
    margin=5Plots.mm,
)

savefig(p_combined, joinpath(@__DIR__, "summary_all.png"))
display(p_combined)

println("Four-time dashboard saved to summary_all.png")
# %% [markdown]


# ## Cell 5: Data Persistence and Verification
#
# ## Ground-Truth Artifact Generation
# Save every exact 2D solution state from the dt_save=0.1 schedule. The NPZ
# file contains the physical coordinates and the full solution tensor; no
# external reference data or error metrics are required.
#
# %%
# Convert from local (x, y, time) storage to (time, x, y) for export.
usol_save = Float64.(permutedims(usol, (3, 1, 2)))
t_save = Float64.(t_array)
x_save = Float64.(x_grid)
y_save = Float64.(y_grid)

npz_path = joinpath(@__DIR__, "burgers.npz")
npzwrite(npz_path, Dict(
    "t" => t_save,
    "x" => x_save,
    "y" => y_save,
    "usol" => usol_save,
))

println("Data saved as NPZ to burgers.npz")
println("Saved usol dimensions: (time, x, y) = ", size(usol_save))

# ----- Verify the saved file ----------------------------------------
println("\n--- File Verification ---")
file_size = filesize(npz_path)
println("File size: $(file_size) bytes ($(round(file_size / 1024 / 1024, digits=2)) MB")

data_check = npzread(npz_path)
println("Keys in file: ", keys(data_check))
println("Shape of usol: ", size(data_check["usol"]))
println("Shape of t: ", size(data_check["t"]))
println("Shape of x: ", size(data_check["x"]))
println("Shape of y: ", size(data_check["y"]))
