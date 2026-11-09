# %% [markdown]
# # Preamble: Solver and Analytical Framework Initialization
#
# ## Framework: Exact Method of Characteristics for 2D Inviscid Burgers
# Initializes the components to generate the exact entropy solution of $u_t + (u^2/2)_x + (u^2/2)_y = 0$ on the doubly periodic domain.
#
# 1. **Analytical Core (`Roots`, `LinearAlgebra`)**:
#     * Reduces the 2D problem along diagonal characteristics to $\sigma=(x+y)/2$, with $V_0(\sigma)=1/3+(2/3)\sin\sigma$.
#     * Solves $\sigma_0+t(1/3+(2/3)\sin\sigma_0)=\sigma$, evaluating the initial condition directly at $t=0$.
#     * Brackets all characteristic roots on $[\sigma-2\pi,\sigma+2\pi]$ using dense sampling and bisection/Brent refinement.
#     * For $t>3/2$, selects the root minimizing the Hopf–Lax (Lax–Oleinik) functional to obtain the entropy solution across shocks.
#
# 2. **Diagnostics (`Plots`, `NPZ`)**:
#     * **`Plots`**: Visualizes the exact 2D field at the four required snapshot times, including the propagating diagonal shocks.
#     * **`NPZ`**: Exports the full space-space-time ground-truth tensor at `dt_save = 0.1` for verification and training.
#
# %%
using Plots
using Roots
using LinearAlgebra
using NPZ
# %% [markdown]

# ## Cell 1: Core Analytical Solver Definition
#
# ## Algorithm: Diagonal Characteristic Tracing with Hopf–Lax Entropy Selection
# The exact 2D solution depends only on $\sigma=(x+y)/2$ and satisfies
# $\sigma_0+t(1/3+(2/3)\sin\sigma_0)=\sigma$.
#
# ### Logic Flow:
# 1. Evaluate $V_0(\sigma)$ directly at $t=0$.
# 2. Bracket every characteristic root on $[\sigma-2\pi,\sigma+2\pi]$ using dense sampling and bracketed root refinement.
# 3. Include stationary points of the foot map in the brackets so that all post-shock roots are captured.
# 4. For $t>3/2$, select the root minimizing the Hopf–Lax functional.
# 5. Evaluate the 2D field as $u(x,y,t)=V((x+y)/2,t)$.
#
# %%
"""
    initial_burgers_state(σ, c0, c1)
Initial diagonal profile `V_0(σ) = c0 + c1 * sin(σ)`.
"""
initial_burgers_state(σ, c0, c1) = c0 + c1 * sin(σ)
"""
    solve_characteristic_root(σ, t, c0, c1)
Return the entropy-selected characteristic foot for a positive time `t`.
All roots are bracketed on `[σ-2π, σ+2π]`; after shock formation, the
Hopf–Lax objective determines the physical branch.
"""
function solve_characteristic_root(σ, t, c0, c1)
    t > 0 || throw(ArgumentError("solve_characteristic_root requires t > 0"))
    g(y) = y + t * (c0 + c1 * sin(y)) - σ
    lower = σ - 2 * π
    upper = σ + 2 * π
    # Dense sampling over two periods of the initial profile.
    nodes = collect(range(lower, upper; length=129))
    # Add stationary points of g. Between these points g is monotone, so
    # sign changes bracket every root, including closely spaced post-shock roots.
    shock_parameter = t * c1
    if shock_parameter > 1
        θ = acos(clamp(-1 / shock_parameter, -1.0, 1.0))
        for base in (-θ, θ)
            k_first = ceil(Int, (lower - base) / (2 * π))
            k_last = floor(Int, (upper - base) / (2 * π))
            for k in k_first:k_last
                push!(nodes, base + 2 * π * k)
            end
        end
    end
    nodes = unique(sort(nodes))
    values = g.(nodes)
    roots = Float64[]
    for i in 1:(length(nodes) - 1)
        a, b = nodes[i], nodes[i + 1]
        ga, gb = values[i], values[i + 1]
        if ga == 0.0
            push!(roots, a)
        elseif ga * gb < 0.0
            push!(roots, find_zero(g, (a, b), Brent()))
        end
    end
    if values[end] == 0.0
        push!(roots, nodes[end])
    end
    roots = unique(sort(roots))
    isempty(roots) && error("No characteristic root could be bracketed.")
    if shock_parameter > 1
        # Primitive of V0 is c0*y - c1*cos(y).
        hopf_lax(y) = (σ - y)^2 / (2 * t) + c0 * y - c1 * cos(y)
        return roots[argmin(hopf_lax.(roots))]
    end
    # Before gradient catastrophe the foot map is monotone and has one root.
    return roots[1]
end
"""
    solve_diagonal_burgers(σ, t; c0=1/3, c1=2/3)
Exact entropy solution of the reduced 1D equation `V_t + V * V_σ = 0`.
"""
function solve_diagonal_burgers(σ::Real, t::Real; c0=1 / 3, c1=2 / 3)
    t < 0 && throw(ArgumentError("The evaluation time must be nonnegative."))
    iszero(t) && return initial_burgers_state(σ, c0, c1)
    σ0 = solve_characteristic_root(σ, t, c0, c1)
    return initial_burgers_state(σ0, c0, c1)
end
"""
    solve_burgers_analytical(x, y, t; c0=1/3, c1=2/3)
Evaluate the exact 2D entropy solution at `(x, y)` and time `t`.
"""
function solve_burgers_analytical(x::Real, y::Real, t::Real; c0=1 / 3, c1=2 / 3)
    σ = (x + y) / 2
    return solve_diagonal_burgers(σ, t; c0=c0, c1=c1)
end
# Grid evaluator: u[i, j] is the solution at (x[i], y[j]).
function solve_burgers_analytical(
    x::AbstractVector,
    y::AbstractVector,
    t::Real;
    c0=1 / 3,
    c1=2 / 3,
)
    u = Matrix{Float64}(undef, length(x), length(y))
    for (j, yj) in enumerate(y)
        for (i, xi) in enumerate(x)
            u[i, j] = solve_burgers_analytical(xi, yj, t; c0=c0, c1=c1)
        end
    end
    return u
end
# %% [markdown]


# ## Cell 2: Problem Configuration and Coordinate Mapping
#
# ## Physical Problem
# * **Domain:** $(x,y)\in[0,4\pi]\times[0,4\pi]$, periodic in both directions.
# * **Initial Condition:** $u_0(x,y)=1/3+(2/3)\sin((x+y)/2)$.
# * **Time Interval:** $t\in[0,2]$, including shock formation at $t=3/2$.
#
# ## Exact Diagonal Reduction
# No spatial or temporal rescaling is required. The analytical solver is evaluated
# directly using $\sigma=(x+y)/2$ at each requested physical time.
#
# %%
# --- 1. Physical constants and domain ---
c0 = 1 / 3
c1 = 2 / 3
t_shock = 3 / 2

x_min, x_max = 0.0, 4 * π
y_min, y_max = 0.0, 4 * π
t_start, t_final = 0.0, 2.0

# --- 2. Default spatial grid and exact evaluation times ---
N_samples = 256
dt_save = 0.1

x_phys = range(x_min, x_max; length=N_samples)
y_phys = range(y_min, y_max; length=N_samples)
t_array = collect(t_start:dt_save:t_final)
N_time_steps = length(t_array)

# Required visualization times; these are evaluation times, not time steps.
snapshot_times = [0.0, 1.0, 1.5, 2.0]

println("Problem configured.")
println("Domain: [0, 4π] × [0, 4π], periodic in x and y.")
println("Grid: $N_samples × $N_samples spatial points.")
println("Ground truth: $N_time_steps saved times with dt_save = $dt_save.")
println("Snapshot times: $snapshot_times")
# %% [markdown]

# ## Cell 3: Time Evolution Loop
#
# ## Batch Processing
# The Method of Characteristics requires no numerical time-stepping. Each saved
# time is evaluated independently on the complete two-dimensional spatial grid.
#
# * **Complexity:** $O(N_x \times N_y \times N_t)$ exact solution evaluations.
# * **Precision:** The same analytical procedure applies before and after shock
#   formation, with Hopf–Lax selection used in the post-shock regime.
#
# %%
# Full ground-truth tensor with dimensions (x, y, time).
usol = zeros(length(x_phys), length(y_phys), length(t_array))

println("Starting exact solution evaluation...")

for (i, t_current) in enumerate(t_array)
    # Evaluate directly at the physical time; no coordinate or time rescaling.
    u_slice = solve_burgers_analytical(
        x_phys,
        y_phys,
        t_current;
        c0=c0,
        c1=c1,
    )

    usol[:, :, i] = u_slice
end

println("Exact solution evaluation complete.")
println("Ground-truth tensor size: $(size(usol))")
# %% [markdown]

# ## Cell 4: Visualization Dashboard
#
# ## Four-Time Snapshot Dashboard
# Displays the exact 2D field $u$ at $t=0.0, 1.0, 1.5, 2.0$.
# Each panel uses automatic color scaling, with colorbars disabled.
#
# %%
snapshot_plots = []

for t_snapshot in snapshot_times
    snapshot_index = findfirst(
        t -> isapprox(t, t_snapshot; atol=1e-12, rtol=0.0),
        t_array,
    )
    isnothing(snapshot_index) &&
        error("Snapshot time $t_snapshot is absent from the saved time array.")

    # usol is indexed as (x, y, time); heatmap expects a (y, x) matrix.
    u_snapshot = permutedims(usol[:, :, snapshot_index])

    p_snapshot = heatmap(
        x_phys,
        y_phys,
        u_snapshot;
        xlabel="x",
        ylabel="y",
        title="Exact u, t = $(round(t_snapshot; digits=1))",
        colorbar=false,
        aspect_ratio=:equal,
    )
    push!(snapshot_plots, p_snapshot)
end

p_combined = plot(
    snapshot_plots...;
    layout=(1, 4),
    size=(1800, 450),
    margin=5 * Plots.mm,
)

savefig(p_combined, joinpath(@__DIR__, "summary_all.png"))
display(p_combined)

println("Four-time snapshot dashboard saved to summary_all.png")
# %% [markdown]

# ## Cell 5: Data Persistence and Verification
#
# ## Artifact Generation and Internal Verification
# Saves the complete 2D ground truth to `burgers.npz` for numerical-error
# assessment and supervised training. The solution array is stored as
# `(time, x, y)` alongside all coordinate and time arrays.
#
# %%
usol_save = permutedims(usol, (3, 1, 2))
t_save = collect(Float64.(t_array))
x_save = collect(Float64.(x_phys))
y_save = collect(Float64.(y_phys))

npz_path = joinpath(@__DIR__, "burgers.npz")
npzwrite(npz_path, Dict(
    "t" => t_save,
    "x" => x_save,
    "y" => y_save,
    "usol" => usol_save,
))

println("Data saved as NPZ to burgers.npz")
println("Saved usol ordering: (time, x, y)")

# ----- Verify the saved file ----------------------------------------
println("\n--- File Verification ---")
file_size = filesize(npz_path)
println("File size: $(file_size) bytes ($(round(file_size / 1024 / 1024; digits=2)) MB)")

data_check = npzread(npz_path)
println("Keys in file: ", keys(data_check))
println("Shape of usol: ", size(data_check["usol"]))
println("Shape of t: ", size(data_check["t"]))
println("Shape of x: ", size(data_check["x"]))
println("Shape of y: ", size(data_check["y"]))

@assert size(data_check["usol"]) ==
        (length(t_save), length(x_save), length(y_save))
@assert length(data_check["t"]) == length(t_save)
@assert length(data_check["x"]) == length(x_save)
@assert length(data_check["y"]) == length(y_save)

# ----- Maximum-principle verification -------------------------------
u_min, u_max = extrema(usol)
bound_tolerance = 1e-10
@assert u_min >= -1 / 3 - bound_tolerance
@assert u_max <= 1 + bound_tolerance
println("\nSolution bounds: [$u_min, $u_max] (expected within [-1/3, 1])")

# ----- Analytic post-shock path verification ------------------------
# At the final time, detect each discontinuity in a narrow reduced-coordinate
# scan and compare its location with σ_s(t) = π + t/3 and its periodic image.
validation_time = t_final
@assert validation_time > t_shock

scan_half_width = 0.25
scan_points = 1001
scan_spacing = 2 * scan_half_width / (scan_points - 1)
base_shock_sigma = π + validation_time / 3

for period_index in 0:1
    expected_sigma = base_shock_sigma + 2 * π * period_index
    σ_scan = range(
        expected_sigma - scan_half_width,
        expected_sigma + scan_half_width;
        length=scan_points,
    )
    u_scan = [
        solve_diagonal_burgers(σ, validation_time; c0=c0, c1=c1)
        for σ in σ_scan
    ]

    jump_index = argmax(abs.(diff(u_scan)))
    detected_sigma = (σ_scan[jump_index] + σ_scan[jump_index + 1]) / 2

    @assert abs(detected_sigma - expected_sigma) <= 2 * scan_spacing

    expected_sum = 2 * expected_sigma
    detected_sum = 2 * detected_sigma
    println(
        "Shock $(period_index + 1): expected x+y = $(round(expected_sum; digits=6)), " *
        "detected x+y = $(round(detected_sum; digits=6))",
    )
end

println("All persistence and internal analytical checks passed.")