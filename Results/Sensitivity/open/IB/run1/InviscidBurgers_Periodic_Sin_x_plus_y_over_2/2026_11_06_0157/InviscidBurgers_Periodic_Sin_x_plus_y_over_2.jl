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
# ## Algorithm: Planar Characteristic Tracing with Entropy Shock Fitting
# Solves U_t + (U^2)_s = 0 for s = x + y using
#     U0(σ) = 1/3 + (2/3) sin(σ/2),
#     F_t(σ) = σ + 2t U0(σ).
# Before shock formation, F_t is monotone on a full period. After shock
# formation, the equal-area rule removes the folded characteristics and
# leaves one monotone entropy-admissible branch.
#
# %%
"""
    initial_planar_profile(σ)

Initial planar profile, periodic with period 4π.
"""
initial_planar_profile(σ) = 1.0 / 3.0 + (2.0 / 3.0) * sin(σ / 2.0)

"""
    characteristic_map(σ, t)

Characteristic image F_t(σ) = σ + 2t U0(σ). It satisfies
F_t(σ + 4π) = F_t(σ) + 4π.
"""
characteristic_map(σ, t) = σ + 2.0 * t * initial_planar_profile(σ)

"""
    planar_shock_position(t)

Exact post-shock position in the planar coordinate, reduced to [0, 4π).
"""
planar_shock_position(t) = mod(2.0 * π + (2.0 / 3.0) * t, 4.0 * π)

"""
    solve_characteristic_root(s_target, t, σ_bracket)

Invert the characteristic map on a monotone bracket using Brent's
bracketed method. Endpoint checks also handle the shock and periodic
boundaries without losing bracketing due to roundoff.
"""
function solve_characteristic_root(s_target, t, σ_bracket)
    σ_lo, σ_hi = σ_bracket
    map_lo = characteristic_map(σ_lo, t)
    map_hi = characteristic_map(σ_hi, t)

    s_target <= map_lo && return σ_lo
    s_target >= map_hi && return σ_hi

    residual(σ) = characteristic_map(σ, t) - s_target
    return find_zero(residual, σ_bracket, Brent())
end

"""
    shock_half_width(t)

Return δ such that the characteristics originating at 2π-δ and 2π+δ
meet at the equal-area shock. Writing q = δ/2, the nonzero post-shock
root satisfies sin(q)/q = 3/(2t).
"""
function shock_half_width(t)
    t <= 1.5 && return 0.0

    function residual(q)
        if q == 0.0
            return 1.0 - 3.0 / (2.0 * t)
        end
        return sin(q) / q - 3.0 / (2.0 * t)
    end

    # The residual is positive at q=0 and negative at q=π. Bracketing
    # away from the trivial q=0 post-shock root selects the shock fold.
    q_shock = find_zero(residual, (0.0, π), Brent())
    return 2.0 * q_shock
end

"""
    solve_burgers_analytical(s_in, t)

Evaluate the exact entropy solution at planar coordinates `s_in`.
The input may be a scalar or a collection of unique s-nodes.

For t > 3/2, the root bracket excludes the folded interval
[2π-δ, 2π+δ]. Its unwrapped image is exactly
[s_sh, s_sh+4π], giving the left state as s approaches s_sh from below
and the right state as s approaches from above. At the shock itself,
the right-state convention is used.
"""
function solve_burgers_analytical(s_in, t)
    t < 0 && throw(ArgumentError("t must be nonnegative"))

    s_values = Float64.(collect(s_in))
    t == 0 && return initial_planar_profile.(s_values)

    period = 4.0 * π
    u_values = similar(s_values)

    if t <= 1.5
        # F_t([0, 4π]) = [offset, offset + 4π], and F_t is monotone.
        offset = (2.0 / 3.0) * t
        σ_bracket = (0.0, period)

        for i in eachindex(s_values)
            s_target = offset + mod(s_values[i] - offset, period)
            σ = solve_characteristic_root(s_target, t, σ_bracket)
            u_values[i] = initial_planar_profile(σ)
        end

        return u_values
    end

    # Equal-area shock fitting. These are unwrapped periodic origins;
    # modulo 4π they are characteristic origins in [0, 4π).
    δ = shock_half_width(t)
    offset = 2.0 * π + (2.0 / 3.0) * t
    σ_bracket = (2.0 * π + δ, 2.0 * π - δ + period)

    for i in eachindex(s_values)
        # This target ordering starts at the shock and traverses the
        # unique monotone entropy branch through one complete period.
        s_target = offset + mod(s_values[i] - offset, period)
        σ = solve_characteristic_root(s_target, t, σ_bracket)
        u_values[i] = initial_planar_profile(σ)
    end

    return u_values
end

solve_burgers_analytical(s::Real, t::Real) =
    solve_burgers_analytical([s], t)[1]
# %% [markdown]


# ## Cell 2: Problem Configuration and Coordinate Mapping
# ## Physical Problem
# * **Domain:** (x, y) ∈ [0, 4π] × [0, 4π], periodic in both directions
# * **Initial Condition:** u₀(x,y) = 1/3 + (2/3) sin((x+y)/2)
# * **Time Interval:** t ∈ [0, 2], with ground-truth saves every 0.1
#
# ## Planar Coordinate Mapping
# The solution depends only on s = x + y modulo 4π. On the uniform
# periodic grid, all grid-point sums reduce to N_samples unique s-nodes.
# `s_index[i,j]` identifies the node associated with (x[i], y[j]).
#
# %%
# --- 1. Define the physical grid and saved times ---
period = 4.0 * π
x_min, x_max = 0.0, period
y_min, y_max = 0.0, period
t_start, t_final = 0.0, 2.0
t_shock = 1.5

N_samples = 256
dt_save = 0.1

# Exclude the duplicate endpoint of each periodic direction.
x_phys = collect(range(x_min, x_max; length=N_samples + 1))[1:N_samples]
y_phys = collect(range(y_min, y_max; length=N_samples + 1))[1:N_samples]

t_array = collect(t_start:dt_save:t_final)
N_time_steps = length(t_array)
snapshot_times = [0.0, 0.75, 1.5, 2.0]

# --- 2. Construct unique planar nodes and diagonal broadcasting map ---
s_nodes = collect(range(0.0, period; length=N_samples + 1))[1:N_samples]
s_index = [
    mod(i + j - 2, N_samples) + 1
    for i in 1:N_samples, j in 1:N_samples
]

println("Problem configured on the periodic domain [0, 4π]^2.")
println("Grid: $N_samples × $N_samples spatial points.")
println("Saved solution times: $N_time_steps (dt_save = $dt_save).")
println("Planar reduction: $(length(s_nodes)) unique s-nodes per time slice.")


# ## Cell 3: Time Evolution Loop
# ## Batch Processing
# The Method of Characteristics permits independent evaluation at every
# requested time. Each time slice requires only N_samples root-finding
# operations on the planar nodes, followed by exact diagonal broadcasting.
#
# * **Ground truth:** usol has dimensions (x, y, time).
# * **Precision:** The same exact solver is used before and after shock formation.
#
# %%
usol = zeros(Float64, N_samples, N_samples, N_time_steps)

println("Starting exact planar-wave evaluation...")

for (k, t_current) in enumerate(t_array)
    # Solve once per unique s-node; no canonical time rescaling is needed.
    u_s = solve_burgers_analytical(s_nodes, t_current)

    # Broadcast the planar profile over all (x, y) grid points.
    usol[:, :, k] .= u_s[s_index]
end

# Evaluate the requested visualization times independently. In particular,
# t=0.75 is not contained in the dt_save=0.1 ground-truth time array.
snapshot_solutions = Vector{Matrix{Float64}}(undef, length(snapshot_times))

for (k, t_current) in enumerate(snapshot_times)
    u_s = solve_burgers_analytical(s_nodes, t_current)
    snapshot_solutions[k] = u_s[s_index]
end

println("Evolution complete.")
println("Ground-truth tensor size: $(size(usol))")
println("Prepared $(length(snapshot_times)) exact visualization snapshots.")


# ## Cell 4: Visualization Dashboard
# ## Exact Planar-Wave Snapshots
# Four 2D views of u show the initial wave, nonlinear steepening,
# gradient catastrophe at t=1.5, and the mature entropy shock at t=2.
# Each heatmap uses its own automatic color scaling.
#
# %%
snapshot_plots = Vector{Plots.Plot}(undef, length(snapshot_times))

for k in eachindex(snapshot_times)
    # usol-style arrays are indexed (x, y); Plots heatmaps expect (y, x).
    u_plot = permutedims(snapshot_solutions[k])

    snapshot_plots[k] = heatmap(
        x_phys, y_phys, u_plot;
        xlabel="x",
        ylabel="y",
        title="t = $(snapshot_times[k])",
        xlims=(x_min, x_max),
        ylims=(y_min, y_max),
        aspect_ratio=:equal,
        colorbar=false,
        legend=false
    )
end

p_combined = plot(
    snapshot_plots...;
    layout=(1, 4),
    size=(1800, 420),
    margin=5Plots.mm
)

savefig(p_combined, joinpath(@__DIR__, "summary_all.png"))
display(p_combined)

println("Four-snapshot dashboard saved to summary_all.png")


# ## Cell 5: Data Persistence and Verification
# ## Artifact Generation and Analytical Verification
# Save the full 2D state at every dt_save=0.1 output time. The NPZ array
# uses the Python-friendly dimension ordering (time, x, y).
#
# %%
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
println("Saved usol ordering: (time, x, y)")

# ----- Internal analytical verification -------------------------------
u_min, u_max = extrema(usol)
@assert u_min >= -1.0 / 3.0 - 1.0e-12
@assert u_max <= 1.0 + 1.0e-12

# Each planar node occurs equally often on the periodic 2D grid. Away
# from a shock, periodic quadrature is highly accurate; after shock
# formation, allow a conservative first-order shock-cell sampling bound.
mean_errors = [
    sum(usol[:, :, k]) / (N_samples * N_samples) - 1.0 / 3.0
    for k in 1:N_time_steps
]
max_mean_error = maximum(abs.(mean_errors))
mean_tolerance = 2.0 * (u_max - u_min) / N_samples
@assert max_mean_error <= mean_tolerance

# Verify the equal-area shock construction at every saved post-shock time.
shock_position_error = 0.0
shock_state_error = 0.0

for t_current in t_array
    t_current <= t_shock && continue

    δ = shock_half_width(t_current)
    σ_left = 2.0 * π - δ
    σ_right = 2.0 * π + δ
    expected_shock = 2.0 * π + (2.0 / 3.0) * t_current

    # Both folded characteristics must meet at the analytical shock.
    shock_position_error = max(
        shock_position_error,
        abs(characteristic_map(σ_left, t_current) - expected_shock),
        abs(characteristic_map(σ_right, t_current) - expected_shock)
    )

    u_left = initial_planar_profile(σ_left)
    u_right = initial_planar_profile(σ_right)
    shock_state_error = max(
        shock_state_error,
        abs(u_left + u_right - 2.0 / 3.0)
    )
end

@assert shock_position_error < 1.0e-10
@assert shock_state_error < 1.0e-12

δ_final = shock_half_width(t_final)
u_left_final = initial_planar_profile(2.0 * π - δ_final)
u_right_final = initial_planar_profile(2.0 * π + δ_final)

println("\n--- Analytical Verification ---")
println("Value range: [$u_min, $u_max]")
println("Maximum discrete mean error: $max_mean_error")
println("Maximum shock-position error: $shock_position_error")
println("Shock at t=$t_final: s = $(planar_shock_position(t_final))")
println("Shock states at t=$t_final: u_L=$u_left_final, u_R=$u_right_final")

# ----- Verify the saved file ------------------------------------------
println("\n--- File Verification ---")
file_size = filesize(npz_path)
println("File size: $(file_size) bytes ($(round(file_size / 1024 / 1024, digits=2)) MB)")

data_check = npzread(npz_path)
println("Keys in file: ", keys(data_check))
println("Shape of usol: ", size(data_check["usol"]))
println("Shape of t: ", size(data_check["t"]))
println("Shape of x: ", size(data_check["x"]))
println("Shape of y: ", size(data_check["y"]))

@assert size(data_check["usol"]) ==
        (N_time_steps, N_samples, N_samples)
@assert size(data_check["t"]) == (N_time_steps,)
@assert size(data_check["x"]) == (N_samples,)
@assert size(data_check["y"]) == (N_samples,)
