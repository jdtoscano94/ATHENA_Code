# %% [markdown]
# # Preamble: Solver and Analytical Framework Initialization
#
# ## Framework: Exact Method of Characteristics for the 2D Inviscid Burgers Equation
# Initializes the components used to generate the **Exact Entropy Solution** on the
# periodic torus $[0,4\pi]^2$.
#
# 1. **Analytical Core (`Roots`, `LinearAlgebra`)**:
#     * Uses the exact diagonal reduction $\xi=(x+y)/2$, reducing the 2D problem to
#       $U_t+U U_\xi=0$ with $U_0(\xi)=\frac13+\frac23\sin\xi$.
#     * Traces characteristics through
#       $\xi=\xi_0+(\frac13+\frac23\sin\xi_0)t$ and solves the implicit equation
#       by periodic bracketed root-finding on $\xi_0\in[0,2\pi)$.
#     * Before breaking at $t_b=3/2$, selects the unique characteristic root.
#       After breaking, uses the Rankine--Hugoniot shock trajectory
#       $\xi_s(t)=\pi+t/3$ and selects the maximum-$u$ root for $\xi<\xi_s$
#       and the minimum-$u$ root for $\xi>\xi_s$, excluding the middle branch.
#
# 2. **Diagnostics (`Plots`, `NPZ`)**:
#     * **`Plots`**: Visualizes the exact 2D field $u(x,y,t)$ at the requested
#       snapshot times, including nonlinear steepening and the diagonal shock.
#     * **`NPZ`**: Exports the full space-time ground-truth tensor and its
#       coordinates for subsequent verification and training workflows.
#
# %%
using Plots
using Roots
using LinearAlgebra
using NPZ

# %% [markdown]

# ## Cell 1: Core Analytical Solver Definition
#
# ## Algorithm: Periodic Characteristic Tracing with Entropy Root Selection
# The 2D solution is evaluated through the exact reduction $\xi=(x+y)/2$.
# Before $t_b=3/2$, the characteristic map has a unique footpoint. After
# breaking, all footpoints are found on monotone brackets and the entropy
# condition selects the appropriate outer branch relative to
# $\xi_s(t)=\pi+t/3$.
#
# %%
const BURGERS_ALPHA = 1.0 / 3.0
const BURGERS_BETA = 2.0 / 3.0
const BURGERS_BREAKING_TIME = 1.5
const BURGERS_TWO_PI = 2.0 * π
@inline reduced_initial_condition(xi0) =
    BURGERS_ALPHA + BURGERS_BETA * sin(xi0)
@inline characteristic_map(xi0, t) =
    xi0 + t * reduced_initial_condition(xi0)
@inline shock_position(t) = π + t / 3.0
"""
    characteristic_roots(xi, t)
Return every characteristic footpoint in the periodic interval that satisfies
    xi0 + t * (1/3 + 2/3*sin(xi0)) = xi + 2π*k
for an integer `k`. The search interval is split at the critical points of
the characteristic map, so every nonlinear solve has a monotone bracket.
"""
function characteristic_roots(xi, t)
    if t <= BURGERS_BREAKING_TIME
        knots = (0.0, BURGERS_TWO_PI)
    else
        # d/dxi0 [xi0 + t*U0(xi0)] = 1 + (2t/3)*cos(xi0).
        critical = acos(-3.0 / (2.0 * t))
        knots = (0.0, critical, BURGERS_TWO_PI - critical, BURGERS_TWO_PI)
    end
    values = map(xi0 -> characteristic_map(xi0, t), knots)
    map_min, map_max = extrema(values)
    # Only levels xi + 2π*k within the global map range can have roots.
    k_min = ceil(Int, (map_min - xi) / BURGERS_TWO_PI)
    k_max = floor(Int, (map_max - xi) / BURGERS_TWO_PI)
    footpoints = Float64[]
    for k in k_min:k_max
        level = xi + BURGERS_TWO_PI * k
        for j in 1:(length(knots) - 1)
            left = knots[j]
            right = knots[j + 1]
            f_left = values[j] - level
            f_right = values[j + 1] - level
            if iszero(f_left)
                push!(footpoints, left)
            elseif iszero(f_right)
                push!(footpoints, right)
            elseif f_left * f_right < 0.0
                residual(xi0) = characteristic_map(xi0, t) - level
                footpoint = find_zero(
                    residual,
                    (left, right),
                    Bisection(),
                )
                push!(footpoints, footpoint)
            end
        end
    end
    return footpoints
end
"""
    solve_reduced_burgers(xi, t)
Evaluate the exact entropy solution of the reduced 1D Burgers problem.
For `t > 3/2`, the maximum-`u` root is selected to the left of the shock
and the minimum-`u` root to its right; the middle branch is never selected.
"""
function solve_reduced_burgers(xi, t)
    t >= 0 || throw(DomainError(t, "The evaluation time must be nonnegative."))
    xi_periodic = mod(xi, BURGERS_TWO_PI)
    if iszero(t)
        return reduced_initial_condition(xi_periodic)
    end
    footpoints = characteristic_roots(xi_periodic, t)
    isempty(footpoints) && error(
        "No characteristic footpoint found for xi=$xi_periodic at t=$t.",
    )
    # Normalize possible endpoint duplicates before evaluating U0.
    u_roots = map(footpoints) do xi0
        reduced_initial_condition(mod(xi0, BURGERS_TWO_PI))
    end
    if t <= BURGERS_BREAKING_TIME
        # The map is invertible through the breaking time. Any duplicate at
        # the periodic seam represents the same footpoint and has the same u.
        return first(u_roots)
    end
    xi_shock = shock_position(t)
    if xi_periodic < xi_shock
        return maximum(u_roots)
    elseif xi_periodic > xi_shock
        return minimum(u_roots)
    else
        # Measure-zero shock point: use the Rankine--Hugoniot mean state.
        return BURGERS_ALPHA
    end
end
"""
    solve_burgers_analytical(x, y, t)
Evaluate the exact 2D entropy solution at one spatial point by lifting the
reduced solution along the diagonal coordinate `xi = (x + y) / 2`.
"""
function solve_burgers_analytical(x::Real, y::Real, t::Real)
    xi = (x + y) / 2.0
    return solve_reduced_burgers(xi, t)
end
# %% [markdown]


# ## Cell 2: Problem Configuration and Coordinate Mapping
#
# ## Physical Problem
# * **Domain:** $(x,y)\in[0,4\pi]\times[0,4\pi]$, periodic in both directions.
# * **Initial Condition:** $u_0(x,y)=\frac13+\frac23\sin((x+y)/2)$.
# * **Time Interval:** $t\in[0,2]$, including the breaking time $t_b=3/2$.
#
# ## Exact Diagonal Reduction
# The analytical solver uses $\xi=(x+y)/2$ and the reduced initial condition
# $U_0(\xi)=\frac13+\frac23\sin\xi$. No spatial or temporal rescaling is
# required: the physical coordinates and times are passed directly to the
# exact characteristic solver.
#
# %%
# --- 1. Define the periodic 2D spatial grid ---
x_min, x_max = 0.0, 4.0 * π
y_min, y_max = 0.0, 4.0 * π
N_samples = 256

x_phys = range(x_min, x_max, length=N_samples)
y_phys = range(y_min, y_max, length=N_samples)

# --- 2. Define the ground-truth save times ---
t_start, t_final = 0.0, 2.0
dt_save = 0.1
t_array = collect(t_start:dt_save:t_final)
N_time_steps = length(t_array)

# --- 3. Define the requested visualization times ---
snapshot_times = (0.0, 0.5, 1.5, 2.0)

println("Problem configured.")
println("Spatial grid: $N_samples × $N_samples points on [0, 4π]².")
println("Ground truth: $N_time_steps save times with dt_save = $dt_save.")
println("Snapshot times: $snapshot_times.")
println("Exact coordinate mapping: ξ = (x + y) / 2.")

# %% [markdown]

# ## Cell 3: Time Evolution Loop
#
# ## Batch Processing
# The Method of Characteristics allows every space-time point to be evaluated
# independently. We loop through the requested save times to construct the full
# ground-truth solution tensor `usol` for visualization and data export.
#
# * **Storage:** `usol[ix, iy, it]` corresponds to `(x_phys[ix], y_phys[iy], t_array[it])`.
# * **Complexity:** $O(N_x \times N_y \times N_t)$ independent exact evaluations.
# * **Precision:** The same analytical procedure applies before and after shock formation.
#
# %%
# Initialize the full solution tensor with dimensions (x, y, time).
usol = zeros(N_samples, N_samples, N_time_steps)

println("Starting exact pointwise solution evaluation...")

for (it, t_current) in enumerate(t_array)
    for (iy, y_current) in enumerate(y_phys)
        for (ix, x_current) in enumerate(x_phys)
            # The solver internally applies xi = (x + y) / 2 and the
            # appropriate smooth or entropy-satisfying characteristic branch.
            usol[ix, iy, it] = solve_burgers_analytical(
                x_current,
                y_current,
                t_current,
            )
        end
    end
end

println("Exact solution evaluation complete.")
println("Solution tensor size: ", size(usol))

# %% [markdown]

# ## Cell 4: Visualization Dashboard
#
# ## Four-Snapshot Exact-Solution Dashboard
# Each panel displays the complete primary field $u(x,y,t)$ on the periodic
# domain. The snapshots show the initial condition, nonlinear steepening,
# gradient catastrophe at $t=1.5$, and the developed entropy shock at $t=2$.
#
# %%
# Locate the requested snapshots in the ground-truth save-time array.
snapshot_indices = map(snapshot_times) do t_snapshot
    index = findfirst(
        t -> isapprox(t, t_snapshot; atol=1.0e-12, rtol=0.0),
        t_array,
    )
    index === nothing && error(
        "Snapshot time $t_snapshot is not present in the save-time array.",
    )
    return index
end

# Create one 2D field panel per requested time. Plots expects the field
# matrix in (y, x) order, while usol is stored in (x, y, time) order.
snapshot_panels = map(snapshot_times, snapshot_indices) do t_snapshot, it
    u_field = permutedims(usol[:, :, it])

    heatmap(
        x_phys,
        y_phys,
        u_field;
        xlabel="x",
        ylabel="y",
        title="t = $t_snapshot",
        aspect_ratio=:equal,
        colorbar=false,
        legend=false,
    )
end

# Combine the snapshots into the required 1 × 4 dashboard. No colormap or
# color limits are specified, so each panel uses automatic color scaling.
p_combined = plot(
    snapshot_panels...;
    layout=(1, 4),
    size=(1600, 400),
    margin=5 * Plots.mm,
)

savefig(p_combined, joinpath(@__DIR__, "summary_all.png"))
display(p_combined)

println("Dashboard saved to summary_all.png")

# %% [markdown]

# ## Cell 5: Data Persistence and Verification
#
# ## Artifact Generation and Internal Validation
# The full exact solution is saved to `burgers.npz` as ground truth. The array
# is stored in `(time, x, y)` order, with explicit coordinate arrays. The final
# section verifies the analytical shock trajectory and jump conditions at t=2.
#
# %%
# Convert from local (x, y, time) storage to (time, x, y) for export.
usol_save = Float64.(permutedims(usol, (3, 1, 2)))
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
# ----- Verify the saved file ----------------------------------------
println("\n--- File Verification ---")
file_size = filesize(npz_path)
println("File size: $(file_size) bytes ($(round(file_size / 1024 / 1024, digits=2)) MB)")
data_check = npzread(npz_path)
println("Keys in file: ", keys(data_check))
println("Shape of usol: ", size(data_check["usol"]))
println("Shape of t: ", size(data_check["t"]))
println("Shape of x: ", size(data_check["x"]))
println("Shape of y: ", size(data_check["y"]))
@assert size(data_check["usol"]) == (N_time_steps, N_samples, N_samples)
@assert size(data_check["t"]) == (N_time_steps,)
@assert size(data_check["x"]) == (N_samples,)
@assert size(data_check["y"]) == (N_samples,)
# ----- Validate the entropy shock at t = 2 ---------------------------
t_check = 2.0
xi_shock_check = shock_position(t_check)
shock_line_check = 2.0 * xi_shock_check
expected_shock_line = 2.0 * π + 4.0 / 3.0
# Nonzero solution of w = (2/3) sin(t*w), giving u_{L,R} = 1/3 ± w.
amplitude_residual(w) = w - BURGERS_BETA * sin(t_check * w)
w_check = find_zero(
    amplitude_residual,
    (1.0e-8, BURGERS_BETA),
    Bisection(),
)
u_left_expected = BURGERS_ALPHA + w_check
u_right_expected = BURGERS_ALPHA - w_check
expected_jump = 2.0 * w_check
# Probe immediately on either side of the measure-zero shock point.
probe_offset = 1.0e-6
u_left_check = solve_reduced_burgers(
    xi_shock_check - probe_offset,
    t_check,
)
u_right_check = solve_reduced_burgers(
    xi_shock_check + probe_offset,
    t_check,
)
jump_check = u_left_check - u_right_check
shock_speed_check = (u_left_check + u_right_check) / 2.0
@assert isapprox(
    shock_line_check,
    expected_shock_line;
    atol=1.0e-12,
    rtol=0.0,
)
@assert isapprox(u_left_check, u_left_expected; atol=1.0e-4, rtol=0.0)
@assert isapprox(u_right_check, u_right_expected; atol=1.0e-4, rtol=0.0)
@assert isapprox(jump_check, expected_jump; atol=1.0e-4, rtol=0.0)
@assert isapprox(jump_check, 1.28; atol=0.01, rtol=0.0)
@assert isapprox(shock_speed_check, 1.0 / 3.0; atol=1.0e-4, rtol=0.0)
println("\n--- Shock Validation at t = 2 ---")
println("Shock line x + y = ", shock_line_check)
println("Left state u_L = ", u_left_check)
println("Right state u_R = ", u_right_check)
println("Jump u_L - u_R = ", jump_check)
println("Rankine-Hugoniot shock speed = ", shock_speed_check)
