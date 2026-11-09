# %% [markdown]
# # Preamble: Solver and Analytical Framework Initialization
# 
# ## Framework: Method of Characteristics (MoC) for 2D Inviscid Burgers' Equation
# Initializes the components needed to generate the **Exact Entropy Solution** on the doubly periodic domain.
# 
# 1.  **Analytical Core (`Roots`, `LinearAlgebra`)**:
#     * Uses the invariant coordinate $s=x+y$ to reduce the 2D problem exactly to $U_t + 2UU_s = 0$.
#     * Solves the implicit characteristic equation $s=\xi+2tU_0(\xi)$ over its bounded root window using dense bracketing and root refinement.
#     * **Entropy Selection**: Enumerates all admissible characteristic roots and selects the minimizer of the Hopf–Lax functional, enforcing the Oleinik entropy condition and the correct Rankine–Hugoniot shocks.
# 
# 2.  **Diagnostics (`Plots`, `NPZ`)**:
#     * **`Plots`**: Visualizes $u(x,y,t)$ at the requested snapshot times to verify smooth steepening and shock formation.
#     * **`NPZ`**: Exports the full space-time ground-truth tensor for subsequent numerical verification and supervised training.
#
# %%
using Plots
using Roots
using LinearAlgebra
using NPZ
# %% [markdown]


# ## Cell 1: Core Analytical Solver Definition
#
# ## Algorithm: Implicit Characteristic Tracing with Entropy Root Selection
# The function `solve_burgers_analytical` evaluates the exact entropy solution
# without time-stepping errors.
#
# ### Logic Flow:
# 1. **Exact Reduction**: Use $s=x+y$ and $U_0(s)=1/3+(2/3)\sin(s/2)$, giving
#    $U_t+2UU_s=0$.
# 2. **Bounded Root Window**: For $t>0$, every characteristic root lies in
#    $[s-2t,s+2t/3]$.
# 3. **Root Finding**: Refine sign-changing brackets with safeguarded Newton
#    iterations. Before breaking, the full window contains the unique root;
#    afterward, a dense scan enumerates the roots of the folded characteristic map.
# 4. **Entropy Enforcement**: Select the root minimizing
#    $G(\xi)=(s-\xi)^2/(4t)+\phi_0(\xi)$, which implements the Oleinik entropy
#    condition and the equal-area rule.

# %%
"""
    initial_state(xi)

Initial profile $U_0(\xi)=1/3+(2/3)\sin(\xi/2)$.
"""
@inline initial_state(xi) = 1.0 / 3.0 + (2.0 / 3.0) * sin(xi / 2.0)

"""
    initial_potential(xi)

Primitive of the initial profile, normalized by $\phi_0(0)=0$.
"""
@inline initial_potential(xi) =
    xi / 3.0 + (4.0 / 3.0) * (1.0 - cos(xi / 2.0))

@inline characteristic_residual(xi, s, t) =
    xi + 2.0 * t * initial_state(xi) - s

@inline characteristic_derivative(xi, t) =
    1.0 + (2.0 * t / 3.0) * cos(xi / 2.0)

"""
    refine_characteristic_root(s, t, left, right)

Refine one root inside a sign-changing bracket. Newton steps are accepted only
when they remain inside the bracket; otherwise bisection is used. This retains
bracketing robustness at the gradient catastrophe, where the derivative can vanish.
"""
function refine_characteristic_root(s, t, left, right)
    residual(xi) = characteristic_residual(xi, s, t)
    derivative(xi) = characteristic_derivative(xi, t)

    f_left = residual(left)
    f_right = residual(right)
    iszero(f_left) && return left
    iszero(f_right) && return right

    lower = left
    upper = right
    f_lower = f_left
    xi = 0.5 * (lower + upper)

    for _ in 1:100
        f_xi = residual(xi)
        iszero(f_xi) && return xi

        if signbit(f_xi) == signbit(f_lower)
            lower = xi
            f_lower = f_xi
        else
            upper = xi
        end

        midpoint = 0.5 * (lower + upper)
        if midpoint == lower || midpoint == upper
            return midpoint
        end

        d_f_xi = derivative(xi)
        candidate = iszero(d_f_xi) ? NaN : xi - f_xi / d_f_xi

        if !isfinite(candidate) ||
           candidate <= lower || candidate >= upper || candidate == xi
            candidate = midpoint
        end
        xi = candidate
    end

    return 0.5 * (lower + upper)
end

function append_unique_root!(roots, candidate)
    tolerance = 64.0 * eps(Float64) * max(1.0, abs(candidate))
    if all(root -> abs(root - candidate) > tolerance, roots)
        push!(roots, candidate)
    end
    return roots
end

"""
    characteristic_roots(s, t)

Find all roots of $\xi+2tU_0(\xi)-s=0$ in the bounded characteristic window.
"""
function characteristic_roots(s, t)
    window_padding = 1.0e-12
    lower = s - 2.0 * t - window_padding
    upper = s + (2.0 * t / 3.0) + window_padding

    # F' >= 1-2t/3 > 0 before breaking. At breaking, the map is still
    # nondecreasing and the full bracket contains its unique root.
    if t <= 1.5
        return [refine_characteristic_root(s, t, lower, upper)]
    end

    roots = Float64[]
    residual(xi) = characteristic_residual(xi, s, t)

    scan_intervals = 256
    scan_grid = range(lower, upper; length=scan_intervals + 1)

    left = lower
    f_left = residual(left)
    iszero(f_left) && append_unique_root!(roots, left)

    for index in 1:scan_intervals
        right = scan_grid[index + 1]
        f_right = residual(right)

        if iszero(f_right)
            append_unique_root!(roots, right)
        elseif !iszero(f_left) && signbit(f_left) != signbit(f_right)
            root = refine_characteristic_root(s, t, left, right)
            append_unique_root!(roots, root)
        end

        left = right
        f_left = f_right
    end

    # The window endpoints have opposite signs, so at least one root must exist.
    if isempty(roots)
        push!(roots, refine_characteristic_root(s, t, lower, upper))
    end

    return roots
end

"""
    solve_burgers_analytical(s, t)
    solve_burgers_analytical(x, y, t)

Exact entropy solution of the reduced Burgers problem. The two-argument form
accepts a scalar or an array of $s=x+y$ values. The three-argument form accepts
vectors of $x$ and $y$ coordinates and returns an array with dimensions
$(N_x,N_y)$. Periodicity follows from the $4\pi$-periodic initial profile.
"""
function solve_burgers_analytical(s::Real, t::Real)
    s_value = Float64(s)
    t_value = Float64(t)

    t_value < 0.0 && throw(ArgumentError("The exact solver requires t >= 0."))
    iszero(t_value) && return initial_state(s_value)

    roots = characteristic_roots(s_value, t_value)

    best_root = first(roots)
    best_value = Inf

    for xi in roots
        hopf_lax_value =
            (s_value - xi)^2 / (4.0 * t_value) + initial_potential(xi)

        if hopf_lax_value < best_value
            best_value = hopf_lax_value
            best_root = xi
        end
    end

    return initial_state(best_root)
end

function solve_burgers_analytical(s::AbstractArray, t::Real)
    return map(s_value -> solve_burgers_analytical(s_value, t), s)
end

function solve_burgers_analytical(
    x::AbstractVector,
    y::AbstractVector,
    t::Real,
)
    s_grid = x .+ transpose(y)
    return solve_burgers_analytical(s_grid, t)
end
# %% [markdown]


# ## Cell 2: Problem Configuration and Coordinate Mapping
#
# ## Physical Problem
# * **Domain:** $(x,y)\in[0,4\pi]\times[0,4\pi]$, periodic in both directions
# * **Initial Condition:** $u_0(x,y)=1/3+(2/3)\sin((x+y)/2)$
# * **Time Interval:** $t\in[0,2]$
#
# The exact solver is evaluated directly in the physical coordinates. No
# canonical spatial mapping, time rescaling, CFL condition, or time stepper
# is required.

# %%
# --- 1. Define the periodic evaluation grid ---
x_min, x_max = 0.0, 4.0 * pi
y_min, y_max = 0.0, 4.0 * pi
t_start, t_final = 0.0, 2.0

# Retain the template's spatial sampling default in each direction.
N_samples = 256
dt_save = 0.1

# Exclude the duplicate endpoint because x=0 and x=4π are identified,
# and likewise for y.
x_phys = range(x_min, x_max; length=N_samples + 1)[1:end-1]
y_phys = range(y_min, y_max; length=N_samples + 1)[1:end-1]

# --- 2. Define output and snapshot times ---
t_array = collect(t_start:dt_save:t_final)
N_output_times = length(t_array)
snapshot_times = [0.0, 1.0, 1.5, 2.0]

@assert N_output_times == 21 "Expected 21 ground-truth output states."
@assert all(t -> any(isapprox(t; atol=1.0e-12), t_array), snapshot_times)

println("Problem configured on the doubly periodic domain [0, 4π]^2.")
println("Grid: $N_samples × $N_samples spatial points.")
println("Saving $N_output_times exact states from t=$t_start to t=$t_final.")
println("Snapshot times: ", snapshot_times)
# %% [markdown]


# ## Cell 3: Time Evolution Loop
#
# ## Batch Processing
# The Method of Characteristics permits independent evaluation at every output
# time. We construct the full ground-truth tensor `usol` for visualization and
# data persistence without time-stepping errors.
#
# * **Storage:** `usol[:, :, i]` is the complete 2D field at `t_array[i]`.
# * **Complexity:** $O(N_x N_y N_t)$ independent exact-solution evaluations.
# * **Precision:** The same entropy-selection procedure is valid before and
#   after shock formation; $t=0$ uses the initial condition directly.

# %%
# Solution tensor with dimensions (x, y, time).
usol = zeros(Float64, N_samples, N_samples, N_output_times)

# The solution depends on space only through s = x + y.
s_grid = x_phys .+ transpose(y_phys)

println("Starting exact solution evaluation at $(N_output_times) output times...")

for (i, t_current) in enumerate(t_array)
    # No canonical time mapping or numerical time stepping is required.
    usol[:, :, i] = solve_burgers_analytical(s_grid, t_current)
end

@assert size(usol) == (N_samples, N_samples, 21)
@assert all(isfinite, usol)

println("Exact solution evaluation complete.")
println("Solution tensor size: ", size(usol))


# ## Cell 4: Visualization Dashboard
#
# ## Four-Snapshot Exact-Solution Dashboard
# Display only the conserved field $u(x,y,t)$ at the initial time, during
# pre-shock steepening, at gradient catastrophe, and after shock formation.
# Each heatmap determines its color scale independently.

# %%
# Locate the requested snapshots among the saved output states.
snapshot_indices = map(snapshot_times) do t_snapshot
    index = findfirst(
        t -> isapprox(t, t_snapshot; atol=1.0e-12),
        t_array,
    )
    @assert index !== nothing "Snapshot time $t_snapshot is not in t_array."
    return index
end

# Create one automatically scaled 2D panel per snapshot.
snapshot_panels = map(snapshot_times, snapshot_indices) do t_snapshot, index
    u_snapshot = usol[:, :, index]

    heatmap(
        x_phys,
        y_phys,
        permutedims(u_snapshot),
        xlabel="x",
        ylabel="y",
        title="u(x,y), t = $t_snapshot",
        colorbar=false,
        aspect_ratio=:equal,
        legend=false,
    )
end

# Combine the four fields into the required 1 × 4 dashboard.
p_combined = plot(
    snapshot_panels...;
    layout=(1, 4),
    size=(1800, 420),
    margin=5 * Plots.mm,
)

savefig(p_combined, joinpath(@__DIR__, "summary_all.png"))
display(p_combined)

println("Four-snapshot dashboard saved to summary_all.png")
# %% [markdown]


# ## Cell 5: Data Persistence and Verification
#
# ## Artifact Generation and Ground-Truth Verification
# Save the high-fidelity exact solution to `burgers.npz` as ground truth for
# numerical error calculations and supervised training. The tensor uses the
# Python-friendly dimension order `(time, x, y)`.

# %%
# Convert from local (x, y, time) storage to (time, x, y).
usol_save = Float64.(permutedims(usol, (3, 1, 2)))
t_save = collect(Float64.(t_array))
x_save = collect(Float64.(x_phys))
y_save = collect(Float64.(y_phys))

@assert size(usol_save) == (21, N_samples, N_samples)
@assert length(t_save) == 21
@assert all(isfinite, usol_save)

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
println("File size: $(file_size) bytes ($(round(file_size / 1024 / 1024; digits=2)) MB)")

data_check = npzread(npz_path)
println("Keys in file: ", keys(data_check))
println("Shape of usol: ", size(data_check["usol"]))
println("Shape of t: ", size(data_check["t"]))
println("Shape of x: ", size(data_check["x"]))
println("Shape of y: ", size(data_check["y"]))

@assert size(data_check["usol"]) == (21, N_samples, N_samples)
@assert size(data_check["t"]) == (21,)
@assert size(data_check["x"]) == (N_samples,)
@assert size(data_check["y"]) == (N_samples,)
@assert all(isfinite, data_check["usol"])

# ----- Analytical shock self-check at t = 2 --------------------------
println("\n--- Shock Validation at t = 2 ---")
validation_time = 2.0
probe_offset = 1.0e-4

shock_positions = [
    2.0 * pi + 2.0 * validation_time / 3.0,
    6.0 * pi + 2.0 * validation_time / 3.0,
]
reference_positions = [7.6165, 20.1829]

for (shock_position, reference_position) in
    zip(shock_positions, reference_positions)

    # Probe the two entropy branches immediately adjacent to the shock.
    u_left = solve_burgers_analytical(
        shock_position - probe_offset,
        validation_time,
    )
    u_right = solve_burgers_analytical(
        shock_position + probe_offset,
        validation_time,
    )
    jump = u_left - u_right

    println(
        "s ≈ $(round(shock_position; digits=6)): " *
        "u_L = $(round(u_left; digits=6)), " *
        "u_R = $(round(u_right; digits=6)), " *
        "jump = $(round(jump; digits=6))",
    )

    @assert isapprox(
        shock_position,
        reference_position;
        atol=5.0e-5,
        rtol=0.0,
    )
    @assert isapprox(u_left, 0.9711; atol=2.0e-3, rtol=0.0)
    @assert isapprox(u_right, -0.3044; atol=2.0e-3, rtol=0.0)
    @assert u_left > u_right
    @assert isapprox(jump, 1.28; atol=0.01, rtol=0.0)
end

println("NPZ verification and analytical shock checks passed.")
