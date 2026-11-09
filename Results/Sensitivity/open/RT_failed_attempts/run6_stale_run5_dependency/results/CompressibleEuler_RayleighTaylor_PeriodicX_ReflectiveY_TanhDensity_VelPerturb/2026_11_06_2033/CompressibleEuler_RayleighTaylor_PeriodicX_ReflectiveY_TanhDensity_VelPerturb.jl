# %% [markdown]

# ## Cell 1: Solver and Discretization Initialization
#
# ## Framework: Method of Lines (MOL) for Hyperbolic Systems
# Initializes the core components to solve conservation laws of the form $\partial_t \mathbf{u} + \nabla \cdot \mathbf{f}(\mathbf{u}) = \mathbf{0}$.
#
# 1.  **Time Integration (`OrdinaryDiffEqLowStorageRK`)**:
#     * Implements **Low-Storage Runge-Kutta (LSRK)** schemes.
#     * Optimized for high-order discretizations where the state vector $\mathbf{U}$ is large, minimizing memory registers while maintaining stability regions for explicit time-stepping ($dt$).
#
# 2.  **Spatial Discretization (`Trixi`)**:
#     * Provides the spatial operator $\mathcal{L}(\mathbf{U})$ for the semi-discrete form $\frac{d\mathbf{U}}{dt} = \mathcal{L}(\mathbf{U})$.
#     * Handles high-order discontinuous Galerkin (DGSEM) or Finite Volume operators, Riemann solvers (flux reconstruction), and adaptive mesh refinement (AMR).
#
# 3.  **Diagnostics (`Plots`, `NPZ`, `Statistics`)**:
#     * Utilities for computing integral norms ($L_2, L_\infty$) and exporting solution snapshots for surrogate modeling or analysis.
#
# %%
using OrdinaryDiffEqLowStorageRK
import DiffEqCallbacks: PeriodicCallback
using Trixi
using Plots
using NPZ
import Printf
using Statistics
# Reinitialize Polyester's per-thread task buffers for the active thread count.
# Without this, Trixi's threaded kernels throw `UndefRefError` in ManualMemory.jl
# when Julia is launched with multiple threads (--threads > 1).
Trixi.Polyester.reset_threads!()
# %% [markdown]

# ## Cell 2: Physics Model Definition (Compressible Navier-Stokes)
# ## Governing Equations: Inviscid Compressible Euler System
# Defines the 2D compressible Euler equations for conservation of mass,
# momentum, and total energy. The ideal-gas equation of state uses γ = 1.4.
# Gravity is introduced separately as a source term in the problem setup.

# %%
equations = CompressibleEulerEquations2D(1.4)
# %% [markdown]

# ## Cell 3: Initial Conditions and Source Terms (Lid-Driven Cavity)
#
# ## Problem Setup: Single-Mode Rayleigh–Taylor Instability
# Initializes a smoothed heavy-over-light density stratification with piecewise
# hydrostatic pressure and a single-mode vertical velocity perturbation.
# Gravity acts in the positive y-direction and enters through a source term.

# %%
@inline function initial_condition_rti(x, t, equations::CompressibleEulerEquations2D)
    x1, x2 = x

    # Heavy fluid (ρ = 2) below the interface; light fluid (ρ = 1) above it.
    rho = 2.0 + (1.0 - 2.0) * 0.5 *
          (1.0 + tanh(50.0 * (x2 - 0.5)))

    # Continuous piecewise hydrostatic pressure.
    p = x2 < 0.5 ? 2.0 * x2 + 1.0 : x2 + 1.5

    u = 0.0
    sound_speed = sqrt(equations.gamma * p / rho)
    v = -0.025 * sound_speed * cos(8.0 * pi * x1) * sin(pi * x2)^6

    return prim2cons(SVector(rho, u, v, p), equations)
end

@inline function source_terms_gravity(u, x, t,
                                      equations::CompressibleEulerEquations2D)
    g = 1.0
    rho, rho_v1, rho_v2, rho_e = u

    # The energy source ρ g v equals g ρv.
    return SVector(0.0, 0.0, g * rho, g * rho_v2)
end

@inline function density_indicator(u, equations)
    return u[1]
end
# %% [markdown]

# ## Cell 4: Spatial Discretization & Hybrid Shock Capturing
#
# ## Methodology: Nodal Discontinuous Galerkin (DGSEM)
# Constructs the spatial operator $\mathcal{L}(\mathbf{U})$ using the **Discontinuous Galerkin Spectral Element Method (DGSEM)**.
#
# 1.  **Function Space (`LobattoLegendreBasis`)**:
#     * Approximates solutions using tensor-product polynomials of degree $N=5$ on **Legendre-Gauss-Lobatto (LGL)** nodes.
#     * This nodal basis satisfies the **Summation-by-Parts (SBP)** property, essential for proving discrete entropy stability.
#
# 2.  **Shock Capturing Strategy (Hybrid DG-FV)**:
#     * **Problem**: High-order polynomials suffer from Gibbs oscillations near discontinuities (shocks).
#     * **Solution**: A convex blending of a high-order DG flux and a robust low-order Finite Volume (FV) flux via `volume_integral`.
#     * **Blending Formula**:
#         $$
#         \mathbf{f}_{numerical} = (1 - \alpha) \mathbf{f}_{DG}^{EC} + \alpha \mathbf{f}_{FV}^{Dissipative}
#         $$
#     * **High-Order Flux ($\mathbf{f}_{DG}^{EC}$)**: Uses `flux_ranocha`, an entropy-conserving/stable flux that preserves thermodynamic consistency in smooth regions.
#     * **Low-Order Flux ($\mathbf{f}_{FV}^{Dissipative}$)**: Uses `flux_lax_friedrichs`, a highly dissipative flux to stabilize shocks.
# 3.  **Interface Flux (`surface_flux=flux_hllc`)**:
#     * **Riemann Solver**: Harten-Lax-van Leer-Contact (HLLC).
#     * **Role**: Computes the numerical flux across element interfaces, including the contact wave omitted by HLL.
#     * **Physics**: Resolves the material contact discontinuity with reduced dissipation while retaining robust shock capturing.
#     * **Physics**: An approximate Riemann solver that estimates the fastest signal velocities (wave speeds). It is robust and positivity-preserving, making it ideal for compressible flows with strong shocks.
#
# 4.  **Indicator Function (`IndicatorHennemannGassner`)**:
#     * Dynamically computes the blending coefficient $\alpha \in [0, 1]$ element-wise.
#      * **Mechanism**: Monitors the spectral decay of energy in the highest modes of the polynomial expansion. If energy in high modes is large (indicating a discontinuity), $\alpha \to 1$ (activates FV sub-cells).
#
# %%
# Shared uniform-grid visualization utilities. These are defined before
# Stage A so that Run 5 is post-processed before Stage C construction.
function visualization_axes(nx=200, ny=800)
    nx >= 200 && ny >= 800 ||
        error("Visualization requires at least 200 x 800 grid points")
    return (range(0.0, 0.25; length=nx),
            range(0.0, 1.0; length=ny))
end

function rasterize_data(x, y, z, nx=200, ny=800)
    length(x) == length(y) == length(z) ||
        error("Coordinate and field arrays must have identical lengths")
    x_grid, y_grid = visualization_axes(nx, ny)
    x_min, x_max = 0.0, 0.25
    y_min, y_max = 0.0, 1.0

    xs = Float64[]
    ys = Float64[]
    zs = Float64[]
    sizehint!(xs, length(x))
    sizehint!(ys, length(y))
    sizehint!(zs, length(z))
    for k in eachindex(x)
        xk, yk, zk = Float64(x[k]), Float64(y[k]), Float64(z[k])
        if isfinite(xk) && isfinite(yk) && isfinite(zk)
            push!(xs, xk)
            push!(ys, yk)
            push!(zs, zk)
        end
    end
    isempty(xs) && error("No finite source samples available for interpolation")

    nbx, nby = 64, 256
    bins = [Int[] for _ in 1:(nbx * nby)]
    for k in eachindex(xs)
        bx = clamp(floor(Int, (xs[k] - x_min) / (x_max - x_min) * nbx) + 1,
                   1, nbx)
        by = clamp(floor(Int, (ys[k] - y_min) / (y_max - y_min) * nby) + 1,
                   1, nby)
        push!(bins[bx + (by - 1) * nbx], k)
    end

    grid = Matrix{Float64}(undef, ny, nx)
    candidates = Int[]
    max_radius = max(nbx, nby)
    for j in 1:ny
        py = y_grid[j]
        by = clamp(floor(Int, (py - y_min) / (y_max - y_min) * nby) + 1,
                   1, nby)
        for i in 1:nx
            px = x_grid[i]
            bx = clamp(floor(Int, (px - x_min) / (x_max - x_min) * nbx) + 1,
                       1, nbx)
            radius = 1
            while true
                empty!(candidates)
                for jb in max(1, by - radius):min(nby, by + radius)
                    for ib in max(1, bx - radius):min(nbx, bx + radius)
                        append!(candidates, bins[ib + (jb - 1) * nbx])
                    end
                end
                if length(candidates) >= 12 || radius >= max_radius
                    break
                end
                radius += 1
            end
            isempty(candidates) && error("Interpolation neighborhood is empty")

            weighted_sum = 0.0
            weight_sum = 0.0
            exact_sum = 0.0
            exact_count = 0
            for k in candidates
                dx = xs[k] - px
                dy = ys[k] - py
                distance_squared = dx * dx + dy * dy
                if distance_squared <= 1.0e-28
                    exact_sum += zs[k]
                    exact_count += 1
                else
                    weight = 1.0 / distance_squared
                    weighted_sum += weight * zs[k]
                    weight_sum += weight
                end
            end
            value = exact_count > 0 ? exact_sum / exact_count :
                    weighted_sum / weight_sum
            isfinite(value) ||
                error("Non-finite interpolated visualization value")
            grid[j, i] = value
        end
    end
    return grid
end

snapshot_times = [0.0, 1.0, 2.0, 3.0]
diagnostic_times = [1.0, 2.0, 3.0]

function _normalize_snapshot(raw, source)
    haskey(raw, "t") || error("Missing time in NPZ snapshot $source")
    raw_t = raw["t"]
    t = raw_t isa Number ? Float64(raw_t) :
        Float64(only(vec(collect(raw_t))))
    snapshot = Dict{String, Any}("t" => t)
    n_points = nothing
    for name in ("x", "y", "rho", "rho_v1", "rho_v2", "rho_e", "p")
        haskey(raw, name) || error("Missing $name in NPZ snapshot $source")
        values = Float64.(vec(collect(raw[name])))
        if n_points === nothing
            n_points = length(values)
        elseif length(values) != n_points
            error("Inconsistent field lengths in NPZ snapshot $source")
        end
        snapshot[name] = values
    end
    n_points > 0 || error("Empty NPZ snapshot $source")
    return snapshot
end

function load_npz_snapshots(directory)
    isdir(directory) || error("Snapshot directory does not exist: $directory")
    files = filter(path -> endswith(path, ".npz"),
                   readdir(directory; join=true))
    isempty(files) && error("No NPZ snapshots found in $directory")
    snapshots = [_normalize_snapshot(npzread(path), path) for path in files]
    sort!(snapshots; by=snapshot -> snapshot["t"])
    return snapshots
end

function load_npz_snapshots(directory, requested_times)
    available = load_npz_snapshots(directory)
    selected = Dict{String, Any}[]
    for requested_time in requested_times
        target = Float64(requested_time)
        distances = [abs(snapshot["t"] - target) for snapshot in available]
        index = argmin(distances)
        actual = available[index]["t"]
        isapprox(actual, target; atol=1.0e-4, rtol=0.0) ||
            error("No snapshot at t = $target in $directory; closest is t = $actual")
        push!(selected, available[index])
    end
    return selected
end

function interpolate_snapshot(snapshot; nx=200, ny=800)
    x_grid, y_grid = visualization_axes(nx, ny)
    rho = snapshot["rho"]
    rho_v2 = snapshot["rho_v2"]
    vertical_velocity = Vector{Float64}(undef, length(rho))
    for k in eachindex(rho)
        if isfinite(rho[k]) && rho[k] != 0.0
            vertical_velocity[k] = rho_v2[k] / rho[k]
        else
            vertical_velocity[k] = NaN
        end
    end
    return (
        t=snapshot["t"],
        x=x_grid,
        y=y_grid,
        rho=rasterize_data(snapshot["x"], snapshot["y"], rho, nx, ny),
        p=rasterize_data(snapshot["x"], snapshot["y"], snapshot["p"], nx, ny),
        v=rasterize_data(snapshot["x"], snapshot["y"], vertical_velocity, nx, ny),
        energy=rasterize_data(snapshot["x"], snapshot["y"],
                              snapshot["rho_e"], nx, ny),
    )
end

function horizontal_stripe_gate(grid, label)
    ndims(grid) == 2 || error("$label is not a two-dimensional image")
    size(grid, 2) >= 200 && size(grid, 1) >= 800 ||
        error("$label does not satisfy the 200 x 800 visualization minimum")
    all(isfinite, grid) ||
        error("$label contains non-finite pixels that could produce white stripes")
    println("Horizontal-stripe gate [$label]: PASS ",
            "(finite, gap-free uniform grid; no missing rows or pixels)")
    return true
end

function _periodicity_residual(grid)
    scale = max(maximum(abs, grid), eps(Float64))
    return maximum(abs.(grid[:, 1] .- grid[:, end])) / scale
end

function density_iso_metrics(rho; iso_level=1.5)
    ny, nx = size(rho)
    ny >= 2 && nx >= 1 ||
        error("Density image is too small for iso-level metrics")
    y_values = range(0.0, 1.0; length=ny)
    crossings = Float64[]
    for i in 1:nx
        for j in 1:(ny - 1)
            r0, r1 = Float64(rho[j, i]), Float64(rho[j + 1, i])
            (isfinite(r0) && isfinite(r1)) || continue
            d0, d1 = r0 - iso_level, r1 - iso_level
            if d0 == 0.0
                push!(crossings, y_values[j])
            end
            if d0 * d1 < 0.0
                fraction = (iso_level - r0) / (r1 - r0)
                push!(crossings,
                      y_values[j] + fraction * (y_values[j + 1] - y_values[j]))
            end
            if j == ny - 1 && d1 == 0.0
                push!(crossings, y_values[ny])
            end
        end
    end
    isempty(crossings) && error("No rho = $iso_level iso-level was found")
    return (spike_tip=maximum(crossings),
            bubble_tip=minimum(crossings),
            interface_location=mean(crossings))
end

function render_panels(items, output_path)
    length(items) == 4 ||
        error("Rendering requires snapshots at t = 0, 1, 2, 3")
    panels = Vector{Any}(undef, 8)
    for k in 1:4
        item = items[k]
        isapprox(item.t, snapshot_times[k]; atol=1.0e-8, rtol=0.0) ||
            error("Rendering snapshot order does not match t = 0, 1, 2, 3")
        all(isfinite, item.rho) && all(isfinite, item.p) ||
            error("Cannot render non-finite density or pressure panels")
        x, y = collect(item.x), collect(item.y)
        panels[k] = heatmap(
            x, y, item.rho;
            title=Printf.@sprintf("ρ, t = %.1f", item.t),
            xlabel="", ylabel="y",
            colorbar=false, aspect_ratio=:auto, framestyle=:box,
            xlims=(0.0, 0.25), ylims=(0.0, 1.0))
        panels[k + 4] = heatmap(
            x, y, item.p;
            title=Printf.@sprintf("p, t = %.1f", item.t),
            xlabel="x", ylabel="y",
            colorbar=false, aspect_ratio=:auto, framestyle=:box,
            xlims=(0.0, 0.25), ylims=(0.0, 1.0))
    end
    figure = plot(panels...; layout=(2, 4), size=(1600, 1200))
    savefig(figure, output_path)
    return output_path
end

function classify_run5_rollups(items)
    length(items) == 4 ||
        error("Stage A classification requires snapshots at t = 0, 1, 2, 3")
    reversal_count = 0
    triplet_count = 0
    intermediate_count = 0
    value_count = 0
    density_min = Inf
    density_max = -Inf

    for k in 3:4
        item = items[k]
        isapprox(item.t, snapshot_times[k]; atol=1.0e-8, rtol=0.0) ||
            error("Stage A classification requires Run 5 data at t = 2 and 3")
        rho = item.rho
        ny, nx = size(rho)
        density_min = min(density_min, minimum(rho))
        density_max = max(density_max, maximum(rho))
        intermediate_count += count(value -> 1.25 < value < 1.75, rho)
        value_count += length(rho)
        for j in 1:ny
            for i in 2:(nx - 1)
                d1 = rho[j, i] - rho[j, i - 1]
                d2 = rho[j, i + 1] - rho[j, i]
                triplet_count += 1
                if d1 * d2 < 0.0 && min(abs(d1), abs(d2)) > 0.02
                    reversal_count += 1
                end
            end
        end
        for j in 2:(ny - 1)
            for i in 1:nx
                d1 = rho[j, i] - rho[j - 1, i]
                d2 = rho[j + 1, i] - rho[j, i]
                triplet_count += 1
                if d1 * d2 < 0.0 && min(abs(d1), abs(d2)) > 0.02
                    reversal_count += 1
                end
            end
        end
    end

    triplet_count > 0 ||
        error("No late-time samples available for classification")
    reversal_rate = reversal_count / triplet_count
    intermediate_fraction = intermediate_count / value_count
    density_overshoot = max(0.0, 1.0 - density_min, density_max - 2.0)
    println("Stage A late-time rollup evidence:")
    Printf.@printf("  Density range at t = 2, 3: [%.16e, %.16e]\n",
                   density_min, density_max)
    Printf.@printf("  Density excursion beyond the initial [1, 2] range: %.16e\n",
                   density_overshoot)
    Printf.@printf("  Strong grid-scale reversal rate: %.16e\n", reversal_rate)
    Printf.@printf("  Intermediate-density fraction: %.16e\n",
                   intermediate_fraction)

    if reversal_rate > 0.01 ||
       (reversal_rate > 0.002 && density_overshoot > 0.05)
        println("Stage A rollup classification: oscillatory")
        return :oscillatory
    else
        println("Stage A rollup classification: smeared/blocky; " *
                "the oscillation criteria were not met")
        return :smeared
    end
end

function run5_baseline_integrals(item)
    x = collect(item.x)
    y = collect(item.y)
    nx, ny = length(x), length(y)
    nx >= 2 && ny >= 2 ||
        error("Run 5 conservation diagnostics require a two-dimensional grid")

    dx = (x[end] - x[1]) / (nx - 1)
    dy = (y[end] - y[1]) / (ny - 1)
    mass = 0.0
    energy = 0.0
    augmented_energy = 0.0

    for j in 1:ny
        wy = (j == 1 || j == ny) ? 0.5 : 1.0
        for i in 1:nx
            wx = (i == 1 || i == nx) ? 0.5 : 1.0
            weight = wx * wy * dx * dy
            rho = item.rho[j, i]
            total_energy = item.energy[j, i]
            mass += weight * rho
            energy += weight * total_energy
            augmented_energy += weight * (total_energy - rho * y[j])
        end
    end

    all(isfinite, (mass, energy, augmented_energy)) ||
        error("Non-finite Run 5 baseline conservation integral")
    return (mass=mass, energy=energy,
            augmented_energy=augmented_energy)
end

function report_run5_stage_b(snapshots, items)
    final_snapshot = snapshots[end]
    final_item = items[end]
    field_names = ("x", "y", "rho", "rho_v1", "rho_v2", "rho_e", "p")

    finite_state = isfinite(final_snapshot["t"]) &&
                   all(name -> all(isfinite, final_snapshot[name]), field_names)
    min_density = minimum(final_snapshot["rho"])
    min_pressure = minimum(final_snapshot["p"])

    initial_integrals = run5_baseline_integrals(items[1])
    final_integrals = run5_baseline_integrals(final_item)
    mass_relative_drift =
        (final_integrals.mass - initial_integrals.mass) /
        max(abs(initial_integrals.mass), eps(Float64))
    energy_relative_drift =
        (final_integrals.augmented_energy - initial_integrals.augmented_energy) /
        max(abs(initial_integrals.augmented_energy), eps(Float64))

    periodicity_residual = max(_periodicity_residual(final_item.rho),
                               _periodicity_residual(final_item.p),
                               _periodicity_residual(final_item.v))
    wall_bottom = maximum(abs, view(final_item.v, 1, :))
    wall_top = maximum(abs, view(final_item.v, size(final_item.v, 1), :))

    println("Stage B: Run 5 baseline diagnostics before Stage C:")
    println("  Finite-value check: ", finite_state ? "PASS" : "FAIL")
    Printf.@printf("  Minimum density: %.16e\n", min_density)
    Printf.@printf("  Minimum pressure: %.16e\n", min_pressure)
    println("  Positivity check: ",
            (min_density > 0.0 && min_pressure > 0.0) ? "PASS" : "FAIL")
    Printf.@printf("  Final simulation time: %.16e\n", final_snapshot["t"])
    println("  Actual time-step range: not stored in the Run 5 NPZ snapshots; " *
            "accepted steps are recorded for the refined run")
    Printf.@printf("  Uniform-grid mass integrals: initial = %.16e, final = %.16e\n",
                   initial_integrals.mass, final_integrals.mass)
    Printf.@printf("  Relative mass drift: %.16e\n", mass_relative_drift)
    Printf.@printf("  Uniform-grid raw energy integrals: initial = %.16e, final = %.16e\n",
                   initial_integrals.energy, final_integrals.energy)
    Printf.@printf("  Uniform-grid gravity-augmented energy: initial = %.16e, final = %.16e\n",
                   initial_integrals.augmented_energy,
                   final_integrals.augmented_energy)
    Printf.@printf("  Relative augmented-energy drift: %.16e\n",
                   energy_relative_drift)
    Printf.@printf("  Normalized x-periodicity residual: %.16e\n",
                   periodicity_residual)
    Printf.@printf("  Wall max |v|: y=0: %.16e; y=1: %.16e\n",
                   wall_bottom, wall_top)

    return (mass_relative_drift=mass_relative_drift,
            energy_relative_drift=energy_relative_drift,
            periodicity_residual=periodicity_residual,
            wall_bottom=wall_bottom, wall_top=wall_top)
end

# ================= PRE-SOLVE: RUN 5 STAGE A =================
println("Stage A: re-rendering Run 5 from saved NPZ data without rerunning.")
run5_npz_dir = joinpath(@__DIR__, "data_npz")
run5_snapshots = load_npz_snapshots(run5_npz_dir, snapshot_times)
run5_items = map(interpolate_snapshot, run5_snapshots)

horizontal_stripe_gate(run5_items[1].rho, "Run 5 t=0 density")
horizontal_stripe_gate(run5_items[1].p, "Run 5 t=0 pressure")

run5_rerender_path = joinpath(@__DIR__, "run5_rerender.png")
render_panels(run5_items, run5_rerender_path)
isfile(run5_rerender_path) || error("Run 5 re-render was not generated")
println("Stage A Run 5 re-render saved to: ", run5_rerender_path)

stage_a_rollup_classification = classify_run5_rollups(run5_items)

# ================= PRE-SOLVE: RUN 5 STAGE B =================
run5_stage_b_diagnostics = report_run5_stage_b(run5_snapshots, run5_items)
run5_metrics = map(
    k -> density_iso_metrics(run5_items[k + 1].rho), 1:3)

println("Stage B Run 5 RTI metrics from the rho = 1.5 iso-level:")
println("  t    spike tip       bubble tip      interface")
for k in 1:3
    metric = run5_metrics[k]
    Printf.@printf("  %.1f  %.10e  %.10e  %.10e\n",
                   diagnostic_times[k], metric.spike_tip,
                   metric.bubble_tip, metric.interface_location)
end

# Publish the Stage A decision only after Stage B has completed.
run5_rollup_classification = stage_a_rollup_classification
run5_preprocessing_complete = true
println("Stages A and B complete; Stage C refinements may now be constructed.")

# ================= STAGE C: SOLVER CONSTRUCTION =================
polydeg = 5
basis = LobattoLegendreBasis(polydeg)

# Smearing: reduce FV blending while retaining CFL = 0.5.
# Oscillations: retain blending; Cell 9 reduces only the CFL number.
if run5_rollup_classification == :smeared
    alpha_max_sc = 0.25
elseif run5_rollup_classification == :oscillatory
    alpha_max_sc = 0.5
else
    error("Stage A must classify Run 5 rollups as :smeared or :oscillatory")
end
alpha_min_sc = 0.001
alpha_smooth_sc = true
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=alpha_max_sc,
                                         alpha_min=alpha_min_sc,
                                         alpha_smooth=alpha_smooth_sc,
                                         variable=density_indicator)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=flux_ranocha,
                                                 volume_flux_fv=flux_lax_friedrichs)
dg = DGSEM(polydeg=polydeg, surface_flux=flux_hllc,
           volume_integral=volume_integral)
# %% [markdown]


# ## Cell 5: Computational Domain & Adaptive Mesh Topology
#
# ## Methodology: Forest-of-Octrees (`p4est`)
# Defines the spatial domain $\Omega$ and the underlying mesh topology using the **`p4est`** library. This framework represents the mesh as a forest of dynamic quadtrees (in 2D), enabling efficient, scalable parallel adaptive mesh refinement (AMR).
#
# 1.  **Geometric Domain**:
#     * Physical domain $\Omega = [0, 1] \times [0, 1]$ (Unit Square).
#     * **Macro-Element Structure**: `trees_per_dimension=(1, 1)` constructs the root connectivity as a single large quadtree covering the entire domain.
#
# 2.  **Discretization Topology (`P4estMesh`)**:
#     * **$h$-Refinement**: The mesh supports dynamic recursive splitting of elements.
#     * **Initial Resolution**: `initial_refinement_level=3` applies uniform recursive refinement to the root tree. The initial grid consists of $2^3 \times 2^3 = 64$ elements.
#     * **Polydeg**: Associates the spectral basis degree $N$ (from Cell 4) with each leaf element of the tree.
#
# 3.  **Boundary Topology**:
#     * `periodicity=(false, false)`: Disables periodic wrapping. This treats the domain boundaries $\partial \Omega$ as physical walls, necessitating explicit boundary conditions (Dirichlet/Neumann) for the Navier-Stokes operator.
#
# %%
coordinates_min = (0.0, 0.0)
coordinates_max = (0.25, 1.0)

# Single root tree; periodic in x, physical wall boundaries in y.
mesh = P4estMesh((1, 1), polydeg=polydeg,
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max,
                 initial_refinement_level=3, # Start coarser, let AMR refine
                 periodicity=(true, false))
# %% [markdown]

# ## Cell 6: General Boundary Condition Specification & System Assembly
#
# ## 1. Boundary Condition Methodology
# Defines the physical constraints at the domain boundaries using a flexible, function-based interface. This approach allows for arbitrary, spatially varying boundary profiles for general fluid dynamics problems.
#
# * **Function-Based Definitions (Closures)**:
#     Boundary values are defined as closures `(x, t, equations) -> values`. This allows BCs to be time-dependent (e.g., oscillating walls) or spatially varying (e.g., parabolic inlet profiles).
#     * **Momentum (`NoSlip`)**: Implements Dirichlet conditions for velocity $\mathbf{v} = \mathbf{v}_{target}(x,t)$.
#         * *Example 1*: A moving boundary (shear driver) where $\mathbf{v} \neq 0$.
#         * *Example 2*: A static wall where $\mathbf{v} = 0$.
#     * **Energy (`Adiabatic` vs `Isothermal`)**:
#         * *Adiabatic*: Neumann condition ($\nabla T \cdot \mathbf{n} = 0$) for insulated walls.
#         * *Isothermal*: Dirichlet condition ($T = T_{wall}$) for heated/cooled walls.
#
# * **Physics Wrappers**:
#     Raw functions are wrapped in physics-specific containers (e.g., `BoundaryConditionNavierStokesWall`) to bundle momentum and energy constraints into a single object usable by the parabolic solver.
#
# ## 2. Topology-Physics Mapping (`Dict`)
# Maps the geometric boundary tags (defined in the Mesh Construction step) to the physical boundary objects.
# * **Inviscid (Hyperbolic) Dictionary**: Defines the base state for the Euler operator (typically `slip_wall` for impermeability).
# * **Viscous (Parabolic) Dictionary**: Defines the viscous corrections (No-Slip/Heat Flux).
# * **Generalization**: Keys (e.g., `:x_pos`, `:y_neg`) must match the boundary names defined by the mesh generator (P4est or Unstructured).
#
# ## 3. Semidiscretization: Constructing the Operator $\mathcal{L}(\mathbf{U})$
# This step fuses the Topology (Mesh), Physics (Equations), and Numerics (DG Solver) into the global semi-discrete operator. The framework selects the appropriate container based on the physics regime:
#
# ### **Case A: Pure Hyperbolic (Euler)**
# $$ \frac{d\mathbf{U}}{dt} = \mathcal{L}_{hyp}(\mathbf{U}) $$
# * **Constructor**: `SemidiscretizationHyperbolic`
# * **Inputs**: Requires only the base `equations` (Euler) and standard boundary conditions.
# * **Role**: Assembles the inviscid spatial operator using the selected Riemann solver (e.g., HLL).
#
# ### **Case B: Hyperbolic-Parabolic (Navier-Stokes)**
# $$ \frac{d\mathbf{U}}{dt} = \mathcal{L}_{hyp}(\mathbf{U}) + \mathcal{L}_{par}(\mathbf{U}, \nabla \mathbf{U}) $$
# * **Constructor**: `SemidiscretizationHyperbolicParabolic`
# * **Inputs**: Requires both `equations` (Euler) and `equations_parabolic` (Diffusion).
# * **Role**:
#     1.  Computes the inviscid residuals.
#     2.  Computes the viscous residuals (gradients + fluxes) using the scheme defined in Cell 2 (e.g., Bassi-Rebay 1).
#     3.  Applies the split boundary conditions (Slip for Euler, No-Slip for Diffusion).
#
# Reflective/free-slip walls at y = 0 and y = 1.
# The x-direction is periodic through the P4estMesh connectivity.
boundary_conditions = Dict(
    :y_neg => boundary_condition_slip_wall,
    :y_pos => boundary_condition_slip_wall
)

semi = SemidiscretizationHyperbolic(mesh, equations,
                                    initial_condition_rti, dg;
                                    boundary_conditions=boundary_conditions,
                                    source_terms=source_terms_gravity)
# %% [markdown]

# ## Cell 7: Dynamic Adaptive Mesh Refinement (AMR) Strategy
#
# ## Methodology: Feature-Driven $h$-Refinement
# Implements an automated feedback loop that dynamically adjusts the local grid resolution ($h$) based on the evolving flow physics. This minimizes computational cost by focusing degrees of freedom only where needed (e.g., boundary layers, vortices).
#
# 1.  **The Sensor: A Posteriori Error Indicator (`IndicatorLöhner`)**
#     * **Algorithm**: Uses the **Löhner Error Estimator**, a classic FEM/DG metric for compressible flows.
#     * **Mechanism**: It estimates the local smoothness of the solution by computing a normalized second derivative (curvature) of the **Velocity Magnitude**.
#     * **Role**: Acts as a "feature detector," identifying regions with high gradients (shear layers) or discontinuities that require higher resolution to resolve.
#
# 2.  **The Controller: Hysteresis Logic (`ControllerThreeLevel`)**
#     * Maps the continuous indicator value $\epsilon$ to discrete mesh operations (Refine/Coarsen/Keep) using a **Three-Level Hysteresis** approach to prevent "chattering" (rapid toggling between levels).
#     * **Refine Condition ($h \to h/2$)**: If error $\epsilon > 0.3$ (`max_threshold`). Splits the element into 4 children (quadtree).
#     * **Coarsen Condition ($h \to 2h$)**: If error $\epsilon < 0.1$ (`med_threshold`). Merges 4 children into 1 parent.
#     * **Deadband**: If $0.1 \le \epsilon \le 0.3$, the mesh is frozen to ensure stability.
#     * **Bounds**: Enforces hard constraints on resolution levels ($L_{min}=4, L_{max}=6$) to guarantee memory limits.
#
# 3.  **The Actuator: Temporal Intervention (`AMRCallback`)**
#     * **Trigger**: Interrupts the time-integration loop every 20 timesteps (`interval=20`).
#     * **Action**: Executes the mesh topology changes and performs the **L2-Projection** (interpolation) of the solution polynomial $\mathbf{U}$ from the old mesh to the new mesh.
#
# %%
amr_indicator = IndicatorLöhner(semi, variable=density_indicator)

# Level 3 is the existing base mesh; only density-selected features refine.
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=3,
                                      med_level=5, med_threshold=0.05,
                                      max_level=6, max_threshold=0.15)

amr_callback = AMRCallback(semi, amr_controller,
                           interval=20,
                           adapt_initial_condition=true)
# %% [markdown]

# ## Cell 8: Data I/O & Surrogate Interoperability
#
# ## Strategy: Cross-Language Serialization (.npz)
# Implements a custom callback to export solution snapshots in **NumPy** format. This bridges the high-performance Julia solver with Python-based analysis tools or ML surrogate models (e.g., the ATHENA Advisor).
#
# 1.  **State Transformation ($\mathbf{u} \to \mathbf{w}$)**:
#     * The solver evolves **Conservative Variables** $\mathbf{u} = [\rho, \rho v_1, \rho v_2, \rho E]^T$.
#     * For analysis, we map these to **Primitive Variables** $\mathbf{w} = [\rho, v_1, v_2, p]^T$ using the `cons2prim` mapping during extraction.
#
# 2.  **Spatial Reconstruction (`PlotData2D`)**:
#     * Projects the solution from the high-order Discontinuous Galerkin polynomial basis (on adaptive quadtrees) onto a structured 2D Cartesian grid suitable for visualization and tensor-based processing.
#
# 3.  **Temporal Sampling**:
#     * **Trigger**: `DiscreteCallback` executes every $N=40$ timesteps or at $t=T_{end}$.
#     * **Output**: Saves fields $(\rho, u, v, p)$ along with grid coordinates $(x, y)$ to `data_npz/`.
# Run 5 snapshots are loaded by the pre-solver Stage A post-processing cell.
# Keep refined output separate; never overwrite or mix it with Run 5.
npz_out = joinpath(@__DIR__, "data_npz_refined")
mkpath(npz_out)
dt_save = 0.1
# Save the complete conserved state and the pressure used for visualization.
function save_npz_file(u, t, semi)
    pd_cons = PlotData2D(u, semi, solution_variables=cons2cons)
    pd_prim = PlotData2D(u, semi, solution_variables=cons2prim)
    # Each PlotData2D state entry contains all variables at one plot node.
    # Select the required component before flattening, without accessing
    # any fields on the scalar PlotDataSeries returned by pd["variable"].
    cons_data = pd_cons.data
    prim_data = pd_prim.data
    x      = Float64.(vec(collect(pd_cons.x)))
    y      = Float64.(vec(collect(pd_cons.y)))
    rho    = Float64.(vec(getindex.(cons_data, 1)))
    rho_v1 = Float64.(vec(getindex.(cons_data, 2)))
    rho_v2 = Float64.(vec(getindex.(cons_data, 3)))
    rho_e  = Float64.(vec(getindex.(cons_data, 4)))
    p      = Float64.(vec(getindex.(prim_data, 4)))
    n_points = length(x)
    (length(y) == n_points &&
     all(length(field) == n_points
         for field in (rho, rho_v1, rho_v2, rho_e, p))) ||
        error("Inconsistent coordinate and field sizes in NPZ export")
    filename = joinpath(npz_out, Printf.@sprintf("time_%08.5f.npz", t))
    npzwrite(filename, Dict("t" => Float64(t),
                            "x" => x,
                            "y" => y,
                            "rho" => rho,
                            "rho_v1" => rho_v1,
                            "rho_v2" => rho_v2,
                            "rho_e" => rho_e,
                            "p" => p))
    println("Saved: $(basename(filename))")
    return nothing
end
# Callback entry point; the explicit-state method also supports post-solve export.
function save_npz_file(integrator)
    return save_npz_file(integrator.u, integrator.t, integrator.p)
end
# Force an initial save at t = 0.
function init_save_npz(cb, u, t, integrator)
    save_npz_file(integrator)
end
# Time-based saves every dt_save, rather than saves based on solver iterations.
save_npz_callback = PeriodicCallback(save_npz_file, dt_save;
                                     save_positions=(false, false),
                                     initialize=init_save_npz)
# %% [markdown]

# ## Cell 9: ODE System Formulation & Time-Loop Control
#
# ## 1. Method of Lines (MOL) Finalization
# Converts the spatially discretized PDE and boundary conditions into a large system of Ordinary Differential Equations (ODEs).
# $$
# \frac{d \mathbf{U}}{dt} = \mathcal{L}(\mathbf{U}, t), \quad \mathbf{U}(0) = \mathbf{U}_0
# $$
# * **`semidiscretize`**: Assembles the global state vector $\mathbf{U}$ and the right-hand side operator $\mathcal{L}$, preparing the system for the LSRK integrator defined in Cell 1.
# * **Time Domain**: `tspan = (0.0, 3.0)` defines the simulation window $t \in [0, T_{end}]$.
#
# ## 2. Stability Control (CFL Condition)
# Explicit time-stepping requires strict adherence to the **Courant-Friedrichs-Lewy (CFL)** condition for stability.
# * **`StepsizeCallback`**: Dynamically adapts the timestep $\Delta t_n$ based on the maximum wave speed in the domain.
#     $$
#     \Delta t_n = \text{CFL} \cdot \min_{\Omega} \left( \frac{h}{|\mathbf{v}| + c} \right)
#     $$
#     Here, we set `cfl=0.5` to ensure the method remains within the stability region of the Runge-Kutta scheme.
#
# ## 3. Observability & Interventions (`CallbackSet`)
# Aggregates the diagnostic and control agents into the main solver loop:
# * **`AnalysisCallback`**: Computes in-situ functional quantities (Entropy, Kinetic Energy, $L_2/L_\infty$ errors) to monitor physical validity.
# * **`SaveSolutionCallback`**: Performs full-state serialization (HDF5-compatible) for restarts.
# * **`AMRCallback`**: (From Cell 7) Triggers mesh adaptation.
# %%
# Stage C ODE construction follows the pre-solver Run 5 Stages A and B.
@isdefined(run5_rollup_classification) ||
    error("Run 5 Stages A and B must complete before Stage C ODE construction")
tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)
summary_callback = SummaryCallback()
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
alive_callback = AliveCallback(analysis_interval = analysis_interval)
# We save fewer solution files to save disk space, but analysis runs often
trixi_out = joinpath(@__DIR__, "trixi_output")
# Save the full conserved state on the prescribed output schedule.
save_solution = SaveSolutionCallback(dt = dt_save,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     output_directory = trixi_out,
                                     solution_variables = cons2cons)
# Stage C: reduce CFL only when Stage A identifies genuine oscillations.
# For smearing, Cell 4 moderates shock capturing and CFL remains unchanged.
if run5_rollup_classification == :oscillatory
    cfl_number = 0.3
elseif run5_rollup_classification == :smeared
    cfl_number = 0.5
else
    error("Stage A must classify Run 5 rollups as :smeared or :oscillatory " *
          "before selecting the refined CFL number")
end
stepsize_callback = StepsizeCallback(cfl=cfl_number)
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        amr_callback,
                        save_npz_callback,
                        stepsize_callback)
# %% [markdown]

# ## Cell 10: Time Integration Loop
#
# ## Methodology: Explicit Time-Stepping
# Evolves the semi-discrete system $\frac{d\mathbf{U}}{dt} = \mathcal{L}(\mathbf{U})$ from $t_0$ to $t_{end}$.
#
# ### 1. The Numerical Integrator: `RDPK3SpFSAL49`
# * **Algorithm**: A 3rd-order, Low-Storage Runge-Kutta (LSRK) method with 4 stages.
# * **Selection Rationale**:
#     * **Memory Efficiency**: As an LSRK method, it minimizes the number of state vector registers required ($2N$ storage vs $4N+$ for classic RK4), which is critical for high-DOF spectral element methods.
#     * **Stability**: Optimized for spectra containing both imaginary eigenvalues (advection) and negative real eigenvalues (diffusion).
#
# ### 2. Adaptive Control Strategy
# * **`adaptive=false`**: We disable the ODE solver's internal error-based adaptivity (PID control).
#     * **Reasoning**: For PDEs, local truncation error control is computationally expensive and often poor at handling shocks.
#     * **Alternative**: We rely entirely on the `StepsizeCallback` (Cell 9), which controls $\Delta t$ based on the physical **CFL condition** (wave propagation speed).
# %%
# Record accepted step sizes separately from sol.t, which also contains saveat times.
dt_extrema = Ref((Inf, 0.0))
function record_actual_timestep(integrator)
    dt = integrator.t - integrator.tprev
    if isfinite(dt) && dt > 0
        current_min, current_max = dt_extrema[]
        dt_extrema[] = (min(current_min, dt), max(current_max, dt))
    end
    return nothing
end
timestep_callback = DiscreteCallback((u, t, integrator) -> true,
                                     record_actual_timestep;
                                     save_positions=(false, false))
# Keep StepsizeCallback last while retaining every callback from Cell 9.
callbacks_with_diagnostics = CallbackSet(summary_callback,
                                         analysis_callback,
                                         alive_callback,
                                         save_solution,
                                         amr_callback,
                                         save_npz_callback,
                                         timestep_callback,
                                         stepsize_callback)
sol = solve(ode, RDPK3SpFSAL49();
            dt=1.0,
            adaptive=false,
            callback=callbacks_with_diagnostics,
            saveat=0.01,
            maxiters=1e7)
# PeriodicCallback does not guarantee an NPZ export at the final time.
final_time = sol.t[end]
isapprox(final_time, tspan[2]; atol=1e-12, rtol=0.0) ||
    error("Solver finished at t = $final_time, expected t = $(tspan[2])")
# Export the actual final state, including all conserved variables and pressure.
save_npz_file(sol.u[end], final_time, semi)
# Use the exported final state for the same nodal fields available to post-processing.
final_snapshot_path = joinpath(npz_out,
                               Printf.@sprintf("time_%08.5f.npz", final_time))
final_snapshot = npzread(final_snapshot_path)
final_rho = final_snapshot["rho"]
final_rho_v1 = final_snapshot["rho_v1"]
final_rho_v2 = final_snapshot["rho_v2"]
final_rho_e = final_snapshot["rho_e"]
final_p = final_snapshot["p"]
final_fields = (final_rho, final_rho_v1, final_rho_v2, final_rho_e, final_p)
finite_state = all(isfinite, sol.u[end]) &&
               all(field -> all(isfinite, field), final_fields)
min_density = minimum(final_rho)
min_pressure = minimum(final_p)
dt_min, dt_max = dt_extrema[]
println("Final-state diagnostics:")
println("  Finite-value check: ", finite_state ? "PASS" : "FAIL")
Printf.@printf("  Minimum density: %.16e\n", min_density)
Printf.@printf("  Minimum pressure: %.16e\n", min_pressure)
println("  Positivity check: ",
        min_density > 0.0 && min_pressure > 0.0 ? "PASS" : "FAIL")
if isfinite(dt_min)
    Printf.@printf("  Actual time-step range: dt_min = %.16e, dt_max = %.16e\n",
                   dt_min, dt_max)
else
    println("  Actual time-step range: no accepted steps were recorded")
end
Printf.@printf("  Final simulation time: %.16e\n", final_time)
# Report resolution, fluxes, AMR, and shock-capturing settings explicitly.
base_level = amr_controller.base_level
max_level = amr_controller.max_level
base_element_count = 4^base_level
maximum_element_count = 4^max_level
println("Implementation summary:")
println("  polydeg = ", polydeg)
println("  Base element count = ", base_element_count,
        " (one tree, base_level = ", base_level, ")")
println("  Maximum element count = ", maximum_element_count,
        " (one tree, max_level = ", max_level, ")")
println("  Surface flux = flux_hllc")
println("  Volume fluxes: DG = flux_ranocha; FV = flux_lax_friedrichs")
println("  AMR: IndicatorLöhner on density; levels = (",
        base_level, ", ", amr_controller.med_level, ", ", max_level,
        "); med_threshold = ", amr_controller.med_threshold,
        "; max_threshold = ", amr_controller.max_threshold,
        "; interval = 20")
println("  Shock capturing: IndicatorHennemannGassner on density; ",
        "alpha_max = ", alpha_max_sc,
        "; alpha_min = ", alpha_min_sc,
        "; alpha_smooth = ", alpha_smooth_sc)
println("  CFL = ", cfl_number)
# %% [markdown]

# ## Cell 11: Data Regularization (Rasterization)
# Post-solve diagnostic utilities.
# Run 5 Stages A and B, shared visualization helpers, and Stage C construction
# have already completed in Cell 4 before the refined solve.
function dg_conservation_integrals(u_ode, semi)
    # Trixi.integrate applies the native DGSEM quadrature and element Jacobians.
    # Do not use rasterized visualization data for conservation diagnostics.
    cache = semi.cache
    gravity = 1.0
    mass = Trixi.integrate(u_ode, semi) do u, i, j, element, equations, dg
        u[1]
    end
    energy = Trixi.integrate(u_ode, semi) do u, i, j, element, equations, dg
        u[4]
    end
    augmented_energy = Trixi.integrate(u_ode, semi) do u, i, j, element,
                                                       equations, dg
        coordinates = Trixi.get_node_coords(cache.elements.node_coordinates,
                                            equations, dg, i, j, element)
        y = coordinates[2]
        # Conserved quantity for gravity g = +1 in y: E - g*rho*y.
        u[4] - gravity * u[1] * y
    end
    integrals = (mass=mass, energy=energy,
                 augmented_energy=augmented_energy)
    all(isfinite, integrals) ||
        error("Non-finite direct DG conservation integral")
    return integrals
end

function report_snapshot_diagnostics(initial_snapshot, final_snapshot,
                                     initial_item, final_item,
                                     initial_integrals, final_integrals,
                                     heading="Snapshot diagnostics:")
    println(heading)
    field_names = ("x", "y", "rho", "rho_v1", "rho_v2", "rho_e", "p")
    finite_state = isfinite(final_snapshot["t"]) &&
                   all(name -> all(isfinite, final_snapshot[name]), field_names)
    min_density = minimum(final_snapshot["rho"])
    min_pressure = minimum(final_snapshot["p"])
    println("  Finite-value check: ", finite_state ? "PASS" : "FAIL")
    Printf.@printf("  Minimum density: %.16e\n", min_density)
    Printf.@printf("  Minimum pressure: %.16e\n", min_pressure)
    println("  Positivity check: ",
            min_density > 0.0 && min_pressure > 0.0 ? "PASS" : "FAIL")
    Printf.@printf("  Final snapshot time: %.16e\n", final_snapshot["t"])

    # These values are direct DG-state integrals, not visualization integrals.
    mass_initial = initial_integrals.mass
    mass_final = final_integrals.mass
    mass_relative_drift = (mass_final - mass_initial) /
                          max(abs(mass_initial), eps(Float64))
    energy_initial = initial_integrals.energy
    energy_final = final_integrals.energy
    augmented_initial = initial_integrals.augmented_energy
    augmented_final = final_integrals.augmented_energy
    energy_relative_drift = (augmented_final - augmented_initial) /
                            max(abs(augmented_initial), eps(Float64))
    Printf.@printf("  DG mass integral: initial = %.16e, final = %.16e\n",
                   mass_initial, mass_final)
    Printf.@printf("  Relative mass drift: %.16e\n", mass_relative_drift)
    Printf.@printf("  DG raw energy integral: initial = %.16e, final = %.16e\n",
                   energy_initial, energy_final)
    Printf.@printf("  DG gravity-augmented energy: initial = %.16e, final = %.16e\n",
                   augmented_initial, augmented_final)
    Printf.@printf("  Relative augmented-energy drift: %.16e\n",
                   energy_relative_drift)

    # Uniform-grid data are used only for boundary residuals, never for the
    # conservation calculations above.
    periodicity_residual = max(_periodicity_residual(final_item.rho),
                               _periodicity_residual(final_item.p),
                               _periodicity_residual(final_item.v))
    wall_bottom = maximum(abs, view(final_item.v, 1, :))
    wall_top = maximum(abs, view(final_item.v, size(final_item.v, 1), :))
    Printf.@printf("  Normalized x-periodicity residual: %.16e\n",
                   periodicity_residual)
    Printf.@printf("  Wall max |v|: y=0: %.16e; y=1: %.16e\n",
                   wall_bottom, wall_top)
    return (mass_relative_drift=mass_relative_drift,
            energy_relative_drift=energy_relative_drift,
            periodicity_residual=periodicity_residual,
            wall_bottom=wall_bottom, wall_top=wall_top)
end
# %% [markdown]


# ## Cell 12: Post-Processing & Temporal Diagnostics
# ================= POST-SOLVE: REFINED STAGE B =================
refined_snapshots = load_npz_snapshots(npz_out, snapshot_times)
refined_items = map(interpolate_snapshot, refined_snapshots)
final_snapshot_time = refined_snapshots[end]["t"]
isapprox(final_snapshot_time, snapshot_times[end];
         atol=1.0e-12, rtol=0.0) ||
    error("Final NPZ snapshot has t = $final_snapshot_time, expected t = 3.0")
horizontal_stripe_gate(refined_items[1].rho, "refined t=0 density")
horizontal_stripe_gate(refined_items[1].p, "refined t=0 pressure")
# Conservation diagnostics use the saved DG states and native DG quadrature.
# Rasterized fields are used only for plotting, boundary residuals, and
# extracting the rho = 1.5 material-interface contour.
initial_integrals = dg_conservation_integrals(sol.u[1], semi)
final_integrals = dg_conservation_integrals(sol.u[end], semi)
report_snapshot_diagnostics(
    refined_snapshots[1], refined_snapshots[end],
    refined_items[1], refined_items[end],
    initial_integrals, final_integrals,
    "Refined conservation and boundary diagnostics:")
refined_metrics = map(
    k -> density_iso_metrics(refined_items[k + 1].rho), 1:3)
# Run 5 was loaded, re-rendered, classified, and measured during pre-solve
# Stage A. Reuse its cached metrics for the resolution-convergence table.
println("RTI metrics from the rho = 1.5 iso-level: Run 5 versus refined run")
println("  t    Run5 spike     Run5 bubble    Run5 interface  " *
        "Refined spike  Refined bubble Refined interface")
for k in 1:3
    run5_metric = run5_metrics[k]
    refined_metric = refined_metrics[k]
    Printf.@printf("  %.1f  %.10e  %.10e  %.10e  %.10e  %.10e  %.10e\n",
                   diagnostic_times[k],
                   run5_metric.spike_tip, run5_metric.bubble_tip,
                   run5_metric.interface_location,
                   refined_metric.spike_tip, refined_metric.bubble_tip,
                   refined_metric.interface_location)
end
println("Saving summary_all.png...")
summary_path = joinpath(@__DIR__, "summary_all.png")
render_panels(refined_items, summary_path)
isfile(summary_path) || error("summary_all.png was not generated")
println("Done.")