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
import Pkg; Pkg.add("QuadGK")
using QuadGK
# %% [markdown]


# ## Cell 1: Core Analytical Solver Definition
# ## Algorithm: Separation of Variables for Spherically Symmetric Wave Equation
# The function `solve_wave_spherical` computes the exact solution u(r,t) via eigenfunction expansion.
#
# ### Logic Flow:
# 1.  **Eigenvalue Problem**: Solves tan(10k) = 10k to find discrete wavenumbers k_n.
# 2.  **Spectral Decomposition**: Projects the initial condition v(r,0) onto the basis X_n(r) = sin(k_n r) to find coefficients a_n.
# 3.  **Series Summation**: Reconstructs v(r,t) and computes u(r,t) = v(r,t)/r.
# 4.  **Regularity**: Handles the removable singularity at r=0 via analytical limits.
# using QuadGK (Moved to Preamble)
"""
    compute_eigenvalues(N_modes)
Finds the first N_modes positive roots of the characteristic equation:
    tan(10k) = 10k
Roots are located in intervals (n*pi/10, n*pi/10 + pi/20).
"""
function compute_eigenvalues(N_modes)
    k_vals = Float64[]
    for n in 1:N_modes
        # Bracket search for the n-th root
        # Asymptotes at pi/20 + n*pi/10. Roots are slightly to the left of asymptotes?
        # tan(x)=x intersects in (n*pi, n*pi + pi/2) for x=10k
        # So 10k in (n*pi, n*pi + pi/2) -> k in (n*pi/10, n*pi/10 + pi/20)
        lower = n * π / 10.0 + 1e-4
        upper = n * π / 10.0 + π / 20.0 - 1e-4
        f(k) = tan(10.0 * k) - 10.0 * k
        # Bisection is robust for bracketed roots
        k_root = find_zero(f, (lower, upper), Roots.Bisection())
        push!(k_vals, k_root)
    end
    return k_vals
end
"""
    compute_coefficients(k_vals)
Computes Fourier coefficients a_n for the initial condition v(r,0).
v(r,0) = e^{-10(r-5)^2} * sin(pi*r/10)
"""
function compute_coefficients(k_vals)
    a_vals = Float64[]
    # Initial condition function for v(r)
    # v(r,0) = r * u(r,0) = e^{-10(r-5)^2} * sin(pi*r/10)
    v0(r) = exp(-10.0 * (r - 5.0)^2) * sin(π * r / 10.0)
    for k in k_vals
        # Numerator: Integral of v0(r) * sin(k*r)
        integrand_num(r) = v0(r) * sin(k * r)
        num, _ = quadgk(integrand_num, 0.0, 10.0)
        # Denominator: Integral of sin^2(k*r)
        integrand_den(r) = sin(k * r)^2
        den, _ = quadgk(integrand_den, 0.0, 10.0)
        push!(a_vals, num / den)
    end
    return a_vals
end
"""
    solve_wave_spherical(r_in, t, k_vals, a_vals)
Computes the exact solution u(r,t) = (1/r) * sum(a_n * cos(k_n*t) * sin(k_n*r)).
Handles r=0 via limit: u(0,t) = sum(a_n * k_n * cos(k_n*t)).
"""
function solve_wave_spherical(r_in, t, k_vals, a_vals)
    r_points = collect(Float64, r_in)
    u_out = zeros(length(r_points))
    for (i, r) in enumerate(r_points)
        sum_v = 0.0
        if r > 1e-10
            # Standard formula for r > 0
            for (k, a) in zip(k_vals, a_vals)
                sum_v += a * cos(k * t) * sin(k * r)
            end
            u_out[i] = sum_v / r
        else
            # Limit at r -> 0
            for (k, a) in zip(k_vals, a_vals)
                sum_v += a * k * cos(k * t)
            end
            u_out[i] = sum_v
        end
    end
    return u_out
end


# ## Cell 2: Problem Configuration and Coordinate Mapping
# ## Physical Problem: Spherically Symmetric 3D Wave Equation
# * **Domain:** r ∈ [0, 10]
# * **Time:** t ∈ [0, 6]
# * **Method:** Separation of Variables (Eigenfunction Expansion)
#
# ## Configuration
# * **Wave Speed:** c = 1.0
# * **Modes:** Truncated series with N_modes terms.

# --- 1. Define Physical Grid ---
r_min, r_max = 0.0, 10.0
t_start, t_final = 0.0, 6.0

# Resolution for Ground Truth
N_r = 500        # Spatial resolution
N_t = 600        # Temporal resolution (high resolution for smooth animation/data)
N_modes = 150    # Number of eigenmodes for series convergence

r_grid = range(r_min, r_max, length=N_r)
t_grid = range(t_start, t_final, length=N_t)

println("Problem configured.")
println("Domain: r ∈ [$r_min, $r_max], t ∈ [$t_start, $t_final]")
println("Grid: $N_r spatial points x $N_t time steps.")
println("Series Expansion: Using $N_modes eigenmodes.")

# ## Cell 3: Time Evolution Loop
#
# ## Batch Processing
# Unlike time-stepping solvers, the Method of Characteristics allows independent evaluation. We loop through the requested time steps to construct the full solution tensor `usol` for visualization and PINN training data.
#
# * **Complexity:** $O(N_x \times N_t)$ independent root-finding operations.
# * **Precision:** Valid at $t=0$ and $t > t_{shock}$ equally.
# %%
# Initialize solution matrix (Space x Time)
usol = zeros(N_r, N_t)

println("Starting pre-computation of eigenmodes...")

# 1. Compute Eigenvalues
k_vals = compute_eigenvalues(N_modes)
println("Computed $(length(k_vals)) eigenvalues.")

# 2. Compute Coefficients
a_vals = compute_coefficients(k_vals)
println("Computed expansion coefficients.")

println("Starting time evolution loop...")

# 3. Time Evolution
for (i, t_current) in enumerate(t_grid)
    # Compute exact solution vector for this time slice
    u_slice = solve_wave_spherical(r_grid, t_current, k_vals, a_vals)
    
    # Store in matrix
    usol[:, i] = u_slice
end

println("Evolution complete.")
# %% [markdown]

# ## Cell 4: Visualization Dashboard
# ----- Visualization: 1x4 Snapshots at t = 0, 2, 4, 6 -----
target_times = [0.0, 2.0, 4.0, 6.0]
plots_array = []

println("Generating snapshots...")

for t_target in target_times
    # Find closest time index in the computed grid
    idx = argmin(abs.(t_grid .- t_target))
    t_actual = t_grid[idx]
    
    # Create subplot
    p = plot(r_grid, usol[:, idx],
        title = "t = $(round(t_actual, digits=1))",
        xlabel = "r",
        ylabel = "u(r,t)",
        linewidth = 2,
        legend = false,
        colorbar = false,
        xlims = (0, 10)
    )
    push!(plots_array, p)
end

# Combine into a single figure
p_combined = plot(plots_array..., 
    layout = (1, 4),
    size = (1200, 300),
    margin = 5Plots.mm
)

savefig(p_combined, joinpath(@__DIR__, "summary_all.png"))
display(p_combined)

println("Dashboard saved to summary_all.png")

# ## Cell 5: Data Persistence and Verification
# ## Artifact Generation
# We save the high-fidelity exact solution to `wave_spherical.npz`.
# This serves as the Ground Truth for the spherically symmetric wave equation.

# Transpose to (Time, Space) for consistency with standard ML formats
usol_save = Float64.(usol')
t_save = collect(Float64.(t_grid))
r_save = collect(Float64.(r_grid))

npz_path = joinpath(@__DIR__, "wave_spherical.npz")
npzwrite(npz_path, Dict(
    "t" => t_save,
    "r" => r_save,
    "u" => usol_save
))

println("Data saved as NPZ to wave_spherical.npz")

# ----- Verify the saved file -----------------------------------------------
println("\n--- File Verification ---")
file_size = filesize(npz_path)
println("File size: $(file_size) bytes ($(round(file_size/1024/1024, digits=2)) MB)")

data_check = npzread(npz_path)
println("Keys in file: ", keys(data_check))
println("Shape of u: ", size(data_check["u"]))
println("Shape of t: ", size(data_check["t"]))
println("Shape of r: ", size(data_check["r"]))