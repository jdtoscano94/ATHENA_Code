# %% [markdown]
# # Preamble: Solver and Analytical Framework Initialization
# 
# ## Framework: Analytical Solution (Method of Characteristics / Exact)
# Initializes the components to generate the **Exact Analytical Solution** for the Black-Scholes parabolic PDE.
# 
# 1.  **Analytical Core (`SpecialFunctions`, `LinearAlgebra`)**:
#     *   Implements the **Black-Scholes Closed-Form Solution** for a European Call option.
#     *   Computes the exact Price $u(x,t)$ and Greeks ($\Delta, \Gamma, \Theta$) using the standard Normal CDF.
#     *   **Physics**: Purely diffusive with variable coefficient. No shocks, but high curvature near strike at maturity.
# 
# 2.  **Diagnostics (`Plots`, `NPZ`)**:
#     *   **`Plots`**: Visualizes the solution and Greeks in a 4x4 grid layout.
#     *   **`NPZ`**: Exports the high-fidelity "Ground Truth" tensor to `.npz` format.
# 
# 
# %%
using Plots
using SpecialFunctions
using LinearAlgebra
using NPZ
# %% [markdown]


# ## Cell 1: Core Analytical Solver Definition
## Algorithm: Black-Scholes Analytical Solution
# Computes the exact solution and Greeks for a European Call option.
#
# ### Logic Flow:
# 1.  **Time Transformation**: Converts calendar time $t$ to time-to-maturity $\tau = 1 - t$.
# 2.  **Singularity Handling**: Checks for maturity ($\tau \to 0$) to return the Payoff function.
# 3.  **d1/d2 Calculation**: Computes the standard Black-Scholes terms.
# 4.  **Greeks Evaluation**: Evaluates closed-form expressions for Price, Delta, Gamma, and Theta.

"""
    solve_black_scholes_components(x_in, t, sigma, r, K)

Computes the exact Black-Scholes solution and Greeks.
Returns a NamedTuple: (u, delta, gamma, theta)
"""
function solve_black_scholes_components(x_in, t, sigma, r, K)
    x = collect(Float64, x_in)
    tau = 1.0 - t
    
    # Initialize outputs
    u_val = zeros(length(x))
    delta_val = zeros(length(x))
    gamma_val = zeros(length(x))
    theta_val = zeros(length(x))

    # Helper functions for Normal distribution
    # Using SpecialFunctions.erf
    normcdf(z) = 0.5 * (1.0 + erf(z / sqrt(2.0)))
    normpdf(z) = exp(-0.5 * z^2) / sqrt(2.0 * π)

    # Handle Maturity (t=1, tau=0)
    if tau <= 1e-12
        # Terminal Condition: Max(x - K, 0)
        u_val .= max.(x .- K, 0.0)
        # Delta is Heaviside step
        delta_val .= ifelse.(x .> K, 1.0, 0.0)
        # Gamma and Theta are distributional (Dirac delta), set to 0 for plotting stability
        gamma_val .= 0.0
        theta_val .= 0.0
        return (u=u_val, delta=delta_val, gamma=gamma_val, theta=theta_val)
    end

    # Standard Black-Scholes Logic
    for i in eachindex(x)
        xi = x[i]
        if xi <= 1e-9
            # Boundary at x=0
            u_val[i] = 0.0
            delta_val[i] = 0.0
            gamma_val[i] = 0.0
            theta_val[i] = 0.0 
        else
            d1 = (log(xi / K) + (r + 0.5 * sigma^2) * tau) / (sigma * sqrt(tau))
            d2 = d1 - sigma * sqrt(tau)

            Nd1 = normcdf(d1)
            Nd2 = normcdf(d2)
            Npd1 = normpdf(d1)

            # Price: u = x N(d1) - K e^{-r tau} N(d2)
            u_val[i] = xi * Nd1 - K * exp(-r * tau) * Nd2

            # Delta: N(d1)
            delta_val[i] = Nd1

            # Gamma: N'(d1) / (x sigma sqrt(tau))
            gamma_val[i] = Npd1 / (xi * sigma * sqrt(tau))

            # Theta: - (x sigma N'(d1)) / (2 sqrt(tau)) - r K exp(-r tau) N(d2)
            theta_val[i] = -(xi * sigma * Npd1) / (2.0 * sqrt(tau)) - r * K * exp(-r * tau) * Nd2
        end
    end

    return (u=u_val, delta=delta_val, gamma=gamma_val, theta=theta_val)
end
# %% [markdown]


# ## Cell 2: Problem Configuration and Coordinate Mapping
# ## Physical Problem: Black-Scholes European Call
# * **Domain:** $x \in [0, 300]$
# * **Time:** $t \in [0, 1]$ (Backward evolution from Maturity $t=1$)
# * **Parameters:** $\sigma=0.2, r=0.05, K=100$

# %%
# --- 1. Define Physical Grid ---
x_min, x_max = 0.0, 300.0
t_start, t_final = 0.0, 1.0
N_samples = 301  # High resolution for smooth Greeks
N_time_steps = 101 # Sufficient for time evolution visualization

x_phys = range(x_min, x_max, length=N_samples)
t_array = range(t_start, t_final, length=N_time_steps)

# --- 2. Define Financial Parameters ---
sigma_val = 0.2
r_val = 0.05
K_val = 100.0

println("Problem configured: Black-Scholes European Call")
println("Domain: x ∈ [$x_min, $x_max], t ∈ [$t_start, $t_final]")
println("Parameters: σ=$sigma_val, r=$r_val, K=$K_val")
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
usol = zeros(N_samples, N_time_steps)
delta_sol = zeros(N_samples, N_time_steps)
gamma_sol = zeros(N_samples, N_time_steps)
theta_sol = zeros(N_samples, N_time_steps)

println("Starting time evolution loop...")

for (i, t_current) in enumerate(t_array)
    # Compute exact solution components for this time slice
    sol_tuple = solve_black_scholes_components(x_phys, t_current, sigma_val, r_val, K_val)
    usol[:, i] = sol_tuple.u
    delta_sol[:, i] = sol_tuple.delta
    gamma_sol[:, i] = sol_tuple.gamma
    theta_sol[:, i] = sol_tuple.theta
end

println("Evolution complete.")
# %% [markdown]


# ## Cell 4: Visualization Dashboard
#
# ## Dual-Panel Analysis
# 1.  **Space-Time Heatmap (Left):** Visualizes the characteristic convergence and shock formation over the full duration.
# 2.  **Final Time Slice (Right):** Provides a precise view of the shock profile at $t=1.0$ to verify the "Equal Area" cut.
# %%
# ## Visualization Dashboard: 4x4 Grid
# Rows: Price (u), Delta, Gamma, Theta
# Cols: Times t = 1.0, 0.75, 0.5, 0.25

# Define target times for columns
target_times = [1.0, 0.75, 0.5, 0.25]
# Find indices closest to these times
time_indices = [argmin(abs.(t_array .- t)) for t in target_times]

# Prepare plot list
plots_list = []

# Row 1: Price (u)
for (idx, t_val) in zip(time_indices, target_times)
    p = plot(x_phys, usol[:, idx], label="t=$t_val", xlabel="x", ylabel="u(x,t)", 
             title="Price t=$t_val", lw=2, legend=:topright, colorbar=false)
    push!(plots_list, p)
end

# Row 2: Delta
for (idx, t_val) in zip(time_indices, target_times)
    p = plot(x_phys, delta_sol[:, idx], label="t=$t_val", xlabel="x", ylabel="Delta", 
             title="Delta t=$t_val", lw=2, legend=:topright, colorbar=false)
    push!(plots_list, p)
end

# Row 3: Gamma
for (idx, t_val) in zip(time_indices, target_times)
    p = plot(x_phys, gamma_sol[:, idx], label="t=$t_val", xlabel="x", ylabel="Gamma", 
             title="Gamma t=$t_val", lw=2, legend=:topright, colorbar=false)
    push!(plots_list, p)
end

# Row 4: Theta
for (idx, t_val) in zip(time_indices, target_times)
    p = plot(x_phys, theta_sol[:, idx], label="t=$t_val", xlabel="x", ylabel="Theta", 
             title="Theta t=$t_val", lw=2, legend=:topright, colorbar=false)
    push!(plots_list, p)
end

# Combine into 4x4 grid
p_combined = plot(plots_list..., layout=(4, 4), size=(1600, 1200), margin=5Plots.mm)

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
usol_save = Float64.(usol')
delta_save = Float64.(delta_sol')
gamma_save = Float64.(gamma_sol')
theta_save = Float64.(theta_sol')

t_save = collect(Float64.(t_array))
x_save = collect(Float64.(x_phys))

npz_path = joinpath(@__DIR__, "black_scholes.npz")
npzwrite(npz_path, Dict(
    "t" => t_save,
    "x" => x_save,
    "u" => usol_save,
    "delta" => delta_save,
    "gamma" => gamma_save,
    "theta" => theta_save
))

println("Data saved as NPZ to black_scholes.npz")

println("\n--- File Verification ---")
file_size = filesize(npz_path)
println("File size: $(file_size) bytes ($(round(file_size/1024/1024, digits=2)) MB)")

data_check = npzread(npz_path)
println("Keys in file: ", keys(data_check))
println("Shape of u: ", size(data_check["u"]))
println("Shape of delta: ", size(data_check["delta"]))
println("Shape of gamma: ", size(data_check["gamma"]))
println("Shape of theta: ", size(data_check["theta"]))
