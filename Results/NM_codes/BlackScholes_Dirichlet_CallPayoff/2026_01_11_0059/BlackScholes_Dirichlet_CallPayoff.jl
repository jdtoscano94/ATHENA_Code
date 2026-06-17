# %% [markdown]
# # Preamble: Solver and Analytical Framework Initialization
# 
# ## Framework: Exact Analytical Solution for Black-Scholes Equation
# Initializes the components to generate the **Exact Solution** for the Black-Scholes parabolic PDE governing European Call Options.
# 
# 1.  **Analytical Core (`SpecialFunctions`)**:
#     *   Implements the closed-form **Black-Scholes formula**: $u(x,t) = x\Phi(d_1) - Ke^{-r(1-t)}\Phi(d_2)$.
#     *   Uses `erf` (Error Function) to compute the Cumulative Distribution Function $\Phi$.
# 
# 2.  **Diagnostics (`Plots`, `NPZ`)**:
#     *   **`Plots`**: Visualizes the option price evolution $u(x,t)$.
#     *   **`NPZ`**: Exports the high-fidelity "Ground Truth" tensor.
# 
# 
# %%
using Plots
using Roots
using LinearAlgebra
using NPZ
using SpecialFunctions
# %% [markdown]

# ## Cell 1: Core Analytical Solver Definition
#
# ## Algorithm: Black-Scholes Closed-Form Solution
# The function `solve_black_scholes_analytical` computes the exact price of a European Call option.
#
# ### Logic Flow:
# 1.  **Time Transformation**: Calculates time to maturity $\tau = 1 - t$.
# 2.  **Singularity Handling**:
#     *   At maturity ($\tau = 0$), returns the payoff $\max(x - K, 0)$.
#     *   At $x=0$, returns $0$ (boundary condition).
# 3.  **Formula Evaluation**: Computes $d_1$ and $d_2$ and evaluates the standard pricing formula using the Normal CDF $\Phi$.
# %%
"""
    solve_black_scholes_analytical(x_in, t, sigma, r, K)
Analytical solution for Black-Scholes European Call Option.
PDE: u_t + 0.5*sigma^2*x^2*u_xx + r*x*u_x - r*u = 0 (Backward)
Domain: x in [0, 300], t in [0, 1]
Maturity at t = 1.
Parameters:
- x_in: Asset price vector
- t: Current time (0.0 to 1.0)
- sigma: Volatility
- r: Risk-free rate
- K: Strike price
"""
function solve_black_scholes_analytical(x_in, t, sigma, r, K)
    # Work on a copy
    x = collect(Float64, x_in)
    # Time to maturity
    tau = 1.0 - t
    # 1. Handle Maturity (t=1, tau=0)
    # Avoid division by zero in d1/d2
    if tau <= 1e-14
        return max.(x .- K, 0.0)
    end
    # 2. Define Normal CDF using erf
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    Phi(z) = 0.5 * (1.0 + erf(z / sqrt(2.0)))
    # 3. Evaluate Formula
    # u = x * Phi(d1) - K * exp(-r * tau) * Phi(d2)
    u = zeros(length(x))
    # Handle x > 0 (log(x) is undefined for x=0)
    # For x=0, u=0, which is already set by initialization
    mask = x .> 1e-14
    if any(mask)
        x_valid = x[mask]
        # Compute d1 and d2
        # d1 = (ln(x/K) + (r + 0.5*sigma^2)*tau) / (sigma*sqrt(tau))
        denom = sigma * sqrt(tau)
        d1 = (log.(x_valid ./ K) .+ (r + 0.5 * sigma^2) * tau) ./ denom
        d2 = d1 .- denom
        # Compute price
        u[mask] = x_valid .* Phi.(d1) .- K * exp(-r * tau) .* Phi.(d2)
    end
    return u
end
# (Leftover Burgers' logic removed)
# %% [markdown]


# ## Cell 2: Problem Configuration

# ## Physical Problem: European Call Option
# * **Domain:** $x \in [0, 300]$
# * **Time:** $t \in [0, 1]$ (Maturity at $t=1$)
# * **Parameters:** $\sigma=0.2, r=0.05, K=100$

# ## Grid Configuration
# We define a spatial grid to evaluate the option price and a time array for evolution.

# %%
# --- 1. Define Physical Grid ---
x_min, x_max = 0.0, 300.0
t_start, t_final = 0.0, 1.0
N_samples = 301  # High resolution for smooth plotting
N_time_steps = 101 # Sufficient for smooth time evolution

x_phys = range(x_min, x_max, length=N_samples) 
t_array = range(t_start, t_final, length=N_time_steps)

# --- 2. Define Problem Parameters ---
sigma_val = 0.2
r_val = 0.05
K_val = 100.0

println("Problem configured: Black-Scholes European Call")
println("Domain: x in [$x_min, $x_max], t in [$t_start, $t_final]")
println("Parameters: sigma=$sigma_val, r=$r_val, K=$K_val")

# ## Cell 3: Time Evolution Loop
# ## Batch Processing
# We evaluate the exact Black-Scholes formula at each time step in the `t_array`.
# This generates the full solution surface $u(x,t)$.
# %%
# Initialize solution matrix (Space x Time)
usol = zeros(N_samples, N_time_steps)
println("Starting time evolution loop...")
for (i, t_current) in enumerate(t_array)
    # Compute exact solution vector for this time slice
    # Parameters: sigma=0.2, r=0.05, K=100
    u_slice = solve_black_scholes_analytical(x_phys, t_current, sigma_val, r_val, K_val)
    
    # Store in matrix
    usol[:, i] = u_slice
end
println("Evolution complete.")


# ## Cell 4: Visualization Dashboard
#
# ## Dual-Panel Analysis
# 1.  **Space-Time Heatmap (Left):** Visualizes the characteristic convergence and shock formation over the full duration.
# 2.  **Final Time Slice (Right):** Provides a precise view of the shock profile at $t=1.0$ to verify the "Equal Area" cut.
# %%
# %%
# Define requested snapshot times
snapshot_times = [1.0, 0.75, 0.5, 0.25, 0.0]
plots_list = []

for t_snap in snapshot_times
    # Find closest index in t_array
    # We use argmin to find the index of the time closest to t_snap
    idx = argmin(abs.(t_array .- t_snap))
    actual_t = t_array[idx]
    
    # Extract solution
    u_snap = usol[:, idx]
    
    # Create plot
    p = plot(x_phys, u_snap,
        title = "t = $(round(actual_t, digits=2))",
        xlabel = "Asset Price (x)",
        ylabel = "Option Price (u)",
        legend = false,
        linewidth = 2,
        color = :blue,
        grid = true
    )
    push!(plots_list, p)
end

# Combine into 1 row x 5 columns
p_combined = plot(plots_list..., 
    layout = (1, 5),
    size = (1500, 300),
    margin = 5Plots.mm
)

savefig(p_combined, joinpath(@__DIR__, "summary_all.png"))
display(p_combined)

println("Dashboard saved to summary_all.png")
# %% [markdown]

# ## Cell 5: Data Persistence and Verification
#
# ## Artifact Generation
# We save the high-fidelity exact solution to `black_scholes.npz`. This file is critical as it serves as the **Ground Truth** for:
# 1.  **Advisor Agent:** To calculate L2 error of numerical runs.
# 2.  **PINN Trainer:** To provide supervised loss data.
# %%
# Transpose to (Time, Space) to match Trixi/Python convention usually preferred by PINNs
usol_save = Float64.(usol')
t_save = collect(Float64.(t_array))
x_save = collect(Float64.(x_phys))
npz_path = joinpath(@__DIR__, "black_scholes.npz")
npzwrite(npz_path, Dict(
    "t" => t_save,
    "x" => x_save,
    "usol" => usol_save
))
println("Data saved as NPZ to black_scholes.npz")
# ----- Verify the saved file ----------------------------------------
println("\n--- File Verification ---")
file_size = filesize(npz_path)
println("File size: $(file_size) bytes ($(round(file_size/1024/1024, digits=2)) MB)")
data_check = npzread(npz_path)
println("Keys in file: ", keys(data_check))
println("Shape of usol: ", size(data_check["usol"]))
println("Shape of t: ", size(data_check["t"]))
println("Shape of x: ", size(data_check["x"]))