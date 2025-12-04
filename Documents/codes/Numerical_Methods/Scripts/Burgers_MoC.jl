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
# # Cell 1: Core Analytical Solver Definition
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
    solve_burgers_analytical(x_in, t, alpha, beta)

Analytical solution for 1D Burgers' equation: u_t + u u_x = 0
Initial condition: u(x,0) = alpha + beta * sin(x)

Handles shock formation and Rankine-Hugoniot conditions using the 
Equal Area Rule logic.
"""
function solve_burgers_analytical(x_in, t, alpha, beta)
     # Work on a copy to avoid modifying the input array
     x = collect(Float64, x_in)
     pi_val = π
 
     # 1. Calculate Shock Location
     # The shock forms at x = π (for sine) and moves with speed 'alpha'
     d2 = pi_val + alpha * t
 
     # 2. Handle Periodicity
     # If the shock has moved past 2π, shift x coordinates to track it
     if (pi_val + alpha * t > 2.0 * pi_val)
         shift_count = floor((pi_val + alpha * t) / (2.0 * pi_val))
         shift_val = 2.0 * pi_val * shift_count
         x .+= shift_val
     end
 
     # 3. Split Domain relative to the Shock
     idx1 = x .<= d2
     idx2 = x .> d2
 
     x1 = x[idx1]
     x2 = x[idx2]
 
     # Prepare output array
     vv = similar(x)
 
     # 4. Solve for Left Side (Pre-shock / Expansion)
     if !isempty(x1)
         xn = x1 .- alpha * t
         u_vals = map(xi -> solve_characteristic_root(xi, t, beta), xn)
         v = alpha .+ beta .* sin.(u_vals)
         vv[idx1] = v
     end
 
     # 5. Solve for Right Side (Post-shock / Compression)
     if !isempty(x2)
         x2_flipped = reverse(2.0 * d2 .- x2)
         xn2 = x2_flipped .- alpha * t
         u2_vals = map(xi -> solve_characteristic_root(xi, t, beta), xn2)
         v2 = alpha .+ beta .* sin.(u2_vals)
         v2_final = reverse(2.0 * alpha .- v2)
         vv[idx2] = v2_final
     end
 
     return vv
 end
 
 """
     solve_characteristic_root(x_target, t, beta)
 
 Solves the implicit characteristic equation: 
     x_target - x0 - beta * sin(x0) * t = 0
 """
 function solve_characteristic_root(x_target, t, beta)
     f(x0) = x_target - x0 - beta * t * sin(x0)
     # Order1 (Secant method) is robust and efficient here
     return find_zero(f, 0.0, Order1()) 
 end

# %% [markdown]
# # Cell 2: Problem Configuration and Coordinate Mapping
# 
# ## Physical Problem
# * **Domain:** $x \in [-1, 1]$
# * **Initial Condition:** $u_0(x) = -\sin(\pi x)$
# * **Target Time:** $t = 1.0$ (Post-shock regime)
# 
# ## Canonical Transformation
# The analytical solver assumes a canonical domain $x \in [0, 2\pi]$ and $u \sim \sin(x)$. We map the physical coordinates:
# 
# 1.  **Space:** $x_{can} = \pi(x_{phys} + 1)$
# 2.  **Time:** $t_{can} = \pi t_{phys}$ (Time scaling to preserve wave speed)
# 3.  **Phase/Amplitude:** The negative sign and amplitude are handled by setting parameters $\alpha=0, \beta=1$ and relying on the coordinate shift.

# %%
# --- 1. Define Physical Grid ---
x_min, x_max = -1.0, 1.0
t_start, t_final = 0.0, 1.0
N_samples = 256 
N_time_steps = 200

x_phys = range(x_min, x_max, length=N_samples) 
t_array = range(t_start, t_final, length=N_time_steps)

# --- 2. Map to Canonical Coordinates (Spatial) ---
x_canonical = @. pi * (x_phys + 1)

println("Problem configured.")
println("Grid: $N_samples points x $N_time_steps time steps.")
println("Mapping [-1, 1] -> [0, 2π] for Exact Solution computation.")

# %% [markdown]
# # Cell 3: Time Evolution Loop
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

println("Starting time evolution loop...")

for (i, t_current) in enumerate(t_array)
    # Map current physical time to canonical time
    t_can = pi * t_current
    
    # Compute exact solution vector for this time slice
    # alpha=0.0, beta=1.0 (phase shift handled by x_canonical)
    u_slice = solve_burgers_analytical(x_canonical, t_can, 0.0, 1.0)
    
    # Store in matrix
    usol[:, i] = u_slice
end

println("Evolution complete.")

# %% [markdown]
# # Cell 4: Visualization Dashboard
# 
# ## Dual-Panel Analysis
# 1.  **Space-Time Heatmap (Left):** Visualizes the characteristic convergence and shock formation over the full duration.
# 2.  **Final Time Slice (Right):** Provides a precise view of the shock profile at $t=1.0$ to verify the "Equal Area" cut.

# %%
# ----- 1. Create Heatmap (Left Panel) -------------------------
p_heatmap = heatmap(t_array, x_phys, usol,
    xlabel="Time (t)",
    ylabel="Space (x)",
    title="Exact Evolution (MoC)",
    color=:jet)

# ----- 2. Create Final Slice (Right Panel) --------------------
p_slice = plot(x_phys, usol[:, end], 
     label="t=$(t_final)", 
     xlabel="x", 
     ylabel="u",
     linewidth=2,
     legend=:topright,
     color=:black,
     linestyle=:solid,
     title="Final Solution (Exact)")

# ----- 3. Combine into Dashboard (1x2) ------------------------
p_combined = plot(p_heatmap, p_slice, 
    layout=(1, 2), 
    size=(1000, 400),
    margin=5Plots.mm,
    )

savefig(p_combined, joinpath(@__DIR__, "summary_all.png"))
display(p_combined)

println("Dashboard saved to summary_all.png")

# %% [markdown]
# # Cell 5: Data Persistence and Verification
# 
# ## Artifact Generation
# We save the high-fidelity exact solution to `burgers.npz`. This file is critical as it serves as the **Ground Truth** for:
# 1.  **Advisor Agent:** To calculate L2 error of numerical runs.
# 2.  **PINN Trainer:** To provide supervised loss data.

# %%
# Transpose to (Time, Space) to match Trixi/Python convention usually preferred by PINNs
usol_save = Float64.(usol') 
t_save = collect(Float64.(t_array))
x_save = collect(Float64.(x_phys))

npz_path = joinpath(@__DIR__, "burgers.npz")
npzwrite(npz_path, Dict(
    "t" => t_save,
    "x" => x_save,
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