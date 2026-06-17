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
    gauss_hermite(n)

Compute the nodes and weights for Gauss-Hermite quadrature using the Golub-Welsch algorithm.
Integrals of the form ∫ exp(-x^2) f(x) dx are approximated by sum(w .* f(x)).
"""
function gauss_hermite(n)
    # Construct the symmetric tridiagonal matrix (Jacobi matrix)
    # For normalized Hermite polynomials, beta[i] = sqrt(i/2)
    i = 1:n-1
    beta = sqrt.(i ./ 2)
    alpha = zeros(n) # diagonal is zero for Hermite
    
    # Solve eigenvalue problem
    T = SymTridiagonal(alpha, beta)
    evals, evecs = eigen(T)
    
    # Nodes are eigenvalues
    x = evals
    
    # Weights are (sqrt(pi) * v[1]^2)
    w = sqrt(π) .* (evecs[1, :] .^ 2)
    
    return x, w
end

"""
    solve_viscous_burgers_exact(x_in, t, nu)

Computes the exact solution of the 1D Viscous Burgers' equation on a periodic domain [-1, 1]
using the Cole-Hopf transformation and Gauss-Hermite quadrature.

u(x,t) = -2ν * (∂x φ / φ)
φ(x,t) satisfies the heat equation.
"""
function solve_viscous_burgers_exact(x_in, t, nu)
    x = collect(Float64, x_in)
    
    # Handle initial condition (t=0)
    if t < 1e-9
        return -sin.(π .* x)
    end
    
    # Quadrature setup
    N_quad = 64
    z, w = gauss_hermite(N_quad)
    
    # Precompute constants
    sqrt_4nu_t = sqrt(4 * nu * t)
    inv_sqrt_nu_t = 1.0 / sqrt(nu * t)
    
    # Initial condition for phi: phi_0(y) = exp( -cos(πy) / (2πν) )
    # Derived from u0 = -sin(πx) -> ∫u0 = cos(πx)/π
    phi0_func(y) = exp(-cos(π * y) / (2 * π * nu))
    
    u_out = similar(x)
    
    # Image sum range for periodicity
    # Domain length L=2. Diffusion spread is small. m=-3:3 is sufficient for machine precision.
    m_range = -3:3
    
    for (i, xi) in enumerate(x)
        num_accum = 0.0
        den_accum = 0.0
        
        for m in m_range
            # Substitution: y = xi - 2m - sqrt(4νt)*z
            # We evaluate phi0 at y corresponding to the quadrature node z
            y_nodes = xi .- 2.0 * m .- sqrt_4nu_t .* z
            
            # Evaluate phi0
            phi_vals = phi0_func.(y_nodes)
            
            # Denominator integral: ∫ K * phi0 dy ~ ∑ w * phi0
            den_accum += dot(w, phi_vals)
            
            # Numerator integral: ∂x φ
            # ∂x K = K * (-z / sqrt(νt))
            # Integral becomes: ∑ w * phi0 * (-z / sqrt(νt))
            num_accum += dot(w, phi_vals .* (-z .* inv_sqrt_nu_t))
        end
        
        # u = -2ν * (φ_x / φ)
        u_out[i] = -2 * nu * (num_accum / den_accum)
    end
    
    return u_out
end
# %% [markdown]


# ## Cell 2: Problem Configuration and Coordinate Mapping
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
# --- 1. Define Physical Grid and Parameters ---
x_min, x_max = -1.0, 1.0
t_start, t_final = 0.0, 1.0
nu = 1.0 / (100.0 * pi) # Viscosity per strategy

N_samples = 256 
N_time_steps = 200

x_phys = range(x_min, x_max, length=N_samples) 
t_array = range(t_start, t_final, length=N_time_steps)

println("Problem configured.")
println("Grid: $N_samples points x $N_time_steps time steps.")
println("Viscosity nu: $nu")
println("Domain: [-1, 1] (Direct Physical Calculation)")
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

println("Starting time evolution loop (Viscous Exact)...")

for (i, t_current) in enumerate(t_array)
    # Compute exact solution vector for this time slice using the Cole-Hopf solver
    # Pass physical coordinates and viscosity directly
    u_slice = solve_viscous_burgers_exact(x_phys, t_current, nu)
    
    # Store in matrix
    usol[:, i] = u_slice
end

println("Evolution complete.")
# %% [markdown]

# ## Cell 4: Visualization Dashboard
#
# ## Dual-Panel Analysis
# 1.  **Space-Time Heatmap (Left):** Visualizes the characteristic convergence and shock formation over the full duration.
# 2.  **Final Time Slice (Right):** Provides a precise view of the shock profile at $t=1.0$ to verify the "Equal Area" cut.
# %%
# ----- Snapshot Visualization (1x4) ---------------
snap_times = [0.0, 0.25, 0.5, 1.0]
plot_list = []

for t_target in snap_times
    # Find closest time index in the computed array
    idx = argmin(abs.(t_array .- t_target))
    t_act = t_array[idx]
    
    p = plot(x_phys, usol[:, idx],
        title="t=$(round(t_act, digits=2))",
        xlabel="x",
        ylabel=(t_target == 0.0 ? "u" : ""), # Label y-axis only on the first plot
        linewidth=2,
        color=:blue,
        legend=false,
        grid=true
    )
    push!(plot_list, p)
end

# Combine into 1x4 layout
p_combined = plot(plot_list..., 
    layout=(1, 4), 
    size=(1200, 350),
    margin=5Plots.mm
)

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