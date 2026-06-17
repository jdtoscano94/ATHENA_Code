# %% [markdown]

# ## Cell 1: Solver and Discretization Initialization
using LinearAlgebra
using Printf
using NPZ
using Plots

# Minmod slope limiter for MUSCL reconstruction
# Returns the smallest magnitude if signs match, otherwise 0 (to prevent oscillations)
function minmod(a, b)
    if a * b > 0
        return sign(a) * min(abs(a), abs(b))
    else
        return 0.0
    end
end


# ## Cell 2: Physics Model Definition 
# Physical flux function for 2D inviscid Burgers
# f(u) = u^2 / 2, g(u) = u^2 / 2
flux_burgers(u) = 0.5 * u^2

# Rusanov (Local Lax-Friedrichs) numerical flux
# Inputs: uL (left state), uR (right state)
# Returns the numerical flux across the interface
function rusanov_flux(uL, uR)
    # Compute physical fluxes
    fL = flux_burgers(uL)
    fR = flux_burgers(uR)
    
    # Local wave speed estimate: max(|f'(u)|) = max(|u|) for Burgers
    alpha = max(abs(uL), abs(uR))
    
    # Rusanov flux formula
    return 0.5 * (fL + fR) - 0.5 * alpha * (uR - uL)
end


# ## Cell 4: Spatial Discretization
# Spatial discretization using Finite Volume Method
# - Periodic Boundary Conditions via modular indexing
# - MUSCL reconstruction (2nd order) with minmod limiter
# - Rusanov numerical flux

function compute_rhs!(du, u, dx, dy, N)
    # Reset RHS accumulator
    du .= 0.0
    
    # --- X-Direction Fluxes ---
    # Loop over all interfaces in X direction
    # Interface i+1/2 is between cell i and cell i+1
    for j in 1:N
        for i in 1:N
            # Periodic indices
            im1 = mod1(i-1, N)
            ip1 = mod1(i+1, N)
            ip2 = mod1(i+2, N)
            
            # 1. Reconstruction
            # Slope in cell i
            diff_L_i = u[i,j] - u[im1,j]
            diff_R_i = u[ip1,j] - u[i,j]
            slope_i = minmod(diff_R_i, diff_L_i)
            
            # Slope in cell i+1
            diff_L_ip1 = u[ip1,j] - u[i,j]
            diff_R_ip1 = u[ip2,j] - u[ip1,j]
            slope_ip1 = minmod(diff_R_ip1, diff_L_ip1)
            
            # Interface states at i+1/2
            # Left state (from cell i)
            uL = u[i,j] + 0.5 * slope_i
            # Right state (from cell i+1)
            uR = u[ip1,j] - 0.5 * slope_ip1
            
            # 2. Numerical Flux (Rusanov)
            F = rusanov_flux(uL, uR)
            
            # 3. Update Residuals (Flux Differencing)
            # F_i+1/2 leaves cell i, enters cell i+1
            du[i,j]   -= F / dx
            du[ip1,j] += F / dx
        end
    end
    
    # --- Y-Direction Fluxes ---
    # Loop over all interfaces in Y direction
    # Interface j+1/2 is between cell j and cell j+1
    for i in 1:N
        for j in 1:N
            # Periodic indices
            jm1 = mod1(j-1, N)
            jp1 = mod1(j+1, N)
            jp2 = mod1(j+2, N)
            
            # 1. Reconstruction
            # Slope in cell j
            diff_L_j = u[i,j] - u[i,jm1]
            diff_R_j = u[i,jp1] - u[i,j]
            slope_j = minmod(diff_R_j, diff_L_j)
            
            # Slope in cell j+1
            diff_L_jp1 = u[i,jp1] - u[i,j]
            diff_R_jp1 = u[i,jp2] - u[i,jp1]
            slope_jp1 = minmod(diff_R_jp1, diff_L_jp1)
            
            # Interface states at j+1/2
            uL = u[i,j] + 0.5 * slope_j
            uR = u[i,jp1] - 0.5 * slope_jp1
            
            # 2. Numerical Flux
            G = rusanov_flux(uL, uR)
            
            # 3. Update Residuals
            du[i,j]   -= G / dy
            du[i,jp1] += G / dy
        end
    end
end


# ## Cell 5: Computational Domain 
# Grid parameters
const N = 256
const L = 4.0 * pi
const dx = L / N
const dy = L / N

# Cell centers
x = [ (i - 0.5) * dx for i in 1:N ]
y = [ (j - 0.5) * dy for j in 1:N ]

# Initial Condition
# u(x,y,0) = 1/3 + 2/3 * sin((x+y)/2)
u = zeros(Float64, N, N)
for j in 1:N
    for i in 1:N
        u[i,j] = 1.0/3.0 + 2.0/3.0 * sin((x[i] + y[j]) / 2.0)
    end
end

# Pre-allocate arrays for RK3
u_stage1 = zeros(Float64, N, N)
u_stage2 = zeros(Float64, N, N)
du = zeros(Float64, N, N)

println("Grid initialized: N=$N, dx=$dx")
println("Initial min/max: ", minimum(u), " / ", maximum(u))


# ## Cell 7: Additional Strategies
# SSP-RK3 Time Stepping Scheme
# Strong Stability Preserving Runge-Kutta (3rd order, 3 stages)
# u_1 = u_n + dt * L(u_n)
# u_2 = 3/4 u_n + 1/4 u_1 + 1/4 dt * L(u_1)
# u_{n+1} = 1/3 u_n + 2/3 u_2 + 2/3 dt * L(u_2)

function step_ssprk3!(u, dt, dx, dy, N, u_stage1, u_stage2, du)
    # Stage 1
    compute_rhs!(du, u, dx, dy, N)
    @. u_stage1 = u + dt * du
    
    # Stage 2
    compute_rhs!(du, u_stage1, dx, dy, N)
    @. u_stage2 = 0.75 * u + 0.25 * u_stage1 + 0.25 * dt * du
    
    # Stage 3
    compute_rhs!(du, u_stage2, dx, dy, N)
    @. u = (1.0/3.0) * u + (2.0/3.0) * u_stage2 + (2.0/3.0) * dt * du
end


# ## Cell 8: Data I/O 
# Data I/O Initialization

# Target output times
target_times = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
next_target_idx = 1

# Storage for snapshots
# We will store the full 2D field for each snapshot
saved_u = []
saved_times = Float64[]

# Save initial condition (t=0)
push!(saved_u, copy(u))
push!(saved_times, 0.0)
next_target_idx += 1

println("Data I/O initialized.")
println("Target times: ", target_times)


# ## Cell 9: Solving
# Main Time Loop

t = 0.0

t_final = 2.0

CFL = 0.4

step = 0

println("Starting time integration...")

while t < t_final && next_target_idx <= length(target_times)
    # 1. Compute adaptive time step
    max_u = maximum(abs.(u))
    if max_u < 1e-6
        max_u = 1e-6 # Avoid division by zero
    end
    
    # CFL condition: dt <= CFL * min(dx, dy) / max_speed
    dt = CFL * min(dx, dy) / max_u
    
    # 2. Adjust dt to hit next target time
    target_t = target_times[next_target_idx]
    if t + dt >= target_t
        dt = target_t - t
        is_target = true
    else
        is_target = false
    end
    
    # 3. Advance solution
    step_ssprk3!(u, dt, dx, dy, N, u_stage1, u_stage2, du)
    global t += dt
    global step += 1
    
    # 4. Save if target reached
    if is_target
        println("Reached target t = ", round(t, digits=4), " at step ", step)
        push!(saved_u, copy(u))
        push!(saved_times, t)
        global next_target_idx += 1
    end
end

println("Simulation complete. Final time: ", t)


# ## Cell 12: Post-Processing 
# --- Post-Processing and Visualization ---

# 1. Consolidate Data
# saved_u is a Vector of (N,N) matrices. Stack into (N, N, n_snaps)
n_snaps = length(saved_u)
u_stack = zeros(Float64, N, N, n_snaps)
for k in 1:n_snaps
    u_stack[:, :, k] = saved_u[k]
end

# Global color limits for consistent visualization
cmin, cmax = minimum(u_stack), maximum(u_stack)

# 2. Generate Summary Plot (2x3 Grid)
println("Generating summary_all.png...")
plts = []

for k in 1:n_snaps
    t_val = saved_times[k]
    
    # Determine axis labels (only on edges to reduce clutter)
    # Row 1 (k=1,2,3), Row 2 (k=4,5,6)
    show_y = (mod1(k, 3) == 1) # Left column
    show_x = (k > 3)           # Bottom row
    
    p = heatmap(x, y, u_stack[:, :, k],
        aspect_ratio = :equal,
        clims = (cmin, cmax),
        title = @sprintf("t = %.2f", t_val),
        xlabel = show_x ? "x" : "",
        ylabel = show_y ? "y" : "",
        colorbar = false,
        xlims = (0, 4*pi),
        ylims = (0, 4*pi),
        c = :viridis
    )
    push!(plts, p)
end

# Combine subplots
summary_fig = plot(plts..., layout=(2, 3), size=(1000, 600))
savefig(summary_fig, "summary_all.png")
println("Saved summary_all.png")

# 3. Save Data to NPZ
npz_path = joinpath(@__DIR__, "burgers.npz")
npzwrite(npz_path, Dict(
    "t" => saved_times,
    "x" => x,
    "y" => y,
    "u" => u_stack
))
println("Data saved to burgers.npz")

# 4. Verification
println("\n--- File Verification ---")
file_size = filesize(npz_path)
println("File size: $(file_size) bytes ($(round(file_size/1024/1024, digits=2)) MB)")

data_check = npzread(npz_path)
println("Keys in file: ", keys(data_check))
println("Shape of u: ", size(data_check["u"]))
