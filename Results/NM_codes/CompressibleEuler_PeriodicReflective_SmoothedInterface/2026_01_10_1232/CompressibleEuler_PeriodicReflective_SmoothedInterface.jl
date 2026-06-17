# %% [markdown]

# ## Cell 1: Solver and Discretization Initialization
using LinearAlgebra
using StaticArrays
using Printf
using NPZ
using Plots

# --------------------------------------------------------------
# Data Structures
# --------------------------------------------------------------

struct Parameters{T}
    gamma::T       # Adiabatic index
    g::T           # Gravitational acceleration magnitude
    cfl::T         # CFL number
    t_end::T       # Final simulation time
end

struct Mesh{T}
    nx::Int
    ny::Int
    x_min::T
    x_max::T
    y_min::T
    y_max::T
    dx::T
    dy::T
    x::Vector{T}   # Cell centers X
    y::Vector{T}   # Cell centers Y
end

# Helper to create mesh
function create_mesh(nx, ny, x_min, x_max, y_min, y_max)
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    x = [x_min + (i - 0.5) * dx for i in 1:nx]
    y = [y_min + (j - 0.5) * dy for j in 1:ny]
    return Mesh(nx, ny, x_min, x_max, y_min, y_max, dx, dy, x, y)
end
# %% [markdown]

# ## Cell 2: Physics Model Definition 
# --------------------------------------------------------------
# Physics: Compressible Euler Equations
# --------------------------------------------------------------

# Convert Primitive variables (ρ, u, v, p) to Conservative variables (ρ, ρu, ρv, E)
function prim2cons(rho, u, v, p, gamma)
    e_kin = 0.5 * rho * (u^2 + v^2)
    e_int = p / (gamma - 1.0)
    E = e_int + e_kin
    return SVector(rho, rho * u, rho * v, E)
end

# Convert Conservative variables to Primitive variables
function cons2prim(u_cons, gamma)
    rho = u_cons[1]
    rhou = u_cons[2]
    rhov = u_cons[3]
    E = u_cons[4]

    u = rhou / rho
    v = rhov / rho
    e_kin = 0.5 * rho * (u^2 + v^2)
    e_int = E - e_kin
    p = (gamma - 1.0) * e_int

    return SVector(rho, u, v, p)
end

function sound_speed(rho, p, gamma)
    return sqrt(gamma * p / rho)
end

# Physical Flux in X direction
function flux_x(u_cons, gamma)
    rho, u, v, p = cons2prim(u_cons, gamma)
    E = u_cons[4]
    return SVector(rho * u, rho * u^2 + p, rho * u * v, (E + p) * u)
end

# Physical Flux in Y direction
function flux_y(u_cons, gamma)
    rho, u, v, p = cons2prim(u_cons, gamma)
    E = u_cons[4]
    return SVector(rho * v, rho * u * v, rho * v^2 + p, (E + p) * v)
end

# Gravitational Source Term
# Strategy: Gravity acts in positive y-direction.
# Momentum Y: + rho * g
# Energy: + rho * v * g
function source_term(u_cons, params)
    rho = u_cons[1]
    rhov = u_cons[3]
    g = params.g
    
    # SVector(Mass, MomX, MomY, Energy)
    return SVector(0.0, 0.0, rho * g, rhov * g)
end

# ## Cell 4: Spatial Discretization
# --------------------------------------------------------------
# Spatial Discretization: HLLC + MUSCL
# --------------------------------------------------------------
function minmod(a, b)
    if a * b > 0
        return sign(a) * min(abs(a), abs(b))
    else
        return 0.0
    end
end
# Helper to handle BCs (Periodic X, Reflective Y)
function get_state(u, i, j, nx, ny)
    # Periodic X
    i_eff = mod1(i, nx)
    # Reflective Y
    if j < 1
        # Bottom Ghost (Reflection)
        j_eff = 1 - j
        val = u[i_eff, j_eff]
        return SVector(val[1], val[2], -val[3], val[4]) # Invert Y-momentum
    elseif j > ny
        # Top Ghost (Reflection)
        j_eff = 2 * ny - j + 1
        val = u[i_eff, j_eff]
        return SVector(val[1], val[2], -val[3], val[4]) # Invert Y-momentum
    else
        return u[i_eff, j]
    end
end
# MUSCL Reconstruction at interface between i and i+1
# Uses stencil: i-1, i, i+1, i+2
function reconstruct_states(u, i, j, nx, ny, dim, params)
    # Dim 1 = X-interface (between i and i+1), Dim 2 = Y-interface (between j and j+1)
    if dim == 1
        # Reconstruct along X
        u_im1 = get_state(u, i-1, j, nx, ny)
        u_i   = get_state(u, i,   j, nx, ny)
        u_ip1 = get_state(u, i+1, j, nx, ny)
        u_ip2 = get_state(u, i+2, j, nx, ny)
    else
        # Reconstruct along Y
        u_im1 = get_state(u, i, j-1, nx, ny)
        u_i   = get_state(u, i, j,   nx, ny)
        u_ip1 = get_state(u, i, j+1, nx, ny)
        u_ip2 = get_state(u, i, j+2, nx, ny)
    end
    # Slope limiting (component-wise)
    slope_L = (u_i - u_im1)
    slope_R = (u_ip1 - u_i)
    slope_R2 = (u_ip2 - u_ip1)
    # Left state at interface (from i)
    uL = u_i + 0.5 * SVector(
        minmod(slope_L[1], slope_R[1]),
        minmod(slope_L[2], slope_R[2]),
        minmod(slope_L[3], slope_R[3]),
        minmod(slope_L[4], slope_R[4])
    )
    # Right state at interface (from i+1)
    uR = u_ip1 - 0.5 * SVector(
        minmod(slope_R[1], slope_R2[1]),
        minmod(slope_R[2], slope_R2[2]),
        minmod(slope_R[3], slope_R2[3]),
        minmod(slope_R[4], slope_R2[4])
    )
    return uL, uR
end
function hllc_flux(uL, uR, gamma, dim)
    # Rotate if Y-direction (dim=2)
    # Swap u and v
    if dim == 2
        uL = SVector(uL[1], uL[3], uL[2], uL[4])
        uR = SVector(uR[1], uR[3], uR[2], uR[4])
    end
    rhoL, uL_vel, vL, pL = cons2prim(uL, gamma)
    rhoR, uR_vel, vR, pR = cons2prim(uR, gamma)
    cL = sound_speed(rhoL, pL, gamma)
    cR = sound_speed(rhoR, pR, gamma)
    # Wave speeds
    SL = min(uL_vel - cL, uR_vel - cR)
    SR = max(uL_vel + cL, uR_vel + cR)
    if 0.0 <= SL
        F = flux_x(uL, gamma)
    elseif 0.0 >= SR
        F = flux_x(uR, gamma)
    else
        # HLLC Star State
        S_star = (pR - pL + rhoL*uL_vel*(SL - uL_vel) - rhoR*uR_vel*(SR - uR_vel)) /
                 (rhoL*(SL - uL_vel) - rhoR*(SR - uR_vel))
        # Star States U_*L
        factorL = rhoL * (SL - uL_vel) / (SL - S_star)
        U_star_L = factorL * SVector(1.0, S_star, vL, (uL[4]/rhoL) + (S_star - uL_vel)*(S_star + pL/(rhoL*(SL - uL_vel))))
        # Star States U_*R
        factorR = rhoR * (SR - uR_vel) / (SR - S_star)
        U_star_R = factorR * SVector(1.0, S_star, vR, (uR[4]/rhoR) + (S_star - uR_vel)*(S_star + pR/(rhoR*(SR - uR_vel))))
        if 0.0 <= S_star
            F = flux_x(uL, gamma) + SL * (U_star_L - uL)
        else
            F = flux_x(uR, gamma) + SR * (U_star_R - uR)
        end
    end
    # Rotate back if Y-direction
    if dim == 2
        F = SVector(F[1], F[3], F[2], F[4])
    end
    return F
end
function compute_rhs!(du, u, mesh, params)
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy
    gamma = params.gamma
    # Reset du
    fill!(du, zero(SVector{4, Float64}))
    # X-Fluxes
    for j in 1:ny
        for i in 1:nx
            uL, uR = reconstruct_states(u, i, j, nx, ny, 1, params)
            F = hllc_flux(uL, uR, gamma, 1)
            du[i, j] -= F / dx
            ip1 = mod1(i+1, nx)
            du[ip1, j] += F / dx
        end
    end
    # Y-Fluxes
    for i in 1:nx
        for j in 0:ny
            uL, uR = reconstruct_states(u, i, j, nx, ny, 2, params)
            F = hllc_flux(uL, uR, gamma, 2)
            if j >= 1
                du[i, j] -= F / dy
            end
            if j < ny
                du[i, j+1] += F / dy
            end
        end
    end
    # Source Terms
    for j in 1:ny
        for i in 1:nx
            S = source_term(u[i, j], params)
            du[i, j] += S
        end
    end
end


# ## Cell 5: Computational Domain 
# --------------------------------------------------------------
# Initialization: Rayleigh-Taylor Instability
# --------------------------------------------------------------

function initialize_rti(mesh, params)
    nx, ny = mesh.nx, mesh.ny
    u = zeros(SVector{4, Float64}, nx, ny)
    
    gamma = params.gamma
    g = params.g
    
    # Base pressure at y=0 (Bottom)
    # Must be high enough so P > 0 everywhere
    P_0 = 2.5
    
    # Pre-calculate density and pressure profiles (Hydrostatic)
    # dp/dy = rho * g
    rho_col = zeros(ny)
    p_col = zeros(ny)
    
    # 1. Density Profile
    for j in 1:ny
        y = mesh.y[j]
        # Heavy (2.0) below 0.5, Light (1.0) above 0.5
        # Smooth transition via tanh
        rho_col[j] = 1.5 - 0.5 * tanh((y - 0.5) / 0.01)
    end
    
    # 2. Pressure Profile (Numerical Integration)
    # P_face[1] = P_0
    # P_face[j+1] = P_face[j] + rho[j] * g * dy
    p_face = P_0
    for j in 1:ny
        # Pressure at cell center (approx)
        p_col[j] = p_face + 0.5 * rho_col[j] * g * mesh.dy
        # Update face pressure for next cell
        p_face += rho_col[j] * g * mesh.dy
    end
    
    # 3. Fill Grid
    Lx = mesh.x_max - mesh.x_min
    
    for j in 1:ny
        for i in 1:nx
            rho = rho_col[j]
            p = p_col[j]
            x = mesh.x[i]
            y = mesh.y[j]
            
            c = sound_speed(rho, p, gamma)
            
            # Velocity Perturbation
            # Trigger instability with single mode
            pert = 0.01 * c * cos(2 * pi * x / Lx) * exp(-100 * (y - 0.5)^2)
            
            u_vel = 0.0
            v_vel = pert
            
            u[i, j] = prim2cons(rho, u_vel, v_vel, p, gamma)
        end
    end
    
    return u
end

# ## Cell 7: Additional Strategies
# --------------------------------------------------------------
# Time Integration: SSP-RK3
# --------------------------------------------------------------

function compute_dt(u, mesh, params)
    nx, ny = mesh.nx, mesh.ny
    gamma = params.gamma
    cfl = params.cfl
    
    max_wave_speed = 0.0
    
    for j in 1:ny
        for i in 1:nx
            rho, u_vel, v_vel, p = cons2prim(u[i, j], gamma)
            c = sound_speed(rho, p, gamma)
            
            ws = max(abs(u_vel), abs(v_vel)) + c
            if ws > max_wave_speed
                max_wave_speed = ws
            end
        end
    end
    
    dt = cfl * min(mesh.dx, mesh.dy) / max_wave_speed
    return dt
end

function time_step!(u, mesh, params, dt)
    nx, ny = mesh.nx, mesh.ny
    
    # Allocate temporary arrays
    u1 = copy(u)
    u2 = copy(u)
    du = zeros(SVector{4, Float64}, nx, ny)
    
    # Stage 1
    compute_rhs!(du, u, mesh, params)
    @. u1 = u + dt * du
    
    # Stage 2
    compute_rhs!(du, u1, mesh, params)
    @. u2 = 0.75 * u + 0.25 * (u1 + dt * du)
    
    # Stage 3
    compute_rhs!(du, u2, mesh, params)
    @. u = (1.0/3.0) * u + (2.0/3.0) * (u2 + dt * du)
end

# ## Cell 8: Data I/O 
# --------------------------------------------------------------
# I/O: Save Data
# --------------------------------------------------------------

output_dir = joinpath(@__DIR__, "rti_output")
if !isdir(output_dir)
    mkdir(output_dir)
end

function save_solution(u, mesh, params, t, step)
    nx, ny = mesh.nx, mesh.ny
    gamma = params.gamma
    
    rho_arr = zeros(nx, ny)
    u_arr = zeros(nx, ny)
    v_arr = zeros(nx, ny)
    p_arr = zeros(nx, ny)
    
    for j in 1:ny
        for i in 1:nx
            rho, u_vel, v_vel, p = cons2prim(u[i, j], gamma)
            rho_arr[i, j] = rho
            u_arr[i, j] = u_vel
            v_arr[i, j] = v_vel
            p_arr[i, j] = p
        end
    end
    
    filename = joinpath(output_dir, @sprintf("step_%04d.npz", step))
    npzwrite(filename, Dict(
        "t" => t,
        "x" => mesh.x,
        "y" => mesh.y,
        "rho" => rho_arr,
        "u" => u_arr,
        "v" => v_arr,
        "p" => p_arr
    ))
    println("Saved: ", filename)
end

# ## Cell 9: Solving
# ------------------------------------------------------------------------------
# Main Simulation Loop
# ------------------------------------------------------------------------------

function solve()
    # 1. Setup
    nx = 64
    ny = 256
    mesh = create_mesh(nx, ny, 0.0, 0.25, 0.0, 1.0)
    
    params = Parameters(
        1.4,   # gamma
        1.0,   # g (gravity) - Normalized? Strategy says g acts in +y. 
               # Usually g is negative if y is up, but problem says "gravity acts in positive y-direction".
               # Wait, "heavy fluid below light fluid... unstable equilibrium where heavy fluid seeks to displace light fluid in direction of gravitational field."
               # If heavy is below (y < 0.5) and light is above (y > 0.5), for this to be unstable (RTI), gravity must point UP (+y).
               # If gravity pointed down (-y), heavy below light is STABLE.
               # So g = +1.0 is correct for this specific setup.
        0.5,   # CFL
        3.0    # t_end
    )
    
    # 2. Initialization
    u = initialize_rti(mesh, params)
    
    t = 0.0
    step = 0
    
    # Save initial state
    save_solution(u, mesh, params, t, step)
    
    # 3. Time Loop
    while t < params.t_end
        dt = compute_dt(u, mesh, params)
        
        if t + dt > params.t_end
            dt = params.t_end - t
        end
        
        time_step!(u, mesh, params, dt)
        
        t += dt
        step += 1
        
        if step % 100 == 0
            @printf("Step: %d, Time: %.4f, dt: %.6f\n", step, t, dt)
        end
        
        # Save roughly every 0.1 time units (approx 30 frames total)
        # Or just save frequently enough to catch 0, 1, 2, 3
        if step % 50 == 0 || abs(t - params.t_end) < 1e-10
             save_solution(u, mesh, params, t, step)
        end
    end
    
    println("Simulation Complete.")
end

# Run
solve()

# ## Cell 12: Post-Processing 
# --------------------------------------------------------------
# Post-Processing: Visualization
# --------------------------------------------------------------

println("Generating summary visualization...")

output_dir = joinpath(@__DIR__, "rti_output")
if isdir(output_dir)
    files = readdir(output_dir, join=true)
    filter!(x -> endswith(x, ".npz"), files)
    sort!(files) # Assumes step_XXXX format sorts correctly

    # Target times
    target_times = [0.0, 1.0, 2.0, 3.0]
    selected_files = String[]

    # Find closest files
    for target in target_times
        best_file = ""
        min_diff = Inf
        
        for f in files
            data = npzread(f)
            t_val = data["t"]
            diff = abs(t_val - target)
            
            if diff < min_diff
                min_diff = diff
                best_file = f
            end
        end
        if best_file != ""
            push!(selected_files, best_file)
        end
    end

    plots_array = []

    for f in selected_files
        data = npzread(f)
        t_val = data["t"]
        
        x = data["x"]
        y = data["y"]
        rho = data["rho"]
        
        # Transpose rho for heatmap (Plots.jl expects z[row, col] ~ (y, x))
        # rho is (nx, ny). We want x on horizontal, y on vertical.
        # So we pass x, y, and rho'.
        
        p = heatmap(x, y, rho', 
            c = :viridis,
            aspect_ratio = :equal,
            xlims = (0, 0.25),
            ylims = (0, 1.0),
            title = @sprintf("t = %.2f", t_val),
            xlabel = "", ylabel = "",
            colorbar = false, # CRITICAL CONSTRAINT
            framestyle = :box,
            grid = false,
            margin = 2Plots.mm
        )
        push!(plots_array, p)
    end

    # Layout: 1 row, 4 columns
    if !isempty(plots_array)
        final_plot = plot(plots_array..., layout=(1, 4), size=(1200, 800), margin=5Plots.mm)
        out_path = joinpath(@__DIR__, "summary_all.png")
        savefig(final_plot, out_path)
        println("Saved summary to: ", out_path)
    else
        println("No data files found to plot.")
    end
else
    println("Output directory not found.")
end