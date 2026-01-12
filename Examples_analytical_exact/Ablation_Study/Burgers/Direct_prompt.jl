using FFTW
using Plots
using LinearAlgebra
using Printf
using NPZ

# ==========================================
# 1. Setup Parameters and Grid
# ==========================================
N = 128                  # Spatial resolution
L = 4π                   # Domain length [0, 4π]
dx = L / N
x_phys = range(0, stop=L-dx, length=N)
y_phys = range(0, stop=L-dx, length=N)

# Create 2D Grids
X = x_phys' .* ones(N)
Y = ones(N)' .* y_phys

# Wavenumbers for FFT
k = [0:N/2-1; -N/2:-1] * (2π/L)
Kx = k' .* ones(N)
Ky = ones(N)' .* k

# ==========================================
# 2. Initial Condition
# ==========================================
u0 = 1/3 .+ (2/3) .* sin.((X .+ Y) ./ 2)

# ==========================================
# 3. RHS Function (Inviscid Burgers)
# ==========================================
function compute_rhs(u, Kx, Ky)
    u_sq = u .^ 2
    u_sq_hat = fft(u_sq)
    # Conservative form: -0.5 * (d/dx(u^2) + d/dy(u^2))
    rhs_hat = -0.5 .* (im .* Kx .* u_sq_hat .+ im .* Ky .* u_sq_hat)
    return real(ifft(rhs_hat))
end

# ==========================================
# 4. Time Stepping with Frequent Saving
# ==========================================
function solve_burgers(u0, Kx, Ky, dx, T_final)
    # CFL Calculation
    u_max = maximum(abs.(u0))
    dt_cfl = 0.1 * dx / u_max
    
    # We want to save exactly every 0.01
    dt_save = 0.01
    
    # Use the CFL dt, but ensure it allows hitting save points roughly
    dt = min(dt_cfl, dt_save / 2.0) 
    
    t = 0.0
    u = copy(u0)
    
    # Storage
    snapshots_list = [copy(u)]
    t_list = [0.0]
    
    next_save_t = dt_save
    
    while t < T_final
        # RK4 Integration
        k1 = compute_rhs(u, Kx, Ky)
        k2 = compute_rhs(u + 0.5*dt*k1, Kx, Ky)
        k3 = compute_rhs(u + 0.5*dt*k2, Kx, Ky)
        k4 = compute_rhs(u + dt*k3, Kx, Ky)
        
        u_new = u + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t_new = t + dt
        
        # Check if we crossed a save point
        if t_new >= next_save_t - 1e-9 # Tolerance for float comparison
            push!(snapshots_list, copy(u_new))
            push!(t_list, t_new)
            next_save_t += dt_save
        end
        
        u = u_new
        t = t_new
    end
    
    # Convert list of matrices to 3D Array (X, Y, Time)
    usol_3d = cat(snapshots_list..., dims=3)
    
    return usol_3d, t_list
end

println("Starting simulation...")
usol_snapshots, t_snapshots = solve_burgers(u0, Kx, Ky, dx, 2.0)
println("Simulation complete. Steps saved: $(length(t_snapshots))")

# ==========================================
# 5. Visualization (summary_all.png)
# ==========================================
println("Generating plot...")

# Select 4 evenly spaced indices for plotting
indices = [1, floor(Int, length(t_snapshots)*0.33), floor(Int, length(t_snapshots)*0.66), length(t_snapshots)]
plots_list = []

for idx in indices
    u_curr = usol_snapshots[:, :, idx]
    t_curr = t_snapshots[idx]
    
    p = heatmap(x_phys, y_phys, u_curr', 
                title=@sprintf("t = %.2f", t_curr),
                xlabel="x", ylabel="y",
                aspect_ratio=:equal,
                c=:viridis,
                clims=(minimum(u0), maximum(u0)))
    push!(plots_list, p)
end

final_plot = plot(plots_list..., layout=(2, 2), size=(800, 800), plot_title="2D Inviscid Burgers")

# STRICTLY SAVE IN @__DIR__
png_path = joinpath(@__DIR__, "summary_all.png")
savefig(final_plot, png_path)
println("Plot saved to: $png_path")

# ==========================================
# 6. Save Results to NPZ (User Format)
# ==========================================

# Convert to Float64 and collect
t_save = collect(Float64.(t_snapshots))
x_save = collect(Float64.(x_phys))
y_save = collect(Float64.(y_phys))

# Permute usol to (Time, X, Y) convention for easier Python/ML loading
# usol_snapshots is currently (X, Y, Time) from the solver
usol_save = permutedims(Float64.(usol_snapshots), (3, 1, 2))

# STRICTLY SAVE IN @__DIR__
npz_path = joinpath(@__DIR__, "burgers.npz")
npzwrite(npz_path, Dict(
    "t" => t_save,
    "x" => x_save,
    "y" => y_save,
    "usol" => usol_save
))

println("Data saved as NPZ to: $npz_path")

# ----- Verify the saved file ----------------------------------------
println("\n--- File Verification ---")
file_size = filesize(npz_path)
println("File size: $(file_size) bytes ($(round(file_size/1024/1024, digits=2)) MB)")

data_check = npzread(npz_path)
println("Keys in file: ", keys(data_check))
println("Shape of usol: ", size(data_check["usol"])) # Should be (Time, X, Y)
println("Shape of t: ", size(data_check["t"]))
println("Shape of x: ", size(data_check["x"]))
println("Shape of y: ", size(data_check["y"]))