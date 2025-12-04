
using FFTW
using OrdinaryDiffEq
using SciMLOperators
using LinearAlgebra
using Plots
using MAT

# ----- Discretisation Parameters --------------------------------
nn = 511                   # Number of spatial collocation points
steps = 500                # Number of time steps for output

# ----- Domain and Grids -----------------------------------------
dom = [-1.0, 1.0]          # Spatial domain
L = dom[2] - dom[1]        # Domain length
dx = L / nn                # Spatial step
x = range(dom[1], dom[2]-dx, length=nn) # Periodic grid

# ----- Time Discretization --------------------------------------
t_span = (0.0, 1.0)        # Time interval
t_save = range(0.0, 1.0, length=steps+1) # Time points for output

# ----- Initial Condition ----------------------------------------
u0 = x.^2 .* cos.(π .* x)

# ----- Spectral Differentiation Setup ---------------------------
# 1. Calculate frequencies for the discrete Fourier transform
k_freq = fftfreq(nn, nn/L)

# 2. Convert frequencies to angular wavenumbers (k = 2*pi*f)
k_angular = 2.0 * π .* k_freq

# 3. Define the second derivative operator in Fourier space: (i*k)^2 = -k^2
k2_fourier = -k_angular.^2

# ----- Define the ODE Problem in Fourier Space ------------------
# Transform initial condition to Fourier space
u0_hat = fft(u0)

# The linear part of the PDE is diagonal in Fourier space
linear_coeffs = 5.0 .+ 0.0001 .* k2_fourier
L_diag = Diagonal(linear_coeffs)
L_op = MatrixOperator(L_diag)

# The nonlinear part is computed in physical space and transformed back.
function nonlinear_part_fourier!(du_hat, u_hat, p, t)
    # Inverse transform to physical space
    u = real(ifft(u_hat))
    # Apply nonlinearity: -5*u^3
    N = -5.0 .* u.^3
    # Transform result back to Fourier space
    du_hat .= fft(N)
end

# ----- Setup the ODE Problem for the Solver ---------------------
# A SplitODEProblem is used to separate the stiff linear part
# from the nonlinear part, which is ideal for ETDRK methods.
prob = SplitODEProblem(L_op, nonlinear_part_fourier!, u0_hat, t_span)

# ----- Solve the System -----------------------------------------
# The ETDRK4 integrator is chosen, which requires a fixed time step.
dt = 1e-5
sol = solve(prob, ETDRK4(), dt=dt, saveat=t_save)

# ----- Process the Solution -------------------------------------
# The solution is returned in Fourier space; transform back to physical space.
usol_fourier = hcat([sol.u[i] for i in 1:length(sol.t)]...) # nn x (steps+1)
usol = zeros(nn, steps+1)
for i in 1:steps+1
    usol[:, i] = real(ifft(usol_fourier[:, i]))
end

# ----- Prepare Data for Plotting and Saving ---------------------
# Create a full grid including the endpoint for visualization
x_plot = range(dom[1], dom[2], length=nn+1)
# Enforce periodicity in the solution data by appending the first row
usol_plot = vcat(usol, usol[1:1, :])

# ----- Visualize the Result -------------------------------------
p = heatmap(t_save, x_plot, usol_plot,
    xlabel="Time (t)",
    ylabel="Space (x)",
    title="Solution of the Allen-Cahn Equation",
    color=:jet,
    aspect_ratio=:auto
)
savefig(p, joinpath(@__DIR__, "allen_cahn_solution.png"))



# ----- Save Data to File ----------------------------------------
# Prepare data as regular Arrays (not ranges or views)
usol_save = Float64.(real.(usol_plot'))
t_array = collect(Float64.(t_save))
x_array = collect(Float64.(x_plot))

# CRITICAL: Use matopen/matclose with NO compress parameter
# This forces MAT v0.10.7 to use v5 format
file_path = joinpath(@__DIR__, "allen_cahn_v5.mat")
file = matopen(file_path, "w")
write(file, "t", t_array)
write(file, "x", x_array)
write(file, "usol", usol_save)
close(file)

println("Data saved at: $file_path")

# ----- Verify the saved file ----------------------------------------
println("\n--- File Verification ---")
file_size = filesize(file_path)
println("File size: $(file_size) bytes ($(round(file_size/1024/1024, digits=2)) MB)")

data_check = matread(file_path)
println("Keys in file: ", keys(data_check))
println("Shape of usol: ", size(data_check["usol"]))
println("Shape of t: ", size(data_check["t"]))
println("Shape of x: ", size(data_check["x"]))