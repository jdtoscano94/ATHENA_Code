using FFTW
using OrdinaryDiffEq
using SciMLOperators
using LinearAlgebra
using Plots
using MAT
# ----- Discretisation Parameters --------------------------------
nn = 511                   # Number of spatial collocation points (an odd number is good for FFT)
steps = 500                # Number of time steps for output

# ----- Domain and Grids -----------------------------------------
dom = [0.0, 1.0]           # Spatial domain [x_min, x_max]
L = dom[2] - dom[1]        # Domain length
dx = L / nn                # Spatial step size
x = range(dom[1], dom[2]-dx, length=nn) # Periodic grid (endpoint is excluded)

# ----- Time Discretization --------------------------------------
t_span = (0.0, 1.0)        # Time interval [t_start, t_end]
t_save = range(0.0, 1.0, length=steps+1) # Time points where the solution is saved

# ----- Initial Condition ----------------------------------------
u0 = sin.(2*π .* x)

# ----- Spectral Differentiation Setup ---------------------------
# 1. Calculate frequencies `f` for the discrete Fourier transform.
k_freq = fftfreq(nn, nn/L)

# 2. Convert frequencies `f` to angular wavenumbers `k = 2*pi*f`.
k_angular = 2.0 * π .* k_freq

# 3. Define the second derivative operator in Fourier space: F(d^2/dx^2) = (i*k)^2 = -k^2.
k2_fourier = -k_angular.^2

# ----- Setup for De-aliasing (2/3 Rule) -------------------------
k_cutoff = floor(Int, nn / 3)
indices_high = (k_cutoff + 2):(nn - k_cutoff) # Indices of modes to be zeroed

# ----- Define the ODE Problem in Fourier Space ------------------
u0_hat = fft(u0)

# The linear part of the PDE (nu * u_xx) is a diagonal matrix in Fourier space.
viscosity = 0.001
linear_coeffs = viscosity .* k2_fourier
L_diag = Diagonal(linear_coeffs)
L_op = MatrixOperator(L_diag)

# The nonlinear part is -0.5 * (u^2)_x. This function computes its Fourier transform.
function nonlinear_part_fourier!(du_hat, u_hat, p, t)
    # Unpack parameters passed by the solver
    k_angular    = p[1]
    indices_high = p[2]

    # 1. Create a de-aliased copy of the solution in Fourier space.
    u_hat_dealiased = copy(u_hat)
    u_hat_dealiased[indices_high] .= 0.0 + 0.0im # Zero out high frequencies

    # 2. Inverse transform back to physical space to compute u.
    u_dealiased = real(ifft(u_hat_dealiased))
    
    # 3. Compute u^2 in physical space. Aliasing is now prevented.
    u_squared = u_dealiased.^2
    
    # 4. Transform u^2 back to Fourier space.
    u_squared_hat = fft(u_squared)

    # 5. Apply the first derivative (i*k) and multiply by -0.5 to get F(-0.5*(u^2)_x).
    du_hat .= -0.5 .* (1im .* k_angular) .* u_squared_hat
end

# ----- Setup the ODE Problem for the Solver ---------------------
parameters = (k_angular, indices_high)
prob = SplitODEProblem(L_op, nonlinear_part_fourier!, u0_hat, t_span, parameters)

# ----- Solve the System -----------------------------------------
dt = 1e-5
sol = solve(prob, ETDRK4(), dt=dt, saveat=t_save)

# ----- Process the Solution -------------------------------------
if sol.retcode != :Success
    error("The solver did not converge. The solution likely became unstable (blew up).")
end

# The solution is in Fourier space; transform each time step back to physical space.
usol_fourier = hcat([sol.u[i] for i in 1:length(sol.t)]...) # nn x (steps+1)
usol = zeros(nn, steps+1)
for i in 1:length(sol.t) # Loop over sol.t to prevent errors if solver stops early
    usol[:, i] = real(ifft(usol_fourier[:, i]))
end

# ----- Prepare Data for Plotting and Saving ---------------------
x_plot = range(dom[1], dom[2], length=nn+1)
usol_plot = vcat(usol, usol[1:1, :])

# ----- Visualize the Result -------------------------------------
my_plot = heatmap(t_save, x_plot, usol_plot,
    xlabel="Time (t)",
    ylabel="Space (x)",
    title="Solution of the Burgers Equation",
    color=:jet,
    aspect_ratio=:auto
)
savefig(my_plot, joinpath(@__DIR__, "burgers_solution.png"))

# ----- Save Data to File ----------------------------------------
# Transpose solution to a (time, space) format for consistency.
usol_save = usol_plot'
matwrite(joinpath(@__DIR__, "burgers_HR.mat"), Dict(
    "t" => collect(t_save),
    "x" => collect(x_plot),
    "usol" => usol_save
))
println("Burgers' equation solved and results saved successfully.")
