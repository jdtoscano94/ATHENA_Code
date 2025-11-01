using StartUpDG
using OrdinaryDiffEq
using Plots
using LinearAlgebra
using Crunch # For the limiter

# ==========================================
# rhs! for VISCOUS Burgers' equation
# ==========================================
function rhs!(du, u, parameters, t)
    (; rd, md, epsilon) = parameters
    (; Pq, Vq, Vf, Dr, LIFT) = rd
    (; nx, J, mapP) = md

    # ==================================
    # Inviscid part (convective term)
    # ==================================
    f = Pq * (Vq * u).^2 / 2
    fM = Vf * f
    fP = fM[mapP]
    uM = Vf * u
    uP = uM[mapP]
    # Lax-Friedrichs numerical flux for the inviscid term
    flux = @. 0.5 * (fP - fM) * nx - 0.5 * max(abs(uP), abs(uM)) * (uP - uM)
    
    # Compute inviscid contribution to the RHS
    du .= -(Dr * f + LIFT * (flux)) ./ J

    # ==================================
    # Viscous part (diffusion term)
    # ==================================
    # This section implements the Local Discontinuous Galerkin (LDG) method
    # to discretize the second derivative term: epsilon * u_xx
    # 1. Compute auxiliary variable sigma ≈ u_x
    u_flux = @. 0.5 * (uP - uM) * nx
    sigma = (Dr * u + LIFT * u_flux) ./ J
    
    # 2. Compute divergence of sigma
    sigmaM = Vf * sigma
    sigmaP = sigmaM[mapP]
    sigma_flux = @. 0.5 * (sigmaP - sigmaM) * nx + (uP - uM)
    
    # Add viscous contribution to the RHS
    du .+= epsilon * (Dr * sigma + LIFT * sigma_flux) ./ J
end

# --- Setup (Periodic) ---
N = 5
num_elements = 128
rd = RefElemData(Line(), N)
(VX,), EToV = uniform_mesh(Line(), num_elements)
md = MeshData(VX, EToV, rd; is_periodic=true)

# --- NEW: Define the viscosity coefficient ---
epsilon = 0.0001

# --- UPDATED: Pass epsilon into the parameters tuple ---
params = (; rd, md, epsilon) 

# --- Initial condition and time span (same as before) ---
(; x) = md
u0(x) = -sin(pi * x)
u = u0.(x)
tspan = (0.0, 1.0) 
ode = ODEProblem(rhs!, u, tspan, params)

# --- Limiter and Solver (same as before) ---
# A limiter is still helpful for small viscosity values to handle sharp gradients.
limiter_callback_P1 = DiscreteCallback(
    (u, t, integrator) -> true, 
    integrator -> limiter_minmod_P1!(integrator.u, integrator.p) 
)

sol = solve(ode, SSPRK432(), callback=limiter_callback_P1, saveat=LinRange(tspan[1], tspan[2], 100))

# --- Plotting (same as before) ---
ylims_plot = extrema(u0.(rd.Vp * md.x)) .+ (0, 0) 
plot(rd.Vp * md.x, rd.Vp * sol.u[end], leg=false, ylims=ylims_plot,
     title="Viscous Burgers' Solution at t=$(tspan[2]) with ε=$epsilon",
     xlabel="x", ylabel="u(x,t)")
