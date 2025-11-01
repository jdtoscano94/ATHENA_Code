using StartUpDG
using OrdinaryDiffEq
using Plots
using LinearAlgebra 
using Crunch
# ==========================================
# rhs! for INVISCID Burgers' 
# ==========================================
function rhs!(du, u, parameters, t)
    (; rd, md) = parameters
    (; Pq, Vq, Vf, Dr, LIFT) = rd
    (; nx, J, mapP) = md

    f = Pq * (Vq * u).^2 / 2
    fM = Vf * f
    fP = fM[mapP]
    uM = Vf * u
    uP = uM[mapP]
    flux = @. 0.5 * (fP - fM) * nx - 0.5 * max(abs(uP), abs(uM)) * (uP - uM)
    du .= -(Dr * f + LIFT * (flux)) ./ J
end

# --- Setup (Periodic) ---
N = 5
num_elements = 128
rd = RefElemData(Line(), N)
(VX,), EToV = uniform_mesh(Line(), num_elements)
md = MeshData(VX, EToV, rd; is_periodic=true)
params = (; rd, md) 
(; x) = md
u0(x) = -sin(pi * x)
u = u0.(x)
tspan = (0.0, 2.0) 
ode = ODEProblem(rhs!, u, tspan, params)


limiter_callback_P1 = DiscreteCallback(
    (u, t, integrator) -> true, 
    integrator -> limiter_minmod_P1!(integrator.u, integrator.p) 
)

sol = solve(ode, SSPRK432(), callback=limiter_callback_P1, saveat=LinRange(tspan[1], tspan[2], 100))

ylims_plot = extrema(u0.(rd.Vp * md.x)) .+ (0, 0) 
plot(rd.Vp * md.x, rd.Vp * sol.u[end], leg=false, ylims=ylims_plot)
