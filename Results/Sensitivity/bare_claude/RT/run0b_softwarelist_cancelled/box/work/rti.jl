using Trixi, OrdinaryDiffEqLowStorageRK, NPZ, LinearAlgebra
const NX = parse(Int, ARGS[1]); const NY = 4NX
const POLY = 3
const g = 1.0
eq = CompressibleEulerEquations2D(1.4)

function ic_rti(x, t, equations::CompressibleEulerEquations2D)
    y = x[2]
    rho = 2.0 + (1.0 - 2.0) * 0.5 * (1 + tanh(50 * (y - 0.5)))
    p = y < 0.5 ? 2y + 1 : y + 1.5
    c = sqrt(equations.gamma * p / rho)
    v1 = 0.0
    v2 = -0.025 * c * cos(8pi * x[1]) * sinpi(y)^6
    return prim2cons(SVector(rho, v1, v2, p), equations)
end
@inline function source_grav(u, x, t, equations::CompressibleEulerEquations2D)
    rho, rho_v1, rho_v2, rho_e = u
    return SVector(0.0, 0.0, g * rho, g * rho_v2)
end

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
basis = LobattoLegendreBasis(POLY)
indicator = IndicatorHennemannGassner(eq, basis; alpha_max = 0.5, alpha_min = 0.001,
                                      alpha_smooth = false, variable = density_pressure)
vol = VolumeIntegralShockCapturingHG(indicator; volume_flux_dg = volume_flux,
                                     volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, vol)
mesh = StructuredMesh((NX, NY), (0.0, 0.0), (0.25, 1.0), periodicity = (true, false))
bcs = (x_neg = boundary_condition_periodic, x_pos = boundary_condition_periodic,
       y_neg = boundary_condition_slip_wall, y_pos = boundary_condition_slip_wall)
semi = SemidiscretizationHyperbolic(mesh, eq, ic_rti, solver;
                                    source_terms = source_grav, boundary_conditions = bcs)
ode = semidiscretize(semi, (0.0, 3.0))
cbs = CallbackSet(SummaryCallback(), AliveCallback(analysis_interval = 2000),
                  StepsizeCallback(cfl = 0.9))
tsave = collect(0.0:0.05:3.0)
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false); dt = 1.0,
            saveat = tsave, callback = cbs, save_everystep = false)

# ---- post-process: global node arrays
N = POLY + 1
nodes_x = semi.cache.elements.node_coordinates  # (2, N, N, nelem)
X = zeros(N*NX, N*NY); Y = similar(X)
el(ex, ey) = ex + (ey - 1) * NX
for ey in 1:NY, ex in 1:NX, j in 1:N, i in 1:N
    X[(ex-1)*N+i, (ey-1)*N+j] = nodes_x[1, i, j, el(ex, ey)]
    Y[(ex-1)*N+i, (ey-1)*N+j] = nodes_x[2, i, j, el(ex, ey)]
end
@assert all(diff(X[:, 1]) .>= 0) && all(diff(Y[1, :]) .>= 0)
w = basis.weights
J = 1 / ((NX / 0.25) * (NY / 1.0)) * 4   # element jacobian: (dx/2)(dy/2)
nt = length(sol.t)
RHO = zeros(nt, N*NX, N*NY); V2 = similar(RHO); P = similar(RHO)
mass = zeros(nt); etot = zeros(nt); ekin = zeros(nt); epot = zeros(nt)
for (k, uvec) in enumerate(sol.u)
    u = Trixi.wrap_array(uvec, semi)
    for ey in 1:NY, ex in 1:NX, j in 1:N, i in 1:N
        e = el(ex, ey)
        uc = SVector(u[1, i, j, e], u[2, i, j, e], u[3, i, j, e], u[4, i, j, e])
        pr = cons2prim(uc, eq)
        I = (ex-1)*N+i; Jj = (ey-1)*N+j
        RHO[k, I, Jj] = pr[1]; V2[k, I, Jj] = pr[3]; P[k, I, Jj] = pr[4]
        wq = w[i] * w[j] * (0.25 / NX / 2) * (1.0 / NY / 2)
        y = nodes_x[2, i, j, e]
        mass[k] += wq * uc[1]
        etot[k] += wq * (uc[4] - g * uc[1] * y)   # E + rho*phi, phi = -g y
        ekin[k] += wq * 0.5 * (uc[2]^2 + uc[3]^2) / uc[1]
        epot[k] += wq * (-g * uc[1] * y)
    end
end
npzwrite("/box/work/rti_N$(NX).npz", Dict("t" => collect(sol.t), "X" => X, "Y" => Y,
         "rho" => RHO, "v" => V2, "p" => P, "mass" => mass, "etot" => etot,
         "ekin" => ekin, "epot" => epot, "w" => collect(w)))
println("done; mass drift = ", maximum(abs.(mass .- mass[1])) / mass[1],
        "  energy drift = ", maximum(abs.(etot .- etot[1])) / abs(etot[1]))
