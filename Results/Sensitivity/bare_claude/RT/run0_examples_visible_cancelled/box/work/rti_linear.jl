using OrdinaryDiffEq, DiffEqCallbacks, Trixi, NPZ, LinearAlgebra, Printf
const NEL = parse(Int, ARGS[1]); const PD = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 3
const gamma = 1.4; const G = 1.0; const AMP = parse(Float64, ARGS[3]); const TEND = 1.5
equations = CompressibleEulerEquations2D(gamma)

function ic_rti(x, t, equations::CompressibleEulerEquations2D)
    y = x[2]
    p = y < 0.5 ? 2y + 1 : y + 1.5
    rho = 2.0 + (1.0 - 2.0) * 0.5 * (1 + tanh(50 * (y - 0.5)))
    c = sqrt(equations.gamma * p / rho)
    v = -AMP * c * cos(8pi * x[1]) * sin(pi * y)^6
    return prim2cons(SVector(rho, 0.0, v, p), equations)
end
@inline function src_rti(u, x, t, equations::CompressibleEulerEquations2D)
    return SVector(0.0, 0.0, G * u[1], G * u[3])      # gravity +y
end

basis = LobattoLegendreBasis(PD)
indicator = IndicatorHennemannGassner(equations, basis; alpha_max = 0.5, alpha_min = 0.001,
                                      alpha_smooth = false, variable = density_pressure)
vol = VolumeIntegralShockCapturingHG(indicator; volume_flux_dg = flux_ranocha, volume_flux_fv = flux_hllc)
solver = DGSEM(basis, flux_hllc, vol)
mapping(xi, eta) = SVector(0.125 * (1 + xi), 0.5 * (1 + eta))
mesh = StructuredMesh((NEL, 4NEL), mapping, periodicity = (true, false))
bcs = (x_neg = boundary_condition_periodic, x_pos = boundary_condition_periodic,
       y_neg = boundary_condition_slip_wall, y_pos = boundary_condition_slip_wall)
semi = SemidiscretizationHyperbolic(mesh, equations, ic_rti, solver; source_terms = src_rti,
                                    boundary_conditions = bcs)
ode = semidiscretize(semi, (0.0, TEND))

# ---------- diagnostics ----------
const nodes = basis.nodes; const w = basis.weights; const n = PD + 1
const xc = semi.cache.elements.node_coordinates
const detJ = (0.25 / NEL) * (1.0 / (4NEL)) / 4
NS = n                     # sub-samples per element per direction (uniform cell centres)
xs = [-1 + (2k - 1) / NS for k in 1:NS]
Vm = Trixi.polynomial_interpolation_matrix(nodes, xs)
function uniform_fields(u_ode)
    u = Trixi.wrap_array(u_ode, semi)
    nx, ny = NEL * NS, 4NEL * NS
    out = zeros(4, nx, ny)   # rho,u,v,p
    for el in 1:(NEL * 4NEL)
        ex = (el - 1) % NEL; ey = (el - 1) ÷ NEL
        for v in 1:4
            prim = [cons2prim(SVector(u[1,i,j,el],u[2,i,j,el],u[3,i,j,el],u[4,i,j,el]), equations)[v] for i in 1:n, j in 1:n]
            out[v, ex*NS+1:(ex+1)*NS, ey*NS+1:(ey+1)*NS] = Vm * prim * Vm'
        end
    end
    out
end
const hist = Float64[]; const times = Float64[]; const stepcount = Ref(0)
function diag!(u_ode, t)
    u = Trixi.wrap_array(u_ode, semi)
    M = 0.0; E = 0.0; P = 0.0; A = 0.0; Ek = 0.0; Ps = 0.0
    for el in axes(u, 4), j in 1:n, i in 1:n
        wq = w[i] * w[j] * detJ
        x = xc[1, i, j, el]
        M += wq * u[1,i,j,el]; E += wq * u[4,i,j,el]; P += wq * u[3,i,j,el]
        A += wq * u[3,i,j,el] / u[1,i,j,el] * cos(8pi * x) * 4 / 0.25 * 2   # ~ amplitude of v mode, normalized
        Ek += wq * 0.5 * (u[2,i,j,el]^2 + u[3,i,j,el]^2) / u[1,i,j,el]
        Ps += wq * G * u[3,i,j,el]   # dE/dt source = g*rho*v
    end
    append!(hist, [t, M, E, P, A, Ek, Ps]); push!(times, t)
end
diag_cb = DiscreteCallback((u, t, integ) -> (stepcount[] += 1; stepcount[] % 4 == 0), integ -> (diag!(integ.u, integ.t); u_modified!(integ, false)),
                           save_positions = (false, false), initialize = (c, u, t, integ) -> diag!(u, t))
snap_t = collect([0.0, TEND]); snaps = Dict{Float64,Array{Float64,3}}()
snap_cb = PresetTimeCallback(snap_t, integ -> (snaps[integ.t] = uniform_fields(integ.u); u_modified!(integ, false)))
stepsize = StepsizeCallback(cfl = 0.9)
cbs = CallbackSet(SummaryCallback(), AliveCallback(analysis_interval = 2000), diag_cb, snap_cb, stepsize)
limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5e-6, 5e-6), variables = (Trixi.density, pressure))
snaps[0.0] = uniform_fields(ode.u0)
t0 = time()
sol = solve(ode, SSPRK43(stage_limiter! = limiter!); dt = 1e-4, adaptive = false, callback = cbs, save_everystep = false)
println("wall time ", time() - t0, "  retcode ", sol.retcode)
H = reshape(hist, 7, :)'
S = cat([snaps[t] for t in sort(collect(keys(snaps)))]...; dims = 4)
npzwrite("lin_N$(NEL)_p$(PD).npz", Dict("hist" => collect(H), "snaps" => S, "snap_t" => sort(collect(keys(snaps))),
         "NEL" => NEL, "PD" => PD))
