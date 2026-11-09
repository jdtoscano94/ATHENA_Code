# locate L2 error: per-element contributions at t=2, N=64
include("common.jl")
using OrdinaryDiffEq: solve
using OrdinaryDiffEqLowStorageRK: CarpenterKennedy2N54
using Printf

N = 64
polydeg = 4
equations = InviscidBurgersEquation2D()
surface_flux = FluxLaxFriedrichs(max_abs_speed_naive)
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis; alpha_max = 0.5,
                                         alpha_min = 0.0, alpha_smooth = true,
                                         variable = first)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = flux_ec,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)
mesh = StructuredMesh((N, N), (0.0, 0.0), (L_DOMAIN, L_DOMAIN))
semi = SemidiscretizationHyperbolic(mesh, equations,
                                    initial_condition_diagonal_sine, solver)
ode = semidiscretize(semi, (0.0, 2.0))
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 0.005, adaptive = false, save_everystep = false, saveat = [2.0])
u_elem = scalar_nodes(sol.u[1], semi)
nc = semi.cache.elements.node_coordinates
nn = polydeg + 1
ne = N * N
hx = L_DOMAIN / N
jac = (hx / 2)^2
contrib = zeros(ne)
for e in 1:ne
    acc = 0.0
    for j in 1:nn, i in 1:nn
        x = nc[1, i, j, e]; y = nc[2, i, j, e]
        ue = u_hl(x + y, 2.0)
        d = u_elem[i, j, e] - ue
        acc += basis.weights[i] * basis.weights[j] * jac * d^2
    end
    contrib[e] = acc
end
ord = sortperm(contrib; rev = true)
tot = sum(contrib)
println("total L2^2 = ", tot, "  L2 = ", sqrt(tot))
println("top 12 elements: (ex, ey) from lower-left corner node:")
for k in 1:12
    e = ord[k]
    xc = nc[1, 1, 1, e]; yc = nc[2, 1, 1, e]
    ex = round(Int, xc / hx); ey = round(Int, yc / hx)
    @printf "  e=%4d (ex=%2d,ey=%2d) x=%.2f y=%.2f s=%.2f contrib=%.4e frac=%.3f\n" e ex ey xc yc xc + yc contrib[k] contrib[k] / tot
end
println("\ncumulative fraction of top elements: ")
c = 0.0
for k in 1:12
    global c += contrib[ord[k]] / tot
    @printf "  k=%2d cumfrac=%.4f\n" k c
end
