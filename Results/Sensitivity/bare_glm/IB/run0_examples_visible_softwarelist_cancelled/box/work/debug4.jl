# inspect the shock profile at t=2 along the line y=0
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
hx = L_DOMAIN / N
emap = element_lookup(nc, 0.0, 0.0, hx, hx, N, N)
xi = basis.nodes
wbar = bary_weights(xi)
println("exact shock (from HL): ")
for s in range(0, 4pi; length = 9)
    @printf "  s=%.2f  u_ex=%.4f\n" s u_hl(s, 2.0)
end
println("\nnumerical profile u(x, y=0.06) and exact at s=x+y:")
y0 = 0.06
ey = 1
eta = 2 * (y0 - 0.0) / hx - 1
for kx in 1:41
    X = (kx - 1) * 4pi / 40
    ex = clamp(floor(Int, X / hx) + 1, 1, N)
    cxi = clamp(2 * (X - (ex - 1) * hx) / hx - 1, -1.0, 1.0)
    e = emap[ex, ey]
    nn = length(xi)
    tmp = [interp1(view(u_elem, :, j, e), wbar, xi, cxi) for j in 1:nn]
    un = interp1(tmp, wbar, xi, eta)
    ue = u_hl(X + y0, 2.0)
    @printf "  x=%.3f  un=%+.4f  ue=%+.4f  d=%+.2e\n" X un ue un - ue
end
