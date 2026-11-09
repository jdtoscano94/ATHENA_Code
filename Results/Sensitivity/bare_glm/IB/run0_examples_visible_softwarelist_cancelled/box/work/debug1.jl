# debug: is the L2 error legitimate discretization error or a bug?
include("common.jl")
using OrdinaryDiffEq: solve
using OrdinaryDiffEqLowStorageRK: CarpenterKennedy2N54
using Printf

function run_case(N; polydeg = 3, sc = true, tend = 0.5, dt = 0.02)
    equations = InviscidBurgersEquation2D()
    surface_flux = FluxLaxFriedrichs(max_abs_speed_naive)
    basis = LobattoLegendreBasis(polydeg)
    if sc
        indicator_sc = IndicatorHennemannGassner(equations, basis;
                                                 alpha_max = 0.5,
                                                 alpha_min = 0.001,
                                                 alpha_smooth = true,
                                                 variable = first)
        volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                          volume_flux_dg = flux_ec,
                                                          volume_flux_fv = surface_flux)
    else
        volume_integral = VolumeIntegralFluxDifferencing(flux_ec)
    end
    solver = DGSEM(basis, surface_flux, volume_integral)
    mesh = StructuredMesh((N, N), (0.0, 0.0), (L_DOMAIN, L_DOMAIN))
    semi = SemidiscretizationHyperbolic(mesh, equations,
                                        initial_condition_diagonal_sine, solver)
    ode = semidiscretize(semi, (0.0, tend))
    sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
                dt = dt, adaptive = false, save_everystep = false,
                saveat = [0.0, tend])
    u_elem = scalar_nodes(sol.u[2], semi)
    node_coords = semi.cache.elements.node_coordinates
    d = node_diagnostics(u_elem, node_coords, basis.weights, L_DOMAIN / N,
                          L_DOMAIN / N, tend)
    # also t=0 projection error
    u0 = scalar_nodes(sol.u[1], semi)
    d0 = node_diagnostics(u0, node_coords, basis.weights, L_DOMAIN / N,
                           L_DOMAIN / N, 0.0)
    return d, d0, u_elem, node_coords
end

for N in (8, 16, 32)
    d, d0, u_elem, nc = run_case(N)
    @printf "N=%3d  t=0 proj: L2=%.3e   t=0.5: L2=%.3e Linf=%.3e umin=%.4f umax=%.4f\n" N d0.l2 d.l2 d.linf d.umin d.umax
end
println("--- without shock capturing ---")
for N in (8, 16)
    d, d0, u_elem, nc = run_case(N; sc = false)
    @printf "N=%3d  t=0 proj: L2=%.3e   t=0.5: L2=%.3e Linf=%.3e umin=%.4f umax=%.4f\n" N d0.l2 d.l2 d.linf d.umin d.umax
end

# error-field profile: print error vs s = x+y for N=8 to see structure
d, d0, u_elem, nc = run_case(8)
hx = L_DOMAIN / 8
ne = 64
println("\nerr samples along y=0 line (element 1..8):")
for e in 1:8
    for i in 1:4
        x = nc[1, i, 1, e]; y = nc[2, i, 1, e]
        un = u_elem[i, 1, e]
        ue = u_exact(x + y, 0.5)
        @printf "  x=%.3f  un=%.6f  ue=%.6f  d=%.2e\n" x un ue abs(un - ue)
    end
end
