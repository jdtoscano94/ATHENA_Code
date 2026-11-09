# post-shock test at t=2.0: SC vs no-SC
include("common.jl")
using OrdinaryDiffEq: solve
using OrdinaryDiffEqLowStorageRK: CarpenterKennedy2N54
using Printf

function run_case(N; polydeg = 4, sc = true, tend = 2.0, dt = 0.005)
    equations = InviscidBurgersEquation2D()
    surface_flux = FluxLaxFriedrichs(max_abs_speed_naive)
    basis = LobattoLegendreBasis(polydeg)
    if sc
        indicator_sc = IndicatorHennemannGassner(equations, basis;
                                                 alpha_max = 0.5, alpha_min = 0.0,
                                                 alpha_smooth = true, variable = first)
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
                saveat = [1.0, 1.5, 1.75, 2.0])
    out = []
    for k in 1:length(sol.t)
        u_elem = scalar_nodes(sol.u[k], semi)
        node_coords = semi.cache.elements.node_coordinates
        d = node_diagnostics(u_elem, node_coords, basis.weights, L_DOMAIN / N,
                             L_DOMAIN / N, sol.t[k])
        push!(out, (sol.t[k], d))
    end
    return out
end

for sc in (true, false)
    println("shock capturing = ", sc)
    for N in (32, 64, 128)
        tt = @elapsed out = run_case(N; sc = sc, dt = 0.005 * 64 / N)
        for (t, d) in out
            @printf "  N=%3d t=%.2f L2=%.3e Linf=%.3e umin=%.4f umax=%.4f mean-1/3=%.1e (%.1fs total)\n" N t d.l2 d.linf d.umin d.umax abs(d.mean - 1 / 3) tt
        end
    end
end
