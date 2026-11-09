# compare SC variants pre-shock (t=1.0) and post-shock (t=2.0)
include("common.jl")
using OrdinaryDiffEq: solve
using OrdinaryDiffEqLowStorageRK: CarpenterKennedy2N54
using Printf

function solve_case(N, t_end; polydeg = 4, mode = :sc_smooth)
    dt = 0.005 * (64 / N)^1.25
    equations = InviscidBurgersEquation2D()
    surface_flux = FluxLaxFriedrichs(max_abs_speed_naive)
    basis = LobattoLegendreBasis(polydeg)
    volume_integral = if mode == :nosc
        VolumeIntegralFluxDifferencing(flux_ec)
    else
        smooth = (mode == :sc_smooth)
        indicator_sc = IndicatorHennemannGassner(equations, basis; alpha_max = 0.5,
                                                 alpha_min = 0.0,
                                                 alpha_smooth = smooth,
                                                 variable = first)
        VolumeIntegralShockCapturingHG(indicator_sc; volume_flux_dg = flux_ec,
                                       volume_flux_fv = surface_flux)
    end
    solver = DGSEM(basis, surface_flux, volume_integral)
    mesh = StructuredMesh((N, N), (0.0, 0.0), (L_DOMAIN, L_DOMAIN))
    semi = SemidiscretizationHyperbolic(mesh, equations,
                                        initial_condition_diagonal_sine, solver)
    ode = semidiscretize(semi, (0.0, t_end))
    sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
                dt = dt, adaptive = false, save_everystep = false, saveat = [t_end])
    u_elem = scalar_nodes(sol.u[1], semi)
    nc = semi.cache.elements.node_coordinates
    h = L_DOMAIN / N
    nn = polydeg + 1
    l2sq = linf = umin = Inf; umax = -Inf
    umin = Inf
    for e in 1:(N * N), j in 1:nn, i in 1:nn
        d = u_elem[i, j, e] - u_exact(nc[1, i, j, e] + nc[2, i, j, e], t_end)
        wq = basis.weights[i] * basis.weights[j] * (h / 2)^2
        l2sq += wq * d^2
        linf = max(linf, abs(d))
        umin = min(umin, u_elem[i, j, e])
        umax = max(umax, u_elem[i, j, e])
    end
    return sqrt(l2sq), linf, umin, umax
end

for mode in (:nosc, :sc_smooth, :sc_nosmooth)
    println("mode = ", mode)
    for (N, t) in [(64, 1.0), (128, 1.0), (64, 2.0), (128, 2.0)]
        l2, linf, umin, umax = solve_case(N, t; mode = mode)
        @printf "  N=%3d t=%.1f  L2=%.4e Linf=%.4e u:[%.4f, %.4f]\n" N t l2 linf umin umax
    end
end
