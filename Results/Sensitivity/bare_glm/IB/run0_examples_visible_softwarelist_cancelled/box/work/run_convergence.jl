###############################################################################
# run_convergence.jl
#
# Grid-convergence study against the exact entropy solution:
#   pre-shock  t = 1.0  (smooth, expect ~ h^{p+1})
#   post-shock t = 2.0  (entropic discontinuity, expect slow L2 convergence)
# polydeg = 4, N in {16, 32, 64, 128}, dt ~ h^1.25 (temporal error subdominant)
###############################################################################
include("common.jl")
using OrdinaryDiffEq: solve
using OrdinaryDiffEqLowStorageRK: CarpenterKennedy2N54
using Printf

function solve_at(N, t_end; polydeg = 4)
    dt = 0.005 * (64 / N)^1.25
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
    ode = semidiscretize(semi, (0.0, t_end))
    sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
                dt = dt, adaptive = false, save_everystep = false,
                saveat = [t_end])
    u_elem = scalar_nodes(sol.u[1], semi)
    node_coords = semi.cache.elements.node_coordinates
    h = L_DOMAIN / N
    return node_diagnostics_l1l2(u_elem, node_coords, basis.weights, h, h, t_end)
end

# L2, Linf, and additionally L1 error
function node_diagnostics_l1l2(u_elem, node_coords, weights, hx, hy, t)
    nn = size(u_elem, 1)
    ne = size(u_elem, 3)
    jac = (hx / 2) * (hy / 2)
    l1 = 0.0
    l2sq = 0.0
    linf = 0.0
    for e in 1:ne, j in 1:nn, i in 1:nn
        x = node_coords[1, i, j, e]
        y = node_coords[2, i, j, e]
        d = u_elem[i, j, e] - u_exact(x + y, t)
        wq = weights[i] * weights[j] * jac
        l1 += wq * abs(d)
        l2sq += wq * d^2
        linf = max(linf, abs(d))
    end
    return (l1 = l1, l2 = sqrt(l2sq), linf = linf)
end

Ns = [16, 32, 64, 128]
for t_end in (1.0, 2.0)
    println("\n=== convergence at t = $t_end (polydeg = 4) ===")
    println("   N     h           L1            L2          rate(L2)      Linf")
    prev = nothing
    for N in Ns
        d = solve_at(N, t_end)
        rate = prev === nothing ? NaN : log2(prev / d.l2)
        @printf "  %4d  %.4f   %.4e   %.4e   %6.2f   %.4e\n" N L_DOMAIN / N d.l1 d.l2 rate d.linf
        prev = d.l2
    end
end
