# smoke test: small run to verify mechanics + exact-solution self-checks
include("common.jl")
using OrdinaryDiffEq: solve
using OrdinaryDiffEqLowStorageRK: CarpenterKennedy2N54
using Printf

# ---- exact solution self-consistency ------------------------------------
println("== exact solution self-checks ==")
# (1) u_hl -> v0 as t -> 0
err0 = maximum(abs(u_hl(s, 1e-8) - v0(s)) for s in range(0, 4pi; length = 501))
@printf "  HL(t=1e-8) vs IC:               max err = %.3e\n" err0
# (2) Newton char vs Hopf-Lax pre-shock
for t in (0.5, 1.0, 1.4)
    e = maximum(abs(u_char(s, t) - u_hl(s, t)) for s in range(0, 4pi; length = 401))
    @printf "  char vs HL at t=%.2f:            max err = %.3e\n" t e
end
# (3) shock time: gradient blowup check at t slightly below 1.5
t = 1.4999
m = maximum(abs(u_char(s, t) - u_char(s + 1e-6, t)) / 1e-6 for s in range(0, 4pi; length = 2001))
@printf "  max |du/ds| at t=1.4999:        = %.3e (blowup => shock at 1.5)\n" m

# ---- small Trixi run -----------------------------------------------------
polydeg = 3
Nx = Ny = 8
dt = 0.02
equations = InviscidBurgersEquation2D()
surface_flux = FluxLaxFriedrichs(max_abs_speed_naive)
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis;
                                         alpha_max = 0.5, alpha_min = 0.001,
                                         alpha_smooth = true, variable = first)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = flux_ec,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)
mesh = StructuredMesh((Nx, Ny), (0.0, 0.0), (L_DOMAIN, L_DOMAIN))
semi = SemidiscretizationHyperbolic(mesh, equations,
                                     initial_condition_diagonal_sine, solver)
ode = semidiscretize(semi, (0.0, 1.0))
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = dt, adaptive = false, save_everystep = false,
            saveat = [0.0, 0.5, 1.0])
println("\n== Trixi smoke run ==")
println("  sol.t = ", sol.t)
u0_wrap = Trixi.wrap_array(sol.u[1], semi)
println("  wrap_array size = ", size(u0_wrap), "  ndims = ", ndims(u0_wrap))
node_coords = semi.cache.elements.node_coordinates
println("  node_coordinates size = ", size(node_coords))
println("  node (1,1,1) = ", node_coords[:, 1, 1, 1])

u_elem = scalar_nodes(sol.u[2], semi)
hx = hy = L_DOMAIN / Nx
emap = element_lookup(node_coords, 0.0, 0.0, hx, hy, Nx, Ny)
println("  emap valid: ", all(emap .> 0), "  (min,max) u at t=0.5: ",
        round(minimum(u_elem), digits = 5), " ", round(maximum(u_elem), digits = 5))
d = node_diagnostics(u_elem, node_coords, basis.weights, hx, hy, 0.5)
@printf "  t=0.5  L2=%.3e  Linf=%.3e  mean=%.15f (should be 1/3)\n" d.l2 d.linf d.mean
u_elem1 = scalar_nodes(sol.u[3], semi)
d1 = node_diagnostics(u_elem1, node_coords, basis.weights, hx, hy, 1.0)
@printf "  t=1.0  L2=%.3e  Linf=%.3e  mean=%.15f\n" d1.l2 d1.linf d1.mean
println("done")
