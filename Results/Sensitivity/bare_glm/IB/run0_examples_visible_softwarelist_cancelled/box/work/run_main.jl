###############################################################################
# run_main.jl
#
# Production run of the 2D inviscid Burgers equation
#   u_t + (u^2/2)_x + (u^2/2)_y = 0 on [0,4pi]^2, periodic,
#   u0 = 1/3 + (2/3) sin((x+y)/2),  t in [0,2]
#
# Method: nodal DGSEM (degree 4, LGL) on 128x128 elements,
#   entropy-conservative two-point volume flux (flux_ec) + Hennemann-Gassner
#   subcell shock capturing (FV flux: local Lax-Friedrichs),
#   surface flux: local Lax-Friedrichs, time: CarpenterKennedy2N54, fixed dt.
#
# Outputs (written to data/):
#   burgers2d_snapshots.npz : snapshots on a 384x384 uniform grid at
#                             t = 0, 0.25, ..., 2.0 + exact counterparts,
#                             plus node-level diagnostics for every snapshot.
###############################################################################
include("common.jl")
using OrdinaryDiffEq: solve
using OrdinaryDiffEqLowStorageRK: CarpenterKennedy2N54
using Printf
using NPZ

out_dir = joinpath(@__DIR__, "data")
isdir(out_dir) || mkpath(out_dir)

# ------------------------------------------------------------- configuration
polydeg = 4
Nx = Ny = 128
dt = 0.0025                       # ~CFL 0.26 (max_dt at CFL 1 is ~0.0098)
saveat = collect(0.0:0.25:2.0)
n_out = 384

# ------------------------------------------------------------------- setup
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
mesh = StructuredMesh((Nx, Ny), (0.0, 0.0), (L_DOMAIN, L_DOMAIN))
semi = SemidiscretizationHyperbolic(mesh, equations,
                                    initial_condition_diagonal_sine, solver)
ode = semidiscretize(semi, (0.0, 2.0))

# summary of the setup (also useful for the report)
summary_callback = SummaryCallback()
println("running 2D simulation: $Nx x $Ny elements, polydeg = $polydeg, dt = $dt")
runtime = @elapsed sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
                               dt = dt, adaptive = false, save_everystep = false,
                               saveat = saveat,
                               callback = CallbackSet(summary_callback))
@printf "done: %d accepted steps in %.1f s\n" sol.stats.naccept runtime

# ------------------------------------------------------- extract + analyze
node_coords = semi.cache.elements.node_coordinates
hx = hy = L_DOMAIN / Nx
emap = element_lookup(node_coords, 0.0, 0.0, hx, hy, Nx, Ny)
xi = basis.nodes
wbar = bary_weights(xi)

n_t = length(sol.t)
U = zeros(Float64, n_t, n_out, n_out)          # numerical snapshots
Uex = zeros(Float64, n_t, n_out, n_out)       # exact snapshots
diag = Dict{Float64,NamedTuple}()
println("snapshot diagnostics (node-level quadrature over DG nodes):")
println("   t        L2          Linf       mean-1/3      entropy      umin      umax")
for k in 1:n_t
    t = sol.t[k]
    u_elem = scalar_nodes(sol.u[k], semi)
    d = node_diagnostics(u_elem, node_coords, basis.weights, hx, hy, t)
    diag[t] = d
    @printf "  %5.2f  %.4e  %.4e  %+.2e  %.12f  %+.5f  %+.5f\n" t d.l2 d.linf d.mean - 1 / 3 d.entropy d.umin d.umax
    U[k, :, :] = to_uniform_grid(u_elem, emap, xi, wbar, 0.0, 0.0, hx, hy, Nx, Ny, n_out)
    xs = range(0.0, L_DOMAIN; length = n_out)
    for (iy, Y) in enumerate(xs), (ix, X) in enumerate(xs)
        Uex[k, iy, ix] = u_exact(X + Y, t)
    end
end

# conservation of mean u: exact (1/3) vs computed at all snapshots
println("\nmax |mean u - 1/3| over snapshots: ",
        maximum(abs(diag[t].mean - 1 / 3) for t in sol.t))
# entropy should be non-increasing (shock dissipates entropy)
ents = [diag[t].entropy for t in sol.t]
println("entropy monotone non-increasing: ", all(diff(ents) .<= 1e-12))

# -------------------------------------------------------------------- save
times = collect(sol.t)
npzwrite(joinpath(out_dir, "burgers2d_snapshots.npz"),
         Dict("times" => times, "u" => U, "u_exact" => Uex,
              "n_elements" => [Nx, Ny], "polydeg" => [polydeg],
              "grid_n" => [n_out],
              "l2" => [diag[t].l2 for t in times],
              "linf" => [diag[t].linf for t in times],
              "mean" => [diag[t].mean for t in times],
              "entropy" => [diag[t].entropy for t in times]))
println("saved ", joinpath(out_dir, "burgers2d_snapshots.npz"))
