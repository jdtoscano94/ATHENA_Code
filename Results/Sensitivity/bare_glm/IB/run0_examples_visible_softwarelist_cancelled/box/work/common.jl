###############################################################################
# common.jl
#
# Custom 2D inviscid Burgers equation (Trixi equation interface, Trixi v0.13)
#
#   u_t + (u^2/2)_x + (u^2/2)_y = 0,   (x,y) in [0,4pi]^2, periodic
#   u(x,y,0) = 1/3 + (2/3) sin((x+y)/2)
#
# Exact solution: since data/flux are isotropic, the entropy solution depends
# only on s = x+y (mod 4pi) and solves the 1D problem
#
#   v_t + (v^2)_s = 0,  v(s,0) = 1/3 + (2/3) sin(s/2)  (period L = 4pi).
#
# Pre-shock  (t < 3/2): implicit characteristic solution
#     v = 1/3 + (2/3) sin((s - 2 v t)/2)        (solved by Newton)
# Post-shock (t >= 3/2): periodic Hopf-Lax (Lax-Oleinik) formula for the
#     entropy solution of the convex-flux HJ equation psi_t + (psi_s)^2 = 0:
#     v(s,t) = (s - m*)/(2t),  m* = argmin_m [ psi0(m) + (s-m)^2/(4t) ],
#     with psi0(m) = m/3 - (4/3)(cos(m/2) - 1)   (valid on all of R,
#     since cos((m mod 4pi)/2) = cos(m/2) for period 4pi and psi0' = v0).
###############################################################################

using Trixi
using StaticArrays
using LinearAlgebra: dot

import Trixi: varnames, flux, max_abs_speed_naive, min_max_speed_naive,
    max_abs_speeds, flux_ec, cons2prim, prim2cons, cons2entropy, entropy2cons,
    entropy, energy_total

# ------------------------------------------------------------------ equations
struct InviscidBurgersEquation2D <: Trixi.AbstractInviscidBurgersEquation{2, 1} end

varnames(::typeof(cons2cons), ::InviscidBurgersEquation2D) = ("scalar",)
varnames(::typeof(cons2prim), ::InviscidBurgersEquation2D) = ("scalar",)

# f(u) = (u^2/2, u^2/2); same flux in both coordinate directions
@inline function flux(u, orientation::Integer, equation::InviscidBurgersEquation2D)
    return SVector(0.5 * u[1]^2)
end

# f(u) .- normal_direction (directional vector is not normalized)
@inline function flux(u, normal_direction::AbstractVector,
                      equation::InviscidBurgersEquation2D)
    return SVector(0.5 * u[1]^2 * (normal_direction[1] + normal_direction[2]))
end

# wave speed estimates
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equation::InviscidBurgersEquation2D)
    return max(abs(u_ll[1]), abs(u_rr[1]))
end

@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equation::InviscidBurgersEquation2D)
    return max(abs(u_ll[1]), abs(u_rr[1])) *
           abs(normal_direction[1] + normal_direction[2])
end

@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equation::InviscidBurgersEquation2D)
    return min(u_ll[1], u_rr[1]), max(u_ll[1], u_rr[1])
end

@inline function min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equation::InviscidBurgersEquation2D)
    a = normal_direction[1] + normal_direction[2]
    if a >= zero(a)
        return a * min(u_ll[1], u_rr[1]), a * max(u_ll[1], u_rr[1])
    else
        return a * max(u_ll[1], u_rr[1]), a * min(u_ll[1], u_rr[1])
    end
end

@inline function max_abs_speeds(u, equation::InviscidBurgersEquation2D)
    return (abs(u[1]), abs(u[1]))
end

# entropy-conservative two-point flux for f(u) = u^2/2 (w.r.t. eta = u^2/2):
# F(U_L,U_R) = (u_L^2 + u_L u_R + u_R^2)/6
@inline function flux_ec(u_ll, u_rr, orientation::Integer,
                         equation::InviscidBurgersEquation2D)
    return SVector((u_ll[1]^2 + u_ll[1] * u_rr[1] + u_rr[1]^2) / 6)
end

@inline function flux_ec(u_ll, u_rr, normal_direction::AbstractVector,
                         equation::InviscidBurgersEquation2D)
    a = normal_direction[1] + normal_direction[2]
    return SVector(a * (u_ll[1]^2 + u_ll[1] * u_rr[1] + u_rr[1]^2) / 6)
end

@inline cons2prim(u, equation::InviscidBurgersEquation2D) = u
@inline prim2cons(u, equation::InviscidBurgersEquation2D) = u
@inline cons2entropy(u, equation::InviscidBurgersEquation2D) = u
@inline entropy2cons(u, equation::InviscidBurgersEquation2D) = u
@inline entropy(u::Real, ::InviscidBurgersEquation2D) = 0.5 * u^2
@inline entropy(u, equation::InviscidBurgersEquation2D) = entropy(u[1], equation)
@inline energy_total(u::Real, ::InviscidBurgersEquation2D) = 0.5 * u^2
@inline function energy_total(u, equation::InviscidBurgersEquation2D)
    return energy_total(u[1], equation)
end

# ------------------------------------------------------------- initial data
const L_DOMAIN = 4 * pi          # domain side length; also period in s = x+y
const U_MEAN = 1 / 3

# initial condition u(x,y,0) (t is unused; the IC is the t=0 state)
function initial_condition_diagonal_sine(x, t, equation::InviscidBurgersEquation2D)
    RealT = eltype(x)
    s = x[1] + x[2]
    u = U_MEAN + (2 / 3) * sin(s / 2)
    return SVector(convert(RealT, u))
end

# ------------------------------------------------------- exact solution kit
v0(s) = U_MEAN + (2 / 3) * sin(s / 2)

# pre-shock implicit characteristic solution (valid t < 1.5; Newton, machine eps)
function u_char(s, t)
    s = mod(s, L_DOMAIN)
    u = v0(s)
    for _ in 1:200
        arg = (s - 2 * u * t) / 2
        res = u - U_MEAN - (2 / 3) * sin(arg)
        dres = 1 + (2 * t / 3) * cos(arg)
        du = res / dres
        u -= du
        abs(du) < 1e-15 && break
    end
    return u
end

# periodic Hopf-Lax entropy solution (valid for all t > 0)
function u_hl(s::Real, t::Real)
    t <= 0 && return v0(s)
    s = mod(s, L_DOMAIN)
    W = 2 * t + 0.5                        # |s - m*| = 2 t |u| <= 2 t (|u|<=1)
    inv4t = 1 / (4 * t)
    mlo = s - W
    n = 2048
    dm = 2 * W / (n - 1)
    # grid search for the global minimum
    gbest = Inf
    ibest = 1
    for i in 0:(n - 1)
        m = mlo + i * dm
        g = m * U_MEAN - (4 / 3) * cos(m / 2) + (s - m)^2 * inv4t
        if g < gbest
            gbest = g
            ibest = i
        end
    end
    # golden-section refinement in [mlo+(ibest-1)dm, mlo+(ibest+1)dm]
    gr = (sqrt(5.0) - 1) / 2
    a = mlo + max(ibest - 1, 0) * dm
    b = mlo + min(ibest + 1, n - 1) * dm
    gm(m) = m * U_MEAN - (4 / 3) * cos(m / 2) + (s - m)^2 * inv4t
    x1 = b - gr * (b - a)
    x2 = a + gr * (b - a)
    f1 = gm(x1)
    f2 = gm(x2)
    for _ in 1:80
        if f1 < f2
            b = x2; x2 = x1; f2 = f1
            x1 = b - gr * (b - a); f1 = gm(x1)
        else
            a = x1; x1 = x2; f1 = f2
            x2 = a + gr * (b - a); f2 = gm(x2)
        end
        (b - a) < 1e-13 && break
    end
    mstar = (a + b) / 2
    return (s - mstar) / (2 * t)
end

# dispatch: exact entropy solution for any (s, t)
function u_exact(s, t)
    if t == 0
        return v0(s)
    elseif t < 1.45
        return u_char(s, t)
    else
        return u_hl(s, t)
    end
end

# --------------------------------------------------- DG postprocessing utils
# barycentric weights for Lagrange interpolation on nodes xi
function bary_weights(xi)
    n = length(xi)
    w = ones(n)
    for j in 1:n, i in 1:n
        if i != j
            w[i] /= (xi[i] - xi[j])
        end
    end
    return w
end

# 1D barycentric interpolation of nodal values u_nodes at query point xq
function interp1(u_nodes, wbar, xi, xq)
    num = 0.0
    den = 0.0
    @inbounds for i in eachindex(xi)
        d = xq - xi[i]
        if abs(d) < 1e-14
            return u_nodes[i]
        end
        b = wbar[i] / d
        num += b * u_nodes[i]
        den += b
    end
    return num / den
end

# extract scalar node values (nnodes_x, nnodes_y, nelements) from Trixi state
function scalar_nodes(u_ode, semi)
    u_wrap = Trixi.wrap_array(u_ode, semi)
    sz = size(u_wrap)
    if ndims(u_wrap) == 4 && sz[1] == 1
        out = Array{eltype(u_wrap),3}(undef, sz[2], sz[3], sz[4])
        out .= u_wrap[1, :, :, :]
        return out
    end
    return copy(u_wrap) # plain Array copy, shape (nnodes_x, nnodes_y, nelements)
end

# map uniform-grid point -> (element id, local coords) on a rectangular mesh
function element_lookup(node_coords, x0, y0, hx, hy, Nx, Ny)
    emap = zeros(Int, Nx, Ny)
    ne = size(node_coords, 4)
    for e in 1:ne
        xc = node_coords[1, 1, 1, e]
        yc = node_coords[2, 1, 1, e]
        ex = round(Int, (xc - x0) / hx) + 1
        ey = round(Int, (yc - y0) / hy) + 1
        emap[ex, ey] = e
    end
    return emap
end

# interpolate DG solution onto a uniform n_out x n_out grid
function to_uniform_grid(u_elem, emap, xi, wbar, x0, y0, hx, hy, Nx, Ny, n_out)
    nn = length(xi)
    xs = range(x0, L_DOMAIN; length = n_out)
    U = zeros(n_out, n_out)                # U[iy, ix] ~ f(x[ix], y[iy])
    tmp = zeros(nn)
    for ky in 1:n_out
        Y = xs[ky]
        ey = clamp(floor(Int, (Y - y0) / hy) + 1, 1, Ny)
        eta = 2 * (Y - (y0 + (ey - 1) * hy)) / hy - 1
        eta = clamp(eta, -1.0, 1.0)
        for kx in 1:n_out
            X = xs[kx]
            ex = clamp(floor(Int, (X - x0) / hx) + 1, 1, Nx)
            cxi = 2 * (X - (x0 + (ex - 1) * hx)) / hx - 1
            cxi = clamp(cxi, -1.0, 1.0)
            e = emap[ex, ey]
            for j in 1:nn
                tmp[j] = interp1(view(u_elem, :, j, e), wbar, xi, cxi)
            end
            U[ky, kx] = interp1(tmp, wbar, xi, eta)
        end
    end
    return U
end

# discrete diagnostics at DG nodes (LGL quadrature on the uniform mesh):
# L2/Linf error vs exact solution, mean, entropy integral, min/max
function node_diagnostics(u_elem, node_coords, weights, hx, hy, t)
    nn = size(u_elem, 1)
    ne = size(u_elem, 3)
    jac = (hx / 2) * (hy / 2)
    l2sq = 0.0
    linf = 0.0
    sum_u = 0.0
    sum_eta = 0.0
    umin = Inf
    umax = -Inf
    for e in 1:ne, j in 1:nn, i in 1:nn
        x = node_coords[1, i, j, e]
        y = node_coords[2, i, j, e]
        un = u_elem[i, j, e]
        ue = u_exact(x + y, t)
        wq = weights[i] * weights[j] * jac
        d = un - ue
        l2sq += wq * d^2
        linf = max(linf, abs(d))
        sum_u += wq * un
        sum_eta += wq * 0.5 * un^2
        umin = min(umin, un)
        umax = max(umax, un)
    end
    area = L_DOMAIN^2
    return (l2 = sqrt(l2sq), linf = linf, mean = sum_u / area,
            entropy = sum_eta, umin = umin, umax = umax)
end
