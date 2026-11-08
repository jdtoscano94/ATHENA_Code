module MoC

using Roots
using LinearAlgebra

export solve_burgers_analytical, solve_characteristic_root

"""
    solve_burgers_analytical(x, t, alpha, beta)

Analytical solution for 1D Burgers' equation: u_t + u u_x = 0
Initial condition: u(x,0) = alpha + beta * sin(x)

Handles shock formation and Rankine-Hugoniot conditions using the 
Equal Area Rule logic.

Arguments:
- `x`: Vector of spatial coordinates
- `t`: Time
- `alpha`: Mean flow velocity (DC offset of initial sine wave)
- `beta`: Perturbation amplitude (Amplitude of initial sine wave)
"""
function solve_burgers_analytical(x_in, t, alpha, beta)
    # Work on a copy to avoid modifying the input array
    x = collect(Float64, x_in)
    pi_val = π

    # 1. Calculate Shock Location
    # The shock forms at x = π (for sine) and moves with speed 'alpha'
    d2 = pi_val + alpha * t

    # 2. Handle Periodicity
    # If the shock has moved past 2π, shift x coordinates to track it
    if (pi_val + alpha * t > 2.0 * pi_val)
        shift_count = floor((pi_val + alpha * t) / (2.0 * pi_val))
        shift_val = 2.0 * pi_val * shift_count
        x .+= shift_val
    end

    # 3. Split Domain relative to the Shock
    idx1 = x .<= d2
    idx2 = x .> d2

    x1 = x[idx1]
    x2 = x[idx2]

    # Prepare output array
    vv = similar(x)

    # 4. Solve for Left Side (Pre-shock / Expansion)
    if !isempty(x1)
        xn = x1 .- alpha * t
        u_vals = map(xi -> solve_characteristic_root(xi, t, beta), xn)
        v = alpha .+ beta .* sin.(u_vals)
        vv[idx1] = v
    end

    # 5. Solve for Right Side (Post-shock / Compression)
    if !isempty(x2)
        x2_flipped = reverse(2.0 * d2 .- x2)
        xn2 = x2_flipped .- alpha * t
        u2_vals = map(xi -> solve_characteristic_root(xi, t, beta), xn2)
        v2 = alpha .+ beta .* sin.(u2_vals)
        v2_final = reverse(2.0 * alpha .- v2)
        vv[idx2] = v2_final
    end

    return vv
end

"""
    solve_characteristic_root(x_target, t, beta)

Solves the implicit characteristic equation: 
    x_target - x0 - beta * sin(x0) * t = 0
"""
function solve_characteristic_root(x_target, t, beta)
    f(x0) = x_target - x0 - beta * t * sin(x0)
    # Order1 (Secant method) is robust and efficient here
    return find_zero(f, 0.0, Order1()) 
end

end # module MoC