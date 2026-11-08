module Limiters
using StartUpDG
export m_minmod, limiter_minmod_P1!
function m_minmod(a1, a2, a3)
    s = sign(a1)
    if s == sign(a2) && s == sign(a3)
        return s * min(abs(a1), abs(a2), abs(a3))
    else
        return 0.0
    end
end

function limiter_minmod_P1!(u_vec, parameters)
    (; rd, md) = parameters
    (; N, Vq, Vf, r) = rd # Need reference nodes 'r'
    (; K, wJq) = md
    tol = 1e-10 # Limiter sensitivity threshold

    u = reshape(u_vec, N + 1, K)

    # 1. Compute cell averages
    u_on_quad_all = Vq * u
    u_avg_vec = sum(u_on_quad_all .* wJq, dims=1) ./ sum(wJq, dims=1)
    u_avg = reshape(u_avg_vec, K)

    # 2. Compute high-order values at element interfaces
    uM = Vf * u
    u_at_left_face = uM[1, :]  # u_{k-1/2}^+
    u_at_right_face = uM[2, :] # u_{k+1/2}^-

    # 3. Apply minmod logic and reconstruct P1 if needed
    for k in 1:K
        u_k = @view u[:, k] # This is the view we modify
        u_avg_k = u_avg[k]

        delta_u_plus_ho = u_at_right_face[k] - u_avg_k
        delta_u_minus_ho = u_avg_k - u_at_left_face[k]

        k_L_nbr = mod1(k - 1, K)
        k_R_nbr = mod1(k + 1, K)
        u_avg_L_nbr = u_avg[k_L_nbr]
        u_avg_R_nbr = u_avg[k_R_nbr]

        slope_forward = u_avg_R_nbr - u_avg_k
        slope_backward = u_avg_k - u_avg_L_nbr

        delta_u_plus_mod = m_minmod(delta_u_plus_ho, slope_forward, slope_backward)
        delta_u_minus_mod = m_minmod(delta_u_minus_ho, slope_forward, slope_backward)

        # Check if limiting occurred
        is_limited = abs(delta_u_plus_mod - delta_u_plus_ho) > tol || abs(delta_u_minus_mod - delta_u_minus_ho) > tol

        if is_limited
            # Reconstruct as P1 polynomial: u(r) = u_avg + slope * r
            limited_slope = (delta_u_plus_mod + delta_u_minus_mod) / 2.0
            @. u_k = u_avg_k + limited_slope * rd.r
        end
        # If not limited, u_k keeps its original P5 values
    end
end
end 