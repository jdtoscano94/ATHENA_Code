"""variational Residual-Based Adaptivity (vRBA) — local loss weighting.

Wiki: PIML > _shared/Optimization_Stage/Loss_function/Local_weights_sampling

Shared across stages — S1/S2/S3 just pass different hyperparameters
(``c_log``, ``gamma``, ``eta``, ``phi``, ``potential_type``).

Provides three primitives:

- ``get_exact_epsilon`` — Newton solver for the per-column potential scale ``eps``,
  enforcing the constraint ``mean(phi(r/eps), axis=0) = 1`` for ``cosh`` /
  ``superexp`` / ``logarithmic`` potentials. Inputs are shape ``(N, m)`` and the
  solver returns one ``eps`` per column ``(1, m)`` so multi-component residuals
  (e.g. ``(u, v, w)``) don't interfere with each other through a shared scale.
- ``update_vRBA`` — EMA update of per-collocation-point weights ``lambdas`` from
  current residuals ``r_i_norm``. Selectable potential ``phi``: ``exponential``
  (default), ``cosh``, ``superexp``, ``logarithmic``, ``logarithmic_simple``,
  ``lp``, ``quadratic``, ``linear``. Per-column ``r_max`` / ``q_max`` keep each
  residual component on its own scale.
- ``get_gamma`` — schedules the EMA decay ``gamma`` from a target ``max_RBA`` cap
  and an inflation rate ``eta``.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax


@partial(jit, static_argnums=(2, 3))
def get_exact_epsilon(values, r_max, potential_type, n_newton=20):
    """Solve for epsilon column-wise in float64.

    Parameters
    ----------
    values : (N, m) array of residual magnitudes.
    r_max  : (1, m) per-column max of ``values``.
    potential_type : 'cosh' | 'superexp' | 'logarithmic'.

    Returns
    -------
    (1, m) array of per-column epsilons.
    """
    N = values.shape[0]
    r_max_safe = jnp.maximum(r_max, 1e-16)

    if potential_type == 'cosh':
        eps_guess = r_max_safe / (jnp.log(2.0 * N) + 1e-16)
    elif potential_type == 'superexp':
        eps_guess = r_max_safe / jnp.sqrt(jnp.log(N) + 1e-16)
    elif potential_type == 'logarithmic':
        eps_guess = jnp.mean(values, axis=0, keepdims=True) / (jnp.e - 1.0)
        eps_guess = jnp.maximum(eps_guess, 1e-16)
    else:
        return jnp.ones_like(r_max)

    def newton_step(i, eps):
        eps_safe = jnp.maximum(eps, 1e-20)
        u = values / eps_safe
        if potential_type == 'cosh':
            s_val = jnp.sinh(u); c_val = jnp.cosh(u)
            val = jnp.mean(s_val, axis=0) - 1.0
            grad = jnp.mean(c_val * (-u / eps_safe), axis=0)
        elif potential_type == 'superexp':
            exp_u2 = jnp.exp(u ** 2)
            term = u * exp_u2
            val = jnp.mean(term, axis=0) - 1.0
            grad = jnp.mean(-(1.0 / eps_safe) * term * (1.0 + 2.0 * u ** 2), axis=0)
        elif potential_type == 'logarithmic':
            val = jnp.mean(jnp.log(u + 1.0), axis=0) - 1.0
            grad = jnp.mean(-(1.0 / eps_safe) * u / (u + 1.0), axis=0)
        else:
            val = 0.0; grad = 1.0
        return eps - val / (grad - 1e-20)

    epsilon_final = lax.fori_loop(0, n_newton, newton_step, eps_guess)
    return jnp.maximum(epsilon_final, 1e-16)


def update_vRBA(r_i_norm, lambdas_batch, it,
                c_log=1.0, gamma=0.999, eta=0.01, phi=1.0,
                potential_type='exponential', p_val=4.0):
    r_val = jnp.abs(r_i_norm)
    r_max = jnp.max(r_val, axis=0, keepdims=True)

    if potential_type == 'exponential':
        epsilon_q = c_log * r_max / jnp.log(it + 2.0)
        beta_it = 1.0 / (epsilon_q + 1e-16)
        q_it = beta_it * jnp.exp(beta_it * r_val)
    elif potential_type == 'cosh':
        epsilon_q = get_exact_epsilon(r_val, r_max, 'cosh', n_newton=20)
        beta_it = 1.0 / (epsilon_q + 1e-16)
        q_it = jnp.sinh(beta_it * r_val)
    elif potential_type == 'superexp':
        epsilon_q = get_exact_epsilon(r_val, r_max, 'superexp', n_newton=20)
        beta_it = 1.0 / (epsilon_q + 1e-16)
        u = beta_it * r_val
        q_it = u * jnp.exp(u ** 2)
    elif potential_type == 'logarithmic':
        epsilon_q = get_exact_epsilon(r_val, r_max, 'logarithmic', n_newton=20)
        beta_it = 1.0 / (epsilon_q + 1e-16)
        q_it = jnp.log(beta_it * r_val + 1.0)
    elif potential_type == 'logarithmic_simple':
        r_mean = jnp.mean(r_val, axis=0, keepdims=True)
        epsilon_q = r_mean / (jnp.e - 1.0)
        beta_it = 1.0 / (epsilon_q + 1e-16)
        q_it = jnp.log(beta_it * r_val + 1.0)
    elif potential_type == 'lp':
        q_it = jnp.power(r_val, p_val - 1.0)
    elif potential_type == 'quadratic':
        q_it = r_val
    elif potential_type == 'linear':
        q_it = jnp.ones_like(r_val)
        return q_it
    else:
        q_it = jnp.ones_like(r_val)

    q_max = jnp.max(q_it, axis=0, keepdims=True)
    q_norm = q_it / (q_max + 1e-20)
    lambda_it = phi * q_norm + (1.0 - phi)
    new_lambdas = gamma * lambdas_batch + eta * lambda_it
    return new_lambdas


def get_gamma(eta, max_RBA):
    return 1 - eta / max_RBA
