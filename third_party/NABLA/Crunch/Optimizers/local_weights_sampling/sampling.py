"""Adaptive collocation-point sampling driven by vRBA weights.

Wiki: PIML > _shared/Optimization_Stage/Loss_function/Local_weights_sampling

Shared across stages.

Both functions below build the same PDF over the collocation pool from the
current ``lambdas`` (vRBA weights) and draw ``batch_size`` indices via
``jax.random.choice``. The exponent ``k`` sharpens the PDF and ``c`` adds a
uniform floor so high-residual regions never get fully starved of samples.

Two shapes are supported:

* ``sample_points_pdf(key, bs, lam, tc, xc, ti, xi, ui, k, c)`` — legacy 1D+t
  with IC pass-through. The IC tuple ``(ti, xi, ui)`` rides through unchanged
  so callers can keep a single ``train_data`` tuple in the training step.
  This is what Allen-Cahn and other 1D+t-with-IC cases use.

* ``sample_points_pdf_coords(key, bs, lam, *coords, k, c)`` — variadic in
  coords; every coord is indexed by ``idx_train``. Use this for cases that
  don't have an IC pass-through (steady 2D Helmholtz / Poisson, anything where
  every coord is part of the collocation pool). Compose IC / BC fields
  manually after the call:

      idx_train, sampled = sample_points_pdf_coords(key, bs, lam, tc, xc, k=k, c=c)
      train_data = (*sampled, ti, xi, ui)   # if a particular case wants IC pass-through

* ``sample_indices_pdf(key, bs, lam, k, c)`` — indices-only entry point. Returns
  just ``idx_train``. Use this when the caller does its own coord indexing
  (multi-domain cPINN dict-keyed orchestration, custom per-key bookkeeping).
  Accepts both 1D ``(N,)`` and 2D ``(N, m)`` lambdas; for 2D the per-column
  weights are summed first so each row gets a single sampling probability —
  this is what cPINN multi-component residuals (``(u, v, w)``) need.

All three share the same core (``_build_pdf_indices``) so the PDF construction
is defined exactly once.
"""
from functools import partial

import jax
import jax.numpy as jnp


def _build_pdf_indices(key, batch_size, lambdas, k, c):
    """PDF construction core — shared by all public entry points.

    Accepts ``lambdas`` of shape ``(N,)`` or ``(N, m)``. For the 2D case the
    per-column weights are summed across ``axis=1`` so each of the ``N`` rows
    gets one sampling probability (correct semantics for multi-component
    residuals; ``.flatten()`` would instead yield ``N*m`` probabilities and
    mismatch ``len(lambdas) = N`` in ``jax.random.choice``).
    """
    if lambdas.ndim > 1:
        lambdas = jnp.sum(lambdas, axis=tuple(range(1, lambdas.ndim)))
    lambdas_k = lambdas ** k
    lambdas_k = lambdas_k / (lambdas_k.mean() + 1e-20) + c
    p = lambdas_k / (lambdas_k.sum() + 1e-20)
    return jax.random.choice(key, len(lambdas), shape=(batch_size,), p=p)


@partial(jax.jit, static_argnames=("batch_size", "k", "c"))
def sample_indices_pdf(key, batch_size, lambdas, k=1, c=0.5):
    """Indices-only sampler. See module docstring."""
    return _build_pdf_indices(key, batch_size, lambdas, k, c)


@partial(jax.jit, static_argnames=("batch_size", "k", "c"))
def sample_points_pdf(key, batch_size, lambdas, tc, xc, ti, xi, ui, k=1, c=0.5):
    """Legacy 1D+t with IC pass-through. See module docstring."""
    idx_train = _build_pdf_indices(key, batch_size, lambdas, k, c)
    train_data = (tc[idx_train], xc[idx_train], ti, xi, ui)
    return idx_train, train_data


@partial(jax.jit, static_argnames=("batch_size", "k", "c"))
def sample_points_pdf_coords(key, batch_size, lambdas, *coords, k=1, c=0.5):
    """Variadic in coords — every coord is indexed by idx_train. See module docstring."""
    idx_train = _build_pdf_indices(key, batch_size, lambdas, k, c)
    return idx_train, tuple(a[idx_train] for a in coords)
