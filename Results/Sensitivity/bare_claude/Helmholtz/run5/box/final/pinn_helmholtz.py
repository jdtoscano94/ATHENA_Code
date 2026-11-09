"""PINN for 2D Helmholtz  u_xx + u_yy + k^2 u = g  on [-1,1]^2, periodic BCs in x and y.
Manufactured solution u = sin(pi*a1*x) sin(pi*a2*y), a1=1, a2=4, k=1.
Periodicity is enforced exactly with an input embedding (cos/sin of pi*x and pi*y, period 2).
Training sees only the PDE residual (the BCs hold by construction). The analytical solution is used only for evaluation.
"""
import json, time
import jax, jax.numpy as jnp, numpy as np, optax
jax.config.update("jax_enable_x64", True)

A1, A2, K = 1.0, 4.0, 1.0
PI = jnp.pi
def u_exact(x, y): return jnp.sin(PI*A1*x)*jnp.sin(PI*A2*y)
def g(x, y): return (K**2 - PI**2*(A1**2 + A2**2))*u_exact(x, y)

LAYERS = [4, 64, 64, 64, 64, 1]
def init(key):
    ps = []
    for i, (m, n) in enumerate(zip(LAYERS[:-1], LAYERS[1:])):
        key, k = jax.random.split(key)
        ps.append((jax.random.normal(k, (m, n))*jnp.sqrt(2.0/(m+n)), jnp.zeros(n)))
    return ps

def net(p, x, y):  # scalar x, y
    h = jnp.array([jnp.cos(PI*x), jnp.sin(PI*x), jnp.cos(PI*y), jnp.sin(PI*y)])
    for W, b in p[:-1]: h = jnp.tanh(h@W + b)
    W, b = p[-1]
    return (h@W + b)[0]

def residual(p, x, y):
    uxx = jax.grad(jax.grad(net, 1), 1)(p, x, y)
    uyy = jax.grad(jax.grad(net, 2), 2)(p, x, y)
    return uxx + uyy + K**2*net(p, x, y) - g(x, y)

res_v = jax.vmap(residual, (None, 0, 0))
net_v = jax.vmap(net, (None, 0, 0))
def loss(p, X): return jnp.mean(res_v(p, X[:, 0], X[:, 1])**2)

def evaluate(p, n=201):
    xs = jnp.linspace(-1, 1, n); Xg, Yg = jnp.meshgrid(xs, xs, indexing="ij")
    up = net_v(p, Xg.ravel(), Yg.ravel()).reshape(n, n); ue = u_exact(Xg, Yg)
    rel = float(jnp.linalg.norm(up-ue)/jnp.linalg.norm(ue)); mx = float(jnp.max(jnp.abs(up-ue)))
    # periodicity check (should be ~machine eps by construction)
    t = jnp.linspace(-1, 1, 501); o = jnp.ones_like(t)
    per = max(float(jnp.max(jnp.abs(net_v(p, -o, t)-net_v(p, o, t)))),
              float(jnp.max(jnp.abs(net_v(p, t, -o)-net_v(p, t, o)))))
    return rel, mx, per, np.array(Xg), np.array(Yg), np.array(up), np.array(ue)

if __name__ == "__main__":
    key = jax.random.PRNGKey(0); key, k0 = jax.random.split(key)
    p = init(k0); N = 8192; hist = []
    # --- Phase 1: Adam with resampled collocation points
    sched = optax.exponential_decay(1e-3, 2000, 0.8, end_value=1e-5)
    opt = optax.adam(sched); st = opt.init(p)
    @jax.jit
    def adam_step(p, st, X):
        l, gr = jax.value_and_grad(loss)(p, X)
        up, st = opt.update(gr, st, p); return optax.apply_updates(p, up), st, l
    t0 = time.time()
    for it in range(20001):
        key, k = jax.random.split(key)
        X = jax.random.uniform(k, (N, 2), minval=-1, maxval=1, dtype=jnp.float64)
        p, st, l = adam_step(p, st, X)
        if it % 1000 == 0:
            rel = evaluate(p)[0]; hist.append(("adam", it, float(l), rel))
            print(f"adam {it:6d} loss {float(l):.3e} relL2 {rel:.3e} t {time.time()-t0:.0f}s", flush=True)
    # --- Phase 2: L-BFGS on a fixed, larger point set
    key, k = jax.random.split(key)
    Xf = jax.random.uniform(k, (16384, 2), minval=-1, maxval=1, dtype=jnp.float64)
    lb = optax.lbfgs(memory_size=50); lst = lb.init(p)
    f = lambda q: loss(q, Xf); vg = optax.value_and_grad_from_state(f)
    @jax.jit
    def lb_step(p, lst):
        l, gr = vg(p, state=lst)
        up, lst = lb.update(gr, lst, p, value=l, grad=gr, value_fn=f)
        return optax.apply_updates(p, up), lst, l
    for it in range(1, 5001):
        p, lst, l = lb_step(p, lst)
        if it % 500 == 0:
            rel = evaluate(p)[0]; hist.append(("lbfgs", it, float(l), rel))
            print(f"lbfgs {it:5d} loss {float(l):.3e} relL2 {rel:.3e} t {time.time()-t0:.0f}s", flush=True)
    # --- Evaluation
    rel, mx, per, Xg, Yg, up, ue = evaluate(p)
    key, k = jax.random.split(key)
    Xt = jax.random.uniform(k, (20000, 2), minval=-1, maxval=1, dtype=jnp.float64)
    test_res = float(jnp.sqrt(loss(p, Xt)))
    out = dict(rel_L2=rel, max_abs_err=mx, periodicity_mismatch=per, test_pde_residual_rms=test_res,
               train_time_s=time.time()-t0, history=hist)
    print(json.dumps({k: v for k, v in out.items() if k != "history"}, indent=1))
    json.dump(out, open("/box/final/metrics.json", "w"), indent=1)
    np.savez("/box/final/solution.npz", x=Xg, y=Yg, u_pinn=up, u_exact=ue)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, Z, tt in zip(ax, [up, ue, np.abs(up-ue)], ["PINN u", "exact u", "|error|"]):
        c = a.pcolormesh(Xg, Yg, Z, shading="auto", cmap="RdBu_r" if tt != "|error|" else "viridis")
        fig.colorbar(c, ax=a); a.set_title(tt); a.set_aspect("equal"); a.set_xlabel("x"); a.set_ylabel("y")
    fig.tight_layout(); fig.savefig("/box/final/solution.png", dpi=120)
    fig, a = plt.subplots(figsize=(6, 4)); a.semilogy([h[3] for h in hist], "o-")
    a.set_xlabel("checkpoint (Adam every 1000 it, then L-BFGS every 500 it)"); a.set_ylabel("rel. L2 error")
    fig.tight_layout(); fig.savefig("/box/final/convergence.png", dpi=120)
