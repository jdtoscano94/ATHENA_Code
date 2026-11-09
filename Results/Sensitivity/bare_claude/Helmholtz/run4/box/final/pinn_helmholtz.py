"""PINN for 2D Helmholtz  u_xx + u_yy + k^2 u = g  on [-1,1]^2, periodic BCs in x and y.
Manufactured solution u = sin(pi a1 x) sin(pi a2 y), a1=1, a2=4, k=1 (used ONLY for g and error evaluation).

Periodicity is imposed exactly (hard constraint) via a Fourier input embedding
  phi(x,y) = [cos(pi m x), sin(pi m x), cos(pi n y), sin(pi n y)],  m,n = 1..M
which makes the network 2-periodic in x and y with all derivatives continuous.
A periodic mismatch is also evaluated (as a diagnostic) to confirm this.
Loss: mean squared (normalised) strong residual + W_MEAN*(domain-mean residual)^2.
Training: Adam (resampled collocation) -> L-BFGS (fixed collocation).
"""
import json, time, sys
import jax, jax.numpy as jnp, numpy as np, optax
jax.config.update("jax_enable_x64", True)

A1, A2, K = 1.0, 4.0, 1.0
PI = jnp.pi
def u_exact(x, y): return jnp.sin(PI*A1*x)*jnp.sin(PI*A2*y)
def g_src(x, y): return (K**2 - (PI*A1)**2 - (PI*A2)**2)*u_exact(x, y)   # source only, from the PDE
G_SCALE = float((PI*A1)**2 + (PI*A2)**2 - K**2)                        # loss normalisation

M = int(sys.argv[1]) if len(sys.argv) > 1 else 4    # number of Fourier harmonics in embedding
WIDTH, DEPTH = 64, 4
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0

def init(key):
    sizes = [4*M] + [WIDTH]*DEPTH + [1]
    params = []
    for i in range(len(sizes)-1):
        key, k = jax.random.split(key)
        W = jax.random.normal(k, (sizes[i], sizes[i+1]))*jnp.sqrt(2.0/(sizes[i]+sizes[i+1]))
        params.append((W, jnp.zeros(sizes[i+1])))
    return params

ms = jnp.arange(1, M+1, dtype=jnp.float64)
def net(params, x, y):
    h = jnp.concatenate([jnp.cos(PI*ms*x), jnp.sin(PI*ms*x), jnp.cos(PI*ms*y), jnp.sin(PI*ms*y)])
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return (h @ W + b)[0]

def residual(params, x, y):
    uxx = jax.grad(jax.grad(net, 1), 1)(params, x, y)
    uyy = jax.grad(jax.grad(net, 2), 2)(params, x, y)
    return uxx + uyy + K**2*net(params, x, y) - g_src(x, y)
res_v = jax.vmap(residual, (None, 0, 0))
net_v = jax.vmap(net, (None, 0, 0))

# Weak-form term (test function v=1): domain average of the residual, computed with the
# trapezoidal rule on a uniform periodic grid (spectrally accurate for periodic integrands).
# It is unnormalised so the constant mode (operator eigenvalue k^2=1 << G_SCALE) is controlled.
W_MEAN = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
_q = jnp.linspace(-1, 1, 64, endpoint=False)
_QX, _QY = [a.ravel() for a in jnp.meshgrid(_q, _q, indexing="ij")]
def loss(params, pts):
    r = res_v(params, pts[:, 0], pts[:, 1]) / G_SCALE
    l = jnp.mean(r**2)
    if W_MEAN > 0:
        l = l + W_MEAN*jnp.mean(res_v(params, _QX, _QY))**2
    return l

def periodic_err(params, n=400):
    s = jnp.linspace(-1, 1, n)
    o = jnp.ones(n)
    ex = net_v(params, -o, s) - net_v(params, o, s)
    ey = net_v(params, s, -o) - net_v(params, s, o)
    dx = jax.vmap(jax.grad(net, 1), (None, 0, 0))
    dy = jax.vmap(jax.grad(net, 2), (None, 0, 0))
    edx = dx(params, -o, s) - dx(params, o, s)
    edy = dy(params, s, -o) - dy(params, s, o)
    return float(jnp.max(jnp.abs(jnp.concatenate([ex, ey, edx, edy]))))

def evaluate(params, n=201):
    s = jnp.linspace(-1, 1, n)
    X, Y = jnp.meshgrid(s, s, indexing="ij")
    up = net_v(params, X.ravel(), Y.ravel()).reshape(n, n)
    ue = u_exact(X, Y)
    rel_l2 = float(jnp.linalg.norm(up-ue)/jnp.linalg.norm(ue))
    linf = float(jnp.max(jnp.abs(up-ue)))
    return rel_l2, linf, np.asarray(X), np.asarray(Y), np.asarray(up), np.asarray(ue)

def main():
    key = jax.random.PRNGKey(SEED)
    params = init(key)
    hist = []
    t0 = time.time()
    # ---- Adam with resampling ----
    N, n_adam = 8192, 20000
    sched = optax.exponential_decay(1e-3, 2000, 0.7, end_value=1e-5)
    opt = optax.adam(sched); st = opt.init(params)
    @jax.jit
    def step(params, st, key):
        pts = jax.random.uniform(key, (N, 2), minval=-1, maxval=1)
        l, gr = jax.value_and_grad(loss)(params, pts)
        up, st = opt.update(gr, st, params)
        return optax.apply_updates(params, up), st, l
    for it in range(n_adam):
        key, k = jax.random.split(key)
        params, st, l = step(params, st, k)
        if it % 1000 == 0:
            e = evaluate(params)[0]; hist.append((it, float(l), e))
            print(f"adam {it:6d} loss {float(l):.3e} relL2 {e:.3e}  t={time.time()-t0:.0f}s", flush=True)
    # ---- L-BFGS on fixed collocation set ----
    key, k = jax.random.split(key)
    pts = jax.random.uniform(k, (16384, 2), minval=-1, maxval=1)
    lfun = lambda p: loss(p, pts)
    lb = optax.lbfgs(memory_size=50)
    lst = lb.init(params)
    vg = optax.value_and_grad_from_state(lfun)
    @jax.jit
    def lstep(params, lst):
        v, gr = vg(params, state=lst)
        up, lst = lb.update(gr, lst, params, value=v, grad=gr, value_fn=lfun)
        return optax.apply_updates(params, up), lst, v
    for it in range(5000):
        params, lst, l = lstep(params, lst)
        if it % 500 == 0:
            e = evaluate(params)[0]; hist.append((n_adam+it, float(l), e))
            print(f"lbfgs {it:5d} loss {float(l):.3e} relL2 {e:.3e}  t={time.time()-t0:.0f}s", flush=True)
    # ---- evaluation ----
    rel_l2, linf, X, Y, up, ue = evaluate(params, 201)
    key, k = jax.random.split(key)
    test = jax.random.uniform(k, (20000, 2), minval=-1, maxval=1)
    test_res = float(jnp.sqrt(jnp.mean(res_v(params, test[:, 0], test[:, 1])**2)))
    per = periodic_err(params)
    out = dict(M=M, w_mean=W_MEAN, seed=SEED, width=WIDTH, depth=DEPTH, rel_l2=rel_l2, linf=linf, final_train_loss=float(l),
               test_residual_rms=test_res, max_periodic_mismatch_u_and_grad=per, time_s=time.time()-t0)
    print(json.dumps(out, indent=1))
    tag = f"M{M}_w{W_MEAN:g}" + (f"_s{SEED}" if SEED else "")
    json.dump(out, open(f"/box/final/metrics_{tag}.json", "w"), indent=1)
    np.savez(f"/box/final/solution_{tag}.npz", X=X, Y=Y, u_pred=up, u_exact=ue, hist=np.array(hist))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    for a, d, t in zip(ax[:3], [ue, up, np.abs(up-ue)], ["exact", "PINN", "|error|"]):
        c = a.pcolormesh(X, Y, d, shading="auto", cmap="RdBu_r" if t != "|error|" else "viridis")
        fig.colorbar(c, ax=a); a.set_title(t); a.set_aspect("equal")
    h = np.array(hist)
    ax[3].semilogy(h[:, 0], h[:, 1], label="train loss"); ax[3].semilogy(h[:, 0], h[:, 2], label="rel L2")
    ax[3].axvline(n_adam, ls="--", c="gray"); ax[3].legend(); ax[3].set_xlabel("iteration")
    fig.suptitle(f"Helmholtz PINN: rel L2 = {rel_l2:.2e}, Linf = {linf:.2e}")
    fig.tight_layout(); fig.savefig(f"/box/final/results_{tag}.png", dpi=110)

if __name__ == "__main__":
    main()
