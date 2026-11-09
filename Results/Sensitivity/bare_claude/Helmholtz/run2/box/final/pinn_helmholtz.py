"""PINN for 2D Helmholtz  u_xx + u_yy + k^2 u = g  on [-1,1]^2, periodic in x and y.
a1=1, a2=4, k=1;  g derived from u_an = sin(pi a1 x) sin(pi a2 y) (used ONLY for g and for error evaluation).
Periodicity is enforced exactly (hard constraint) through the input embedding
  phi(x,y) = [cos(pi x), sin(pi x), cos(pi y), sin(pi y)]   (period 2 = domain length),
so u and all derivatives match on opposite edges. Only the fundamental frequency is embedded;
the network must synthesize the sin(4 pi y) content itself.
"""
import json, time, sys
import numpy as np
import jax, jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)
A1, A2, K = 1.0, 4.0, 1.0
PI = jnp.pi
LAM = (A1**2 + A2**2) * np.pi**2 - K**2          # g = -LAM * u_an
SCALE = LAM                                       # residual normalisation

def u_exact(x, y): return jnp.sin(PI*A1*x) * jnp.sin(PI*A2*y)
def g(x, y): return -LAM * u_exact(x, y)          # = (u_an)_xx + (u_an)_yy + k^2 u_an

WIDTH, DEPTH = int(sys.argv[1]) if len(sys.argv) > 1 else 64, 4
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
W_MEAN = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0

def init(key):
    sizes = [4] + [WIDTH]*DEPTH + [1]
    params = []
    for i in range(len(sizes)-1):
        key, k1 = jax.random.split(key)
        W = jax.random.normal(k1, (sizes[i], sizes[i+1])) * jnp.sqrt(2.0/(sizes[i]+sizes[i+1]))
        params.append((W, jnp.zeros(sizes[i+1])))
    return params

def net(params, x, y):
    h = jnp.stack([jnp.cos(PI*x), jnp.sin(PI*x), jnp.cos(PI*y), jnp.sin(PI*y)])
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return (h @ W + b)[0]

def residual(params, x, y):
    uxx = jax.grad(jax.grad(net, 1), 1)(params, x, y)
    uyy = jax.grad(jax.grad(net, 2), 2)(params, x, y)
    return (uxx + uyy + K**2 * net(params, x, y) - g(x, y)) / SCALE

# uniform periodic grid (endpoint excluded): trapezoid mean is spectrally accurate for periodic integrands
_q = jnp.linspace(-1, 1, 65)[:-1]; QX, QY = [a.ravel() for a in jnp.meshgrid(_q, _q, indexing="ij")]
def loss(params, pts):
    r = jax.vmap(residual, (None, 0, 0))(params, pts[:, 0], pts[:, 1])
    rq = jax.vmap(residual, (None, 0, 0))(params, QX, QY)
    # mean-residual term: integral of the PDE over the periodic cell (Laplacian integrates to 0)
    # gives k^2 <u> = <g>; the constant mode is otherwise only weakly penalised (eigenvalue k^2 << LAM).
    return jnp.mean(r**2) + W_MEAN * jnp.mean(rq)**2

# ---------------- evaluation grid (analytic solution used only here) ----------------
ng = 201
xs = jnp.linspace(-1, 1, ng); X, Y = jnp.meshgrid(xs, xs, indexing="ij")
Ue = u_exact(X, Y)
def evaluate(params):
    U = jax.vmap(jax.vmap(net, (None, 0, 0)), (None, 0, 0))(params, X, Y)
    err = U - Ue
    return float(jnp.linalg.norm(err)/jnp.linalg.norm(Ue)), float(jnp.max(jnp.abs(err))), U

key = jax.random.PRNGKey(SEED)
params = init(key)
N = 8192
t0 = time.time()
# ---------------- Adam with resampling ----------------
n_adam = 20000
sched = optax.exponential_decay(2e-3, 2000, 0.8, end_value=1e-5)
opt = optax.adam(sched); st = opt.init(params)
@jax.jit
def adam_step(params, st, pts):
    l, gr = jax.value_and_grad(loss)(params, pts)
    up, st = opt.update(gr, st, params)
    return optax.apply_updates(params, up), st, l
hist = []
for it in range(n_adam):
    if it % 100 == 0:
        key, k1 = jax.random.split(key)
        pts = jax.random.uniform(k1, (N, 2), minval=-1, maxval=1)
    params, st, l = adam_step(params, st, pts)
    if it % 1000 == 0 or it == n_adam-1:
        rl2, mx, _ = evaluate(params)
        hist.append(("adam", it, float(l), rl2, mx))
        print(f"adam {it:6d} loss {float(l):.3e} relL2 {rl2:.3e} maxerr {mx:.3e}", flush=True)

# ---------------- L-BFGS on a fixed (larger) point set ----------------
key, k1 = jax.random.split(key)
pts = jax.random.uniform(k1, (16384, 2), minval=-1, maxval=1)
lb = optax.lbfgs(memory_size=50)
lst = lb.init(params)
vg = optax.value_and_grad_from_state(lambda p: loss(p, pts))
@jax.jit
def lbfgs_step(params, lst):
    l, gr = vg(params, state=lst)
    up, lst = lb.update(gr, lst, params, value=l, grad=gr, value_fn=lambda p: loss(p, pts))
    return optax.apply_updates(params, up), lst, l
n_lb = 5000
for it in range(n_lb):
    params, lst, l = lbfgs_step(params, lst)
    if it % 500 == 0 or it == n_lb-1:
        rl2, mx, _ = evaluate(params)
        hist.append(("lbfgs", it, float(l), rl2, mx))
        print(f"lbfgs {it:5d} loss {float(l):.3e} relL2 {rl2:.3e} maxerr {mx:.3e}", flush=True)
train_time = time.time() - t0

# ---------------- final checks ----------------
rl2, mx, U = evaluate(params)
# held-out residual on fresh points
key, k1 = jax.random.split(key)
tp = jax.random.uniform(k1, (20000, 2), minval=-1, maxval=1)
test_res = float(loss(params, tp))
# periodicity check: u and normal derivatives on opposite edges
s = jnp.linspace(-1, 1, 101); one = jnp.ones_like(s)
vn = jax.vmap(net, (None, 0, 0)); dx = jax.vmap(jax.grad(net, 1), (None, 0, 0)); dy = jax.vmap(jax.grad(net, 2), (None, 0, 0))
per = max(float(jnp.max(jnp.abs(vn(params, -one, s) - vn(params, one, s)))),
          float(jnp.max(jnp.abs(vn(params, s, -one) - vn(params, s, one)))),
          float(jnp.max(jnp.abs(dx(params, -one, s) - dx(params, one, s)))),
          float(jnp.max(jnp.abs(dy(params, s, -one) - dy(params, s, one)))))
res = dict(width=WIDTH, depth=DEPTH, seed=SEED, w_mean=W_MEAN, rel_L2=rl2, max_abs_err=mx,
           test_residual_mse_normalised=test_res, periodicity_max_mismatch=per,
           train_time_s=train_time, grid=f"{ng}x{ng}")
print(json.dumps(res, indent=1))
tag = f"w{WIDTH}_s{SEED}_m{W_MEAN:g}"
json.dump(dict(result=res, history=hist), open(f"/box/final/results_{tag}.json", "w"), indent=1)
np.savez(f"/box/final/solution_{tag}.npz", x=np.array(xs), U_pinn=np.array(U), U_exact=np.array(Ue))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for a, Z, t in zip(ax, [U, Ue, jnp.abs(U-Ue)], ["PINN u", "exact u", "|error|"]):
    im = a.imshow(np.array(Z).T, origin="lower", extent=[-1, 1, -1, 1], cmap="RdBu_r" if t != "|error|" else "viridis")
    a.set_title(t); a.set_xlabel("x"); a.set_ylabel("y"); fig.colorbar(im, ax=a)
fig.suptitle(f"Helmholtz periodic, rel L2 = {rl2:.2e}"); fig.tight_layout()
fig.savefig(f"/box/final/solution_{tag}.png", dpi=120)
fig, a = plt.subplots(figsize=(6, 4))
h = np.array([[r[2], r[3]] for r in hist]); a.semilogy(h[:, 0], label="residual loss"); a.semilogy(h[:, 1], label="rel L2 error")
a.set_xlabel("checkpoint (Adam every 1000 it, then L-BFGS every 500 it)"); a.legend(); fig.tight_layout()
fig.savefig(f"/box/final/history_{tag}.png", dpi=120)
