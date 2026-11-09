"""PINN for 2D Poisson: -Lap u = f on [0,1]^2, u = u_ex on boundary.
u_ex = sin(4 pi x) sin(4 pi y)  ->  f = 32 pi^2 sin(4 pi x) sin(4 pi y).
The exact solution is used ONLY for the boundary data (as required by the BC) and for error evaluation.
"""
import time, json
import jax, jax.numpy as jnp, numpy as np, optax
jax.config.update("jax_enable_x64", True)   # x64 available for the L-BFGS stage; Adam runs in float32
F32 = lambda t: jax.tree_util.tree_map(lambda a: a.astype(jnp.float32), t)
F64 = lambda t: jax.tree_util.tree_map(lambda a: a.astype(jnp.float64), t)

PI = jnp.pi
def u_exact(x, y): return jnp.sin(4*PI*x) * jnp.sin(4*PI*y)
def f_src(x, y):   return 32*PI**2 * jnp.sin(4*PI*x) * jnp.sin(4*PI*y)

# ---------------- network: Gaussian Fourier features + tanh MLP ----------------
key = jax.random.PRNGKey(0)
NF, SIGMA, WIDTH, DEPTH = 64, 2.0, 128, 4
key, kb = jax.random.split(key)
B = SIGMA * jax.random.normal(kb, (2, NF), dtype=jnp.float32)           # fixed (not trained)

def init_params(key):
    sizes = [2*NF] + [WIDTH]*DEPTH + [1]
    params = []
    for i in range(len(sizes)-1):
        key, k = jax.random.split(key)
        W = jax.random.normal(k, (sizes[i], sizes[i+1]), dtype=jnp.float32) * jnp.sqrt(2.0/(sizes[i]+sizes[i+1]))
        params.append((W, jnp.zeros(sizes[i+1], dtype=jnp.float32)))
    return params

def net(params, x, y):
    z = 2*PI*(jnp.stack([x, y]) @ B.astype(x.dtype))
    h = jnp.concatenate([jnp.sin(z), jnp.cos(z)])
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return (h @ W + b)[0]

def laplacian(params, x, y):
    uxx = jax.grad(jax.grad(net, 1), 1)(params, x, y)
    uyy = jax.grad(jax.grad(net, 2), 2)(params, x, y)
    return uxx + uyy

v_net = jax.vmap(net, (None, 0, 0))
v_lap = jax.vmap(laplacian, (None, 0, 0))

# ---------------- collocation points ----------------
N_INT, N_BC = 8192, 1024   # interior points, points per edge
LAM_BC = 100.0

def sample(key):
    k1, k2 = jax.random.split(key)
    xi = jax.random.uniform(k1, (N_INT, 2))
    t = jax.random.uniform(k2, (4, N_BC))
    z, o = jnp.zeros(N_BC), jnp.ones(N_BC)
    xb = jnp.concatenate([t[0], t[1], z, o]); yb = jnp.concatenate([z, o, t[2], t[3]])
    return F32((xi[:, 0], xi[:, 1], xb, yb))

def loss_fn(params, pts):
    xi, yi, xb, yb = pts
    r = -v_lap(params, xi, yi) - f_src(xi, yi)
    rb = v_net(params, xb, yb) - u_exact(xb, yb)
    # residual normalised by (32 pi^2) so both terms are O(1)
    return jnp.mean((r/(32*PI**2))**2) + LAM_BC*jnp.mean(rb**2), (jnp.mean(r**2), jnp.mean(rb**2))

# ---------------- evaluation ----------------
ng = 201
gx, gy = jnp.meshgrid(jnp.linspace(0, 1, ng), jnp.linspace(0, 1, ng), indexing="ij")
GX, GY = gx.ravel(), gy.ravel(); UEX = u_exact(GX, GY)
@jax.jit
def evaluate(params):
    up = v_net(params, GX, GY)
    return jnp.linalg.norm(up-UEX)/jnp.linalg.norm(UEX), jnp.max(jnp.abs(up-UEX))

# ---------------- stage 1: Adam with resampling ----------------
key, kp = jax.random.split(key)
params = init_params(kp)
N_ADAM = 100000
sched = optax.exponential_decay(1e-3, transition_steps=5000, decay_rate=0.75, end_value=1e-6)
opt = optax.adam(sched); st = opt.init(params)

@jax.jit
def adam_step(params, st, pts):
    (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, pts)
    upd, st = opt.update(g, st, params)
    return optax.apply_updates(params, upd), st, l, aux

hist = []
t0 = time.time()
for it in range(N_ADAM + 1):
    if it % 100 == 0:
        key, ks = jax.random.split(key); pts = sample(ks)
    params, st, l, aux = adam_step(params, st, pts)
    if it % 5000 == 0:
        e2, einf = evaluate(params)
        hist.append(dict(stage="adam", it=it, loss=float(l), pde_mse=float(aux[0]), bc_mse=float(aux[1]),
                         rel_l2=float(e2), linf=float(einf)))
        print(hist[-1], f"{time.time()-t0:.1f}s", flush=True)

# ---------------- stage 2: L-BFGS on a fixed (larger) point set ----------------
key, ks = jax.random.split(key)
N_INT, N_BC = 16384, 1024
pts = F64(sample(ks)); params = F64(params)
lb = optax.lbfgs(memory_size=50)
lst = lb.init(params)
fval = lambda p: loss_fn(p, pts)[0]
vg = optax.value_and_grad_from_state(fval)

@jax.jit
def lbfgs_step(params, lst):
    v, g = vg(params, state=lst)
    upd, lst = lb.update(g, lst, params, value=v, grad=g, value_fn=fval)
    return optax.apply_updates(params, upd), lst, v

for it in range(1, 5001):
    params, lst, v = lbfgs_step(params, lst)
    if it % 500 == 0:
        e2, einf = evaluate(params); _, aux = loss_fn(params, pts)
        hist.append(dict(stage="lbfgs", it=it, loss=float(v), pde_mse=float(aux[0]), bc_mse=float(aux[1]),
                         rel_l2=float(e2), linf=float(einf)))
        print(hist[-1], f"{time.time()-t0:.1f}s", flush=True)
train_time = time.time() - t0

# ---------------- final evaluation & outputs ----------------
e2, einf = evaluate(params)
up = np.array(v_net(params, GX, GY)).reshape(ng, ng); ue = np.array(UEX).reshape(ng, ng)
# independent check on cell-midpoint grid (offset from evaluation grid)
mx = (jnp.arange(400)+0.5)/400; MX, MY = [a.ravel() for a in jnp.meshgrid(mx, mx, indexing="ij")]
um = v_net(params, MX, MY); uem = u_exact(MX, MY)
e2m = float(jnp.linalg.norm(um-uem)/jnp.linalg.norm(uem))
# independent check on random points not used in training
kk = jax.random.PRNGKey(12345); xr = jax.random.uniform(kk, (100000, 2), dtype=jnp.float64)
ur = v_net(params, xr[:, 0], xr[:, 1]); uer = u_exact(xr[:, 0], xr[:, 1])
e2r = float(jnp.linalg.norm(ur-uer)/jnp.linalg.norm(uer))
res = dict(rel_l2_grid=float(e2), linf_grid=float(einf), rel_l2_random100k=e2r, rel_l2_midpoint400=e2m,
           mean_abs_err=float(np.mean(np.abs(up-ue))), train_time_s=train_time,
           config=dict(NF=NF, SIGMA=SIGMA, WIDTH=WIDTH, DEPTH=DEPTH, LAM_BC=LAM_BC, N_ADAM=N_ADAM))
print(res)
json.dump(dict(final=res, history=hist), open("/box/final/results.json", "w"), indent=1)
np.savez("/box/final/solution_grid.npz", x=np.array(gx), y=np.array(gy), u_pinn=up, u_exact=ue)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for a, d, t in zip(ax, [up, ue, np.abs(up-ue)], ["PINN u", "Exact u", "|error|"]):
    im = a.imshow(d.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis" if t != "|error|" else "magma")
    a.set_title(t); a.set_xlabel("x"); a.set_ylabel("y"); plt.colorbar(im, ax=a)
plt.tight_layout(); plt.savefig("/box/final/solution.png", dpi=120)
fig, a = plt.subplots(figsize=(6, 4))
its = np.arange(len(hist)); a.semilogy(its, [h["rel_l2"] for h in hist], label="rel L2 error")
a.semilogy(its, [h["loss"] for h in hist], label="training loss")
a.axvline(N_ADAM//5000 + 0.5, color="gray", ls="--"); a.set_xlabel("checkpoint (Adam every 5k | L-BFGS every 500)")
a.legend(); plt.tight_layout(); plt.savefig("/box/final/convergence.png", dpi=120)
