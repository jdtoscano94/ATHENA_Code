"""PINN for 2D Helmholtz  u_xx + u_yy + k^2 u = g  on [-1,1]^2, periodic BCs in x and y.
Manufactured solution u = sin(pi*a1*x) sin(pi*a2*y), a1=1, a2=4, k=1 (used for g and for error only).
Periodic BCs are imposed softly: u, u_x, u_y must match on opposite edges.
"""
import time, json, sys
import jax, jax.numpy as jnp, numpy as np, optax
jax.config.update("jax_enable_x64", True)

A1, A2, K = 1.0, 4.0, 1.0
PI = jnp.pi
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MODE = sys.argv[2] if len(sys.argv) > 2 else "hard"  # "hard": periodic embedding; "soft": raw (x,y) + BC penalty
LAYERS = [4 if MODE == "hard" else 2, 64, 64, 64, 64, 1]
N_COL, N_BC = 4096, 256
ADAM_STEPS, LBFGS_STEPS = 20000, 10000
W_BC = 10.0
SCALE = (A1**2 + A2**2) * PI**2  # residual normalisation

def u_exact(x, y): return jnp.sin(A1*PI*x) * jnp.sin(A2*PI*y)
def g(x, y): return (K**2 - (A1**2 + A2**2) * PI**2) * u_exact(x, y)

def init(key):
    params = []
    for din, dout in zip(LAYERS[:-1], LAYERS[1:]):
        key, k1 = jax.random.split(key)
        params.append((jax.random.normal(k1, (din, dout)) * jnp.sqrt(2.0/(din+dout)), jnp.zeros(dout)))
    return params

def net(p, x, y):
    if MODE == "hard":  # exact 2-periodicity via fundamental-mode embedding only
        h = jnp.array([jnp.cos(PI*x), jnp.sin(PI*x), jnp.cos(PI*y), jnp.sin(PI*y)])
    else:
        h = jnp.array([x, y])
    for W, b in p[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = p[-1]
    return (h @ W + b)[0]

def residual(p, x, y):
    uxx = jax.grad(jax.grad(net, 1), 1)(p, x, y)
    uyy = jax.grad(jax.grad(net, 2), 2)(p, x, y)
    return uxx + uyy + K**2 * net(p, x, y) - g(x, y)

def bc_terms(p, s):
    """periodic mismatch (used as loss in soft mode, diagnostic only in hard mode) at x=-1 vs x=1 (param y=s) and y=-1 vs y=1 (param x=s)"""
    ux = jax.grad(net, 1); uy = jax.grad(net, 2)
    o = jnp.ones_like(s)
    v = jax.vmap
    dx = [v(f, (None, 0, 0))(p, -o, s) - v(f, (None, 0, 0))(p, o, s) for f in (net, ux, uy)]
    dy = [v(f, (None, 0, 0))(p, s, -o) - v(f, (None, 0, 0))(p, s, o) for f in (net, ux, uy)]
    # derivative mismatches scaled by characteristic wavenumber
    sc = [1.0, 1.0/(A1*PI), 1.0/(A2*PI)]
    return sum(jnp.mean((d*c)**2) for d, c in zip(dx, sc)) + sum(jnp.mean((d*c)**2) for d, c in zip(dy, sc))

def loss_fn(p, xc, yc, s):
    r = jax.vmap(residual, (None, 0, 0))(p, xc, yc) / SCALE
    lr = jnp.mean(r**2); lb = bc_terms(p, s)
    return lr + (W_BC * lb if MODE == "soft" else 0.0), (lr, lb)

def sample(key):
    k1, k2, k3 = jax.random.split(key, 3)
    xc = jax.random.uniform(k1, (N_COL,), minval=-1, maxval=1)
    yc = jax.random.uniform(k2, (N_COL,), minval=-1, maxval=1)
    s = jax.random.uniform(k3, (N_BC,), minval=-1, maxval=1)
    return xc, yc, s

# evaluation grid (exact solution used ONLY here)
ng = 201
gx = jnp.linspace(-1, 1, ng); X, Y = jnp.meshgrid(gx, gx, indexing="ij")
Ue = u_exact(X, Y)
def rel_l2(p):
    U = jax.vmap(jax.vmap(net, (None, 0, 0)), (None, 0, 0))(p, X, Y)
    return float(jnp.linalg.norm(U - Ue) / jnp.linalg.norm(Ue)), float(jnp.max(jnp.abs(U - Ue))), U

key = jax.random.PRNGKey(SEED)
key, k0 = jax.random.split(key)
params = init(k0)
hist = []

# ---- Adam with resampling ----
sched = optax.exponential_decay(1e-3, 2000, 0.8, end_value=1e-5)
opt = optax.adam(sched); st = opt.init(params)
@jax.jit
def adam_step(p, st, xc, yc, s):
    (l, aux), gr = jax.value_and_grad(loss_fn, has_aux=True)(p, xc, yc, s)
    up, st = opt.update(gr, st, p)
    return optax.apply_updates(p, up), st, l, aux
t0 = time.time()
for it in range(ADAM_STEPS):
    if it % 100 == 0:
        key, ks = jax.random.split(key); batch = sample(ks)
    params, st, l, (lr, lb) = adam_step(params, st, *batch)
    if it % 1000 == 0 or it == ADAM_STEPS - 1:
        e, emax, _ = rel_l2(params)
        hist.append(dict(phase="adam", it=it, loss=float(l), lres=float(lr), lbc=float(lb), rel_l2=e))
        print(f"adam {it:6d} loss {l:.3e} res {lr:.3e} bc {lb:.3e} relL2 {e:.3e}  t={time.time()-t0:.0f}s", flush=True)

# ---- L-BFGS on a fixed, larger point set ----
N_COL, N_BC = 16384, 1024
key, ks = jax.random.split(key); batch = sample(ks)
f = lambda p: loss_fn(p, *batch)[0]
lb_opt = optax.lbfgs(memory_size=50)
lst = lb_opt.init(params)
vg = optax.value_and_grad_from_state(f)
@jax.jit
def lbfgs_step(p, s):
    v, gr = vg(p, state=s)
    up, s = lb_opt.update(gr, s, p, value=v, grad=gr, value_fn=f)
    return optax.apply_updates(p, up), s, v
for it in range(LBFGS_STEPS):
    params, lst, l = lbfgs_step(params, lst)
    if it % 1000 == 0 or it == LBFGS_STEPS - 1:
        lr, lbv = loss_fn(params, *batch)[1]
        e, emax, _ = rel_l2(params)
        hist.append(dict(phase="lbfgs", it=it, loss=float(l), lres=float(lr), lbc=float(lbv), rel_l2=e))
        print(f"lbfgs {it:6d} loss {l:.3e} res {lr:.3e} bc {lbv:.3e} relL2 {e:.3e}  t={time.time()-t0:.0f}s", flush=True)

e, emax, U = rel_l2(params)
# independent residual check on fresh random points (unnormalised)
key, ks = jax.random.split(key); xc, yc, s = sample(ks)
rr = jax.vmap(residual, (None, 0, 0))(params, xc, yc)
res = dict(seed=SEED, mode=MODE, rel_l2=e, max_abs_err=emax, rms_pde_residual_fresh=float(jnp.sqrt(jnp.mean(rr**2))),
           rms_g=float(jnp.sqrt(jnp.mean(g(xc, yc)**2))), bc_mismatch_fresh=float(bc_terms(params, s)),
           train_time_s=time.time()-t0)
print(json.dumps(res, indent=1))
json.dump(dict(result=res, history=hist), open(f"/box/final/results_{MODE}_seed{SEED}.json", "w"), indent=1)
np.savez(f"/box/final/solution_{MODE}_seed{SEED}.npz", X=np.array(X), Y=np.array(Y), U_pinn=np.array(U), U_exact=np.array(Ue))
