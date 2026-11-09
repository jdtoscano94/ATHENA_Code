"""PINN for 2D Helmholtz u_xx+u_yy+k^2 u = g on [-1,1]^2, periodic BCs (hard-enforced
via fundamental-period Fourier embedding). Exact solution used ONLY for error evaluation."""
import jax, jax.numpy as jnp, numpy as np, optax, time, json, sys
jax.config.update("jax_enable_x64", True)
a1, a2, k = 1.0, 4.0, 1.0
pi = jnp.pi
def u_exact(x, y): return jnp.sin(a1*pi*x)*jnp.sin(a2*pi*y)
def g(x, y): return (k**2 - (a1*pi)**2 - (a2*pi)**2)*jnp.sin(a1*pi*x)*jnp.sin(a2*pi*y)

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
key = jax.random.PRNGKey(seed)
layers = [4, 64, 64, 64, 64, 1]
def init(key):
    ps = []
    for din, dout in zip(layers[:-1], layers[1:]):
        key, sk = jax.random.split(key)
        ps.append((jax.random.normal(sk, (din, dout))*jnp.sqrt(2/(din+dout)), jnp.zeros(dout)))
    return ps
def net(p, x, y):
    # period-2 embedding: exact periodicity of u and all derivatives in x and y
    h = jnp.stack([jnp.cos(pi*x), jnp.sin(pi*x), jnp.cos(pi*y), jnp.sin(pi*y)])
    for W, b in p[:-1]: h = jnp.tanh(h @ W + b)
    W, b = p[-1]; return (h @ W + b)[0]
def residual(p, x, y):
    uxx = jax.grad(jax.grad(net, 1), 1)(p, x, y)
    uyy = jax.grad(jax.grad(net, 2), 2)(p, x, y)
    return uxx + uyy + k**2*net(p, x, y) - g(x, y)
res_v = jax.vmap(residual, (None, 0, 0))
scale = float((a1*pi)**2 + (a2*pi)**2 - k**2)  # normalise residual
def loss(p, X): return jnp.mean((res_v(p, X[:, 0], X[:, 1])/scale)**2)

def sample(key, n): return jax.random.uniform(key, (n, 2), minval=-1., maxval=1.)
params = init(key)
N = 8000
# --- Adam with resampling
sched = optax.exponential_decay(1e-3, 2000, 0.8)
opt = optax.adam(sched); st = opt.init(params)
@jax.jit
def adam_step(p, st, X):
    l, gr = jax.value_and_grad(loss)(p, X)
    up, st = opt.update(gr, st, p); return optax.apply_updates(p, up), st, l
hist = []
t0 = time.time()
for it in range(15000):
    if it % 100 == 0: key, sk = jax.random.split(key); X = sample(sk, N)
    params, st, l = adam_step(params, st, X)
    if it % 1000 == 0: print(f"adam {it} loss {l:.3e} t={time.time()-t0:.0f}s", flush=True); hist.append(("adam", it, float(l)))
# --- L-BFGS on fixed larger set
key, sk = jax.random.split(key); X = sample(sk, 16000)
lb = optax.lbfgs(); lst = lb.init(params)
vg = optax.value_and_grad_from_state(lambda p: loss(p, X))
@jax.jit
def lb_step(p, s):
    l, gr = vg(p, state=s)
    up, s = lb.update(gr, s, p, value=l, grad=gr, value_fn=lambda q: loss(q, X))
    return optax.apply_updates(p, up), s, l
for it in range(5000):
    params, lst, l = lb_step(params, lst)
    if it % 500 == 0: print(f"lbfgs {it} loss {l:.3e} t={time.time()-t0:.0f}s", flush=True); hist.append(("lbfgs", it, float(l)))

# --- evaluation (exact solution used only here)
n = 201
xs = jnp.linspace(-1, 1, n); XX, YY = jnp.meshgrid(xs, xs, indexing="ij")
U = jax.vmap(net, (None, 0, 0))(params, XX.ravel(), YY.ravel()).reshape(n, n)
UE = u_exact(XX, YY)
rel_l2 = float(jnp.linalg.norm(U-UE)/jnp.linalg.norm(UE)); linf = float(jnp.max(jnp.abs(U-UE)))
R = res_v(params, XX.ravel(), YY.ravel())
# periodicity check on boundary (value and normal derivative)
t = jnp.linspace(-1, 1, 401); one = jnp.ones_like(t)
nv = jax.vmap(net, (None, 0, 0)); dx = jax.vmap(jax.grad(net, 1), (None, 0, 0)); dy = jax.vmap(jax.grad(net, 2), (None, 0, 0))
per = dict(ux=float(jnp.max(jnp.abs(nv(params, -one, t)-nv(params, one, t)))),
           uy=float(jnp.max(jnp.abs(nv(params, t, -one)-nv(params, t, one)))),
           dux=float(jnp.max(jnp.abs(dx(params, -one, t)-dx(params, one, t)))),
           duy=float(jnp.max(jnp.abs(dy(params, t, -one)-dy(params, t, one)))))
out = dict(seed=seed, rel_l2=rel_l2, linf=linf, final_loss=float(l),
           pde_res_rms=float(jnp.sqrt(jnp.mean(R**2))), periodicity_max_jump=per, time_s=time.time()-t0)
print(json.dumps(out, indent=1))
json.dump(dict(out, history=hist), open(f"/box/final/results_seed{seed}.json", "w"), indent=1)
np.savez(f"/box/final/solution_seed{seed}.npz", x=np.array(xs), u_pinn=np.array(U), u_exact=np.array(UE))
