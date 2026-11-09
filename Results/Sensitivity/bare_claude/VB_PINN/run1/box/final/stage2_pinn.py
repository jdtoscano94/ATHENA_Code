"""Stage 2: PINN for viscous Burgers, trained on PDE residual + IC + periodic BC ONLY.
The Stage-1 reference is loaded solely to monitor/measure error (never enters the loss).
 - Periodic BC enforced exactly: x enters only via Fourier features cos(k pi x), sin(k pi x).
 - IC enforced exactly: u(x,t) = -sin(pi x) + t * N(x,t)   (hard constraint).
 - Modified MLP (Wang et al. 2021), tanh; causal residual weighting (Wang et al. 2022);
   SOAP optimizer with warmup + exponential decay; fresh random collocation points each step."""
import jax, jax.numpy as jnp, numpy as np, optax, time, json, sys, pickle
from soap_jax import soap
jax.config.update("jax_default_matmul_precision", "highest")
cfg = dict(K=10, width=256, depth=5, steps=60000, batch_t=64, batch_x=256, n_chunks=32,
           eps_causal=1.0, lr=2e-3, decay_rate=0.9, decay_steps=2000, warmup=1000, seed=0, opt="soap",
           tag="run")
for a in sys.argv[1:]:
    k, v = a.split("="); cfg[k] = type(cfg[k])(v) if not isinstance(cfg[k], bool) else v == "True"
print(cfg, flush=True)
nu = 1.0/(100*np.pi)
ref = np.load("/box/final/burgers_reference.npz")
Xr, Tr = np.meshgrid(ref["x"][::8], ref["t"], indexing="ij")         # 512 x 201 monitor grid
Ur = ref["u"][::8, :]

def init(key):
    din = 2*cfg["K"] + 1; W = cfg["width"]
    def dense(k, i, o): return dict(W=jax.random.normal(k, (i, o))*np.sqrt(2/(i+o)), b=jnp.zeros(o))
    ks = jax.random.split(key, cfg["depth"] + 3)
    return dict(U=dense(ks[0], din, W), V=dense(ks[1], din, W),
                hidden=[dense(ks[2+i], din if i == 0 else W, W) for i in range(cfg["depth"])],
                out=dense(ks[-1], W, 1))

kpi = jnp.pi*jnp.arange(1, cfg["K"]+1)
def net(p, x, t):                       # scalar x,t -> scalar u
    feat = jnp.concatenate([jnp.cos(kpi*x), jnp.sin(kpi*x), jnp.array([2*t-1])])
    U = jnp.tanh(feat@p["U"]["W"] + p["U"]["b"]); V = jnp.tanh(feat@p["V"]["W"] + p["V"]["b"])
    h = feat
    for l in p["hidden"]:
        z = jnp.tanh(h@l["W"] + l["b"]); h = z*U + (1-z)*V
    return -jnp.sin(jnp.pi*x) + t*(h@p["out"]["W"] + p["out"]["b"])[0]

def residual(p, x, t):
    u = net(p, x, t)
    ux = jax.grad(net, 1)(p, x, t); ut = jax.grad(net, 2)(p, x, t)
    uxx = jax.grad(jax.grad(net, 1), 1)(p, x, t)
    return ut + u*ux - nu*uxx
res_v = jax.vmap(residual, (None, 0, 0))
u_v = jax.jit(jax.vmap(net, (None, 0, 0)))

def loss_fn(p, key):
    M, bt, bx = cfg["n_chunks"], cfg["batch_t"], cfg["batch_x"]
    k1, k2 = jax.random.split(key)
    # stratified t: bt/M samples per chunk... we use M chunks each with (bt//M or 1) times x bx points
    tt = (jnp.arange(M)[:, None] + jax.random.uniform(k1, (M, bt//M if bt >= M else 1)))/M   # (M, m)
    xx = jax.random.uniform(k2, (M, tt.shape[1], bx), minval=-1, maxval=1)
    T = jnp.broadcast_to(tt[..., None], xx.shape)
    r = res_v(p, xx.ravel(), T.ravel()).reshape(M, -1)
    Lc = jnp.mean(r**2, axis=1)                                  # per-chunk residual loss
    w = jax.lax.stop_gradient(jnp.exp(-cfg["eps_causal"]*jnp.cumsum(Lc) + cfg["eps_causal"]*Lc))
    return jnp.mean(w*Lc), (Lc, w)

sched = optax.warmup_exponential_decay_schedule(0, cfg["lr"], cfg["warmup"], cfg["decay_steps"], cfg["decay_rate"])
opt = soap(learning_rate=sched) if cfg["opt"] == "soap" else optax.adam(sched)
opt = optax.chain(optax.clip_by_global_norm(1.0), opt)

@jax.jit
def step(p, s, key):
    (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, key)
    up, s = opt.update(g, s, p); return optax.apply_updates(p, up), s, l, aux

def rel_l2(p):
    up = np.asarray(u_v(p, jnp.array(Xr.ravel()), jnp.array(Tr.ravel()))).reshape(Xr.shape)
    return np.linalg.norm(up-Ur)/np.linalg.norm(Ur), np.abs(up-Ur).max()

key = jax.random.PRNGKey(cfg["seed"]); p = init(key); s = opt.init(p)
hist = []; t0 = time.time(); best = (1e9, None)
for it in range(cfg["steps"]+1):
    key, sk = jax.random.split(key)
    p, s, l, (Lc, w) = step(p, s, sk)
    if it % 1000 == 0:
        e, emax = rel_l2(p)                     # monitoring only
        hist.append(dict(it=it, loss=float(l), res=float(Lc.mean()), wmin=float(w.min()), relL2=float(e), maxerr=float(emax)))
        print(f"{it:6d} loss {l:.3e} res {Lc.mean():.3e} wmin {w.min():.3f} relL2 {e:.3e} max {emax:.3e} [{time.time()-t0:.0f}s]", flush=True)
pickle.dump(jax.device_get(p), open(f"/box/final/pinn_params_{cfg['tag']}.pkl", "wb"))
json.dump(dict(cfg=cfg, hist=hist), open(f"/box/final/pinn_hist_{cfg['tag']}.json", "w"), indent=1)
