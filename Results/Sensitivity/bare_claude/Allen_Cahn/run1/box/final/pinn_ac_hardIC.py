"""PINN for Allen-Cahn: u_t - eps u_xx + k (u^3 - u) = 0, x in [-1,1] periodic, t in [0,1],
u(x,0) = x^2 cos(pi x), eps=1e-4, k=5.

Ingredients:
 * exact periodicity via Fourier embedding of x (cos/sin(pi m x)),
 * hard initial condition  u = u0(x) + t * N(x,t),
 * modified-MLP (Wang et al. 2021) with tanh,
 * causal residual weighting (Wang, Sankaran, Perdikaris 2022),
 * SOAP optimizer, fresh random collocation points every step.
Training uses ONLY the PDE residual (IC/BC are built into the ansatz).
The reference solution is loaded only for error evaluation.
"""
import sys, time, json, pickle
import numpy as np, jax, jax.numpy as jnp, optax, scipy.io
from soap_jax import soap

cfg = dict(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 60000, width=256, depth=4, M_x=10,
           n_chunks=32, pts_per_chunk=256, eps_causal=1.0, lr=1e-3, seed=0,
           out=sys.argv[2] if len(sys.argv) > 2 else "/box/final/run")
EPS, K = 1e-4, 5.0
REF = "/oscar/data/gk/jdtoscan/References/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms/Documents/codes/Data/allen_cahn.mat"

def glorot(key, i, o):
    return jax.random.normal(key, (i, o)) * jnp.sqrt(2.0 / (i + o))

def init(key):
    din = 2 * cfg["M_x"] + 1
    W = cfg["width"]
    ks = jax.random.split(key, cfg["depth"] + 4)
    p = {"U": (glorot(ks[0], din, W), jnp.zeros(W)), "V": (glorot(ks[1], din, W), jnp.zeros(W)),
         "H0": (glorot(ks[2], din, W), jnp.zeros(W)),
         "H": [(glorot(ks[3 + i], W, W), jnp.zeros(W)) for i in range(cfg["depth"] - 1)],
         "out": (glorot(ks[-1], W, 1), jnp.zeros(1))}
    return p

ms = jnp.arange(1, cfg["M_x"] + 1) * jnp.pi

def net(p, x, t):  # scalar x, t
    z = jnp.concatenate([jnp.cos(ms * x), jnp.sin(ms * x), jnp.array([2.0 * t - 1.0])])
    U = jnp.tanh(z @ p["U"][0] + p["U"][1]); V = jnp.tanh(z @ p["V"][0] + p["V"][1])
    h = jnp.tanh(z @ p["H0"][0] + p["H0"][1])
    h = h * U + (1 - h) * V
    for Wl, bl in p["H"]:
        h = jnp.tanh(h @ Wl + bl)
        h = h * U + (1 - h) * V
    return (h @ p["out"][0] + p["out"][1])[0]

def u0(x):
    return x ** 2 * jnp.cos(jnp.pi * x)

def u(p, x, t):
    return u0(x) + t * net(p, x, t)

def residual(p, x, t):
    ux_fn = lambda xx: jax.jvp(lambda s: u(p, s, t), (xx,), (1.0,))[1]
    uval, ut = jax.jvp(lambda s: u(p, x, s), (t,), (1.0,))
    uxx = jax.jvp(ux_fn, (x,), (1.0,))[1]
    return ut - EPS * uxx + K * (uval ** 3 - uval)

res_v = jax.vmap(residual, (None, 0, 0))
u_v = jax.jit(jax.vmap(u, (None, 0, 0)))

nc, npc = cfg["n_chunks"], cfg["pts_per_chunk"]

def sample(key):
    kx, kt = jax.random.split(key)
    x = jax.random.uniform(kx, (nc, npc), minval=-1.0, maxval=1.0)
    t = (jnp.arange(nc)[:, None] + jax.random.uniform(kt, (nc, npc))) / nc
    return x.ravel(), t.ravel()

def loss_fn(p, key, eps_c):
    x, t = sample(key)
    r = res_v(p, x, t).reshape(nc, npc)
    Li = jnp.mean(r ** 2, axis=1)
    w = jax.lax.stop_gradient(jnp.exp(-eps_c * jnp.cumsum(jnp.concatenate([jnp.zeros(1), Li[:-1]]))))
    return jnp.mean(w * Li), (Li, w)

def main():
    d = scipy.io.loadmat(REF)
    xr, tr, Uref = d["x"].ravel(), d["t"].ravel(), d["usol"]  # Uref[t, x]
    XX, TT = np.meshgrid(xr, tr)
    Xf, Tf = jnp.array(XX.ravel(), jnp.float32), jnp.array(TT.ravel(), jnp.float32)

    def rel_l2(p):
        up = np.asarray(u_v(p, Xf, Tf)).reshape(Uref.shape)
        return np.linalg.norm(up - Uref) / np.linalg.norm(Uref), up

    key = jax.random.PRNGKey(cfg["seed"])
    key, k0 = jax.random.split(key)
    p = init(k0)
    S = cfg["steps"]
    sched = optax.join_schedules([optax.linear_schedule(0, cfg["lr"], 2000),
                                  optax.exponential_decay(cfg["lr"], 2000, 0.9, end_value=1e-6)], [2000])
    opt = optax.chain(optax.clip_by_global_norm(1.0), soap(learning_rate=sched, b1=0.99, b2=0.999))
    st = opt.init(p)

    @jax.jit
    def step(p, st, key, eps_c):
        (l, (Li, w)), g = jax.value_and_grad(loss_fn, has_aux=True)(p, key, eps_c)
        upd, st = opt.update(g, st, p)
        return optax.apply_updates(p, upd), st, l, Li, w

    eps_c = cfg["eps_causal"]
    hist = []
    t0 = time.time()
    for it in range(S + 1):
        key, sk = jax.random.split(key)
        p, st, l, Li, w = step(p, st, sk, eps_c)
        if it % 1000 == 0:
            err, _ = rel_l2(p)
            wmin = float(w.min())
            hist.append(dict(it=it, loss=float(l), res=float(Li.mean()), wmin=wmin, eps_c=eps_c, rel_l2=float(err)))
            print(f"{it:6d} loss {float(l):.3e} res {float(Li.mean()):.3e} wmin {wmin:.3f} eps_c {eps_c:g} relL2 {err:.3e} ({time.time()-t0:.0f}s)", flush=True)
            # causality annealing: once all chunks are "active", tighten eps_c
            if wmin > 0.99 and eps_c < 100.0:
                eps_c *= 10.0
    err, up = rel_l2(p)
    print("FINAL rel L2", err, "max abs", np.abs(up - Uref).max())
    out = cfg["out"]
    np.savez(out + "_pred.npz", x=xr, t=tr, u_pred=up, u_ref=Uref)
    json.dump(dict(cfg=cfg, hist=hist, final_rel_l2=float(err), max_abs=float(np.abs(up - Uref).max())),
              open(out + "_hist.json", "w"), indent=1)
    pickle.dump(jax.device_get(p), open(out + "_params.pkl", "wb"))

if __name__ == "__main__":
    main()
