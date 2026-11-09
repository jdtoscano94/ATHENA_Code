import os, sys, time, json, argparse
import jax, jax.numpy as jnp, numpy as np, optax
from functools import partial
p = argparse.ArgumentParser()
p.add_argument('--steps', type=int, default=40000)
p.add_argument('--width', type=int, default=128)
p.add_argument('--depth', type=int, default=5)
p.add_argument('--M', type=int, default=8)        # Fourier modes in x
p.add_argument('--N', type=int, default=8192)     # collocation pts / step
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--opt', default='soap')
p.add_argument('--rba', type=int, default=1)      # residual-based attention weights
p.add_argument('--nu0', type=float, default=0.0)  # curriculum artificial viscosity start (decays to 0)
p.add_argument('--eps', type=float, default=0.0)  # causal weighting tolerance
p.add_argument('--Nt', type=int, default=32)
p.add_argument('--kw', type=float, default=0.0)  # WE-PINN shock weight 1/(1+kw(|u_x|-u_x))
p.add_argument('--decay', type=float, default=0.9)
p.add_argument('--wm', type=float, default=0.0)  # weight of mass-conservation loss (d/dt int u dx = 0)
p.add_argument('--sym', type=int, default=0)  # hard odd symmetry u(-x,t)=-u(x,t) (symmetry of PDE+IC)
p.add_argument('--nm', type=int, default=1024)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', default='/box/work/run')
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
REF = '/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes/Burgers_Periodic_SinPiX/2025_11_22_1730/burgers.npz'
d = np.load(REF); tr, xr, ur = d['t'], d['x'], d['usol']
TT, XX = np.meshgrid(tr, xr, indexing='ij')
Xe = jnp.array(np.stack([XX.ravel(), TT.ravel()], 1), jnp.float32)

def init(key):
    sizes = [2*a.M + 1] + [a.width]*a.depth + [1]
    ps = []
    for i in range(len(sizes)-1):
        key, k = jax.random.split(key)
        W = jax.random.normal(k, (sizes[i], sizes[i+1])) * jnp.sqrt(2/(sizes[i]+sizes[i+1]))
        ps.append((W, jnp.zeros(sizes[i+1])))
    # modified-MLP gates U,V
    key, k1, k2 = jax.random.split(key, 3)
    U = (jax.random.normal(k1, (sizes[0], a.width))*jnp.sqrt(2/(sizes[0]+a.width)), jnp.zeros(a.width))
    V = (jax.random.normal(k2, (sizes[0], a.width))*jnp.sqrt(2/(sizes[0]+a.width)), jnp.zeros(a.width))
    return {'layers': ps, 'U': U, 'V': V}

def feats(x, t):
    k = jnp.arange(1, a.M+1) * jnp.pi
    return jnp.concatenate([jnp.cos(k*x), jnp.sin(k*x), jnp.array([2*t-1])])  # exactly periodic in x

def net0(params, x, t):
    h0 = feats(x, t)
    u_ = jnp.tanh(h0 @ params['U'][0] + params['U'][1])
    v_ = jnp.tanh(h0 @ params['V'][0] + params['V'][1])
    h = h0
    for W, b in params['layers'][:-1]:
        z = jnp.tanh(h @ W + b)
        h = z*u_ + (1-z)*v_
    W, b = params['layers'][-1]
    o = (h @ W + b)[0]
    return -jnp.sin(jnp.pi*x) + t*o                  # hard initial condition

def net(params, x, t):
    if a.sym: return 0.5*(net0(params, x, t) - net0(params, -x, t))
    return net0(params, x, t)

def resid(params, x, t, nu):
    u = net(params, x, t)
    ux = jax.grad(net, 1)(params, x, t)
    ut = jax.grad(net, 2)(params, x, t)
    uxx = jax.grad(jax.grad(net, 1), 1)(params, x, t) if True else 0.
    r = ut + u*ux - nu*uxx
    w = jax.lax.stop_gradient(1.0/(1.0 + a.kw*(jnp.abs(ux) - ux)))
    return w*r

R = jax.vmap(resid, (None, 0, 0, None))
U = jax.jit(jax.vmap(net, (None, 0, 0)))

MASS0 = 0.0  # int_{-1}^{1} -sin(pi x) dx
def mass_loss(params, key):
    tm = jax.random.uniform(key, (64,))
    xm = (jnp.arange(a.nm) + jax.random.uniform(jax.random.fold_in(key, 1), ())) / a.nm * 2 - 1  # shifted uniform grid -> spectrally exact quadrature of periodic fn
    um = jax.vmap(lambda tt: U(params, xm, jnp.full_like(xm, tt)))(tm)
    return jnp.mean((2*jnp.mean(um, 1) - MASS0)**2)

def loss_fn(params, X, lam, nu, key):
    lm = mass_loss(params, key) if a.wm > 0 else 0.0
    r = R(params, X[:, 0], X[:, 1], nu)
    if a.eps > 0:   # causal weighting: X sorted into Nt equal time-chunks
        Lc = jnp.mean((lam*r).reshape(a.Nt, -1)**2, 1)
        w = jax.lax.stop_gradient(jnp.exp(-a.eps*jnp.concatenate([jnp.zeros(1), jnp.cumsum(Lc)[:-1]])))
        return jnp.mean(w*Lc) + a.wm*lm, (r, w.min())
    return jnp.mean((lam*r)**2) + a.wm*lm, (r, 1.0)

if a.opt == 'soap':
    from soap_jax import soap
    base = soap
else:
    base = optax.adam
sched = optax.warmup_exponential_decay_schedule(0, a.lr, 1000, 2000, a.decay, end_value=1e-6)
opt = optax.chain(optax.clip_by_global_norm(1.0), base(learning_rate=sched) if a.opt != 'soap' else soap(learning_rate=sched, b1=0.95, b2=0.95, weight_decay=0.0, precondition_frequency=2))
key = jax.random.PRNGKey(a.seed)
params = init(key); st = opt.init(params)
# fixed collocation set (for RBA weights), with periodic resample of a fraction
Ncol = a.N
key, k = jax.random.split(key)
Xc = jnp.concatenate([jax.random.uniform(k, (Ncol, 1), minval=-1, maxval=1),
                      jax.random.uniform(jax.random.fold_in(k, 1), (Ncol, 1))], 1)
lam = jnp.ones(Ncol)

@jax.jit
def step(params, st, lam, X, nu, key):
    (l, (r, wmin)), g = jax.value_and_grad(loss_fn, has_aux=True)(params, X, lam, nu, key)
    up, st = opt.update(g, st, params); params = optax.apply_updates(params, up)
    ra = jnp.abs(r)
    lam = 0.999*lam + 0.01*ra/jnp.max(ra) if a.rba else lam
    return params, st, lam, (l, wmin)

def err(params):
    up = np.array(U(params, Xe[:, 0], Xe[:, 1])).reshape(ur.shape)
    return np.linalg.norm(up-ur)/np.linalg.norm(ur), up

hist = []; t0 = time.time()
for it in range(a.steps+1):
    nu = a.nu0 * max(0.0, 1 - it/(0.6*a.steps))   # viscosity curriculum -> 0 for last 40% of steps
    if a.rba == 0:
        key, k1, k2 = jax.random.split(key, 3)
        tt = (jnp.arange(a.Nt)[:, None] + jax.random.uniform(k2, (a.Nt, Ncol//a.Nt))).reshape(-1, 1)/a.Nt  # stratified, sorted by chunk
        Xc = jnp.concatenate([jax.random.uniform(k1, (Ncol, 1), minval=-1, maxval=1), tt], 1)
    params, st, lam, l = step(params, st, lam, Xc, nu, jax.random.fold_in(key, it))
    if it % 1000 == 0:
        e, _ = err(params); l, wm = l
        hist.append((it, float(l), e, nu, float(wm))); print(f'{it} loss {float(l):.3e} wmin {float(wm):.2e} relL2 {e:.4e} nu {nu:.2e} {time.time()-t0:.0f}s', flush=True)
e, up = err(params)
np.save(f'{a.out}/hist.npy', np.array(hist)); np.save(f'{a.out}/upred.npy', up)
import pickle; pickle.dump(jax.device_get(params), open(f'{a.out}/params.pkl', 'wb'))
json.dump({'relL2': e, 'maxabs': float(np.abs(up-ur).max()), 'args': vars(a)}, open(f'{a.out}/result.json', 'w'))
print('FINAL relL2', e)
