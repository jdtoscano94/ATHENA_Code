"""Residual MSE of the Opus 5.5 (Claude Code) PINN runs, computed the same way as ATHENA's Table 1 MSE.

ATHENA's MSE = mean(pde_residual_grid**2), the raw (unscaled) PDE residual evaluated on the grid stored in each
ATHENA result file (Plots/PINNs.ipynb). Here the same residual formula is evaluated on the same grid for the
Opus networks, in float64:
  Allen-Cahn  u_t - 1e-4 u_xx + 5u^3 - 5u           512 x 501 on [-1,1] x [0,1]     (Allen_Cahn/run1 final_s0_params.pkl)
  Burgers     u_t + u u_x - (0.01/pi) u_xx          2001 x 1001 on [-1,1] x [0,1]   (VB_PINN/run1 pinn_params_B.pkl)
  KdV         u_t + 6 u u_x + u_xxx                 201 x 101 on [0,20] x [0,5]     (KdV/run1 model_A.pkl, 10 windows)
  Inv.Burgers u_t + u u_x                           256 x 200 on [-1,1] x [0,1]     (IB_PINN/run1 model/params.pkl)
Helmholtz run2 and Poisson saved no network parameters, only the predicted field, so their residual is
computed from that field with derivatives accurate far below the residual level:
  Helmholtz   u_xx + u_yy + u - g   (Fourier spectral derivatives; the network is exactly periodic)   200 x 200
  Poisson     -u_xx - u_yy - f      (10th-order central / one-sided finite differences)            201 x 201
Run inside athena_release.sif (needs jax + soap-free code only):  python opus_residual_mse.py
"""
import pickle, sys, json
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

B = '../Results/Sensitivity/bare_claude/'
f64 = lambda p: jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float64), p)
out = {}

def grid_eval(res, xs, ts):
    X, T = np.meshgrid(xs, ts, indexing='ij')
    f = jax.jit(jax.vmap(res)); xf, tf = X.ravel(), T.ravel(); cs = 65536
    return np.concatenate([np.asarray(f(jnp.asarray(xf[i:i + cs]), jnp.asarray(tf[i:i + cs])))
                           for i in range(0, len(xf), cs)])

# ---------------- Allen-Cahn (copied from pinn_ac.py, cfg of final_s0) ----------------
p = f64(pickle.load(open(B + 'Allen_Cahn/run1/box/final/final_s0_params.pkl', 'rb')))
ms = jnp.arange(1, 11) * jnp.pi
def ac_net(x, t):
    z = jnp.concatenate([jnp.cos(ms * x), jnp.sin(ms * x), jnp.array([2.0 * t - 1.0])])
    U = jnp.tanh(z @ p["U"][0] + p["U"][1]); V = jnp.tanh(z @ p["V"][0] + p["V"][1])
    h = jnp.tanh(z @ p["H0"][0] + p["H0"][1]); h = h * U + (1 - h) * V
    for Wl, bl in p["H"]:
        h = jnp.tanh(h @ Wl + bl); h = h * U + (1 - h) * V
    return (h @ p["out"][0] + p["out"][1])[0]
def ac_res(x, t):
    u = ac_net(x, t); ut = jax.grad(ac_net, 1)(x, t); uxx = jax.grad(jax.grad(ac_net, 0), 0)(x, t)
    return ut - 1e-4 * uxx + 5.0 * u ** 3 - 5.0 * u
r = grid_eval(ac_res, np.linspace(-1, 1, 512), np.linspace(0, 1, 501)); out['Allen-Cahn'] = np.mean(r ** 2)

# ---------------- Viscous Burgers (copied from stage2_pinn.py, cfg B: K=10, width 256, depth 5) ----------------
p = f64(pickle.load(open(B + 'VB_PINN/run1/box/final/pinn_params_B.pkl', 'rb')))
kpi = jnp.pi * jnp.arange(1, 11)
def vb_net(x, t):
    feat = jnp.concatenate([jnp.cos(kpi * x), jnp.sin(kpi * x), jnp.array([2 * t - 1])])
    U = jnp.tanh(feat @ p["U"]["W"] + p["U"]["b"]); V = jnp.tanh(feat @ p["V"]["W"] + p["V"]["b"])
    h = feat
    for l in p["hidden"]:
        z = jnp.tanh(h @ l["W"] + l["b"]); h = z * U + (1 - z) * V
    return -jnp.sin(jnp.pi * x) + t * (h @ p["out"]["W"] + p["out"]["b"])[0]
nu = 1.0 / (100 * np.pi)
def vb_res(x, t):
    u = vb_net(x, t); ux = jax.grad(vb_net, 0)(x, t); ut = jax.grad(vb_net, 1)(x, t)
    uxx = jax.grad(jax.grad(vb_net, 0), 0)(x, t)
    return ut + u * ux - nu * uxx
r = grid_eval(vb_res, np.linspace(-1, 1, 2001), np.linspace(0, 1, 1001)); out['Burgers'] = np.mean(r ** 2)

# ---------------- Inviscid Burgers (copied from pinn.py, args M=16, sym=1) ----------------
p = f64(pickle.load(open(B + 'IB_PINN/run1/box/final/model/params.pkl', 'rb')))
kib = jnp.arange(1, 17) * jnp.pi
def ib_net0(x, t):
    h0 = jnp.concatenate([jnp.cos(kib * x), jnp.sin(kib * x), jnp.array([2 * t - 1])])
    u_ = jnp.tanh(h0 @ p['U'][0] + p['U'][1]); v_ = jnp.tanh(h0 @ p['V'][0] + p['V'][1])
    h = h0
    for W, b in p['layers'][:-1]:
        z = jnp.tanh(h @ W + b); h = z * u_ + (1 - z) * v_
    W, b = p['layers'][-1]
    return -jnp.sin(jnp.pi * x) + t * (h @ W + b)[0]
ib_net = lambda x, t: 0.5 * (ib_net0(x, t) - ib_net0(-x, t))
def ib_res(x, t):
    u = ib_net(x, t); ux = jax.grad(ib_net, 0)(x, t); ut = jax.grad(ib_net, 1)(x, t)
    return ut + u * ux
r = grid_eval(ib_res, np.linspace(-1, 1, 256), np.linspace(0, 1, 200)); out['Inviscid Burgers'] = np.mean(r ** 2)

# ---------------- KdV (lib.py make_net, model_A.pkl: 10 windows of dT = 0.5) ----------------
m = pickle.load(open(B + 'KdV/run1/box/final/model_A.pkl', 'rb'))
NW, dT, Lk = m['NW'], 5.0 / m['NW'], 20.0
xs, ts = np.linspace(0, 20, 201), np.linspace(0, 5, 101)
r_all = np.zeros((201, 101))
for j, t in enumerate(ts):
    k = min(int(t / dT), NW - 1)
    t0, pk = m['models'][k]; pk = f64(pk)
    def kn(x, tt, pk=pk, t0=t0):
        h = jnp.stack([x / Lk * 2 - 1, (tt - t0) / dT * 2 - 1])
        for W, b in pk[:-1]: h = jnp.tanh(h @ W + b)
        W, b = pk[-1]; return (h @ W + b)[0]
    def kr(x, tt, kn=kn):
        u = kn(x, tt); ux = jax.grad(kn, 0)(x, tt); ut = jax.grad(kn, 1)(x, tt)
        uxxx = jax.grad(jax.grad(jax.grad(kn, 0), 0), 0)(x, tt)
        return ut + 6.0 * u * ux + uxxx
    r_all[:, j] = np.asarray(jax.vmap(kr, (0, None))(jnp.asarray(xs), t))
out['KdV'] = np.mean(r_all ** 2)

# ---------------- Helmholtz run2 seed 0 (spectral derivatives of the saved periodic field) ----------------
d = np.load(B + 'Helmholtz/run2/box/final/solution_w64_s0_m100.npz')
x = d['x']; U = d['U_pinn']                  # 201 x 201 on [-1,1], endpoint included
x0 = x[:-1]; Up = U[:-1, :-1]                # one period
n = len(x0); kk = 2 * np.pi * np.fft.fftfreq(n, d=2.0 / n)
Uh = np.fft.fft2(Up)
lap = np.real(np.fft.ifft2(-(kk[:, None] ** 2 + kk[None, :] ** 2) * Uh))
Xg, Yg = np.meshgrid(x0, x0, indexing='ij')  # check orientation against the exact field below
Ue = d['U_exact'][:-1, :-1]
g_ij = -(17 * np.pi ** 2 - 1) * np.sin(np.pi * Xg) * np.sin(4 * np.pi * Yg)
g_ji = -(17 * np.pi ** 2 - 1) * np.sin(np.pi * Yg) * np.sin(4 * np.pi * Xg)
g = g_ij if np.abs(Ue - g_ij / -(17 * np.pi ** 2 - 1)).max() < 1e-10 else g_ji
r = lap + Up - g; out['Helmholtz'] = np.mean(r ** 2)
lap_e = np.real(np.fft.ifft2(-(kk[:, None] ** 2 + kk[None, :] ** 2) * np.fft.fft2(Ue)))
print('Helmholtz derivative check on exact field: residual MSE %.2e' % np.mean((lap_e + Ue - g) ** 2))

# ---------------- Poisson (10th-order finite differences of the saved field) ----------------
d = np.load(B + 'Poisson/run1/box/final/solution_grid.npz')
xg, yg, Uq, Ueq = d['x'], d['y'], d['u_pinn'], d['u_exact']
h = xg[1, 0] - xg[0, 0] if xg[1, 0] != xg[0, 0] else xg[0, 1] - xg[0, 0]
def d2(u, axis, h, order=10):
    # second derivative, central stencil inside, one-sided stencils at the edges (Fornberg weights)
    from numpy.polynomial import polynomial as P
    def weights(z, x0):  # Fornberg for 2nd derivative at x0 with nodes z
        nz = len(z); A = np.vander(np.asarray(z, float) - x0, nz, increasing=True).T
        b = np.zeros(nz); b[2] = 2.0
        return np.linalg.solve(A, b)
    u = np.moveaxis(u, axis, 0); N = u.shape[0]; npts = order + 1; half = npts // 2
    out_ = np.zeros_like(u)
    for i in range(N):
        s = min(max(i - half, 0), N - npts - 1)
        idx = np.arange(s, s + npts + 1)
        w = weights(idx, i)
        out_[i] = np.tensordot(w, u[idx], axes=(0, 0)) / h ** 2
    return np.moveaxis(out_, 0, axis)
f_src = 32 * np.pi ** 2 * np.sin(4 * np.pi * xg) * np.sin(4 * np.pi * yg)
r = -(d2(Uq, 0, h) + d2(Uq, 1, h)) - f_src; out['Poisson'] = np.mean(r ** 2)
re = -(d2(Ueq, 0, h) + d2(Ueq, 1, h)) - f_src
print('Poisson derivative check on exact field: residual MSE %.2e' % np.mean(re ** 2))

print(json.dumps({k: float('%.3e' % v) for k, v in out.items()}, indent=1))
