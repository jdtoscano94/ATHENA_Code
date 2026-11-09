"""Stage 3: evaluate trained PINN vs Stage-1 spectral reference (full 4096x201 grid),
time slices, error norms, periodicity/smoothness checks, residual map, figures."""
import sys, json, pickle, numpy as np, jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
jax.config.update("jax_default_matmul_precision", "highest")
tag = sys.argv[1]
sys.argv = [sys.argv[0]] + [f"{k}={v}" for k, v in json.load(open(f"/box/final/pinn_hist_{tag}.json"))["cfg"].items()] + ["steps=-1"]
# reuse model definition from stage2 without training (steps=-1 -> loop body skipped except save)
import importlib.util
src = open("/box/final/stage2_pinn.py").read().split("key = jax.random.PRNGKey")[0]
g = {}; exec(src, g)
p = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(f"/box/final/pinn_params_{tag}.pkl", "rb")))
net, u_v, res_v = g["net"], g["u_v"], jax.jit(g["res_v"])
ref = np.load("/box/final/burgers_reference.npz"); x, t, U = ref["x"], ref["t"], ref["u"]
X, T = np.meshgrid(x, t, indexing="ij")
P = np.concatenate([np.asarray(u_v(p, jnp.array(X[:, j]), jnp.array(T[:, j]))) [:, None] for j in range(len(t))], 1)
E = P - U
out = {}
out["rel_L2_spacetime"] = float(np.linalg.norm(E)/np.linalg.norm(U))
out["max_abs_err"] = float(np.abs(E).max()); out["mean_abs_err"] = float(np.abs(E).mean())
i, j = np.unravel_index(np.abs(E).argmax(), E.shape); out["max_err_location"] = [float(x[i]), float(t[j])]
out["rel_L2_vs_ColeHopf"] = float(np.linalg.norm(P-ref["u_colehopf"])/np.linalg.norm(ref["u_colehopf"]))
slices = {}
for ts in [0.0, 0.25, 0.5, 0.75, 1.0]:
    jj = int(round(ts*200)); e = E[:, jj]
    slices[ts] = dict(relL2=float(np.linalg.norm(e)/np.linalg.norm(U[:, jj])), maxabs=float(np.abs(e).max()))
out["time_slices"] = slices
# shock region vs smooth region (|x|<0.05 after t>0.3)
shock = (np.abs(X) < 0.05) & (T > 0.3)
out["max_err_shock_region"] = float(np.abs(E[shock]).max()); out["max_err_elsewhere"] = float(np.abs(E[~shock]).max())
out["rel_L2_excluding_shock_region"] = float(np.linalg.norm(E[~shock])/np.linalg.norm(U[~shock]))
# arbitrary off-grid points (random) vs reference via spectral interpolation of the stage-1 field
rng = np.random.default_rng(1); xs = rng.uniform(-1, 1, 2000); js = rng.integers(0, len(t), 2000)
Uhat = np.fft.rfft(U[:, js].T, axis=1); k = np.pi*np.arange(Uhat.shape[1])
ue = (Uhat[:, 0].real + 2*np.real((Uhat[:, 1:]*np.exp(1j*k[None, 1:]*(xs[:, None]+1))).sum(1)))/len(x)
up = np.asarray(u_v(p, jnp.array(xs), jnp.array(t[js])))
out["offgrid_random_rel_L2"] = float(np.linalg.norm(up-ue)/np.linalg.norm(ue))
# periodicity checks
tt = jnp.linspace(0, 1, 101)
ux = jax.vmap(jax.grad(net, 1), (None, 0, 0)); uxx = jax.vmap(jax.grad(jax.grad(net, 1), 1), (None, 0, 0))
m1, p1 = -jnp.ones_like(tt), jnp.ones_like(tt)
out["periodicity_max|u(-1)-u(1)|"] = float(jnp.abs(u_v(p, m1, tt)-u_v(p, p1, tt)).max())
out["periodicity_max|ux(-1)-ux(1)|"] = float(jnp.abs(ux(p, m1, tt)-ux(p, p1, tt)).max())
out["IC_max_err"] = float(np.abs(E[:, 0]).max())
# PDE residual on the grid (subsampled x) and gradient comparison
R = np.concatenate([np.asarray(res_v(p, jnp.array(X[::4, j]), jnp.array(T[::4, j])))[:, None] for j in range(len(t))], 1)
out["pde_residual_rms"] = float(np.sqrt((R**2).mean())); out["pde_residual_max"] = float(np.abs(R).max())
Ux_ref = np.real(np.fft.irfft(1j*np.pi*np.fft.rfftfreq(len(x), 1/len(x))[:, None]*np.fft.rfft(U, axis=0), n=len(x), axis=0))
Ux_p = np.concatenate([np.asarray(ux(p, jnp.array(X[:, j]), jnp.array(T[:, j])))[:, None] for j in range(len(t))], 1)
out["min_ux_ref"] = float(Ux_ref.min()); out["min_ux_pinn"] = float(Ux_p.min())
out["rel_L2_ux"] = float(np.linalg.norm(Ux_p-Ux_ref)/np.linalg.norm(Ux_ref))
json.dump(out, open(f"/box/final/metrics_{tag}.json", "w"), indent=1); print(json.dumps(out, indent=1))
# figures
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for a, Z, ttl, cm in [(ax[0], U, "Reference (spectral)", "RdBu_r"), (ax[1], P, "PINN", "RdBu_r"), (ax[2], np.abs(E), "|PINN - ref|", "viridis")]:
    im = a.pcolormesh(t, x, Z, cmap=cm, shading="auto", **({} if cm == "viridis" else dict(vmin=-1, vmax=1)))
    a.set_xlabel("t"); a.set_ylabel("x"); a.set_title(ttl); plt.colorbar(im, ax=a)
plt.tight_layout(); plt.savefig(f"/box/final/fig_fields_{tag}.png", dpi=130); plt.close()
fig, ax = plt.subplots(2, 5, figsize=(20, 6.5), sharex=True)
for c, ts in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
    jj = int(round(ts*200))
    ax[0, c].plot(x, U[:, jj], "k-", lw=2, label="reference"); ax[0, c].plot(x, P[:, jj], "r--", lw=1.5, label="PINN")
    ax[0, c].set_title(f"t = {ts}"); ax[1, c].semilogy(x, np.abs(E[:, jj]) + 1e-12, color="C0"); ax[1, c].set_xlabel("x")
ax[0, 0].legend(); ax[0, 0].set_ylabel("u"); ax[1, 0].set_ylabel("|error|")
plt.tight_layout(); plt.savefig(f"/box/final/fig_slices_{tag}.png", dpi=130); plt.close()
h = json.load(open(f"/box/final/pinn_hist_{tag}.json"))["hist"]
fig, a = plt.subplots(figsize=(6, 4)); it = [r["it"] for r in h]
a.semilogy(it, [r["res"] for r in h], label="mean PDE residual loss"); a.semilogy(it, [r["relL2"] for r in h], label="rel. L2 error vs ref (monitor only)")
a.set_xlabel("iteration"); a.legend(); plt.tight_layout(); plt.savefig(f"/box/final/fig_training_{tag}.png", dpi=130); plt.close()
