import numpy as np, json, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from burgers2d import *

res = {}
# --- sanity of exact solution: implicit relation u = u0(s - u t) and shock location/jump
s = np.linspace(0, 4*np.pi, 20001)
for t in [0.5, 1.0, 1.5, 2.0]:
    ue = exact_1d(s, t)
    resid = np.abs(ue - (1/3 + 2/3*np.sin(s - ue*t))).max()
    res[f"exact_char_residual_t{t}"] = float(resid)
ue = exact_1d(np.array([np.pi + 2/3 - 1e-9, np.pi + 2/3 + 1e-9]), 2.0)  # shock at s = pi + t/3
res["exact_shock_states_t2"] = ue.tolist()

# --- convergence study
Ns = [64, 128, 256, 512, 1024]
times = [1.0, 1.5, 2.0]
def one(N):
    t0 = time.time()
    _, snaps = solve(N, 2.0, snaps=times)
    row = {}
    for t in times:
        ue = exact_nodes(N, t); e = snaps[t] - ue
        row[t] = dict(L1=float(np.mean(np.abs(e))), L2=float(np.sqrt(np.mean(e**2))),
                      Linf=float(np.abs(e).max()))
    u2 = snaps[2.0]
    row["mass_drift"] = float(abs(u2.mean() - 1/3))
    row["max"] = float(u2.max()); row["min"] = float(u2.min())
    row["sym_err"] = float(np.abs(u2 - u2.T).max())
    row["wall"] = time.time() - t0
    print(N, row, flush=True)
    return row

from multiprocessing import Pool
with Pool(len(Ns)) as p:
    conv = dict(zip(Ns, p.map(one, Ns[::-1])[::-1]))
res["convergence"] = {str(k): {str(a): b for a, b in v.items()} for k, v in conv.items()}
for t in times:
    for nrm in ["L1", "L2", "Linf"]:
        e = [conv[N][t][nrm] for N in Ns]
        res[f"order_{nrm}_t{t}"] = [float(np.log2(e[i]/e[i+1])) for i in range(len(e)-1)]
json.dump(res, open("results.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k.startswith("order") or k.startswith("exact")}, indent=1))

# --- snapshot figure at N=512
N = 512; ts = [0.0, 0.5, 1.0, 1.5, 1.75, 2.0]
_, snaps = solve(N, 2.0, snaps=ts[1:]); snaps[0.0] = exact_nodes(N, 0.0)
np.savez_compressed("snapshots_N512.npz", **{f"t{t}": snaps[t].astype(np.float32) for t in ts})
xc = (np.arange(N)+0.5)*L/N
fig = plt.figure(figsize=(18, 10))
for k, t in enumerate(ts):
    ax = fig.add_subplot(3, 6, k+1)
    im = ax.imshow(snaps[t].T, origin="lower", extent=[0, L, 0, L], cmap="viridis", vmin=-1/3, vmax=1)
    ax.set_title(f"u(x,y), t={t}"); ax.set_xlabel("x"); ax.set_ylabel("y")
fig.colorbar(im, ax=fig.axes[:6], shrink=0.8)
# diagonal slice y = x (s = x) compared to exact
for k, t in enumerate(ts):
    ax = fig.add_subplot(3, 6, 7+k)
    ax.plot(xc, np.diag(snaps[t]), "o", ms=2, label="WENO5 N=512")
    sf = np.linspace(0, L, 4000); ax.plot(sf, exact_1d(sf, t), "k-", lw=1, label="exact")
    ax.set_title(f"diagonal y=x, t={t}"); ax.set_xlabel("x"); ax.set_ylim(-0.45, 1.1)
    if k == 0: ax.legend(fontsize=7)
# convergence + error map
ax = fig.add_subplot(3, 3, 7)
for t in times:
    ax.loglog(Ns, [conv[n][t]["L1"] for n in Ns], "o-", label=f"L1, t={t}")
ax.loglog(Ns, 3e3*np.array(Ns, float)**-5, "k--", label="N^-5")
ax.loglog(Ns, 2*np.array(Ns, float)**-1, "k:", label="N^-1")
ax.set_xlabel("N"); ax.set_ylabel("error vs exact"); ax.legend(fontsize=7); ax.set_title("grid convergence")
ax = fig.add_subplot(3, 3, 8)
e = snaps[2.0] - exact_nodes(N, 2.0)
im = ax.imshow(np.log10(np.abs(e).T+1e-16), origin="lower", extent=[0, L, 0, L], cmap="magma", vmin=-12, vmax=0)
fig.colorbar(im, ax=ax); ax.set_title("log10|u - u_exact|, t=2, N=512")
ax = fig.add_subplot(3, 3, 9)
for t in ts:
    ax.plot(xc, np.diag(snaps[t]), label=f"t={t}")
ax.set_xlim(2.5, 5.5); ax.set_title("zoom: shock formation (diagonal)"); ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("summary_all.png", dpi=110)
