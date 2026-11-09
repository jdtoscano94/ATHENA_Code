import numpy as np, json, sys
sys.path.insert(0, '/box/final')
from burgers2d import *
times = [0.0, 0.5, 1.0, 1.5, 1.75, 2.0]
res = {}
for N in [64, 128, 256, 512, 1024]:
    sn = solve(N, times)
    res[N] = {}
    for t in times:
        ue = point_exact(N, t)
        e = sn[t]-ue
        res[N][t] = dict(L1=float(np.mean(np.abs(e))), Linf=float(np.max(np.abs(e))),
                         mean=float(sn[t].mean()), max=float(sn[t].max()), min=float(sn[t].min()),
                         sym=float(np.max(np.abs(sn[t]-sn[t].T))))
        print(N, t, res[N][t], flush=True)
    if N == 512: np.savez('/box/final/snaps_N512.npz', **{f"t{t}": sn[t] for t in times})
json.dump({str(k): {str(t): v for t, v in d.items()} for k, d in res.items()}, open('/box/final/errors.json','w'), indent=1)
