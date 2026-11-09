"""Shock-position and away-from-shock error at t=2."""
import numpy as np, json
from burgers2d import solve, exact_xy, L
out = {}
for N in [64, 128, 256, 512]:
    u = solve(N, 2.0)[2.0]; h = L/N; xc = (np.arange(N)+0.5)*h
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    e = np.abs(u - exact_xy(X, Y, 2.0))
    s = 0.5*(X+Y); ss = np.pi + 2.0/3.0          # exact shock location in s (mod 2pi)
    d = np.abs(np.mod(s - ss + np.pi, 2*np.pi) - np.pi)
    mask = d > 0.5                                  # exclude |s - s_shock| <= 0.5
    # numerical shock location along diagonal: where u crosses mean 1/3 with negative slope
    ud = u.diagonal(); sd = xc
    i = np.where((ud[:-1] > 1/3) & (ud[1:] <= 1/3))[0][0]
    s_num = sd[i] + (ud[i]-1/3)/(ud[i]-ud[i+1])*(sd[i+1]-sd[i])
    out[N] = dict(L1_away=float(e[mask].mean()), Linf_away=float(e[mask].max()),
                  shock_s_num=float(s_num), shock_s_exact=ss, shock_err=float(s_num-ss))
    print(N, out[N])
json.dump(out, open("shock_check.json", "w"), indent=1)
