# Exact entropy solution via characteristics (verification only). For x>0, u(x,t) = -sin(pi x0) with
# x = x0 - t sin(pi x0), x0 in (x, 1] taking the largest root (characteristics left of x0 merge into the stationary shock at x=0).
import numpy as np
from scipy.optimize import brentq
def exact(x, t):
    if t == 0: return -np.sin(np.pi*x)
    s = np.sign(x); x = abs(x)
    if x == 0 or x == 1: return 0.0
    f = lambda x0: x0 - t*np.sin(np.pi*x0) - x
    g = np.linspace(x, 1, 4001); v = f(g)
    i = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    j = i[-1]
    x0 = brentq(f, g[j], g[j+1], xtol=1e-14)
    return s*(-np.sin(np.pi*x0))
if __name__ == '__main__':
    import sys
    d = np.load(sys.argv[1]); t, x, u = d['t'], d['x'], d['usol']
    ue = np.array([[exact(xx, tt) for xx in x] for tt in t])
    np.save('/box/work/uexact.npy', ue)
    print('max|ref-exact|', np.abs(ue-u).max(), 'relL2', np.linalg.norm(ue-u)/np.linalg.norm(u))
