import sys; sys.path.insert(0,'/box/final'); from burgers2d import *
for cfl in [0.4, 0.05]:
    prev=None
    for N in [64,128,256]:
        u=solve(N,[1.0],cfl=cfl)[1.0]; e=np.mean(np.abs(u-point_exact(N,1.0)))
        print(f"cfl={cfl} N={N} L1(t=1)={e:.3e}" + (f" order={np.log2(prev/e):.2f}" if prev else ""), flush=True); prev=e
