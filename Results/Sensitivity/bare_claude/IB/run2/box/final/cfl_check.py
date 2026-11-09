from burgers2d import *
from multiprocessing import Pool
def f(a):
    N,c=a; _,s=solve(N,1.0,cfl=c,snaps=[1.0]); e=np.abs(s[1.0]-exact_nodes(N,1.0)); return N,c,e.mean(),e.max()
with Pool(4) as p:
    for r in p.map(f,[(512,0.4),(512,0.2),(512,0.1),(1024,0.1)]): print(r)
