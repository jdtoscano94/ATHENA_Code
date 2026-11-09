import numpy as np
def exact2(x,y,t):
    th=(x+y)/2; a=2*t/3
    eta=np.mod(th-t/3+np.pi,2*np.pi)-np.pi   # in [-pi,pi)
    s=np.sign(eta); e=np.abs(eta)
    zc=np.pi if a<=1 else np.arccos(-1/a)
    lo=np.zeros_like(e); hi=np.full_like(e,zc)
    for _ in range(60):
        m=.5*(lo+hi); f=m+a*np.sin(m)-e
        lo=np.where(f<0,m,lo); hi=np.where(f<0,hi,m)
    z=s*.5*(lo+hi)
    return 1/3+2/3*np.sin(z)
