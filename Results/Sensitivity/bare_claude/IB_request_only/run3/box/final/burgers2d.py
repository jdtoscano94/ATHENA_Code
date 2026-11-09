import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, sys, json
L=4*np.pi
def u0(x,y): return 1/3+2/3*np.sin((x+y)/2)

def exact(x,y,t):
    """Entropy solution via Hopf-Lax in theta=(x+y)/2: u_t+u u_theta=0."""
    th=((x+y)/2).ravel()
    if t==0: return u0(x,y)
    # minimise U0(z)+(th-z)^2/(2t), U0=z/3-2/3 cos z ; candidates = all characteristic roots
    zg=np.linspace(-4*np.pi,4*np.pi,4001)[None,:]+np.round(th/(2*np.pi))[:,None]*2*np.pi*0
    # shift: work with th mod 2pi, then u is periodic
    thm=np.mod(th,2*np.pi)[:,None]
    z=np.linspace(thm.min()-2*np.pi,2*np.pi+2*np.pi,1)  # placeholder
    zg=thm+np.linspace(-np.pi-1.5*t,np.pi+1.5*t,2001)[None,:]*1.0
    G=zg/3-2/3*np.cos(zg)+(thm-zg)**2/(2*t)
    zs=zg[np.arange(len(thm)),G.argmin(1)][:,None]
    # Newton polish on characteristic eq: z + t*u0(z) - th = 0
    for _ in range(30):
        f=zs+t*(1/3+2/3*np.sin(zs))-thm; fp=1+t*2/3*np.cos(zs)
        zs=zs-f/np.where(np.abs(fp)<1e-12,1e-12,fp)
    return (1/3+2/3*np.sin(zs)).reshape(x.shape)

def weno_flux(u,ax):
    # global Lax-Friedrichs splitting, WENO5-JS, flux at i+1/2
    a=np.abs(u).max(); f=0.5*u*u
    fp=0.5*(f+a*u); fm=0.5*(f-a*u)
    def rec(v,s):  # reconstruct at i+1/2 from left-biased stencil (s=+1) or right (s=-1)
        g=lambda k: np.roll(v,-k*s if s==1 else -k,axis=ax)
        if s==1: vm2,vm1,v0,vp1,vp2=g(-2),g(-1),g(0),g(1),g(2)
        else:    vm2,vm1,v0,vp1,vp2=[np.roll(v,-k,axis=ax) for k in (3,2,1,0,-1)]
        p0=(2*vm2-7*vm1+11*v0)/6; p1=(-vm1+5*v0+2*vp1)/6; p2=(2*v0+5*vp1-vp2)/6
        b0=13/12*(vm2-2*vm1+v0)**2+.25*(vm2-4*vm1+3*v0)**2
        b1=13/12*(vm1-2*v0+vp1)**2+.25*(vm1-vp1)**2
        b2=13/12*(v0-2*vp1+vp2)**2+.25*(3*v0-4*vp1+vp2)**2
        e=1e-6; a0=.1/(e+b0)**2; a1=.6/(e+b1)**2; a2=.3/(e+b2)**2
        return (a0*p0+a1*p1+a2*p2)/(a0+a1+a2)
    return rec(fp,1)+rec(fm,-1)

def rhs(u,h):
    r=0
    for ax in (0,1):
        F=weno_flux(u,ax); r=r-(F-np.roll(F,1,axis=ax))/h
    return r

def solve(N,times,cfl=0.4):
    h=L/N; xc=np.arange(N)*h; X,Y=np.meshgrid(xc,xc,indexing='ij')
    # finite-difference (Shu-Osher) WENO: unknowns are point values at nodes
    u=u0(X,Y)
    t=0; out={0.0:u.copy()}
    for T in times:
        while t<T-1e-14:
            dt=min(cfl*h/(2*np.abs(u).max()),T-t)
            u1=u+dt*rhs(u,h); u2=.75*u+.25*(u1+dt*rhs(u1,h)); u=u/3+2/3*(u2+dt*rhs(u2,h)); t+=dt
        out[T]=u.copy()
    return X,Y,h,out

def exact_avg(X,Y,h,t):
    gp,gw=np.polynomial.legendre.leggauss(4); s=0
    for a,wa in zip(gp,gw):
        for b,wb in zip(gp,gw): s=s+wa*wb/4*exact(X+a*h/2,Y+b*h/2,t)
    return s
