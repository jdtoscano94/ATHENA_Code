# Compressible linear stability (adiabatic perturbations ~exp(sigma t + i k x)) of the
# hydrostatic diffuse profile rho0 = tanh profile, p0' = rho0 g (g=+1 in +y), walls w(0)=w(1)=0.
#   w'  = -[p1 (s^2 + c^2 k^2)/s + p0' w]/(gam p0)
#   p1' = g rho1 - s rho0 w ,  rho1 = -(rho0' w + rho0 (k^2 p1/(s rho0) + w'))/s
# Shooting in s from y=0 (w=0,p1=1) -> root of w(1).
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq
k=8*np.pi; g=1.0
rho=lambda y:2.0-0.5*(1+np.tanh(50*(y-0.5)))
drho=lambda y:-25.0/np.cosh(50*(y-0.5))**2
# hydrostatic p0 with same p(0)=1 (task's p is the sharp-interface version; differs by O(1e-2) near y=0.5)
yy=np.linspace(0,1,20001); 
from scipy.integrate import cumulative_trapezoid
P=1+cumulative_trapezoid(rho(yy),yy,initial=0)
p0=lambda y:np.interp(y,yy,P)
def shoot(s,gam):
    def f(y,X):
        w,p1=X; r=rho(y); p=p0(y); c2=gam*p/r
        wp=-(p1*(s*s+c2*k*k)/s + r*g*w)/(gam*p)
        r1=-(drho(y)*w + r*(k*k*p1/(s*r)+wp))/s
        return [wp, g*r1 - s*r*w]
    sol=solve_ivp(f,[0,1],[0.0,1.0],rtol=1e-10,atol=1e-14,method='DOP853',max_step=2e-3)
    return sol.y[0,-1]/np.abs(sol.y[0]).max()
def growth(gam):
    ss=np.linspace(1.0,3.0,41); vals=[shoot(s,gam) for s in ss]
    roots=[brentq(shoot,ss[i],ss[i+1],args=(gam,),xtol=1e-10) for i in range(len(ss)-1) if vals[i]*vals[i+1]<0]
    return max(roots)
if __name__=="__main__":
    print("gamma=1e4 (incompressible limit):",growth(1e4))
    print("gamma=1.4 compressible        :",growth(1.4))
