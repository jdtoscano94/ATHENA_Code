# Incompressible linear stability of the diffuse RT profile, walls at y=0,1, gravity +y:
#   (rho w')' - k^2 rho w = (g k^2/sigma^2) rho' w ,  w(0)=w(1)=0
# Symmetric-definite form: (-g k^2 rho') w = sigma^2 (-(rho w')' + k^2 rho w); largest sigma^2.
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
def growth(N,k=8*np.pi,g=1.0):
    y=np.linspace(0,1,N+1); h=y[1]-y[0]
    rho=lambda y:2.0-0.5*(1+np.tanh(50*(y-0.5)))
    drho=lambda y:-25.0/np.cosh(50*(y-0.5))**2
    yi=y[1:-1]; rp=rho(yi+h/2); rm=rho(yi-h/2)
    A=sp.diags([(rp+rm)/h**2+k**2*rho(yi),-rp[:-1]/h**2,-rm[1:]/h**2],[0,1,-1],format='csc')
    B=sp.diags(-g*k**2*drho(yi),format='csc')
    mu=sla.eigsh(B,k=1,M=A,which='LA')[0]
    return np.sqrt(mu[0])
if __name__=="__main__":
    for N in [1000,2000,4000,8000]: print(N,growth(N))
    print("sharp interface sqrt(Akg)=",np.sqrt(8*np.pi/3))
