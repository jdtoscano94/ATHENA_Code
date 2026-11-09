# Incompressible linear RT growth rate for the smoothed profile, walls at y=0,1, gravity +y.
# (rho w')' - k^2 rho w = (g k^2 / s^2) rho' w   ->  generalized eigenproblem A w = lam B w, lam = g k^2/s^2
import numpy as np, scipy.linalg as sl
def growth(k=8*np.pi, g=1.0, n=4000):
    y = np.linspace(0, 1, n+1); h = y[1]-y[0]
    rho = lambda y: 2.0 - 0.5*(1+np.tanh(50*(y-0.5)))
    drho = lambda y: -25.0/np.cosh(50*(y-0.5))**2
    yi = y[1:-1]; rp = rho(yi+h/2); rm = rho(yi-h/2)
    A = (np.diag(-(rp+rm)/h**2 - k**2*rho(yi)) + np.diag(rp[:-1]/h**2, 1) + np.diag(rm[1:]/h**2, -1))
    B = np.diag(drho(yi))
    lam = sl.eigvals(A, B).real
    lam = lam[(lam > 0) & np.isfinite(lam)]
    return np.sqrt(g*k**2/lam.min())
if __name__ == "__main__":
    for n in (1000, 2000, 4000): print(n, growth(n=n))
    print("sharp interface sqrt(Akg tanh)", np.sqrt(1/3*8*np.pi*np.tanh(8*np.pi*0.5)))
