# Compressible linear stability of the stratified base state (gravity +y, walls at y=0,1).
# q = (rho', uhat, v, p') ~ exp(ikx + s t), u = -i*uhat. Staggered FD: v on faces, others at centers.
#  s rho' = -k rho0 uhat - (rho0 v)_y
#  s rho0 uhat = k p'
#  s rho0 v = -p'_y + g rho'
#  s p' = -v dp0/dy - gamma p0 (k uhat + v_y)
import numpy as np, scipy.linalg as sl
def comp_growth(n=400, k=8*np.pi, g=1.0, gam=1.4, consistent=True):
    h = 1.0/n; yc = (np.arange(n)+0.5)*h; yf = np.arange(1, n)*h   # interior faces
    rho = lambda y: 2.0 - 0.5*(1+np.tanh(50*(y-0.5)))
    if consistent:   # p0 integrated from dp/dy = rho g with p0(0)=1 (exact antiderivative)
        P = lambda y: 1 + g*(2*y - 0.5*(y + np.log(np.cosh(50*(y-0.5))/np.cosh(25))/50))
    else:
        P = lambda y: np.where(y < 0.5, 2*y+1, y+1.5)
    rc, pc, rf, pf = rho(yc), P(yc), rho(yf), P(yf)
    dpf = g*rf if consistent else np.where(yf < 0.5, 2.0, 1.0)
    nc, nf = n, n-1
    # D: faces->centers derivative (v=0 at walls), G: centers->faces derivative, Ic: centers->faces avg, If: faces->centers avg
    D = np.zeros((nc, nf)); G = np.zeros((nf, nc)); Af = np.zeros((nf, nc)); Ac = np.zeros((nc, nf))
    for j in range(nf):
        D[j, j] += -1/h; D[j+1, j] += 1/h
        G[j, j] = -1/h; G[j, j+1] = 1/h
        Af[j, j] = Af[j, j+1] = 0.5
        Ac[j, j] += 0.5; Ac[j+1, j] += 0.5
    Z = np.zeros
    N = 3*nc + nf
    L = np.zeros((N, N)); M = np.zeros((N, N))
    ir, iu, iv, ip = slice(0, nc), slice(nc, 2*nc), slice(2*nc, 2*nc+nf), slice(2*nc+nf, N)
    M[ir, ir] = np.eye(nc); L[ir, iu] = -k*np.diag(rc); L[ir, iv] = -D @ np.diag(rf)
    M[iu, iu] = np.diag(rc); L[iu, ip] = k*np.eye(nc)
    M[iv, iv] = np.diag(rf); L[iv, ip] = -G; L[iv, ir] = g*Af
    M[ip, ip] = np.eye(nc); L[ip, iv] = -(Ac @ np.diag(dpf)) - np.diag(gam*pc) @ D; L[ip, iu] = -gam*k*np.diag(pc)
    ev = sl.eigvals(L, M)
    ev = ev[np.isfinite(ev)]
    return ev[np.argmax(ev.real)]
if __name__ == "__main__":
    for n in (200, 400, 800):
        print(n, comp_growth(n), comp_growth(n, consistent=False))
