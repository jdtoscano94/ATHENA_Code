"""Linear stability (normal mode) analysis of the compressible RTI base state used in the simulation.
Perturbations ~ exp(i k x + sigma t), k = 8*pi, walls (v=0) at y=0,1. Staggered FD in y (2nd order),
Richardson-extrapolated in N. Prints the most unstable growth rate."""
import numpy as np, scipy.linalg as sla
gam, g, k = 1.4, 1.0, 8*np.pi
rho0 = lambda y: 2.0 - 0.5*(1+np.tanh(50*(y-0.5)))
drho0 = lambda y: -25.0/np.cosh(50*(y-0.5))**2
p0 = lambda y: np.where(y < 0.5, 2*y+1, y+1.5)
dp0 = lambda y: np.where(y < 0.5, 2.0, 1.0)
def growth(N, use_hydro_dp=False):
    h = 1.0/N; yc = (np.arange(N)+0.5)*h; yf = np.arange(1, N)*h      # centers, interior faces
    nc, nf = N, N-1
    # variables: rho(nc), u(nc), v(nf), p(nc)
    iR, iU, iV, iP = 0, nc, 2*nc, 2*nc+nf
    n = 3*nc+nf; L = np.zeros((n, n))
    # Dv at centers (v=0 on walls): (v_{j+1/2}-v_{j-1/2})/h ; avg v at centers
    D = np.zeros((nc, nf)); Av = np.zeros((nc, nf))
    for j in range(nc):
        if j < nf: D[j, j] += 1/h; Av[j, j] += 0.5
        if j-1 >= 0: D[j, j-1] -= 1/h; Av[j, j-1] += 0.5
    G = np.zeros((nf, nc)); Ac = np.zeros((nf, nc))   # gradient / average center->face
    for f in range(nf):
        G[f, f] = -1/h; G[f, f+1] = 1/h; Ac[f, f] = Ac[f, f+1] = 0.5
    r0c, r0f, p0c = rho0(yc), rho0(yf), p0(yc)
    dp = g*rho0(yc) if use_hydro_dp else dp0(yc)
    L[iR:iR+nc, iU:iU+nc] = -np.diag(r0c)*1j*k if False else 0
    # use real formulation: u = i*uhat -> sigma uhat = -(-k) ... write u = i*ut so ik u = -k ut
    # sigma rho = -rho0' v - rho0(-k ut + Dv);  sigma ut = -k p/rho0 ; sigma v = -Gp/rho0f + g Ac rho/rho0f
    # sigma p  = -p0' v - gam p0 (-k ut + Dv)
    L[iR:iR+nc, iU:iU+nc] = np.diag(r0c*k)
    L[iR:iR+nc, iV:iV+nf] = -np.diag(drho0(yc)) @ Av - np.diag(r0c) @ D
    L[iU:iU+nc, iP:iP+nc] = -np.diag(k/r0c)
    L[iV:iV+nf, iP:iP+nc] = -np.diag(1/r0f) @ G
    L[iV:iV+nf, iR:iR+nc] = g*np.diag(1/r0f) @ Ac
    L[iP:iP+nc, iU:iU+nc] = np.diag(gam*p0c*k)
    L[iP:iP+nc, iV:iV+nf] = -np.diag(dp) @ Av - np.diag(gam*p0c) @ D
    ev = sla.eigvals(L)
    return ev[np.argmax(ev.real)]
if __name__ == "__main__":
    res = {}
    for N in (250, 500, 1000):
        s = growth(N); res[N] = s.real; print(N, s)
    rich = res[1000] + (res[1000]-res[500])/3
    print("Richardson sigma =", rich)
    print("with exactly hydrostatic dp0/dy=rho0 g:", growth(1000, True))
    print("incompressible sharp interface sqrt(Akg) =", np.sqrt(k*g/3))
    np.save("sigma_lin.npy", rich)
