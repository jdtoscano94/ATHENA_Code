"""Linear stability of the RT base state (compressible Euler, gravity +y, walls at y=0,1).
Perturbations ~ f(y) exp(i k x + sigma t). Staggered FD grid in y; v on faces (v=0 at walls),
rho', p', u on cell centres. Solves sigma X = A X with shift-invert eigs for the largest growth rate.
Also solves the incompressible (Rayleigh) problem with the same smooth density profile, and the
sharp-interface formula sqrt(A g k)."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla

gam, g = 1.4, 1.0
k = 8 * np.pi


def rho0(y): return 2.0 - 0.5 * (1 + np.tanh(50 * (y - 0.5)))
def drho0(y): return -0.5 * 50 / np.cosh(50 * (y - 0.5)) ** 2
def p0(y): return np.where(y < 0.5, 2 * y + 1, y + 1.5)
def dp0(y): return np.where(y < 0.5, 2.0, 1.0)


def compressible(N, use_actual_dp=True):
    h = 1.0 / N
    yc = (np.arange(N) + 0.5) * h
    yf = np.arange(1, N) * h  # interior faces
    nf = N - 1
    # unknowns: r (N), u (N), p (N), v (nf); u stored as i*u_hat -> real system
    # with U = i u:  sigma rho0 U = k p'   (since sigma rho0 u = -i k p => sigma rho0 (i u) = k p)
    # div = i k u + v_y = k*... : i k u = k U -> careful: i k u = k (i u) = k U
    # Rayleigh: rho0 sigma u = -ik p  -> U = i u: sigma rho0 U = k p
    # div term: i k u + v_y = k U + v_y ... (since i k u = k (i u)) -- sign: ik u = k*(i u) = k U. ok
    r0c, r0f = rho0(yc), rho0(yf)
    dr0c = drho0(yc)
    dp0c = dp0(yc) if use_actual_dp else g * r0c
    p0c = p0(yc)
    Dv = sp.diags([np.ones(N), -np.ones(N)], [0, -1], shape=(N, nf)) / h  # face->centre derivative
    Dv = sp.lil_matrix(Dv); Dv = Dv.tocsr()
    # check: centre j uses v_{j+1/2} (face index j) - v_{j-1/2} (face j-1)
    Av = sp.diags([0.5 * np.ones(N), 0.5 * np.ones(N)], [0, -1], shape=(N, nf))  # face->centre average
    Dp = sp.diags([-np.ones(nf), np.ones(nf)], [0, 1], shape=(nf, N)) / h  # centre->face derivative
    Ap = sp.diags([0.5 * np.ones(nf), 0.5 * np.ones(nf)], [0, 1], shape=(nf, N))
    I = sp.identity(N)
    Z = None
    # sigma r = -drho0 v - rho0 (k U + v_y)
    Rr = [None, -k * sp.diags(r0c), None, -(sp.diags(dr0c) @ Av + sp.diags(r0c) @ Dv)]
    # sigma U = k p / rho0
    Ru = [None, None, k * sp.diags(1 / r0c), None]
    # sigma p = -dp0 v - gam p0 (k U + v_y)
    Rp = [None, -gam * k * sp.diags(p0c), None, -(sp.diags(dp0c) @ Av + gam * sp.diags(p0c) @ Dv)]
    # sigma v = (-p_y + g r)/rho0  (faces)
    Rv = [g * sp.diags(1 / r0f) @ Ap, None, -sp.diags(1 / r0f) @ Dp, None]
    A = sp.bmat([Rr, Ru, Rp, Rv], format='csc')
    vals = sla.eigs(A, k=6, sigma=3.0, return_eigenvectors=False)
    vals = vals[np.abs(vals.imag) < 1e-8].real
    return vals.max()


def incompressible(N):
    # (rho0 w')' - k^2 rho0 w = -(g k^2 / sigma^2) rho0' w  with gravity in +y (sign flipped vs. -y)
    # generalized eigenproblem for lambda = sigma^2:  lambda * L w = M w,  L = (rho w')' - k^2 rho w,
    # M = -(-g) k^2 rho0' w   [gravity +y => effective g_y = +g]
    h = 1.0 / N
    y = np.arange(1, N) * h
    yh = (np.arange(N) + 0.5) * h
    rh = rho0(yh)
    L = sp.diags([rh[1:-0 or None][:-0 or None][:N - 2] / h**2 if False else rh[1:N - 1] / h**2,
                  -(rh[:-1] + rh[1:]) / h**2 - k**2 * rho0(y),
                  rh[1:N - 1] / h**2], [-1, 0, 1]).tocsc()
    M = sp.diags(g * k**2 * drho0(y)).tocsc()
    # derived: d/dy(rho w') - k^2 rho w = (g k^2/sigma^2) * rho0' w  for gravity +y
    # => sigma^2 L w = g k^2 rho0' w
    vals = sla.eigs(M, k=4, M=None, sigma=None, which='LR', return_eigenvectors=False) if False else None
    import scipy.linalg as la
    lam = la.eigvals(M.toarray(), L.toarray())
    lam = lam[np.isfinite(lam)].real
    return np.sqrt(lam.max())


if __name__ == '__main__':
    A = (2 - 1) / 3
    print(f"sharp interface incompressible sqrt(A g k) = {np.sqrt(A*g*k):.4f}")
    for N in [400, 800, 1600]:
        print(f"N={N}: incompressible smooth = {incompressible(N):.5f}, "
              f"compressible (actual base dp/dy) = {compressible(N):.5f}, "
              f"compressible (dp/dy=rho g) = {compressible(N, False):.5f}")
