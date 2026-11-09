"""Compressible linear stability of the hydrostatic tanh profile (gravity +y, walls at y=0,1).
Perturbations ~ exp(i k x + n t):
  n r = -rho0 (i k u + v_y) - rho0_y v
  n u = -i k p / rho0
  n v = (-p_y + g r) / rho0
  n p = -p0_y v - gamma p0 (i k u + v_y)
Staggered FD: r,u,p at cell centres, v at faces (v=0 at walls). Shift-invert sparse eig.
"""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigs
from scipy.integrate import cumulative_trapezoid

GAM, G, k = 1.4, 1.0, 8 * np.pi


def growth(N, sigma=2.5):
    h = 1.0 / N
    yc = (np.arange(N) + 0.5) * h; yf = np.arange(1, N) * h
    rho = lambda s: 2.0 - 0.5 * (1 + np.tanh(50 * (s - 0.5)))
    drho = lambda s: -25 * (1 - np.tanh(50 * (s - 0.5)) ** 2)
    yy = np.linspace(0, 1, 200001); P = 2.0 + cumulative_trapezoid(rho(yy) * G, yy, initial=0)
    P -= np.interp(0.5, yy, P) - 2.0
    p0 = lambda s: np.interp(s, yy, P)
    rc, pc, rcy, pcy = rho(yc), p0(yc), drho(yc), G * rho(yc)
    rf = rho(yf)
    # operators: Dv: faces(N-1) -> centres(N); Dp: centres -> interior faces; Avc: faces -> centres; Acf: centres->faces
    e = np.ones(N)
    Dv = sp.diags([e[:N - 1], -e[:N - 1]], [0, -1], shape=(N, N - 1)) / h
    Dp = sp.diags([-e[:N - 1], e[:N - 1]], [0, 1], shape=(N - 1, N)) / h
    Avc = 0.5 * abs(Dv * h)
    Acf = 0.5 * abs(Dp * h)
    I = sp.identity(N)
    D = sp.diags
    Z = lambda a, b: sp.csr_matrix((a, b))
    ik = 1j * k
    # unknowns [r(N), u(N), v(N-1), p(N)]
    rows = [
        [Z(N, N), -ik * D(rc), -(D(rc) @ Dv + D(rcy) @ Avc), Z(N, N)],
        [Z(N, N), Z(N, N), Z(N, N - 1), -ik * D(1 / rc)],
        [G * D(1 / rf) @ Acf, Z(N - 1, N), Z(N - 1, N - 1), -D(1 / rf) @ Dp],
        [Z(N, N), -ik * GAM * D(pc), -(D(pcy) @ Avc + GAM * D(pc) @ Dv), Z(N, N)],
    ]
    M = sp.bmat(rows, format="csc")
    vals = eigs(M, k=6, sigma=sigma, return_eigenvectors=False)
    return vals[np.argmax(vals.real)].real


if __name__ == "__main__":
    for N in (1000, 2000, 4000, 8000):
        print(N, growth(N))
