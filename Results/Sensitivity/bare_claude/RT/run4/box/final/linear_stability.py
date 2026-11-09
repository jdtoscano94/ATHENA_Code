"""Compressible linear stability of the RT base state (normal modes ~ exp(ikx + sigma t)).
Base: rho0(y) = tanh profile, p0 hydrostatic (dp0/dy = rho0 g), walls v=0 at y=0,1, k=8*pi.
Staggered FD: rho',u_hat,p' at centres, v at faces (u = i*u_hat makes the matrix real)."""
import numpy as np, sys

gam, g, k = 1.4, 1.0, 8 * np.pi


def growth(N):
    dy = 1.0 / N
    yc = (np.arange(N) + 0.5) * dy
    yf = np.arange(N + 1) * dy
    rho = lambda y: 2.0 - 0.5 * (1 + np.tanh(50 * (y - 0.5)))
    # hydrostatic p0 (analytic integral of rho), p0(0)=1
    P = lambda y: 1.0 + 1.5 * y - 0.5 * (np.log(np.cosh(50 * (y - 0.5))) - np.log(np.cosh(-25.0))) / 50
    rc, rf, pc = rho(yc), rho(yf), P(yc)
    pyc = rc * g
    nv = N - 1  # interior faces
    n = 3 * N + nv
    iR, iU, iP, iV = 0, N, 2 * N, 3 * N
    A = np.zeros((n, n))
    for j in range(N):
        # rho' eq
        if j + 1 <= N - 1: A[iR + j, iV + j] += -rf[j + 1] / dy
        if j >= 1: A[iR + j, iV + j - 1] += rf[j] / dy
        A[iR + j, iU + j] += k * rc[j]
        # u_hat eq
        A[iU + j, iP + j] = -k / rc[j]
        # p' eq
        A[iP + j, iU + j] += gam * pc[j] * k
        if j + 1 <= N - 1:
            A[iP + j, iV + j] += -gam * pc[j] / dy - 0.5 * pyc[j]
        if j >= 1:
            A[iP + j, iV + j - 1] += gam * pc[j] / dy - 0.5 * pyc[j]
    for m in range(nv):
        jf = m + 1  # face index, between centres jf-1 and jf
        A[iV + m, iP + jf] += -1 / (dy * rf[jf])
        A[iV + m, iP + jf - 1] += 1 / (dy * rf[jf])
        A[iV + m, iR + jf] += 0.5 * g / rf[jf]
        A[iV + m, iR + jf - 1] += 0.5 * g / rf[jf]
    ev = np.linalg.eigvals(A)
    return ev[np.argmax(ev.real)]


if __name__ == "__main__":
    for N in [200, 400, 800]:
        s = growth(N)
        print(f"N={N}: sigma = {s.real:.6f} (imag {s.imag:.1e})")
    A = 1 / 3
    print("sharp incompressible sqrt(Agk) =", np.sqrt(A * g * k))
