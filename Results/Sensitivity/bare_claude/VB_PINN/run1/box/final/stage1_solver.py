"""Stage 1: high-fidelity Fourier pseudo-spectral solver for viscous Burgers (periodic, [-1,1]).
Space: Fourier collocation with N modes, 2/3 dealiasing. Time: ETDRK4 (Kassam & Trefethen 2005)
with contour-integral evaluation of phi-functions. Verified by (i) resolution/time-step convergence
and (ii) the exact Cole-Hopf solution evaluated with high-order Gauss-Hermite quadrature."""
import numpy as np, sys
nu = 1.0/(100*np.pi)

def solve(N, dt, t_out):
    L = 2.0
    x = -1 + L*np.arange(N)/N
    k = 2*np.pi*np.fft.rfftfreq(N, d=L/N)       # wavenumbers
    Lin = -nu*k**2
    dealias = (np.abs(k) < (2/3)*np.max(np.abs(k))).astype(float)
    g = -0.5j*k*dealias                         # N(u) = -(u^2/2)_x
    E, E2 = np.exp(dt*Lin), np.exp(dt*Lin/2)
    M = 64; r = np.exp(1j*np.pi*(np.arange(1, M+1)-0.5)/M)
    LR = dt*Lin[:, None] + r[None, :]
    Q  = dt*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
    f1 = dt*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
    f2 = dt*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
    f3 = dt*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))
    NL = lambda v: g*np.fft.rfft(np.fft.irfft(v, n=N)**2)
    v = np.fft.rfft(-np.sin(np.pi*x))
    out = [np.fft.irfft(v, n=N)]
    nsteps_per = [int(round((t_out[j+1]-t_out[j])/dt)) for j in range(len(t_out)-1)]
    for ns in nsteps_per:
        for _ in range(ns):
            Nv = NL(v); a = E2*v + Q*Nv; Na = NL(a)
            b = E2*v + Q*Na; Nb = NL(b)
            c = E2*a + Q*(2*Nb-Nv); Nc = NL(c)
            v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
        out.append(np.fft.irfft(v, n=N))
    return x, np.array(out)   # (Nt, N)

def cole_hopf(x, t, nq=20001):
    """Exact Cole-Hopf solution
       u(x,t) = -int sin(pi(x-eta)) F(x-eta) G(eta) d eta / int F(x-eta) G(eta) d eta,
       F(y)=exp(-cos(pi y)/(2 pi nu)), G = exp(-eta^2/(4 nu t)).
    Evaluated by the trapezoid rule on eta in [-W, W], W = sqrt(4 nu t * 160) (F varies by e^{1/(pi nu)}=e^100,
    so the Gaussian must decay by >e^-160 at the window edge) (spectrally accurate
    for this smooth, effectively compactly supported integrand), in log-sum-exp form."""
    s = np.sqrt(2*nu*t)
    W = np.sqrt(4*nu*t*160.0)
    eta = np.linspace(-W, W, nq)
    out = np.empty_like(x)
    for i0 in range(0, len(x), 256):
        X = x[i0:i0+256, None] - eta[None, :]
        expo = -np.cos(np.pi*X)/(2*np.pi*nu) - eta[None, :]**2/(4*nu*t)
        f = np.exp(expo - expo.max(axis=1, keepdims=True))
        out[i0:i0+256] = -(np.sin(np.pi*X)*f).sum(1)/f.sum(1)
    return out

if __name__ == "__main__":
    t_out = np.linspace(0, 1, 201)               # dt_out = 0.005
    runs = {}
    for N, dt in [(1024, 1e-4), (2048, 5e-5), (4096, 2.5e-5), (4096, 1.25e-5)]:
        x, U = solve(N, dt, t_out); runs[(N, dt)] = (x, U)
        print(f"N={N} dt={dt} done, min u_x at t=1: {np.min(np.gradient(U[-1], x)):.2f}", flush=True)
    xr, Ur = runs[(4096, 1.25e-5)]
    for (N, dt), (x, U) in runs.items():
        s = 4096//N
        d = U - Ur[:, ::s]
        print(f"N={N:5d} dt={dt:.2e}: max|u-u_ref| = {np.abs(d).max():.3e}")
    # Cole-Hopf check (convergence in nq, then compare)
    errs = []
    for tj in [0.25, 0.5, 1/np.pi, 0.75, 1.0]:
        j = None
        if np.isclose(tj*200, round(tj*200)):
            j = int(round(tj*200)); ex = cole_hopf(xr, tj, 20001); ex2 = cole_hopf(xr, tj, 40001)
            e = np.abs(Ur[j]-ex).max(); errs.append(e)
            print(f"t={tj:.4f}: max|spectral - ColeHopf| = {e:.3e}   (quadrature self-check {np.abs(ex-ex2).max():.1e})")
    # full-grid check against Cole-Hopf
    EX = np.array([cole_hopf(xr, t) if t > 0 else -np.sin(np.pi*xr) for t in t_out])
    print("full grid max err vs Cole-Hopf:", np.abs(EX-Ur).max(),
          " rel L2:", np.linalg.norm(EX-Ur)/np.linalg.norm(EX))
    np.savez_compressed("/box/final/burgers_reference.npz", x=xr, t=t_out, u=Ur.T, u_colehopf=EX.T, nu=nu,
                        note="u[i,j]=u(x_i,t_j); x periodic grid on [-1,1) (x=1 identical to x=-1)")
