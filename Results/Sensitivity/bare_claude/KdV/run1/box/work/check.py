from exact import *
import numpy as np
ux=jax.grad(u_exact,0); uxx=jax.grad(ux,0); uxxx=jax.grad(uxx,0); ut=jax.grad(u_exact,1)
r=jax.vmap(lambda x,t: ut(x,t)+6*u_exact(x,t)*ux(x,t)+uxxx(x,t))
X,T=np.meshgrid(np.linspace(0,20,201),np.linspace(0,5,51))
print("max residual",np.abs(r(X.ravel(),T.ravel())).max())
U=np.array(jax.vmap(u_exact)(X.ravel(),T.ravel())).reshape(X.shape)
for i in range(0,51,5): print(T[i,0], U[i].max().round(3), X[0][U[i].argmax()], "u(0)=%.2e u(20)=%.2e"%(U[i,0],U[i,-1]))
