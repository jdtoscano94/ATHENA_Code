import sys, json, jax, jax.numpy as jnp
from lib import *
cfg=dict(width=128,depth=5,nr=8000,wic=100,wbc=100,iters=10000,lr=1e-3,opt='adam',lbfgs=0)
cfg.update(json.loads(sys.argv[1])); t0=float(sys.argv[2]); dT=float(sys.argv[3])
ic=lambda x: uex(x,jnp.full_like(x,t0))
p,net=train_window(t0,dT,ic,jax.random.PRNGKey(1),cfg)
print(sys.argv[1],t0,dT,"relL2=%.3e max=%.3e"%window_err(p,net,t0,dT),flush=True)
