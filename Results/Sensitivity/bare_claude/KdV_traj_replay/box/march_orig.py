import sys, json, pickle, time, jax, jax.numpy as jnp, numpy as np
from lib import *
cfg=dict(width=128,depth=5,nr=8000,wic=100,wbc=100,iters=20000,lr=3e-3,decay=0.6,opt='soap',lbfgs=0,logev=5000)
NW=int(sys.argv[1]); cfg.update(json.loads(sys.argv[2])); tag=sys.argv[3]
T=5.; dT=T/NW; key=jax.random.PRNGKey(cfg.get("seed",0))
ic=lambda x: uex(x,jnp.zeros_like(x)); models=[]; tic=time.time()
for k in range(NW):
    t0=k*dT; key,kk=jax.random.split(key)
    p,net=train_window(t0,dT,ic,kk,cfg,log=lambda s: print(s,flush=True))
    models.append((t0,jax.tree.map(np.asarray,p)))
    Unet=jax.jit(jax.vmap(net,(None,0,0)))
    ic=(lambda P,F,tt: (lambda x: F(P,x,jnp.full_like(x,tt))))(p,Unet,t0+dT)
    print(f"window {k} [{t0:.2f},{t0+dT:.2f}] relL2=%.3e max=%.3e  t={time.time()-tic:.0f}s"%window_err(p,net,t0,dT),flush=True)
    pickle.dump({"NW":NW,"cfg":cfg,"models":models},open(f"model_{tag}.pkl","wb"))
