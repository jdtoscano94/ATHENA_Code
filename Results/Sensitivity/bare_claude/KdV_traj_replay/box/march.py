import sys, json, pickle, time, jax, jax.numpy as jnp, numpy as np
from lib import *
cfg=dict(width=128,depth=5,nr=8000,wic=100,wbc=100,iters=20000,lr=3e-3,decay=0.6,opt='soap',lbfgs=0,logev=5000)
NW=int(sys.argv[1]); cfg.update(json.loads(sys.argv[2])); tag=sys.argv[3]
T=5.; dT=T/NW; key=jax.random.PRNGKey(cfg.get("seed",0))
ic=lambda x: uex(x,jnp.zeros_like(x)); models=[]; tic=time.time()
# REPLAY: error trajectory of the marched solution, on the window_err grid of each window (401 x 26).
# cum relL2 at (window k, iter i) = sqrt((sum sq err of finished windows + current window) / (same for exact))
def _sq(p,net,t0,dT,nx=401,nt=26):
    Unet=jax.jit(jax.vmap(net,(None,0,0)))
    X,TT=np.meshgrid(np.linspace(0,L,nx),np.linspace(t0,t0+dT,nt))
    up=np.array(Unet(p,jnp.array(X.ravel(),jnp.float32),jnp.array(TT.ravel(),jnp.float32)))
    ue=np.array(uex(jnp.array(X.ravel(),jnp.float32),jnp.array(TT.ravel(),jnp.float32)))
    return float(np.sum((up-ue)**2)), float(np.sum(ue**2))
SE,SU=0.,0.; traj=[]
for k in range(NW):
    t0=k*dT; key,kk=jax.random.split(key)
    def _track(i,p_,net_,k=k,t0=t0):
        e,u=_sq(p_,net_,t0,dT); traj.append((k*cfg['iters']+i,k,np.sqrt((SE+e)/(SU+u)),np.sqrt(e/u)))
    p,net=train_window(t0,dT,ic,kk,cfg,log=lambda s: print(s,flush=True),track=_track)
    e,u=_sq(p,net,t0,dT); SE+=e; SU+=u   # frozen (best check-loss) parameters of this window
    traj.append(((k+1)*cfg['iters'],k,np.sqrt(SE/SU),np.sqrt(e/u)))
    np.save(f"traj_{tag}.npy",np.array(traj))   # columns: global iter, window, cumulative relL2, window relL2
    models.append((t0,jax.tree.map(np.asarray,p)))
    Unet=jax.jit(jax.vmap(net,(None,0,0)))
    ic=(lambda P,F,tt: (lambda x: F(P,x,jnp.full_like(x,tt))))(p,Unet,t0+dT)
    print(f"window {k} [{t0:.2f},{t0+dT:.2f}] relL2=%.3e max=%.3e  t={time.time()-tic:.0f}s"%window_err(p,net,t0,dT),flush=True)
    pickle.dump({"NW":NW,"cfg":cfg,"models":models},open(f"model_{tag}.pkl","wb"))
