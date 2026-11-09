import sys, pickle, json, jax, jax.numpy as jnp, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lib import make_net, uex, L
tag=sys.argv[1]; out=sys.argv[2] if len(sys.argv)>2 else "."
d=pickle.load(open(f"model_{tag}.pkl","rb")); NW=d["NW"]; dT=5./NW
nets=[(t0,jax.tree.map(jnp.asarray,p),jax.jit(jax.vmap(make_net(t0,dT),(None,0,0)))) for t0,p in d["models"]]
def predict(x,t):
    x=np.asarray(x,float); t=np.asarray(t,float); u=np.zeros_like(x)
    k=np.minimum((t/dT).astype(int),NW-1)
    for i,(t0,p,f) in enumerate(nets):
        m=k==i
        if m.any(): u[m]=np.asarray(f(p,jnp.array(x[m]),jnp.array(t[m])))
    return u
nx,nt=1001,501
xg=np.linspace(0,L,nx); tg=np.linspace(0,5,nt); X,T=np.meshgrid(xg,tg)
U=predict(X.ravel(),T.ravel()).reshape(X.shape)
Ue=np.asarray(uex(jnp.array(X.ravel()),jnp.array(T.ravel()))).reshape(X.shape)
E=U-Ue
m={"tag":tag,"n_windows":NW,"grid":f"{nx}x{nt}","rel_L2":float(np.linalg.norm(E)/np.linalg.norm(Ue)),
   "max_abs_err":float(np.abs(E).max()),"mean_abs_err":float(np.abs(E).mean()),
   "rel_L2_per_time":{f"{t:.1f}":float(np.linalg.norm(E[i])/np.linalg.norm(Ue[i])) for i,t in enumerate(tg) if i%50==0}}
# continuity across window interfaces
jumps=[]
for i in range(1,NW):
    tt=i*dT; a=np.asarray(nets[i-1][2](nets[i-1][1],jnp.array(xg),jnp.full(nx,tt))); b=np.asarray(nets[i][2](nets[i][1],jnp.array(xg),jnp.full(nx,tt)))
    jumps.append(float(np.abs(a-b).max()))
m["max_interface_jump"]=max(jumps) if jumps else 0.
print(json.dumps(m,indent=1))
json.dump(m,open(f"{out}/metrics_{tag}.json","w"),indent=1)
np.savez_compressed(f"{out}/solution_{tag}.npz",x=xg,t=tg,u_pinn=U,u_exact=Ue)
fig,ax=plt.subplots(1,3,figsize=(16,4.2),constrained_layout=True)
for a,Z,ttl,cm in [(ax[0],U,"PINN u(x,t)","viridis"),(ax[1],Ue,"Exact u(x,t)","viridis"),(ax[2],np.abs(E),"|error|","magma")]:
    im=a.pcolormesh(xg,tg,Z,cmap=cm,shading="auto"); a.set_xlabel("x"); a.set_ylabel("t"); a.set_title(ttl); fig.colorbar(im,ax=a)
fig.savefig(f"{out}/fields_{tag}.png",dpi=130)
fig,ax=plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
for a,tv in zip(ax.ravel(),[0,1,2,3,3.5,5]):
    i=int(round(tv/5*(nt-1))); a.plot(xg,Ue[i],color="0.2",lw=2.5,label="exact"); a.plot(xg,U[i],"--",color="tab:orange",lw=1.6,label="PINN")
    a.set_title(f"t = {tv}"); a.set_xlabel("x"); a.set_ylabel("u")
ax[0,0].legend(); fig.savefig(f"{out}/snapshots_{tag}.png",dpi=130)
fig,a=plt.subplots(figsize=(6,3.5),constrained_layout=True)
a.semilogy(tg,np.linalg.norm(E,axis=1)/np.linalg.norm(Ue,axis=1)); a.set_xlabel("t"); a.set_ylabel("relative L2 error in x"); a.grid(alpha=.3)
for i in range(1,NW): a.axvline(i*dT,color="0.8",lw=.6)
fig.savefig(f"{out}/error_vs_time_{tag}.png",dpi=130)
# PDE residual on a dense grid (independent of exact solution)
rmax=0; rms=[]
for t0,p,_ in nets:
    net=make_net(t0,dT); ux=jax.grad(net,1); uxx=jax.grad(ux,1); uxxx=jax.grad(uxx,1); ut=jax.grad(net,2)
    R=jax.jit(jax.vmap(lambda x,t: ut(p,x,t)+6*net(p,x,t)*ux(p,x,t)+uxxx(p,x,t)))
    Xr,Tr=np.meshgrid(np.linspace(0,L,401),np.linspace(t0,t0+dT,51))
    r=np.asarray(R(jnp.array(Xr.ravel()),jnp.array(Tr.ravel()))); rmax=max(rmax,np.abs(r).max()); rms.append(np.mean(r**2))
m["pde_residual_rms"]=float(np.sqrt(np.mean(rms))); m["pde_residual_max"]=float(rmax)
# BC errors
tb=np.linspace(0,5,1001)
from lib import uex_x
m["bc_err_x0_max"]=float(np.abs(predict(np.zeros_like(tb),tb)-np.asarray(uex(jnp.zeros(1001),jnp.array(tb)))).max())
m["bc_err_x20_max"]=float(np.abs(predict(np.full_like(tb,L),tb)-np.asarray(uex(jnp.full(1001,L),jnp.array(tb)))).max())
print("residual rms %.3e max %.3e; BC max err x=0 %.2e x=20 %.2e"%(m["pde_residual_rms"],rmax,m["bc_err_x0_max"],m["bc_err_x20_max"]))
json.dump(m,open(f"{out}/metrics_{tag}.json","w"),indent=1)
