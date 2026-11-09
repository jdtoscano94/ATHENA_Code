from burgers2d import *
times=[0.5,1.0,1.5,2.0]; res={}
for N in [32,64,128,256,512,1024]:
    X,Y,h,out=solve(N,times); res[N]={}
    for T in times:
        e=out[T]-exact(X,Y,T)
        res[N][T]=dict(L1=float(np.abs(e).mean()),Linf=float(np.abs(e).max()),
            mass_err=float(abs(out[T].mean()-out[0].mean())),max=float(out[T].max()),min=float(out[T].min()))
    print(N,{T:"L1=%.2e Linf=%.2e"%(r['L1'],r['Linf']) for T,r in res[N].items()},flush=True)
    if N==1024: keep=(X,Y,h,out)
json.dump({str(k):{str(t):v for t,v in d.items()} for k,d in res.items()},open('errors.json','w'),indent=1)
Ns=sorted(res)
for T in times:
    print("t=%.1f L1 orders:"%T,["%.2f"%np.log2(res[a][T]['L1']/res[b][T]['L1']) for a,b in zip(Ns,Ns[1:])])
X,Y,h,out=keep; ts=[0.0]+times
fig,axs=plt.subplots(2,5,figsize=(24,9.5))
for k,T in enumerate(ts):
    ax=axs[0,k]; im=ax.pcolormesh(X,Y,out[T],cmap='RdBu_r',vmin=-1/3,vmax=1,shading='auto')
    ax.set_aspect('equal'); ax.set_title(f"u(x,y,t={T}) numerical, N=1024"); ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.colorbar(im,ax=ax,fraction=.046)
    ax=axs[1,k]; s=np.linspace(0,L,2000)
    ax.plot(X.diagonal()+Y.diagonal(),out[T].diagonal(),'o',ms=2.5,color='C0',label='numerical (diagonal x=y)')
    ax.plot(2*s,exact(s,s,T),'-',color='k',lw=1,label='exact entropy solution')
    ax.set_xlabel('x+y'); ax.set_ylabel('u'); ax.set_title(f"t={T}: slice along x=y"); ax.grid(alpha=.3)
    if k==0: ax.legend(fontsize=8)
fig.suptitle("2D inviscid Burgers $u_t+uu_x+uu_y=0$, $u_0=1/3+2/3\\sin((x+y)/2)$, periodic $[0,4\\pi]^2$ (shock forms at $t=1.5$)")
fig.tight_layout(); fig.savefig('summary_all.png',dpi=110)
np.savez_compressed('snapshots_N1024.npz',x=X[:,0],y=Y[0],**{f"u_t{T}":out[T] for T in ts})
