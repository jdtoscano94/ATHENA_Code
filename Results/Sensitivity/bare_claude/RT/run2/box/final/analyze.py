import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys, os
RES={'64×256':('r64',64,256,'#2a78d6'),'128×512':('r128',128,512,'#eb6834'),'256×1024':('r256',256,1024,'#1baf7a')}
cmap=LinearSegmentedColormap.from_list('b',['#f4f8fe','#cde2fb','#86b6ef','#3987e5','#1c5cab','#0d366b'])
def load(d,nx,ny,k):
    a=np.fromfile(f'{d}/s_{k:03d}.bin'); t=a[0]; U=a[1:].reshape(4,ny,nx); return t,U
def width(U,ny):
    # integral mixing width  h = 6 * int <f>(1-<f>) dy , f = light-fluid fraction from x-averaged rho
    f=np.clip(2-U[0].mean(1),0,1); return 6*np.sum(f*(1-f))/ny
def tips(U,nx,ny):
    y=(np.arange(ny)+.5)/ny; r=U[0]
    # rho=1.5 contour crossings: bubble (light moves to -y) min y where rho<1.5; spike max y where rho>1.5
    light=r<1.5
    yb=np.where(light.any(0),[y[np.argmax(col)] for col in light.T],1).min()
    heavy=r>1.5
    ys=max(y[ny-1-np.argmax(col[::-1])] for col in heavy.T if col.any())
    return yb,ys
out={}
for lab,(d,nx,ny,col) in RES.items():
    if not os.path.exists(f'{d}/s_030.bin'): continue
    T=[];B=[];S=[];W=[]
    for k in range(31):
        t,U=load(d,nx,ny,k); yb,ys=tips(U,nx,ny); T.append(t);B.append(yb);S.append(ys);W.append(width(U,ny))
    diag=np.loadtxt(f'{d}/s_diag.txt')
    out[lab]=dict(T=np.array(T),B=np.array(B),S=np.array(S),W=np.array(W),diag=diag,d=d,nx=nx,ny=ny,col=col)
    m=diag[:,1];E=diag[:,2]
    # linear growth rate from KE ~ exp(2 sigma t)
    sel=(diag[:,0]>0.3)&(diag[:,0]<0.9); sig=np.polyfit(diag[sel,0],np.log(diag[sel,4]),1)[0]/2
    print(f"{lab}: mass drift {abs(m-m[0]).max()/m[0]:.2e}, energy drift {abs(E-E[0]).max()/E[0]:.2e}, "
          f"sigma(KEy fit 0.3-0.9)={sig:.3f}, bubble y(3)={B[-1]:.4f}, spike y(3)={S[-1]:.4f}, "
          f"rho min/max(3) {U[0].min():.4f}/{U[0].max():.4f}, KE(3)={diag[-1,3]:.4e}")
# convergence of final density field: L1 diff to finest after block-averaging
if len(out)==3:
    _,Uf=load('r256',256,1024,30)
    for k in [10,20,30]:
        _,Uf=load('r256',256,1024,k)
        for lab,(d,nx,ny,_) in list(RES.items())[:2]:
            _,U=load(d,nx,ny,k); f=256//nx
            Uc=Uf[0].reshape(ny,f,nx,f).mean((1,3))
            print(f"t={k/10:.1f} L1(rho_{lab} - rho_256 coarsened) = {np.abs(U[0]-Uc).mean():.4e}")
# ---------- figure
finest=list(out)[-1]; o=out[finest]
times=[0,5,10,15,20,25,30]
fig=plt.figure(figsize=(17,11),facecolor='#fcfcfb')
gs=fig.add_gridspec(2,len(times)+1,height_ratios=[1.35,1],hspace=0.32,wspace=0.25,width_ratios=[1]*len(times)+[0.08])
for n,k in enumerate(times):
    ax=fig.add_subplot(gs[0,n]); t,U=load(o['d'],o['nx'],o['ny'],k)
    im=ax.imshow(U[0],origin='lower',extent=[0,.25,0,1],cmap=cmap,vmin=1,vmax=2,interpolation='nearest')
    ax.set_title(f't = {t:.1f}',fontsize=11,color='#0b0b0b'); ax.set_xticks([0,.25]); ax.tick_params(labelsize=8,colors='#52514e')
    if n: ax.set_yticks([])
    else: ax.set_ylabel('y  (gravity → +y)',color='#52514e')
cax=fig.add_subplot(gs[0,-1]); cb=fig.colorbar(im,cax=cax); cb.set_label('density ρ',color='#52514e')
fig.text(0.5,0.955,f'Rayleigh–Taylor instability, compressible Euler (γ=1.4, g=+1 ŷ), density at {finest} — MUSCL-HLLC-RK3',ha='center',fontsize=14,color='#0b0b0b')
def style(ax):
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.grid(alpha=.25); ax.tick_params(colors='#52514e',labelsize=9)
gb=gs[1,:].subgridspec(1,4,wspace=0.32)
ax=fig.add_subplot(gb[0]); 
for lab,q in out.items():
    ax.plot(q['T'],q['S'],'-',color=q['col'],lw=2,label=lab); ax.plot(q['T'],q['B'],'--',color=q['col'],lw=2)
ax.axhline(.5,color='#52514e',lw=.8); ax.set_xlabel('t'); ax.set_ylabel('y of ρ=1.5 contour tip')
ax.set_title('Spike (solid, ↑) & bubble (dashed, ↓) tips',fontsize=10); ax.legend(fontsize=8,frameon=False); style(ax)
ax=fig.add_subplot(gb[1])
for lab,q in out.items(): ax.plot(q['T'],q['W'],color=q['col'],lw=2,label=lab)
ax.set_xlabel('t'); ax.set_ylabel('h = 6∫⟨f⟩(1−⟨f⟩)dy'); ax.set_title('Integral mixing-layer width',fontsize=10); ax.legend(fontsize=8,frameon=False); style(ax)
ax=fig.add_subplot(gb[2])
from growth import modeamp
SIG_C,SIG_I=2.4509,2.3850   # compressible / incompressible linear theory (linstab_comp.py, linstab.py)
for lab,d,nx,ny,col in [('64×256','lin64',64,256,'#2a78d6'),('128×512','lin128',128,512,'#eb6834')]:
    if not os.path.exists(f'{d}/s_060.bin'): continue
    T,A=modeamp(d,nx,ny,60); s_=(T>=1.8-1e-9); sg=np.polyfit(T[s_],np.log(A[s_]),1)[0]
    print(f"linear run {lab}: sigma fit [1.8,3] = {sg:.4f}  (theory {SIG_C}, err {100*(sg-SIG_C)/SIG_C:+.2f}%)")
    ax.semilogy(T,A,color=col,lw=2,label=f'{lab}: σ_fit={sg:.3f}')
tt=np.linspace(1.0,3.0,5); A2=A[T>=2-1e-9][0]
ax.semilogy(tt,A2*np.exp(SIG_C*(tt-2)),':',color='#0b0b0b',lw=1.5,label=f'compressible theory σ={SIG_C}')
ax.semilogy(tt,A2*np.exp(np.sqrt(8*np.pi/3)*(tt-2)),'--',color='#52514e',lw=1,label='sharp interface √(Akg)=2.894')
ax.set_xlabel('t'); ax.set_ylabel('rms over y of k=8π mode of v'); ax.set_title('Linear growth (seed amplitude ×10⁻³)',fontsize=10); ax.legend(fontsize=7,frameon=False); style(ax)
ax=fig.add_subplot(gb[3])
for lab,q in out.items():
    d=q['diag']; ax.semilogy(d[1:,0],np.maximum.accumulate(np.abs(d[1:,1]-d[0,1])/d[0,1])+1e-17,color=q['col'],lw=2,label=f'{lab} mass')
    ax.semilogy(d[1:,0],np.maximum.accumulate(np.abs(d[1:,2]-d[0,2])/d[0,2])+1e-17,'--',color=q['col'],lw=1.5,label=f'{lab} energy')
ax.set_xlabel('t'); ax.set_ylabel('max relative drift so far'); ax.set_title('Conservation: mass, ∫(E−ρgy)',fontsize=10); ax.legend(fontsize=7,frameon=False,ncol=1); style(ax)
fig.savefig(sys.argv[1] if len(sys.argv)>1 else 'summary_all.png',dpi=130,facecolor=fig.get_facecolor())
