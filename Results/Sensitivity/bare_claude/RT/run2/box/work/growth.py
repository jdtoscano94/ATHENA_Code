import numpy as np, sys
def modeamp(d,nx,ny,n):
    T=[];A=[]
    x=(np.arange(nx)+.5)/nx*0.25; cx=np.cos(8*np.pi*x)
    for k in range(n+1):
        a=np.fromfile(f'{d}/s_{k:03d}.bin'); U=a[1:].reshape(4,ny,nx)
        v=U[2]/U[0]; vh=2*(v*cx).mean(1)            # k=8π cosine coefficient of v(y)
        T.append(a[0]); A.append(np.sqrt((vh**2).mean()))
    return np.array(T),np.array(A)
if __name__=="__main__":
    d,nx,ny,n=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
    T,A=modeamp(d,nx,ny,n)
    loc=np.gradient(np.log(A),T)
    for t,a,l in zip(T,A,loc): print(f"{t:.2f} {a:.4e} {l:.4f}")
    for t0,t1 in [(1.0,2.0),(1.5,2.5),(1.0,2.5)]:
        s=(T>=t0-1e-9)&(T<=t1+1e-9); print(f"fit [{t0},{t1}] sigma={np.polyfit(T[s],np.log(A[s]),1)[0]:.4f}")
