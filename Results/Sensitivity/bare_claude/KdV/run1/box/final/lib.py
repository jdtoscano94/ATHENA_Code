import jax, jax.numpy as jnp, numpy as np, optax, time
from exact import u_exact
import soap_jax
L=20.
def _ue(x,t): return u_exact(x.astype(jnp.float64),t.astype(jnp.float64)).astype(jnp.float32)
uex=jax.jit(jax.vmap(_ue)); uex_x=jax.jit(jax.vmap(jax.grad(_ue,0)))

def init(key,layers):
    ps=[]
    for a,b in zip(layers[:-1],layers[1:]):
        key,k=jax.random.split(key)
        ps.append((jax.random.normal(k,(a,b))*jnp.sqrt(2/(a+b)),jnp.zeros(b)))
    return ps
def make_net(t0,dT):
    def net(p,x,t):
        h=jnp.stack([x/L*2-1,(t-t0)/dT*2-1])
        for W,b in p[:-1]: h=jnp.tanh(h@W+b)
        W,b=p[-1]; return (h@W+b)[0]
    return net

def train_window(t0,dT,ic_fn,key,cfg,log=print):
    layers=[2]+[cfg['width']]*cfg['depth']+[1]
    net=make_net(t0,dT)
    ux=jax.grad(net,1); uxx=jax.grad(ux,1); uxxx=jax.grad(uxx,1); ut=jax.grad(net,2)
    r1=lambda p,x,t: ut(p,x,t)+6*net(p,x,t)*ux(p,x,t)+uxxx(p,x,t)
    res=jax.vmap(r1,(None,0,0))
    U=jax.vmap(net,(None,0,0)); UX=jax.vmap(ux,(None,0,0))
    xi=jnp.linspace(0,L,cfg.get('nic',2001)); ui=ic_fn(xi)
    tb=jnp.linspace(t0,t0+dT,cfg.get('nbc',401))
    g1=uex(jnp.zeros_like(tb),tb); g2=uex(jnp.full_like(tb,L),tb); g3=uex_x(jnp.full_like(tb,L),tb)
    NR=cfg['nr']; wi,wb=cfg['wic'],cfg['wbc']
    def loss(p,xr,tr):
        lr=jnp.mean(res(p,xr,tr)**2)
        li=jnp.mean((U(p,xi,jnp.full_like(xi,t0))-ui)**2)
        lb=jnp.mean((U(p,jnp.zeros_like(tb),tb)-g1)**2)+jnp.mean((U(p,jnp.full_like(tb,L),tb)-g2)**2)+jnp.mean((UX(p,jnp.full_like(tb,L),tb)-g3)**2)
        return lr+wi*li+wb*lb,(lr,li,lb)
    def uni(key,n):
        k1,k2=jax.random.split(key)
        return jax.random.uniform(k1,(n,))*L, t0+jax.random.uniform(k2,(n,))*dT
    @jax.jit
    def rad(p,key):  # residual-based adaptive distribution, half uniform
        k1,k2,k3=jax.random.split(key,3)
        xc,tc=uni(k1,8*NR); r=jnp.abs(res(p,xc,tc))
        w=r**2/jnp.mean(r**2)+1.0
        idx=jax.random.choice(k2,8*NR,(NR//2,),p=w/w.sum(),replace=False)
        xu,tu=uni(k3,NR-NR//2)
        return jnp.concatenate([xc[idx],xu]),jnp.concatenate([tc[idx],tu])
    p=init(key,layers)
    N=cfg['iters']
    sched=optax.warmup_exponential_decay_schedule(0,cfg['lr'],200,N//10,cfg.get('decay',0.8),end_value=1e-6)
    opt=soap_jax.soap(sched,qr_dtype=jnp.float64,mu_dtype=jnp.float64) if cfg['opt']=='soap' else optax.adam(sched)
    st=opt.init(p)
    @jax.jit
    def step(p,st,xr,tr):
        (l,aux),g=jax.value_and_grad(loss,has_aux=True)(p,xr,tr)
        up,st=opt.update(g,st,p); return optax.apply_updates(p,up),st,l,aux
    tic=time.time()
    key,k=jax.random.split(key); xr,tr=uni(k,NR)
    xv,tv=uni(jax.random.PRNGKey(999),NR); lossv=jax.jit(lambda p: loss(p,xv,tv)[0])
    best=(jnp.inf,p)
    for i in range(N):
        if i%cfg.get('resample',100)==0:
            key,k=jax.random.split(key)
            xr,tr=rad(p,k) if (cfg.get('rad',True) and i>=1000) else uni(k,NR)
        p,st,l,aux=step(p,st,xr,tr)
        if i%250==0 or i==N-1:
            lv=lossv(p)
            if lv<best[0]: best=(lv,p)
        if i%cfg.get('logev',2000)==0 or i==N-1: log(f"  it{i} L={l:.3e} r={aux[0]:.2e} ic={aux[1]:.2e} bc={aux[2]:.2e} ({time.time()-tic:.0f}s)")
    log(f"  best check-loss {best[0]:.3e}"); p=best[1]
    if cfg.get('lbfgs',0)>0:
        key,k=jax.random.split(key); xr,tr=rad(p,k); xr2,tr2=uni(jax.random.split(k)[1],NR)
        xr=jnp.concatenate([xr,xr2]); tr=jnp.concatenate([tr,tr2])
        f=lambda p: loss(p,xr,tr)[0]
        lb=optax.lbfgs(); s2=lb.init(p); vg=optax.value_and_grad_from_state(f)
        @jax.jit
        def lstep(p,s):
            v,g=vg(p,state=s); up,s=lb.update(g,s,p,value=v,grad=g,value_fn=f)
            return optax.apply_updates(p,up),s,v
        for i in range(cfg['lbfgs']): p,s2,v=lstep(p,s2)
        l,aux=jax.jit(loss)(p,xr,tr); log(f"  LBFGS L={l:.3e} r={aux[0]:.2e} ic={aux[1]:.2e} bc={aux[2]:.2e} ({time.time()-tic:.0f}s)")
    return p,net

def window_err(p,net,t0,dT,nx=401,nt=26):
    Unet=jax.jit(jax.vmap(net,(None,0,0)))
    X,TT=np.meshgrid(np.linspace(0,L,nx),np.linspace(t0,t0+dT,nt))
    up=np.array(Unet(p,jnp.array(X.ravel(),jnp.float32),jnp.array(TT.ravel(),jnp.float32)))
    ue=np.array(uex(jnp.array(X.ravel(),jnp.float32),jnp.array(TT.ravel(),jnp.float32)))
    return np.linalg.norm(up-ue)/np.linalg.norm(ue), np.abs(up-ue).max()
