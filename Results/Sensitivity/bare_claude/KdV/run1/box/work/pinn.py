import jax, jax.numpy as jnp, numpy as np, optax, time, sys, pickle
from exact import u_exact, c1
jax.config.update("jax_enable_x64", False)
NW=int(sys.argv[1]) if len(sys.argv)>1 else 10
ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 20000
LBFGS=int(sys.argv[3]) if len(sys.argv)>3 else 5000
TAG=sys.argv[4] if len(sys.argv)>4 else "run"
L,T=20.,5.; dT=T/NW
layers=[2]+[128]*5+[1]
uex=jax.jit(jax.vmap(lambda x,t: u_exact(x.astype(jnp.float64),t.astype(jnp.float64)).astype(jnp.float32)))
uex_x=jax.jit(jax.vmap(jax.grad(lambda x,t: u_exact(x.astype(jnp.float64),t.astype(jnp.float64)).astype(jnp.float32),0)))

def init(key):
    ps=[]
    for a,b in zip(layers[:-1],layers[1:]):
        key,k=jax.random.split(key)
        ps.append((jax.random.normal(k,(a,b))*jnp.sqrt(2/(a+b)),jnp.zeros(b)))
    return ps
def make_net(t0):
    def net(p,x,t):
        h=jnp.stack([x/L*2-1,(t-t0)/dT*2-1])
        for W,b in p[:-1]: h=jnp.tanh(h@W+b)
        W,b=p[-1]; return (h@W+b)[0]
    return net

def train_window(k,t0,ic_fn,key):
    net=make_net(t0)
    ux=jax.grad(net,1); uxx=jax.grad(ux,1); uxxx=jax.grad(uxx,1); ut=jax.grad(net,2)
    res=jax.vmap(lambda p,x,t: ut(p,x,t)+6*net(p,x,t)*ux(p,x,t)+uxxx(p,x,t),(None,0,0))
    U=jax.vmap(net,(None,0,0)); UX=jax.vmap(ux,(None,0,0))
    # fixed data
    xi=jnp.linspace(0,L,2001); ui=ic_fn(xi)
    tb=jnp.linspace(t0,t0+dT,401)
    g1=uex(jnp.zeros_like(tb),tb); g2=uex(jnp.full_like(tb,L),tb); g3=uex_x(jnp.full_like(tb,L),tb)
    NR=10000
    def loss(p,xr,tr):
        lr=jnp.mean(res(p,xr,tr)**2)
        li=jnp.mean((U(p,xi,jnp.full_like(xi,t0))-ui)**2)
        lb=jnp.mean((U(p,jnp.zeros_like(tb),tb)-g1)**2)+jnp.mean((U(p,jnp.full_like(tb,L),tb)-g2)**2)+jnp.mean((UX(p,jnp.full_like(tb,L),tb)-g3)**2)
        return lr+100*li+100*lb,(lr,li,lb)
    def sample(key):
        k1,k2=jax.random.split(key)
        return jax.random.uniform(k1,(NR,))*L, t0+jax.random.uniform(k2,(NR,))*dT
    p=init(key)
    sched=optax.exponential_decay(1e-3,ITERS//10,0.7,end_value=1e-5)
    opt=optax.adam(sched); st=opt.init(p)
    @jax.jit
    def step(p,st,key):
        xr,tr=sample(key)
        (l,aux),g=jax.value_and_grad(loss,has_aux=True)(p,xr,tr)
        up,st=opt.update(g,st,p); return optax.apply_updates(p,up),st,l,aux
    for i in range(ITERS):
        key,k=jax.random.split(key); p,st,l,aux=step(p,st,k)
        if i%5000==0: print(f"  it{i} L={l:.3e} r={aux[0]:.2e} ic={aux[1]:.2e} bc={aux[2]:.2e}",flush=True)
    if LBFGS>0:
        key,k=jax.random.split(key); xr,tr=sample(k)
        xr2,tr2=sample(jax.random.split(k)[0]); xr=jnp.concatenate([xr,xr2]); tr=jnp.concatenate([tr,tr2])
        f=lambda p: loss(p,xr,tr)[0]
        lb=optax.lbfgs(); st=lb.init(p); vg=optax.value_and_grad_from_state(f)
        @jax.jit
        def lstep(p,st):
            v,g=vg(p,state=st); up,st=lb.update(g,st,p,value=v,grad=g,value_fn=f)
            return optax.apply_updates(p,up),st,v
        for i in range(LBFGS):
            p,st,v=lstep(p,st)
        l,aux=jax.jit(loss)(p,xr,tr); print(f"  LBFGS L={l:.3e} r={aux[0]:.2e} ic={aux[1]:.2e} bc={aux[2]:.2e}",flush=True)
    return p,net

key=jax.random.PRNGKey(0)
ic=lambda x: uex(x,jnp.zeros_like(x))
models=[]; t_start=time.time()
xg=np.linspace(0,L,401)
for k in range(NW):
    t0=k*dT; key,kk=jax.random.split(key)
    p,net=train_window(k,t0,ic,kk)
    models.append((t0,p))
    Unet=jax.jit(jax.vmap(net,(None,0,0)))
    ic=(lambda P,F: (lambda x: F(P,x,jnp.full_like(x,t0+dT))))(p,Unet)
    # window error
    tt=np.linspace(t0,t0+dT,26); X,TT=np.meshgrid(xg,tt)
    up=np.array(Unet(p,jnp.array(X.ravel(),jnp.float32),jnp.array(TT.ravel(),jnp.float32)))
    ue=np.array(uex(jnp.array(X.ravel()),jnp.array(TT.ravel())))
    print(f"window {k} [{t0:.2f},{t0+dT:.2f}] relL2={np.linalg.norm(up-ue)/np.linalg.norm(ue):.3e} maxerr={np.abs(up-ue).max():.3e} time={time.time()-t_start:.0f}s",flush=True)
pickle.dump({"NW":NW,"layers":layers,"models":[(t0,jax.tree.map(np.array,p)) for t0,p in models]},open(f"model_{TAG}.pkl","wb"))
