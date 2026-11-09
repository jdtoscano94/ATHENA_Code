import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, json
d=np.load("solution_seed0.npz"); x=d["x"]; ext=[-1,1,-1,1]
fig,ax=plt.subplots(1,3,figsize=(15,4.3))
for a,Z,t in zip(ax,[d["u_pinn"],d["u_exact"],np.abs(d["u_pinn"]-d["u_exact"])],["PINN u","Exact u (eval only)","|error|"]):
    im=a.imshow(Z.T,origin="lower",extent=ext,cmap="viridis" if t!="|error|" else "magma"); a.set_title(t); a.set_xlabel("x"); a.set_ylabel("y"); fig.colorbar(im,ax=a)
plt.tight_layout(); plt.savefig("solution_error.png",dpi=120)
h=json.load(open("results_seed0.json"))["history"]
plt.figure(figsize=(6,4)); its=[i if s=="adam" else 15000+i for s,i,_ in h]
plt.semilogy(its,[l for *_,l in h],"o-"); plt.axvline(15000,ls="--",c="gray"); plt.xlabel("iteration (Adam | L-BFGS)"); plt.ylabel("normalised PDE loss"); plt.title("Training loss"); plt.tight_layout(); plt.savefig("loss.png",dpi=120)
