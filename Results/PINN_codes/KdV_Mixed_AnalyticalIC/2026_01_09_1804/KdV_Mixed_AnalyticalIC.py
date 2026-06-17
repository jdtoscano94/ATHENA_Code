# ## Cell 1: Core Imports and Project Setup
import os
import sys
import time
import random
from functools import partial
from typing import Sequence, Callable 

# --- JAX, Flax, and Optax ---
import jax
import jax.numpy as jnp
from jax import Array
from jax import jit, jvp, vjp, value_and_grad, lax, vmap
from jax.flatten_util import ravel_pytree
from flax import linen as nn
import optax

# --- NumPy, SciPy, and Plotting ---
import numpy as np
import scipy
import scipy.io as sio
from scipy import sparse
from scipy.linalg import cholesky, LinAlgError
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors

# --- Other Third-Party ---
from tqdm import tqdm
from pyDOE import lhs


#from jax import config
jax.config.update("jax_enable_x64", True)
# force jax to use one device
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

# ## Cell 2: Hyperparameter Configuration (argparse)
class Args:
    # --- Domain & Sampling ---
    t_domain = [0.0, 5.0]
    x_domain = [0.0, 20.0]
    N_f = 2000  # Collocation points
    N_ic = 200   # Initial condition points
    N_b1 = 200   # Boundary Left points
    N_b2 = 200   # Boundary Right points (Dirichlet)
    N_b3 = 200   # Boundary Right points (Neumann)
    # --- Analytical Solution Parameters (No change) ---
    c1 = 6.0
    c2 = 2.0
    x1 = -2.0
    x2 = 2.0
    # --- Model Architecture (Standard MLP) ---
    # Input: (t, x), 5 hidden layers of 128, 1 output u
    layers = [2] + [128] * 8 + [1]
    activation = nn.tanh
    # --- Optimization & Loss Annealing ---
    key = jax.random.PRNGKey(1234)
    iterations = 50000
    learning_rate_start = 1e-3
    learning_rate_end = 1e-5
    clip_norm = 1.0
    # --- Loss Weights & Annealing Schedule ---
    lambda_bc = 100.0  # Constant high weight for all IC/BC terms
    anneal_start = 5000   # Iteration to start ramping up PDE loss
    anneal_end = 10000    # Iteration to finish ramping up PDE loss
    # --- RAR Parameters ---
    rar_every = 2500      # How often to refine
    rar_start = 10000     # Iteration to start refining
    rar_candidates = 10000 # Number of candidate points to evaluate
    rar_add = 200         # Number of high-residual points to add
    # --- Reporting ---
    report_every = 500
    # --- File Naming ---
    Name = "KdV_PINN_MLP_Annealing_RAR"
args = Args()

# ## Cell 3: Data Generation
# --- Analytical and Ground Truth Data Generation ---
@jit
def u_analytical_scalar(t, x):
    """
    Analytical two-soliton solution for the KdV equation form: u_t + 6uu_x + u_xxx = 0.
    """
    # Parameters from args
    c1, c2, x1, x2 = args.c1, args.c2, args.x1, args.x2
    
    # Intermediate variables
    sqrt_c1 = jnp.sqrt(c1)
    sqrt_c2 = jnp.sqrt(c2)
    zeta1 = x - c1 * t - x1
    zeta2 = x - c2 * t - x2
    
    # Terms
    term1 = c1 * jnp.cosh(sqrt_c2 * zeta2 / 2)**2
    term2 = c2 * jnp.sinh(sqrt_c1 * zeta1 / 2)**2
    numerator = 2 * (c1 - c2) * (term1 + term2)
    
    denom_term1 = (sqrt_c1 - sqrt_c2) * jnp.cosh((sqrt_c1 * zeta1 + sqrt_c2 * zeta2) / 2)
    denom_term2 = (sqrt_c1 + sqrt_c2) * jnp.cosh((sqrt_c1 * zeta1 - sqrt_c2 * zeta2) / 2)
    denominator = (denom_term1 + denom_term2)**2
    
    u = numerator / denominator
    return u
# Vectorize the scalar function to handle entire arrays of (t, x) points
u_analytical = jit(vmap(u_analytical_scalar, in_axes=(0, 0), out_axes=0))
# Create the Neumann boundary condition ground truth function u_x(t, x) via automatic differentiation
@jit
def u_x_analytical_scalar(t, x):
    return jax.grad(u_analytical_scalar, argnums=1)(t, x)
u_x_analytical = jit(vmap(u_x_analytical_scalar, in_axes=(0, 0), out_axes=0))
# --- Training Data Sampling ---
# Domain boundaries
t_min, t_max = args.t_domain
x_min, x_max = args.x_domain
# Collocation points (for PDE residual) using Latin Hypercube Sampling
# The default for pyDOE lhs is 'random', which satisfies the "do not use maximin" requirement.
lhs_samples = lhs(2, samples=args.N_f)
t_f = t_min + (t_max - t_min) * lhs_samples[:, 0:1]
x_f = x_min + (x_max - x_min) * lhs_samples[:, 1:2]

# Create a new random key for sampling to ensure reproducibility
key, subkey = jax.random.split(args.key)

# Initial condition points (t=0)
x_ic = jax.random.uniform(subkey, (args.N_ic, 1), minval=x_min, maxval=x_max)
t_ic = jnp.full_like(x_ic, t_min)
u_ic = u_analytical(t_ic.flatten(), x_ic.flatten()).reshape(-1, 1)

# Left boundary points (x=0)
key, subkey = jax.random.split(key)
t_bc1 = jax.random.uniform(subkey, (args.N_b1, 1), minval=t_min, maxval=t_max)
x_bc1 = jnp.full_like(t_bc1, x_min)
u_bc1 = u_analytical(t_bc1.flatten(), x_bc1.flatten()).reshape(-1, 1)

# Right boundary points (x=20) for Dirichlet
key, subkey = jax.random.split(key)
t_bc2 = jax.random.uniform(subkey, (args.N_b2, 1), minval=t_min, maxval=t_max)
x_bc2 = jnp.full_like(t_bc2, x_max)
u_bc2 = u_analytical(t_bc2.flatten(), x_bc2.flatten()).reshape(-1, 1)

# Right boundary points (x=20) for Neumann
key, subkey = jax.random.split(key)
t_bc3 = jax.random.uniform(subkey, (args.N_b3, 1), minval=t_min, maxval=t_max)
x_bc3 = jnp.full_like(t_bc3, x_max)
u_x_bc3 = u_x_analytical(t_bc3.flatten(), x_bc3.flatten()).reshape(-1, 1)
# --- Validation Data ---
# Create a high-resolution grid for validation and plotting
t_star = jnp.linspace(t_min, t_max, 256)
x_star = jnp.linspace(x_min, x_max, 100)
T, X = jnp.meshgrid(t_star, x_star, indexing='ij')
t_val = T.flatten()
x_val = X.flatten()
Exact = u_analytical(t_val, x_val).reshape(T.shape)
print(f"Generated {args.N_f} collocation points.")
print(f"Generated {args.N_ic} initial condition points (t=0).")
print(f"Generated {args.N_b1} (x=0), {args.N_b2} (x=20, Dirichlet), and {args.N_b3} (x=20, Neumann) boundary points.")
print(f"Validation grid shape: {Exact.shape}")

# ## Cell 4: Model
class MLP(nn.Module):
    """A standard Multi-Layer Perceptron."""
    features: Sequence[int]
    t_domain: Sequence[float]
    x_domain: Sequence[float]
    @nn.compact
    def __call__(self, t, x):
        # --- Input Normalization ---
        # Normalize t and x to the range [-1, 1]
        t_min, t_max = self.t_domain
        x_min, x_max = self.x_domain
        t_norm = 2.0 * (t - t_min) / (t_max - t_min) - 1.0
        x_norm = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
        X = jnp.hstack([t_norm, x_norm])
        # --- MLP Layers ---
        # The Glorot normal initializer is specified by the strategy.
        initializer = nn.initializers.glorot_normal()
        for i, units in enumerate(self.features[:-1]):
            X = nn.Dense(features=units, kernel_init=initializer)(X)
            X = nn.tanh(X)
        # Final layer (no activation)
        X = nn.Dense(features=self.features[-1], kernel_init=initializer)(X)
        return X
# Instantiate the model
# The first element of args.layers (input dim) is handled implicitly,
# so we pass the subsequent layer dimensions.
model = MLP(
    features=args.layers[1:],
    t_domain=args.t_domain,
    x_domain=args.x_domain
)
# Initialize parameters
# Create a dummy input with the correct shape (N, 1) for t and x
dummy_t = jnp.ones((1, 1))
dummy_x = jnp.ones((1, 1))
# The new model is deterministic and doesn't require a 'params' RNG stream for feature generation.
# The key is passed directly for weight initialization.
params = model.init(args.key, dummy_t, dummy_x)['params']
# Define the model's apply function for predictions
apply_fn = jit(model.apply)
print(f"Model initialized with {sum(p.size for p in jax.tree_util.tree_leaves(params))} parameters.")

# ## Cell 5: Optimizer Setup
# --- PDE Residual Function (Updated for deterministic model) ---
def PDE_residuals(params, t, x, apply_fn):
    """
    Computes the residual of the KdV equation: f = u_t + 6*u*u_x + u_xxx
    """
    def u_scalar(t_s, x_s):
        # The model is now deterministic, no RNG key needed for apply_fn
        return apply_fn({'params': params}, t_s.reshape(1, -1), x_s.reshape(1, -1))[0, 0]
    # Higher-order derivatives via repeated jax.grad
    u_t_fn = jax.grad(u_scalar, argnums=0)
    u_x_fn = jax.grad(u_scalar, argnums=1)
    u_xx_fn = jax.grad(u_x_fn, argnums=1)
    u_xxx_fn = jax.grad(u_xx_fn, argnums=1)
    # Vectorize the derivative functions to apply them over the entire batch
    u_t_vmap = vmap(u_t_fn)
    u_x_vmap = vmap(u_x_fn)
    u_xxx_vmap = vmap(u_xxx_fn)
    # Compute model predictions and derivatives
    u_pred = apply_fn({'params': params}, t, x)
    u_t = u_t_vmap(t.flatten(), x.flatten()).reshape(-1, 1)
    u_x = u_x_vmap(t.flatten(), x.flatten()).reshape(-1, 1)
    u_xxx = u_xxx_vmap(t.flatten(), x.flatten()).reshape(-1, 1)
    residual = u_t + 6.0 * u_pred * u_x + u_xxx
    return residual
# --- Unified Loss Function with Annealing ---
@jit
def loss_fn(params, it, t_f, x_f):
    """Unified loss function with annealed PDE weight."""
    # 1. PDE Loss
    f_pred = PDE_residuals(params, t_f, x_f, apply_fn)
    loss_pde = jnp.mean(f_pred ** 2)
    # 2. Initial Condition Loss
    u_ic_pred = apply_fn({'params': params}, t_ic, x_ic)
    loss_ic = jnp.mean((u_ic_pred - u_ic) ** 2)
    # 3. Boundary Condition Losses (Dirichlet)
    u_bc1_pred = apply_fn({'params': params}, t_bc1, x_bc1)  # Left
    u_bc2_pred = apply_fn({'params': params}, t_bc2, x_bc2)  # Right
    loss_bc_dirichlet = jnp.mean((u_bc1_pred - u_bc1) ** 2) + jnp.mean((u_bc2_pred - u_bc2) ** 2)
    # 4. Neumann Boundary Condition Loss
    def u_scalar_for_grad(t_s, x_s):
        return apply_fn({'params': params}, t_s.reshape(1, -1), x_s.reshape(1, -1))[0, 0]
    u_x_model_fn = vmap(jax.grad(u_scalar_for_grad, argnums=1))
    u_x_bc3_pred = u_x_model_fn(t_bc3.flatten(), x_bc3.flatten()).reshape(-1, 1)
    loss_neumann = jnp.mean((u_x_bc3_pred - u_x_bc3) ** 2)
    # Combine all data-driven losses
    loss_data = loss_ic + loss_bc_dirichlet + loss_neumann
    # 5. Annealing logic for PDE loss weight
    progress = jnp.clip((it - args.anneal_start) / (args.anneal_end - args.anneal_start), 0.0, 1.0)
    lambda_pde_annealed = lax.cond(it < args.anneal_start, lambda: 0.0, lambda: progress)
    # 6. Total Loss Calculation
    total_loss = args.lambda_bc * loss_data + lambda_pde_annealed * loss_pde
    loss_components = {
        "total": total_loss, "pde": loss_pde, "ic": loss_ic,
        "bc": loss_bc_dirichlet, "neumann": loss_neumann, "lambda_pde": lambda_pde_annealed
    }
    return total_loss, loss_components
# --- Optimizer Setup with Cosine Decay and Gradient Clipping ---
lr_schedule = optax.cosine_decay_schedule(
    init_value=args.learning_rate_start,
    decay_steps=args.iterations,
    alpha=args.learning_rate_end / args.learning_rate_start
)
optimizer = optax.chain(
    optax.clip_by_global_norm(args.clip_norm),
    optax.adam(learning_rate=lr_schedule)
)
opt_state = optimizer.init(params)
# --- JIT-Compiled Unified Training Step ---
@jit
def train_step(params, opt_state, it, t_f, x_f):
    (loss_val, loss_components), grads = value_and_grad(loss_fn, argnums=0, has_aux=True)(params, it, t_f, x_f)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val, loss_components
print("Optimizer with annealing and unified training step initialized.")


# ## Cell 6: Training
# --- Main Training Loop ---
print("Starting training with loss annealing...")
start_time = time.time()
# Lists to store metrics
all_its = []
all_loss = []
all_errors = []
all_L_2 = []
all_L_infty = []
# Use tqdm for a progress bar
pbar = tqdm(range(args.iterations), desc="Training")
# The main key needs to be accessible and updatable inside the loop for RAR
key = args.key

for it in pbar:
    # --- Residual-based Adaptive Refinement (RAR) ---
    if it % args.rar_every == 0 and it >= args.rar_start:
        # Generate candidate points for refinement
        key, subkey = jax.random.split(key)
        t_cand = jax.random.uniform(subkey, (args.rar_candidates, 1), minval=args.t_domain[0], maxval=args.t_domain[1])
        key, subkey = jax.random.split(key)
        x_cand = jax.random.uniform(subkey, (args.rar_candidates, 1), minval=args.x_domain[0], maxval=args.x_domain[1])

        # Compute residuals on the candidate points
        f_cand = PDE_residuals(params, t_cand, x_cand, apply_fn)
        f_cand_abs = jnp.abs(f_cand).flatten()

        # Find the indices of the points with the highest residuals
        top_indices = jnp.argsort(f_cand_abs)[-args.rar_add:]

        # Add these high-residual points to the training set
        t_f = jnp.vstack([t_f, t_cand[top_indices]])
        x_f = jnp.vstack([x_f, x_cand[top_indices]])
        
        # Update description to show new number of points
        pbar.set_description(f"Training (N_f={t_f.shape[0]})")

    # --- Unified Training Step ---
    # Pass the (potentially updated) t_f and x_f to the train_step
    params, opt_state, loss_val, loss_components = train_step(params, opt_state, it, t_f, x_f)
    # --- Reporting and Error Calculation ---
    if it % args.report_every == 0 or it == args.iterations - 1:
        # Predict solution on the validation grid (no RNG key needed)
        u_pred_val = apply_fn({'params': params}, t_val.reshape(-1, 1), x_val.reshape(-1, 1))
        u_pred_val = u_pred_val.reshape(Exact.shape)
        # Calculate relative L2 error
        error_l2 = jnp.linalg.norm(u_pred_val - Exact) / jnp.linalg.norm(Exact)
        # Calculate L-infinity error
        error_linf = jnp.max(jnp.abs(u_pred_val - Exact))
        # Store metrics
        all_its.append(it)
        all_loss.append(loss_val.item())
        all_errors.append(error_l2.item())
        all_L_2.append(jnp.linalg.norm(u_pred_val - Exact).item())
        all_L_infty.append(error_linf.item())
        # Update progress bar description
        pbar.set_postfix({
            "Loss": f"{loss_val.item():.4e}",
            "L2 Error": f"{error_l2.item():.4e}",
            "PDE": f"{loss_components['pde']:.2e}",
            "IC": f"{loss_components['ic']:.2e}",
            "BC": f"{loss_components['bc']:.2e}",
            "Neumann": f"{loss_components['neumann']:.2e}",
            "λ_pde": f"{loss_components['lambda_pde']:.2f}"
        })
# --- Finalization ---
training_time = time.time() - start_time
print(f"\nTraining finished in {training_time:.2f} seconds.")
# Store the final trained parameters for evaluation and plotting
current_params_for_eval = params

# ## Cell 7: Final Visualization & Error Reporting
# The T, X grid for plotting is already created during data generation (Cell 3)
# T, X were created from t_star, x_star
# Ground Truth (already float64 due to T, X)
t = T.flatten()[:,None]
x = X.flatten()[:,None]
# The ground truth solution on the grid is stored in 'Exact'
print('Solution:')
# The apply_fn expects params in a dictionary. The new model is deterministic.
u = apply_fn({'params': current_params_for_eval}, t, x)
u = u.reshape(T.shape)
fig, axes = plt.subplots(2, 3, figsize=(14, 6))
ax_tl = axes[0, 0]
ax_tl.plot(all_its, all_loss, label='Total Loss', color='blue')
ax_tl.set_yscale('log')
ax_tl.set_title('Total Loss History (Log Scale)')
ax_tl.set_xlabel('Iterations')
ax_tl.set_ylabel('Loss (log scale)')
ax_tl.grid(True, which="both", ls="--")
ax_tl.legend()
ax_tm = axes[0, 1]
ax_tm.plot(all_its, all_errors, label='Rel. L2 Error', color='green')
ax_tm.set_yscale('log')
ax_tm.set_title('Relative $L^2$ Error History (Log Scale)')
ax_tm.set_xlabel('Iterations')
ax_tm.set_ylabel('Rel. $L^2$ Error (log scale)')
ax_tm.grid(True, which="both", ls="--")
ax_tm.legend()
ax_tr = axes[0, 2]
print("Calculating PDE residuals for plotting...")
print("Calculating PDE residuals for plotting...")
res = PDE_residuals(current_params_for_eval, t, x, apply_fn)
res_grid = jnp.abs(res.reshape(T.shape))
print(f"Residuals calculated. Min: {res_grid.min()}, Max: {res_grid.max()}")
res_log_norm = colors.LogNorm(vmin=res_grid.min()+1e-16, vmax=res_grid.max() + 1e-16)
mesh_res = ax_tr.pcolormesh(T, X, res_grid, cmap='jet', norm=res_log_norm, shading='gouraud')
fig.colorbar(mesh_res, ax=ax_tr)
ax_tr.set_xlabel('t')
ax_tr.set_ylabel('x')
ax_tr.set_title('|PDE Residual| (Log Scale)')
global_vmin = np.min([u, Exact])
global_vmax = np.max([u, Exact])
shared_levels = np.linspace(global_vmin, global_vmax, 500)
ax_bl = axes[1, 0]
contour1 = ax_bl.contourf(T, X, Exact, cmap='jet', levels=shared_levels, extend='both')
fig.colorbar(contour1, ax=ax_bl)
ax_bl.set_xlabel('t')
ax_bl.set_ylabel('x')
ax_bl.set_title('Reference')
ax_bm = axes[1, 1]
contour2 = ax_bm.contourf(T, X, u, cmap='jet', levels=shared_levels, extend='both')
fig.colorbar(contour2, ax=ax_bm)
ax_bm.set_xlabel('t')
ax_bm.set_title('Prediction')
ax_br = axes[1, 2]
error_data = np.abs(Exact - u)
err_log_norm = colors.LogNorm(vmin=error_data.min()+1e-16, vmax=error_data.max() + 1e-16)
mesh_err = ax_br.pcolormesh(T, X, error_data, cmap='jet', norm=err_log_norm, shading='gouraud')
fig.colorbar(mesh_err, ax=ax_br)
ax_br.set_xlabel('t')
ax_br.set_title('Pointwise Error (Log Scale)')
plt.tight_layout()
plt.savefig('summary_all.png', dpi=150) # 150 DPI is a good balance for token size
plt.show()
print('Minimum achieved  RL2 error:',np.nanmin(all_errors))

# ## Cell 8: Save Experiment Results to .mat File
#
results_dict = {
    'model_name': args.Name,
    'all_errors': all_errors,
    'all_its': all_its,
    'all_loss': all_loss,
    'all_L_infty': all_L_infty,
    'all_L_2': all_L_2,
}

# Define the filename and save the dictionary
filename = f"{args.Name}_results.mat"
sio.savemat(filename, results_dict)

print(f"Experiment results saved to {filename}")