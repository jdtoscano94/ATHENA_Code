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
class Config:
    # --- Problem and Domain ---
    DATA_PATH = "/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes/Burgers_Periodic_SinPiX/2025_11_22_1730/burgers.npz"
    T_DOMAIN = [0.0, 1.0]
    X_DOMAIN = [-1.0, 1.0]
    # NU = 0.01 / np.pi  # Artificial Viscosity
    # --- Sampling (Initial) ---
    N_f = 2500  # Initial collocation points for PDE residual
    N_ic = 500    # Initial condition points
    N_bc = 500    # Boundary condition points
    # --- Model Architecture ---
    # --- Model Architecture ---
    OUTPUT_DIM = 1
    LAYERS = [256] * 5 + [OUTPUT_DIM]
    # ACTIVATION = jax.nn.tanh # This will be handled by the adaptive model
    # --- Training (RAR Curriculum) ---
    SEED = 42
    # Continuous training loop, no more cycles. Resampling is frequency-based.
    RESAMPLE_FREQUENCY = 1000
    RAR_N_CANDIDATES = 50000
    RAR_K_TOP = 500
    LAMBDA_IC = 100.0
    LAMBDA_BC = 100.0
    # --- Optimizer ---
    LEARNING_RATE = 1e-3
    END_LEARNING_RATE = 1e-5
    TOTAL_TRAIN_STEPS = 60000
    # --- Reporting ---
    LOG_FREQUENCY = 500
# Instantiate the config
args = Config()
# Set random seed for reproducibility
key = jax.random.PRNGKey(args.SEED)

# ## Cell 3: Data Generation
# --- 1. Load Validation Data ---
# Load data from the .npz file specified in the config
data = np.load(args.DATA_PATH)
t_coords, x_coords, u_sol = data['t'], data['x'], data['usol']
# Create a meshgrid and flatten it for validation during training
T_test, X_test = np.meshgrid(t_coords, x_coords, indexing='ij')
t_test = T_test.flatten()[:, None]
x_test = X_test.flatten()[:, None]
u_test = u_sol.flatten()[:, None]
# Store in a dictionary for easy access
test_data = {
    't': jnp.array(t_test),
    'x': jnp.array(x_test),
    'u': jnp.array(u_test)
}
# --- 2. Generate Training Collocation Points ---
# Define domain boundaries from the config
t_min, t_max = args.T_DOMAIN
x_min, x_max = args.X_DOMAIN
# --- PDE Residual Points (Interior) using LHS ---
key, subkey = jax.random.split(key)
# Generate points using Latin Hypercube Sampling
np.random.seed(args.SEED)  # Set NumPy seed for pyDOE's LHS
lhs_samples = lhs(2, samples=args.N_f)
# Scale samples to the t and x domains
t_f = jnp.array(t_min + (0.1 - t_min) * lhs_samples[:, 0:1])
x_f = jnp.array(x_min + (x_max - x_min) * lhs_samples[:, 1:2])
# --- Initial Condition Points (t=0) ---
key, subkey = jax.random.split(key)
x_ic = jax.random.uniform(subkey, shape=(args.N_ic, 1), minval=x_min, maxval=x_max)
t_ic = jnp.zeros_like(x_ic)
# Compute the exact solution at t=0 for the IC loss
u_ic = -jnp.sin(jnp.pi * x_ic)
# --- Boundary Condition Points (x=-1 and x=1) ---
key, subkey = jax.random.split(key)
t_bc = jax.random.uniform(subkey, shape=(args.N_bc, 1), minval=t_min, maxval=0.1)
# Create points at the left (x=-1) and right (x=1) boundaries
x_bc_left = jnp.full_like(t_bc, x_min)
x_bc_right = jnp.full_like(t_bc, x_max)
# --- 3. Assemble Training Data Dictionary ---
train_data = {
't_f': t_f, 'x_f': x_f,
    't_ic': t_ic, 'x_ic': x_ic, 'u_ic': u_ic,
't_bc': t_bc, 'x_bc_left': x_bc_left, 'x_bc_right': x_bc_right
}
# --- 4. Print Summary ---
print("--- Data Generation Summary ---")
print(f"PDE points (t_f, x_f): {train_data['t_f'].shape}, {train_data['x_f'].shape}")
print(f"IC points (t_ic, x_ic, u_ic): {train_data['t_ic'].shape}, {train_data['x_ic'].shape}, {train_data['u_ic'].shape}")
print(f"BC points (t_bc): {train_data['t_bc'].shape}")
print(f"Test data (t, x, u): {test_data['t'].shape}, {test_data['x'].shape}, {test_data['u'].shape}")


# ## Cell 4: Model
class GaussianFourierFeatures(nn.Module):
    """Gaussian Fourier Feature embedding layer."""
    degree: int
    s: float
    @nn.compact
    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        # Infer input dimension dynamically from the input tensor X.
        input_features = X.shape[-1]
        # Define the initializer for the projection matrix B
        def b_init(key, shape, dtype=jnp.float32):
            # Sample from a Gaussian distribution with scale 's'
            return self.s * jax.random.normal(key, shape, dtype)
        # Create the non-trainable projection matrix B
        B = self.param('B', b_init, (input_features, self.degree))
        # Project the input features according to the strategy: 2 * pi * B * v
        X_proj = 2 * jnp.pi * jnp.dot(X, B)
        # Concatenate the cosine and sine components
        return jnp.concatenate([jnp.cos(X_proj), jnp.sin(X_proj)], axis=-1)
class AdaptiveFourierFeatureMLP(nn.Module):
    """An Adaptive Fourier Feature MLP with trainable activation scalers."""
    features: Sequence[int]

    @nn.compact
    def __call__(self, t, x):
        """Forward pass for the Adaptive Fourier Feature MLP."""
        # Concatenate inputs
        in_arr = jnp.concatenate([t, x], axis=-1)
        
        # Apply a single Gaussian Fourier embedding layer as per the strategy
        x_embedded = GaussianFourierFeatures(degree=64, s=5.0)(in_arr)
        
        # MLP backbone with adaptive activation
        initializer = nn.initializers.xavier_uniform()
        x_processed = x_embedded
        for i, feat in enumerate(self.features[:-1]):
            x_processed = nn.Dense(features=feat, kernel_init=initializer)(x_processed)
            
            # --- Adaptive Activation ---
            # Create a trainable scalar 'alpha' for this layer, initialized to 1.0
            alpha = self.param(f'alpha_{i}', nn.initializers.ones, ())
            x_processed = jax.nn.tanh(alpha * x_processed)
        
        # Output layer
        output = nn.Dense(features=self.features[-1], kernel_init=initializer)(x_processed)
        return output
# Instantiate the model based on the configuration
model = AdaptiveFourierFeatureMLP(
    features=args.LAYERS
)
# Initialize model parameters
print("--- Initializing Model ---")
key, subkey = jax.random.split(key)
# Use dummy inputs with the correct feature dimension to initialize
dummy_t = jnp.ones((1, 1))
dummy_x = jnp.ones((1, 1))
params = model.init(subkey, dummy_t, dummy_x)['params']
# Define a concise apply function for inference
apply_fn = jit(model.apply)
# Print a summary of the initialized parameters
param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"Model initialized with {param_count:,} trainable parameters.")

# ## Cell 5: Optimizer Setup
# --- Optimizer Setup ---
# Define the learning rate schedule as per the strategy
lr_schedule = optax.cosine_decay_schedule(
    init_value=args.LEARNING_RATE,
    decay_steps=args.TOTAL_TRAIN_STEPS,
    alpha=args.END_LEARNING_RATE / args.LEARNING_RATE
)
# Initialize the Adam optimizer
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=lr_schedule)
)
# Initialize the optimizer state with the model parameters
opt_state = optimizer.init(params)
print("--- Optimizer Summary ---")
print(f"Optimizer: Adam with Global Norm Clipping (max_norm=1.0)")
print(f"Initial Learning Rate: {args.LEARNING_RATE}")
print(f"End Learning Rate: {args.END_LEARNING_RATE}")
print(f"LR Decay Schedule: Cosine Decay (steps={args.TOTAL_TRAIN_STEPS})")

# ## Cell 6: Training
# --- 1. Define PDE Residual Function ---
def PDE_residuals(params, t, x, apply_fn):
    """Computes the residual of the 1D Inviscid Burgers' equation."""
    # Define a wrapper for the model's forward pass to handle single-argument differentiation
    u_fn = lambda t, x: apply_fn({'params': params}, t, x).squeeze()
    # Calculate u(t, x)
    u = u_fn(t, x)
    # --- Calculate first derivatives (u_t, u_x) using JVP ---
    # Temporal derivative u_t
    u_t_fn = lambda t_inp: u_fn(t_inp, x)
    _, u_t = jvp(u_t_fn, (t,), (jnp.ones_like(t),))
    # Spatial derivative u_x
    u_x_fn = lambda x_inp: u_fn(t, x_inp)
    _, u_x = jvp(u_x_fn, (x,), (jnp.ones_like(x),))
    # Inviscid Burgers' equation residual: u_t + u * u_x = 0
    residual = u_t + u * u_x
    return residual
# --- 2. Define Loss Functions ---
def total_loss_fn(params, train_data, apply_fn):
    """Computes the total loss as a sum of PDE, IC, and BC losses."""
    # PDE Loss (on collocation points)
    res_f = PDE_residuals(params, train_data['t_f'], train_data['x_f'], apply_fn)
    loss_f = jnp.mean(res_f**2)
    # Initial Condition Loss (at t=0)
    u_pred_ic = apply_fn({'params': params}, train_data['t_ic'], train_data['x_ic'])
    loss_ic = jnp.mean((u_pred_ic - train_data['u_ic'])**2)
    # Periodic Boundary Condition Loss (at x=-1 and x=1)
    u_pred_bc_left = apply_fn({'params': params}, train_data['t_bc'], train_data['x_bc_left'])
    u_pred_bc_right = apply_fn({'params': params}, train_data['t_bc'], train_data['x_bc_right'])
    loss_bc = jnp.mean((u_pred_bc_left - u_pred_bc_right)**2)
    # Total loss is the sum of individual losses
    total_loss = loss_f + args.LAMBDA_IC * loss_ic + args.LAMBDA_BC * loss_bc
    # Return component losses as auxiliary data for logging
    return total_loss, (loss_f, loss_ic, loss_bc)
# --- 3. Define Metrics ---
@partial(jit, static_argnums=(2,))
def l2_error(params, test_data, apply_fn):
    """Calculates the relative L2 error against the test data."""
    u_pred = apply_fn({'params': params}, test_data['t'], test_data['x'])
    error = jnp.linalg.norm(u_pred - test_data['u']) / jnp.linalg.norm(test_data['u'])
    return error
# --- 4. Define JIT-Compiled Training Step ---
@partial(jit, static_argnums=(3,))
def train_step(params, opt_state, train_data, apply_fn):
    """Performs a single optimization step."""
    # Calculate loss and gradients. `has_aux=True` allows returning component losses.
    (loss_val, component_losses), grads = value_and_grad(total_loss_fn, has_aux=True)(params, train_data, apply_fn)
    # Update the optimizer state and model parameters
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val, component_losses
# --- 5. Main Training Loop (Causal Curriculum) ---
print("--- Starting Training with Causal Curriculum ---")
start_time = time.time()
# Initialize lists to store metrics history
all_errors, all_its, all_loss = [], [], []
# Define domain boundaries for convenience
t_min, t_full_max = args.T_DOMAIN
x_min, x_max = args.X_DOMAIN

# Initialize empty arrays to act as a buffer for RAR points
t_rar = jnp.array([]).reshape(0, 1)
x_rar = jnp.array([]).reshape(0, 1)

pbar = tqdm(range(args.TOTAL_TRAIN_STEPS), desc="Training", unit="iteration")
for it in pbar:
    # --- Causal Curriculum: Dynamically update T_max ---
    if it <= 40000:
        T_max = 0.1 + 0.9 * (it / 40000.0)
    else:
        T_max = t_full_max

    # --- 1. Resample base and BC points EVERY iteration within the current time horizon ---
    # Resample boundary condition points dynamically
    key, subkey = jax.random.split(key)
    train_data['t_bc'] = jax.random.uniform(subkey, (args.N_bc, 1), minval=t_min, maxval=T_max)

    # Resample base collocation points using LHS
    np.random.seed(it) # Use iteration for seed to ensure different samples
    lhs_samples = lhs(2, samples=args.N_f)
    t_f_new = jnp.array(t_min + (T_max - t_min) * lhs_samples[:, 0:1])
    x_f_new = jnp.array(x_min + (x_max - x_min) * lhs_samples[:, 1:2])

    # --- 2. Periodically update the RAR points buffer ---
    if it % args.RESAMPLE_FREQUENCY == 0:
        key, subkey = jax.random.split(key)
        t_cand = jax.random.uniform(subkey, (args.RAR_N_CANDIDATES, 1), minval=t_min, maxval=T_max)
        key, subkey = jax.random.split(key)
        x_cand = jax.random.uniform(subkey, (args.RAR_N_CANDIDATES, 1), minval=x_min, maxval=x_max)

        residuals_cand = PDE_residuals(params, t_cand, x_cand, apply_fn)
        abs_residuals = jnp.abs(residuals_cand).squeeze()
        _, top_k_indices = jax.lax.top_k(abs_residuals, k=args.RAR_K_TOP)

        # Update the RAR points buffer
        t_rar = t_cand[top_k_indices]
        x_rar = x_cand[top_k_indices]

    # --- 3. Combine the new base points with the current RAR points ---
    train_data['t_f'] = jnp.concatenate([t_f_new, t_rar])
    train_data['x_f'] = jnp.concatenate([x_f_new, x_rar])

    # --- 4. Perform one training step ---
    params, opt_state, loss_val, component_losses = train_step(params, opt_state, train_data, apply_fn)

    # --- 5. Log metrics at the specified frequency ---
    if it % args.LOG_FREQUENCY == 0:
        rel_l2_err = l2_error(params, test_data, apply_fn)
        loss_f, loss_ic, loss_bc = component_losses
        all_its.append(it)
        all_loss.append(loss_val.item())
        all_errors.append(rel_l2_err.item())
        pbar.set_description(
            f"It: {it:6d} | T_max: {T_max:.2f} | Loss: {loss_val:.3e} | L2 Error: {rel_l2_err:.3e} | "
            f"PDE: {loss_f:.2e} | IC: {loss_ic:.2e} | BC: {loss_bc:.2e}"
        )

# --- 6. Post-Training Summary ---
end_time = time.time()
training_time = end_time - start_time
print(f"\n--- Training Finished ---")
print(f"Total training time: {training_time:.2f} seconds")
# Store final parameters for evaluation in the next cell
current_params_for_eval = params


# ## Cell 7: Final Visualization & Error Reporting
T, X = np.meshgrid(t_coords, x_coords, indexing='ij')
# Ground Truth (already float64 due to T, X)
t = T.flatten()[:,None]
x = X.flatten()[:,None]
print('Solution:')
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
res = PDE_residuals(current_params_for_eval, t, x, apply_fn)
res_grid = jnp.abs(res.reshape(T.shape))
print(f"Residuals calculated. Min: {res_grid.min()}, Max: {res_grid.max()}")
res_log_norm = colors.LogNorm(vmin=res_grid.min()+1e-16, vmax=res_grid.max() + 1e-16)
mesh_res = ax_tr.pcolormesh(T, X, res_grid, cmap='jet', norm=res_log_norm, shading='gouraud')
fig.colorbar(mesh_res, ax=ax_tr)
ax_tr.set_xlabel('t')
ax_tr.set_ylabel('x')
ax_tr.set_title('|PDE Residual| (Log Scale)')
global_vmin = np.min([u, u_sol])
global_vmax = np.max([u, u_sol])
shared_levels = np.linspace(global_vmin, global_vmax, 500)
ax_bl = axes[1, 0]
contour1 = ax_bl.contourf(T, X, u_sol, cmap='jet', levels=shared_levels, extend='both')
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
error_data = np.abs(u_sol - u)
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
results_dict = {
    'all_errors': np.array(all_errors),
    'all_its': np.array(all_its),
    'all_loss': np.array(all_loss),
}
sio.savemat('training_results.mat', results_dict)
print("\n--- Results Saved ---")
print("Training history (errors, iterations, loss) saved to 'training_results.mat'")