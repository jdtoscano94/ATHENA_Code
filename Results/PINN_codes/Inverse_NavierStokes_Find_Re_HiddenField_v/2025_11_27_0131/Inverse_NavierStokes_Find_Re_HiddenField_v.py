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

# --- Local Application Libraries ---
from Crunch.Models.layers import * 
from Crunch.Auxiliary.metrics import relative_l2 
from Crunch.Optimizers.minimize import minimize
from Crunch.Auxiliary.utils import static_options_SSBroyden


#from jax import config
jax.config.update("jax_enable_x64", True)
# force jax to use one device
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

# ## Cell 2: Hyperparameter Configuration (argparse)
#
import argparse
# Set up argument parser
parser = argparse.ArgumentParser(description='Tuning Parameters')
parser.add_argument('--Equation', type=str, default='Navier_Stokes', help='Name of equation')
parser.add_argument('--Name', type=str, default='SS_Broyden', help='Name of the experiment')
parser.add_argument('--NC', type=int, default=200000, help='Number of samples for training')
parser.add_argument('--NI', type=int, default=512, help='Number of iterations')
parser.add_argument('--NB', type=int, default=512, help='Batch size')
parser.add_argument('--NC_TEST', type=int, default=100, help='Number of test samples')
parser.add_argument('--SEED', type=int, default=9999, help='Random seed')
parser.add_argument('--EPOCHS', type=int, default=10000, help='Number of training epochs')
parser.add_argument('--N_LAYERS', type=int, default=3, help='Number of layers in the network')
parser.add_argument('--HIDDEN', type=int, default=30, help='Number of hidden units per layer')
parser.add_argument('--FEATURES', type=int, default=2, help='Feature size')
parser.add_argument('--embedding_degree', type=int, default=0, help='Degree of features')
parser.add_argument('--lr_fact', type=float, default=0.2, help='Scale Lr')
parser.add_argument('--eta', type=float, default=0.01, help='Learning rate or step size for adaptive gamma')
parser.add_argument('--gamma', type=float, default=0.999, help='Decay rate for adaptive gamma')
parser.add_argument('--gamma_bfgs', type=float, default=0.1, help='Decay rate for adaptive gamma')
parser.add_argument('--gamma_grads', type=float, default=0.99, help='Decay rate for adaptive gamma')
parser.add_argument('--alpha', type=float, default=0.9999, help='memory rate for moving average of the global weights')
parser.add_argument('--cap_RBA', type=float, default=20, help='Cap limit for RBA')
parser.add_argument('--max_RBA', type=float, help='Maximum RBA value, default calculated as eta / (1 - gamma)')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for learning rate schedule')
parser.add_argument('--LR', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--decay_step', type=int,default=5000, help='Decay step size')
parser.add_argument('--Note', type=str, default='', help='In case')
parser.add_argument('--batch_size', type=int, default=20000, help='batch_size')
parser.add_argument('--k_samp', type=float, default=1.0, help='Enhance outliers smoothing factor')
parser.add_argument('--c_samp', type=float, default=0.0, help='homogenize')
parser.add_argument('--phi', type=float, default=1, help='Enhance outliers smoothing factor')
parser.add_argument('--c_log', type=float, default=1.0, help='homogenize')
parser.add_argument('--N_change', type=int, default=100, help='homogenize')
# Parse arguments and display them
args, unknown = parser.parse_known_args()
for arg, value in vars(args).items():
    print(f'{arg}: {value}')
# Data
NC = args.NC
NI = args.NI
NB = args.NB
NC_TEST = args.NC_TEST
#Representation Model
#- Input transfromations
embedding_degree = args.embedding_degree
# - Architecture
N_LAYERS = args.N_LAYERS # Number of Layers
HIDDEN = args.HIDDEN #hidden neurons per layer
FEATURES = args.FEATURES # Number of output fetures
# - 1st Stage Optmization
## Adam
EPOCHS = args.EPOCHS
eta = args.eta
decay_rate = args.decay_rate
LR = args.LR
lr0 = LR
decay_step = args.decay_step
## vRBA Params
gamma = args.gamma
alpha = args.alpha
c_log=args.c_log
phi=args.phi
gamma_bfgs=args.gamma_bfgs
eta_bfgs = 1-gamma_bfgs
##  We use a self scaling algoirhtm to automatically tune the global weights and to increase the cap of RBA to improve learning dunamics
max_RBA0 = args.max_RBA if args.max_RBA is not None else eta / (1 - gamma)
cap_RBA = args.cap_RBA
gamma_grads=args.gamma_grads
## vRBA Params Second order_ Sampling + SSBroycen
k_samp_bfgs = args.k_samp
c_samp_bfgs = args.c_samp
Nchange = args.N_change  # Number of SSBroyden steps before resampling
batch_size=args.batch_size
Nbfgs=30000 # Number of SSBroyden BFGs iterations
# random key
SEED = args.SEED
key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)
# Initialize NumPy seed
np.random.seed(SEED)
# Name of the project
args.Name=args.Name+f'vRBA-phi:{phi:.2f}-c_log:{c_log:.2f}-bs:{batch_size}_k:{k_samp_bfgs:.2f}_N:{Nchange:.2f}_Seed:{SEED}'+args.Note
print(args.Name)
#Define folder
file_path = os.getcwd()
project_root = os.path.dirname(os.path.dirname(file_path))

# ## Cell 3: Data Loading and Ground Truth Definition
available_data_sets_path='/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Documents/codes/Data/'
data = np.load(available_data_sets_path+'ldc_Re500.npz')
print(data.keys())
Re_true = 500.0 # True Reynolds number for evaluation
# Extract raw 1D data for evaluation
x_raw = data['x'].flatten().astype(jnp.float64)
y_raw = data['y'].flatten().astype(jnp.float64)
u_raw = data['u'].flatten().astype(jnp.float64)
v_raw = data['v'].flatten().astype(jnp.float64)
p_raw = data['p'].flatten().astype(jnp.float64)
# For plotting, interpolate to a regular grid
# Define a finer grid for smoother plots
num_plot_points = 200
x_grid_plot = np.linspace(x_raw.min(), x_raw.max(), num_plot_points)
y_grid_plot = np.linspace(y_raw.min(), y_raw.max(), num_plot_points)
X_plot, Y_plot = np.meshgrid(x_grid_plot, y_grid_plot)
# Interpolate u, v, p onto the regular grid for plotting
points = np.vstack((x_raw, y_raw)).T
u_plot_gt = scipy.interpolate.griddata(points, u_raw, (X_plot, Y_plot), method='cubic')
v_plot_gt = scipy.interpolate.griddata(points, v_raw, (X_plot, Y_plot), method='cubic')
p_plot_gt = scipy.interpolate.griddata(points, p_raw, (X_plot, Y_plot), method='cubic')
# Prepare 1D arrays for model evaluation and error computation
# These are the original unstructured points, reshaped for model input
x_eval = x_raw[:, None]
y_eval = y_raw[:, None]
u_gt_eval = u_raw[:, None]
v_gt_eval = v_raw[:, None]
p_gt_eval = p_raw[:, None] # For plotting, not error calculation

# --- Data for L_data (500 noisy u observations) ---
N_data_points = 500
key, subkey_data = jax.random.split(key)
data_indices = jax.random.choice(subkey_data, len(x_raw), shape=(N_data_points,), replace=False)

x_data = x_raw[data_indices][:, None]
y_data = y_raw[data_indices][:, None]
u_data_true = u_raw[data_indices][:, None] # True u at data points for noise calculation

# Add 5% Gaussian noise ONLY to u_data
noise_level = 0.05
u_data_noise = noise_level * jnp.std(u_data_true) * jax.random.normal(subkey_data, u_data_true.shape, dtype=jnp.float64)
u_data_noisy = u_data_true + u_data_noise

# ## Cell 4: Data Processing, Sampling, and Visualization
# Collocation points
lb_D = jnp.array([0.0, 0.0])  # x_min, y_min
ub_D = jnp.array([1.0, 1.0])  # x_max, y_max
# Generate collocation points and ensure they are float64
X_c = lb_D + (ub_D - lb_D) * lhs(2, NC)
X_c = X_c.astype(jnp.float64)
x_pde = X_c[:, 0:1]
y_pde = X_c[:, 1:2]
# Initial conditions are not applicable for steady-state problems
# Boundary Conditions
# Use keys for random sampling
key, *bc_keys = jax.random.split(key, 5)  # Split key for 4 BCs + 1 for future use
# Bottom Wall (y=0, 0<=x<=1)
x_bc_bottom = jax.random.uniform(bc_keys[0], (NB, 1), minval=lb_D[0], maxval=ub_D[0]).astype(jnp.float64)
y_bc_bottom = jnp.zeros_like(x_bc_bottom)
u_bc_bottom = jnp.zeros_like(x_bc_bottom)
v_bc_bottom = jnp.zeros_like(x_bc_bottom)
# Top Wall (y=1, 0<=x<=1) - Moving Lid
x_bc_top = jax.random.uniform(bc_keys[1], (NB, 1), minval=lb_D[0], maxval=ub_D[0]).astype(jnp.float64)
y_bc_top = jnp.ones_like(x_bc_top)
u_bc_top = 1 - jnp.cosh(50 * (x_bc_top - 0.5)) / jnp.cosh(25.0)
v_bc_top = jnp.zeros_like(x_bc_top)
# Left Wall (x=0, 0<=y<=1)
y_bc_left = jax.random.uniform(bc_keys[2], (NB, 1), minval=lb_D[1], maxval=ub_D[1]).astype(jnp.float64)
x_bc_left = jnp.zeros_like(y_bc_left)
u_bc_left = jnp.zeros_like(y_bc_left)
v_bc_left = jnp.zeros_like(y_bc_left)
# Right Wall (x=1, 0<=y<=1)
y_bc_right = jax.random.uniform(bc_keys[3], (NB, 1), minval=lb_D[1], maxval=ub_D[1]).astype(jnp.float64)
x_bc_right = jnp.ones_like(y_bc_right)
u_bc_right = jnp.zeros_like(y_bc_right)
v_bc_right = jnp.zeros_like(y_bc_right)
# Pressure Gauge Point (p(0,0) = 0) - Removed as per strategy (no pressure BCs)
# x_gauge = jnp.array([[0.0]]).astype(jnp.float64)
# y_gauge = jnp.array([[0.0]]).astype(jnp.float64)
# p_gauge = jnp.array([[0.0]]).astype(jnp.float64)
# Combine all BC points for convenience in train_data
x_bcs = jnp.concatenate([x_bc_bottom, x_bc_top, x_bc_left, x_bc_right], axis=0)
y_bcs = jnp.concatenate([y_bc_bottom, y_bc_top, y_bc_left, y_bc_right], axis=0)
u_bcs = jnp.concatenate([u_bc_bottom, u_bc_top, u_bc_left, u_bc_right], axis=0)
v_bcs = jnp.concatenate([v_bc_bottom, v_bc_top, v_bc_left, v_bc_right], axis=0)
# This data tuple is now consistently float64
train_data = (x_pde, y_pde, x_bcs, y_bcs, u_bcs, v_bcs, x_data, y_data, u_data_noisy)
lambdas = jnp.ones((x_pde.shape[0], 2)) * 0.1  # Initialize lambdas for PDE points with shape (NC, 2)
# Plotting code
fig = plt.figure(figsize=(20, 5))
# Scatter plot for collocation, initial, and boundary points
ax1 = fig.add_subplot(141)
ax1.scatter(x_pde, y_pde, color='blue', label='Collocation Points', s=0.2, alpha=0.5)
ax1.scatter(x_bcs, y_bcs, color='red', label='Dirichlet BCs', s=5, alpha=0.8)
ax1.scatter(x_data, y_data, color='green', label='Data Points (u_noisy)', s=5, alpha=0.8)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Training Points Distribution')
ax1.set_xlim([0,1])
ax1.set_ylim([0,1])
ax1.set_aspect('equal', adjustable='box')
ax1.legend(loc='upper right')
# Surface plot for Exact u solution
ax2 = fig.add_subplot(142)
contour_u = ax2.contourf(X_plot, Y_plot, u_plot_gt, cmap='jet', levels=50)
fig.colorbar(contour_u, ax=ax2)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Exact u Solution')
ax2.set_aspect('equal', adjustable='box')
# Surface plot for Exact v solution
ax3 = fig.add_subplot(143)
contour_v = ax3.contourf(X_plot, Y_plot, v_plot_gt, cmap='jet', levels=50)
fig.colorbar(contour_v, ax=ax3)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Exact v Solution')
ax3.set_aspect('equal', adjustable='box')
# Surface plot for Exact p solution
ax4 = fig.add_subplot(144)
contour_p = ax4.contourf(X_plot, Y_plot, p_plot_gt, cmap='jet', levels=50)
fig.colorbar(contour_p, ax=ax4)
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_title('Exact p Solution')
ax4.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig('Boundaries_data.png', dpi=300)
plt.show()

# ## Cell 5: Model Architecture Definition (The PINN Structure)
class Random_Fourier_Embedding(nn.Module):
    """Random Fourier Feature embedding layer."""
    degree: int
    s: float = 10.0
    @nn.compact
    def __call__(self, X: Array) -> Array:
        # Infer input dimension dynamically from the input tensor X.
        input_features = X.shape[-1]
        def b_init(key, shape, dtype=jnp.float32):
            return self.s * jax.random.normal(key, shape, dtype)
        B = self.param('B', b_init, (input_features, self.degree))
        X_proj = jnp.dot(X, B)
        # Concatenate original features with their sinusoidal projections.
        return jnp.concatenate([X, jnp.sin(X_proj), jnp.cos(X_proj)], axis=-1)
class MLP(nn.Module):
    features: Sequence[int]
    activation: Callable = nn.tanh
    @nn.compact
    def __call__(self, Z: Array) -> Array:
        init = nn.initializers.glorot_normal()
        for fs in self.features[:-1]:
            Z = nn.Dense(fs, kernel_init=init)(Z)
            # Use the activation function from the attribute
            Z = self.activation(Z)
        Z = nn.Dense(self.features[-1], kernel_init=init)(Z)
        return Z
class PINN(nn.Module):
    features: Sequence[int]
    embedding_degree: int = 1
    Lx: float = 2.0
    activation: Callable = nn.tanh
    def setup(self):
        if self.embedding_degree > 0:
            self.enbedding = Random_Fourier_Embedding(degree=self.embedding_degree)
        self.MLP = MLP(features=self.features, activation=self.activation)
        # Inverse problem parameter: alpha for Re_est
        self.alpha = self.param('alpha', nn.initializers.constant(-10.0), (1,))
    @nn.compact
    def __call__(self, x: Array, y: Array) -> Array:
        # Helper function to prepare input for MLP, including embedding if enabled
        def prepare_input(x_coord, y_coord):
            if self.embedding_degree > 0:
                x_embedded = self.enbedding(x_coord)
                y_embedded = self.enbedding(y_coord)
                return jnp.concatenate([x_embedded, y_embedded], axis=-1)
            else:
                # Fallback to original scaling if no embedding is used
                return jnp.concatenate([2.0 * (x_coord - 0.5), 2.0 * (y_coord - 0.5)], axis=-1)
        # Initial MLP call for psi and p using the prepared input
        X_concat = prepare_input(x, y)
        output = self.MLP(X_concat)
        psi = output[:, 0:1]
        p = output[:, 1:2]
        # Derive u and v from psi using JAX's jvp
        # u = d(psi)/dy
        psi_y_fn = lambda y_val: self.MLP(prepare_input(x, y_val))[:, 0:1]
        u = jvp(psi_y_fn, (y,), (jnp.ones_like(y),))[1]
        # v = -d(psi)/dx
        psi_x_fn = lambda x_val: self.MLP(prepare_input(x_val, y))[:, 0:1]
        v = -jvp(psi_x_fn, (x,), (jnp.ones_like(x),))[1]
        Re_est = 300 + (525 - 300) * jax.nn.sigmoid(self.alpha)  # Sigmoid scaling for Re with new bounds
        return u, v, p, psi, Re_est


# ## Cell 6: Model and Optimizer Initialization
feat_sizes = tuple([HIDDEN for _ in range(N_LAYERS)] + [FEATURES])
print(feat_sizes)
# make & init model
model = PINN(feat_sizes,activation=nn.tanh,embedding_degree=embedding_degree)
params = model.init(subkey, jnp.ones((NC, 1), dtype=jnp.float64), jnp.ones((NC, 1), dtype=jnp.float64))
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), params)
optimizers = {}
for key_params in params['params'].keys():
    if key_params == 'alpha': # Specific optimizer for alpha
        optimizers[key_params] = optax.adam(optax.exponential_decay(lr0 * args.lr_fact, decay_step, decay_rate, staircase=False))
    else:
        optimizers[key_params] = optax.adam(optax.exponential_decay(lr0, decay_step, decay_rate, staircase=False))
# Initialize optimizer states for each parameter group
states = {key_params: optim.init(params['params'][key_params]) for key_params, optim in optimizers.items()}
# forward & loss function
apply_fn = jax.jit(model.apply)
total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(total_params )

# ## Cell 7: PDE Residual Definition
def hvp_fwdfwd(f, primals, tangents):
    # Compute the gradient of f at primals in the direction of tangents
    _, grad_f = jvp(f, primals, tangents)
    # Now compute the Hessian-vector product by differentiating grad_f
    _, hvp = jvp(lambda p: jvp(f, (p,), tangents)[1], primals, tangents)
    return hvp
def Navier_Stokes_Residuals(params, x, y, apply_fn):
    # Get u, v, p, psi, Re_est values directly from apply_fn
    u, v, p, psi, Re_est = apply_fn(params, x, y)
    # Define helper functions for differentiation of u, v, p
    # These functions take a single coordinate (x or y) and return the corresponding field
    u_x_fn = lambda x_val: apply_fn(params, x_val, y)[0]
    u_y_fn = lambda y_val: apply_fn(params, x, y_val)[0]
    v_x_fn = lambda x_val: apply_fn(params, x_val, y)[1]
    v_y_fn = lambda y_val: apply_fn(params, x, y_val)[1]
    p_x_fn = lambda x_val: apply_fn(params, x_val, y)[2]
    p_y_fn = lambda y_val: apply_fn(params, x, y_val)[2]
    # Tangent vectors for jvp (forward-mode autodiff)
    v_x_tangent = jnp.ones_like(x)
    v_y_tangent = jnp.ones_like(y)
    # First derivatives for u
    ux = jvp(u_x_fn, (x,), (v_x_tangent,))[1]
    uy = jvp(u_y_fn, (y,), (v_y_tangent,))[1]
    # Second derivatives for u
    uxx = hvp_fwdfwd(u_x_fn, (x,), (v_x_tangent,))
    uyy = hvp_fwdfwd(u_y_fn, (y,), (v_y_tangent,))
    # First derivatives for v
    vx = jvp(v_x_fn, (x,), (v_x_tangent,))[1]
    vy = jvp(v_y_fn, (y,), (v_y_tangent,))[1]
    # Second derivatives for v
    vxx = hvp_fwdfwd(v_x_fn, (x,), (v_x_tangent,))
    vyy = hvp_fwdfwd(v_y_fn, (y,), (v_y_tangent,))
    # First derivatives for p
    px = jvp(p_x_fn, (x,), (v_x_tangent,))[1]
    py = jvp(p_y_fn, (y,), (v_y_tangent,))[1]
    # Reynolds number is now estimated by the model
    nu = 1.0 / Re_est
    # Navier-Stokes Momentum Residuals (Continuity is satisfied by streamfunction)
    e_momentum_x = u * ux + v * uy + px - nu * (uxx + uyy)  # X-momentum
    e_momentum_y = u * vx + v * vy + py - nu * (vxx + vyy)  # Y-momentum
    # Concatenate only the momentum residuals
    return jnp.concatenate([e_momentum_x, e_momentum_y], axis=-1)
PDE_residuals=Navier_Stokes_Residuals

# ## Cell 9: vRBA Steps 1 & 3 - Weight & Multiplier Generation


def update_vRBA(
    r_i_norm, lambdas_batch, it,
    c_log=1.0, gamma=0.999, eta=0.01, phi=1.0,
    potential_type='exponential'):
    
    if potential_type == 'exponential':
        epsilon_q = c_log * jnp.max(r_i_norm) / jnp.log(it + 2.0)
        beta_it = 1.0 / (epsilon_q + 1e-20)
        q_it = beta_it * jnp.exp(beta_it * r_i_norm)
    elif potential_type == 'quadratic':# Standard RBA
        q_it = r_i_norm
    elif potential_type == 'linear': # No weights
        q_it = jnp.ones_like(r_i_norm)
    
    lambda_it = phi * q_it / (q_it.max() + 1e-20) + (1.0 - phi)
    new_lambdas = gamma * lambdas_batch + eta * lambda_it# EMA
    return new_lambdas

METHOD = 'exponential' 
c_log = 1.0 
phi = 1.0

update_fn_with_baked_in_type = partial(
    update_vRBA,
    c_log=c_log,
    phi=phi,
    potential_type=METHOD
)
update_weights_fn = jax.jit(update_fn_with_baked_in_type)

# ## Cell 10: Optax Model Update Step


@partial(jax.jit, static_argnums=(0, 1))  # key and optimizer are static
def update_model(key_params, optimizer, gradient, params, state):
    updates, new_state = optimizer.update(gradient['params'][key_params], state)
    new_params = optax.apply_updates(params['params'][key_params], updates)
    params['params'][key_params] = new_params
    return params, new_state

# ## Cell 11: Main Training Step (Adam) & Gradient Calculation
@partial(jax.jit, static_argnums=(0, 1))
def apply_model(
    apply_fn,
    update_weights_fn,
    params, lambdas, lamB,
    all_grads,  #
    idx_train, it,
    hyperparams,
    *train_data
):
    lamE, gamma, eta, gamma_grads, alpha = hyperparams
    x_pde, y_pde, x_bcs, y_bcs, u_bcs, v_bcs, x_data, y_data, u_data_noisy = train_data
    lambdas_it = lambdas[idx_train]
    def loss_pde(params, lambdas):
        residuals = PDE_residuals(params, x_pde, y_pde, apply_fn)
        r_i_norm = lax.stop_gradient(jnp.abs(residuals))
        new_lambdas = update_weights_fn(r_i_norm, lambdas, it,
            gamma=gamma,
            eta=eta)
        pde_loss = jnp.mean((new_lambdas * residuals) ** 2)
        return pde_loss, new_lambdas

    def loss_bcs(params):
        # Dirichlet BCs for u and v
        u_pred_bcs, v_pred_bcs, _, _, _ = apply_fn(params, x_bcs, y_bcs)
        dirichlet_u_loss = jnp.mean((u_bcs - u_pred_bcs) ** 2)
        dirichlet_v_loss = jnp.mean((v_bcs - v_pred_bcs) ** 2)
        # Combine all BC losses
        bcs_loss = dirichlet_u_loss + dirichlet_v_loss
        return bcs_loss

    def loss_data(params):
        u_pred_data, _, _, _, _ = apply_fn(params, x_data, y_data)
        data_loss = jnp.mean((u_data_noisy - u_pred_data) ** 2)
        return data_loss

    (pde_loss, new_lambdas), gradient_pde = jax.value_and_grad(loss_pde, has_aux=True)(params, lambdas_it)
    bcs_loss, gradient_bcs = jax.value_and_grad(loss_bcs)(params)
    data_loss, gradient_data = jax.value_and_grad(loss_data)(params)
    lambdas = lambdas.at[idx_train].set(new_lambdas)
    pde_gradient_flat, _ = ravel_pytree(gradient_pde)
    bcs_gradient_flat, _ = ravel_pytree(gradient_bcs)
    data_gradient_flat, _ = ravel_pytree(gradient_data)
    grad_PDE_norm = jnp.linalg.norm(pde_gradient_flat) + 1e-20
    grad_bcs_norm = jnp.linalg.norm(bcs_gradient_flat) + 1e-20
    grad_data_norm = jnp.linalg.norm(data_gradient_flat) + 1e-20
    grad_data_norm = jnp.linalg.norm(data_gradient_flat) + 1e-20  # Duplicate line, but keeping as per "do no harm" unless it causes an error.
    # This line now correctly uses the 'all_grads' that was passed in
    grad_avg_PDE = gamma_grads * all_grads['grad_bar_PDE'] + (1.0 - gamma_grads) * grad_PDE_norm
    grad_avg_BCs = gamma_grads * all_grads['grad_bar_BCs'] + (1.0 - gamma_grads) * grad_bcs_norm
    grad_avg_Data = gamma_grads * all_grads['grad_bar_Data'] + (1.0 - gamma_grads) * grad_data_norm
    lamB_BCs = alpha * lamB[0] + (1.0 - alpha) * grad_avg_PDE / (grad_avg_BCs + 1e-20)
    lamB_Data = alpha * lamB[1] + (1.0 - alpha) * grad_avg_PDE / (grad_avg_Data + 1e-20)
    lamB = jnp.array([lamB_BCs, lamB_Data])
    total_gradient = jax.tree_util.tree_map(lambda g_pde, g_bcs, g_data: lamE * g_pde + lamB[0] * g_bcs + lamB[1] * g_data, gradient_pde, gradient_bcs, gradient_data)
    all_loss = {
        'loss_PDE': pde_loss,
        'loss_BCs': bcs_loss,
        'loss_Data': data_loss,
        'Loss': lamE * pde_loss + lamB[0] * bcs_loss + lamB[1] * data_loss,
    }
    all_grads_updated = {
        'grad_bar_PDE': grad_avg_PDE,
        'grad_bar_BCs': grad_avg_BCs,
        'grad_bar_Data': grad_avg_Data,
    }
    return all_loss, total_gradient, lambdas, lamB, all_grads_updated
#

# ## Cell 12: vRBA Implementation via Importance Sampling
#
@partial(jax.jit, static_argnames=("batch_size", "k", "c"))
def sample_points_pdf(key, batch_size, lambdas, x_pde, y_pde, k=1, c=0.5):
    # Adjust lambdas with exponent k and normalize
    lambdas_k = lambdas**k
    # Sum across the residual dimension to get a single importance value per point
    lambdas_k_summed = jnp.sum(lambdas_k, axis=-1, keepdims=True)
    lambdas_k_summed = lambdas_k_summed / (lambdas_k_summed.mean() + 1e-20) + c
    lambdas_k_summed = lambdas_k_summed.flatten()
    p = lambdas_k_summed / (lambdas_k_summed.sum() + 1e-20)
    idx_train = jax.random.choice(
        key, len(lambdas), shape=(batch_size,), p=p
    )
    train_data = (x_pde[idx_train], y_pde[idx_train])
    return idx_train, train_data

# ## Cell 13: Main Adam Training Loop
def get_gamma(eta, max_RBA):
    gamma_it = 1-eta/max_RBA
    return gamma_it
all_errors = []
all_L_infty = []
all_L_2 = []
all_its = []
all_loss = []
all_gamma = []
all_lamB = []
all_max_RBA = []
all_lambdas=[]
all_idx=[]
all_Re_est = []  # Store estimated Reynolds number
start = time.time()
pbar = tqdm(range(1, EPOCHS + 1), desc='Training Progress')
gamma_it=get_gamma(eta, max_RBA0)
#Global weights
max_RBA=max_RBA0
lamE = 1.0
lamB = jnp.ones((2,))  # lamB for BCs and Data
#RBA
step_RBA = (cap_RBA - max_RBA0) / (EPOCHS / 50000 - 1)
# initialize grads container
all_grads = {
    'grad_bar_PDE': 1.0,
    'grad_bar_BCs': 1.0,
    'grad_bar_Data': 1.0,
}
for e in pbar:
    key, subkey = jax.random.split(key)
    idx_train, (x_pde_batch, y_pde_batch) = sample_points_pdf(
        subkey, batch_size, lambdas, x_pde, y_pde, k=args.k_samp, c=args.c_samp
    )
    # single run
    hyperparams = (lamE,gamma_it, eta, gamma_grads, alpha)
    all_loss_it, gradient, lambdas, lamB, all_grads = apply_model(
            apply_fn, update_weights_fn,
            params, lambdas,
            lamB,
            all_grads,
            idx_train,
            e,
            hyperparams,
*(x_pde_batch, y_pde_batch, x_bcs, y_bcs, u_bcs, v_bcs, x_data, y_data, u_data_noisy)
        )
    for key_params in params['params']:
        params, states[key_params] = update_model(key_params, optimizers[key_params], gradient, params, states[key_params])
    log_frequency = 1 if e < 500 else 100 if e <= 5000 else 500
    if e % log_frequency == 0:
        # Update RBA
        max_RBA=max_RBA0+step_RBA*e//50000
        gamma_it=get_gamma(eta, max_RBA)
        all_lambdas.append(np.array(lambdas))  #
        all_idx.append(idx_train)
        #Compute errors
        #Compute errors
        u_pred_all, v_pred_all, _, _, Re_est_it = apply_fn(params, x_eval, y_eval)
        error_u = relative_l2(u_pred_all, u_gt_eval)
        error_v = relative_l2(v_pred_all, v_gt_eval)
        error = (error_u + error_v) / 2.0  # Average error for u and v
        error_vec_u = u_pred_all - u_gt_eval
        error_vec_v = v_pred_all - v_gt_eval
        L_infty = jnp.max(jnp.concatenate([jnp.abs(error_vec_u), jnp.abs(error_vec_v)], axis=0))
        L_2 = jnp.sqrt(jnp.mean(error_vec_u**2) + jnp.mean(error_vec_v**2))
        # Updating the tqdm progress bar with loss and other metrics
        pbar.set_description(f"It: {e}/{EPOCHS} | Error: {error:.3e} | lam_max: {lambdas.max():.3f}| max_RBA: {max_RBA:.3f}| lamB_BCs: {lamB[0]:.3f}| lamB_Data: {lamB[1]:.3f}|")
        all_errors.append(error)
        all_its.append(e)
        all_lambdas.append(np.array(lambdas))
        all_loss.append(all_loss_it['Loss'])
        all_gamma.append(gamma_it)
        all_lamB.append(lamB)
        all_max_RBA.append(max_RBA)
        all_Re_est.append(Re_est_it[0])  # Re_est is a scalar, take the first element
        # Error norms
        all_L_infty.append(L_infty)
        all_L_2.append(L_2)
end = time.time()
print(f'Runtime: {((end - start) / EPOCHS * 1000):.2f} ms/iter.')

# ## Cell 14: Visualization & Evaluation (Adam Phase)
#
# Adjusting the plot to use a log scale for both loss and error
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(all_its, all_loss, label='Total Loss', color='blue')
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Total Loss (log scale)')
plt.title('Total Loss History (Log Scale)')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(all_its, all_errors, label='Avg Rel. L2 Error (u,v)', color='green')
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Relative $L^2$ Error (log scale)')
plt.title('Average Relative $L^2$ Error History (Log Scale)')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(all_its, all_Re_est, label='Estimated Re', color='purple')
plt.axhline(y=Re_true, color='red', linestyle='--', label=f'True Re ({Re_true})')
plt.xlabel('Iterations')
plt.ylabel('Estimated Reynolds Number')
plt.title('Reynolds Number Evolution (Adam Phase)')
plt.grid(True)
plt.legend()

# Placeholder for the 4th subplot, or remove if not needed
ax_dummy = plt.subplot(2, 2, 4)
ax_dummy.set_visible(False)

plt.tight_layout()
plt.savefig('Adam_Phase_Metrics.png')
plt.show()
plt.show()
u_pred_eval, v_pred_eval, _, _, Re_est_adam = apply_fn(params, x_eval, y_eval)
error_u = relative_l2(u_pred_eval, u_gt_eval)
error_v = relative_l2(v_pred_eval, v_gt_eval)
print(f'RL2 error u: {error_u:.8f}, RL2 error v: {error_v:.8f}')
print(f'Estimated Re (Adam): {Re_est_adam[0]:.3f}, True Re: {Re_true:.3f}, Relative Error Re: {jnp.abs(Re_est_adam[0] - Re_true) / Re_true:.3e}')
it=-1
print('Solution:')
# (Removed lines 23-24)
lambdas_grid=all_lambdas[it]
idx_it=all_idx[it]
# Get predictions for plotting
u_pred_plot, v_pred_plot, p_pred_plot, _, Re_est_plot = apply_fn(params, X_plot.flatten()[:,None], Y_plot.flatten()[:,None])
u_pred_plot = u_pred_plot.reshape(X_plot.shape)
v_pred_plot = v_pred_plot.reshape(X_plot.shape)
p_pred_plot = p_pred_plot.reshape(X_plot.shape)
# Plotting code
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
levels = 100
# U-velocity plots
ax_u_gt = axes[0, 0]
contour_u_gt = ax_u_gt.contourf(X_plot, Y_plot, u_plot_gt, cmap='jet', levels=levels)
fig.colorbar(contour_u_gt, ax=ax_u_gt)
ax_u_gt.set_title('Reference U-velocity')
ax_u_gt.set_xlabel('x')
ax_u_gt.set_ylabel('y')
ax_u_pred = axes[0, 1]
contour_u_pred = ax_u_pred.contourf(X_plot, Y_plot, u_pred_plot, cmap='jet', levels=levels)
fig.colorbar(contour_u_pred, ax=ax_u_pred)
ax_u_pred.set_title('Predicted U-velocity')
ax_u_pred.set_xlabel('x')
ax_u_pred.set_ylabel('y')
ax_u_err = axes[0, 2]
error_u_plot = np.abs(u_plot_gt - u_pred_plot)
err_u_log_norm = colors.LogNorm(vmin=error_u_plot.min()+1e-16, vmax=error_u_plot.max() + 1e-16)
mesh_u_err = ax_u_err.pcolormesh(X_plot, Y_plot, error_u_plot, cmap='jet', norm=err_u_log_norm, shading='gouraud')
fig.colorbar(mesh_u_err, ax=ax_u_err)
ax_u_err.set_title('Pointwise Error U (Log Scale)')
ax_u_err.set_xlabel('x')
ax_u_err.set_ylabel('y')
# V-velocity plots
ax_v_gt = axes[1, 0]
contour_v_gt = ax_v_gt.contourf(X_plot, Y_plot, v_plot_gt, cmap='jet', levels=levels)
fig.colorbar(contour_v_gt, ax=ax_v_gt)
ax_v_gt.set_title('Reference V-velocity')
ax_v_gt.set_xlabel('x')
ax_v_gt.set_ylabel('y')
ax_v_pred = axes[1, 1]
contour_v_pred = ax_v_pred.contourf(X_plot, Y_plot, v_pred_plot, cmap='jet', levels=levels)
fig.colorbar(contour_v_pred, ax=ax_v_pred)
ax_v_pred.set_title('Predicted V-velocity')
ax_v_pred.set_xlabel('x')
ax_v_pred.set_ylabel('y')
ax_v_err = axes[1, 2]
error_v_plot = np.abs(v_plot_gt - v_pred_plot)
err_v_log_norm = colors.LogNorm(vmin=error_v_plot.min()+1e-16, vmax=error_v_plot.max() + 1e-16)
mesh_v_err = ax_v_err.pcolormesh(X_plot, Y_plot, error_v_plot, cmap='jet', norm=err_v_log_norm, shading='gouraud')
fig.colorbar(mesh_v_err, ax=ax_v_err)
ax_v_err.set_title('Pointwise Error V (Log Scale)')
ax_v_err.set_xlabel('x')
ax_v_err.set_ylabel('y')
# Pressure plots
ax_p_gt = axes[2, 0]
contour_p_gt = ax_p_gt.contourf(X_plot, Y_plot, p_plot_gt, cmap='jet', levels=levels)
fig.colorbar(contour_p_gt, ax=ax_p_gt)
ax_p_gt.set_title('Reference Pressure')
ax_p_gt.set_xlabel('x')
ax_p_gt.set_ylabel('y')
ax_p_pred = axes[2, 1]
contour_p_pred = ax_p_pred.contourf(X_plot, Y_plot, p_pred_plot, cmap='jet', levels=levels)
fig.colorbar(contour_p_pred, ax=ax_p_pred)
ax_p_pred.set_title('Predicted Pressure')
ax_p_pred.set_xlabel('x')
ax_p_pred.set_ylabel('y')
ax_p_dummy = axes[2, 2] # Placeholder for pressure error, as per strategy, no error for p
ax_p_dummy.set_visible(False) # Hide this subplot
plt.tight_layout()
plt.savefig('Errorss_Adam.png', dpi=300)
plt.show()
#

# ## Cell 15: Preparation for Second-Order Optimization (SSBroyden)
initial_weights, unflatten_func = ravel_pytree(params)
params_test=unflatten_func(initial_weights)
u_pred_eval, v_pred_eval, _, _, Re_est_adam = apply_fn(params, x_eval, y_eval)
error_u = relative_l2(u_pred_eval, u_gt_eval)
error_v = relative_l2(v_pred_eval, v_gt_eval)
print(f'RL2 error u: {error_u:.8f}, RL2 error v: {error_v:.8f}, Re_est: {Re_est_adam[0]:.3f}')
u_pred_test, v_pred_test, _, _, Re_est_test = apply_fn(params_test, x_eval, y_eval)
error_u_test = relative_l2(u_pred_test, u_gt_eval)
error_v_test = relative_l2(v_pred_test, v_gt_eval)
print(f'Reconstructed parameters: RL2 error u: {error_u_test:.8f}, RL2 error v: {error_v_test:.8f}, Re_est: {Re_est_test[0]:.3f}')
# X=np.hstack([tc.astype(jnp.float64),xc.astype(jnp.float64)]) # Not needed for 2D steady-state
# Capture final lambdas and lamB from Adam phase for SSBroyden (frozen weights)
lambdas_frozen = lambdas
lamB_frozen = lamB

# ## Cell 16: Defining the Loss Functions for Second-Order Optimization
@partial(jax.jit, static_argnums=(0,))
def apply_model_2nd_Order(apply_fn, params, *train_data_with_weights):
    x_pde_batch, y_pde_batch, x_bcs, y_bcs, u_bcs, v_bcs, x_data, y_data, u_data_noisy, lambdas_batch, lamB_frozen_bcs, lamB_frozen_data = train_data_with_weights
    def loss_pde(params):
        # Compute residuals
        residuals = PDE_residuals(params, x_pde_batch, y_pde_batch, apply_fn)
        pde_loss = jnp.mean((lambdas_batch * residuals) ** 2)
        return pde_loss
    def loss_bcs(params):
        # Dirichlet BCs for u and v
        u_pred_bcs, v_pred_bcs, _, _, _ = apply_fn(params, x_bcs, y_bcs)
        dirichlet_u_loss = jnp.mean((u_bcs - u_pred_bcs) ** 2)
        dirichlet_v_loss = jnp.mean((v_bcs - v_pred_bcs) ** 2)
        # Combine all BC losses
        bcs_loss = dirichlet_u_loss + dirichlet_v_loss
        return bcs_loss
    def loss_data(params):
        u_pred_data, _, _, _, _ = apply_fn(params, x_data, y_data)
        data_loss = jnp.mean((u_data_noisy - u_pred_data) ** 2)
        return data_loss
    def loss(params):
        return loss_pde(params) + lamB_frozen_bcs * loss_bcs(params) + lamB_frozen_data * loss_data(params)
    loss_value = loss(params)
    return loss_value
@partial(jax.jit, static_argnums=(1, 2))
def loss_and_gradient(weights, N_arg, unflatten_func_arg, *train_data_tuple_arg_with_weights):
    flat_jax_array = weights
    params_current = unflatten_func_arg(flat_jax_array)
    loss_val_jax = apply_model_2nd_Order(N_arg, params_current, *train_data_tuple_arg_with_weights)
    return loss_val_jax
# update_lambdas function is removed as per strategy: freeze weights in SSBroyden phase

# ## Cell 17: The SSBroyden Optimization Step Function
@partial(jax.jit, static_argnames=("apply_fn", "unflatten_func", "batch_size", "gamma", "eta", "k", "c", "static_options"))
def SSBroyden_step(
    initial_weights, H0, key,
    x_pde, y_pde, lambdas, lamB_frozen_bcs, lamB_frozen_data,
    apply_fn, unflatten_func, batch_size, gamma, eta, k, c, static_options, it,
    x_bcs, y_bcs, u_bcs, v_bcs, x_data, y_data, u_data_noisy ):
    key, subkey = jax.random.split(key)
    params_it = unflatten_func(initial_weights)
    # 1. Global Update of Weights
    # lambdas are frozen in SSBroyden phase, no update here.
    # 2. Sample Batch based on NEW weights
    idx_train, (x_pde_batch, y_pde_batch) = sample_points_pdf(
        subkey, batch_size, lambdas, x_pde, y_pde, k, c
    )
    # 3. Use Unweighted Loss for Optimization
    lambdas_ones = jnp.ones_like(lambdas[idx_train])
    # 4. Minimize
    current_options = dict(static_options)
    current_options['initial_H'] = H0
    result = minimize(
        fun=loss_and_gradient,
        x0=initial_weights,
        args=(apply_fn, unflatten_func, x_pde_batch, y_pde_batch, x_bcs, y_bcs, u_bcs, v_bcs, x_data, y_data, u_data_noisy, lambdas[idx_train], lamB_frozen_bcs, lamB_frozen_data),
        method='BFGS',
        options=current_options)
    # 5. Recycle Hessian
    new_H0 = result.hess_inv
    new_H0 = (new_H0 + jnp.transpose(new_H0)) / 2
    try:
        L = jnp.linalg.cholesky(new_H0)
        is_failed = jnp.any(jnp.isnan(L))
        final_H0 = jax.lax.cond(
            is_failed,
            lambda op: jnp.eye(op.shape[0], dtype=op.dtype),
            lambda op: op,
            operand=new_H0)
    except LinAlgError:
        final_H0 = jnp.eye(new_H0.shape[0], dtype=new_H0.dtype)
    return result.x, final_H0, key, result.fun, result.nit

# ## Cell 18: Main SSBroyden Training Loop
# Initialize the Hessian
H0 = jnp.eye(len(initial_weights), dtype=jnp.float64)
num_outer_iterations = Nbfgs // Nchange
## SSBroyden Hyperparameters
static_options = static_options_SSBroyden
static_options['maxiter']=Nchange
static_options_tuple = tuple(static_options.items())
# Training
pbar = tqdm(range(num_outer_iterations), desc="SSBroyden Training")
effective_steps=0
Adam_Steps=all_its[-1]
# Initialize lambdas and lamB for SSBroyden phase with frozen values from Adam phase
current_lambdas = lambdas_frozen
current_lamB = lamB_frozen
for it in pbar:
    initial_weights, H0, key, loss_val, nit = SSBroyden_step(
        initial_weights, H0, key,
        x_pde, y_pde, current_lambdas, current_lamB[0], current_lamB[1],
        apply_fn, unflatten_func, batch_size, gamma_bfgs, eta_bfgs, k_samp_bfgs, c_samp_bfgs,
        static_options_tuple, it,
        x_bcs, y_bcs, u_bcs, v_bcs, x_data, y_data, u_data_noisy
    )
    effective_steps = effective_steps + nit
    current_params_for_eval = unflatten_func(initial_weights)
    current_params_for_eval = unflatten_func(initial_weights)
    u_pred_eval, v_pred_eval, _, _, Re_est_it = apply_fn(current_params_for_eval, x_eval, y_eval)
    error_u = relative_l2(u_pred_eval, u_gt_eval)
    error_v = relative_l2(v_pred_eval, v_gt_eval)
    error = (error_u + error_v) / 2.0
    error_vec_u = u_pred_eval - u_gt_eval
    error_vec_v = v_pred_eval - v_gt_eval
    L_infty = jnp.max(jnp.concatenate([jnp.abs(error_vec_u), jnp.abs(error_vec_v)], axis=0))
    L_2 = jnp.sqrt(jnp.mean(error_vec_u**2) + jnp.mean(error_vec_v**2))
    all_errors.append(error)
    all_loss.append(loss_val)
    all_its.append(effective_steps + Adam_Steps)
    all_L_infty.append(L_infty)
    all_L_2.append(L_2)
    all_Re_est.append(Re_est_it[0])
    pbar.set_postfix({
        'It': effective_steps,
        'Loss': f'{loss_val:.3e}',
        'RL2': f'{float(error):.3e}',
        'Re_est': f'{Re_est_it[0]:.3f}',
    })
#

# ## Cell 19: Final Visualization & Error Reporting
# 1. CALCULATE PREDICTIONS WITH FINAL PARAMETERS (Crucial Step Missing previously)
# We must regenerate the plot data using current_params_for_eval
u_pred_plot, v_pred_plot, p_pred_plot, _, Re_est_final = apply_fn(current_params_for_eval, X_plot.flatten()[:, None], Y_plot.flatten()[:, None])
# Reshape to grid for contour plotting
u_pred_plot = u_pred_plot.reshape(X_plot.shape)
v_pred_plot = v_pred_plot.reshape(X_plot.shape)
p_pred_plot = p_pred_plot.reshape(X_plot.shape)
print('Solution:')
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
levels = 100
# --- Row 1: Loss and Error History ---
ax_tl = axes[0, 0]
ax_tl.plot(all_its, all_loss, label='Total Loss', color='blue')
ax_tl.set_yscale('log')
ax_tl.set_title('Total Loss History (Log Scale)')
ax_tl.set_xlabel('Iterations')
ax_tl.set_ylabel('Loss (log scale)')
ax_tl.grid(True, which="both", ls="--")
ax_tl.legend()
ax_tm = axes[0, 1]
ax_tm.plot(all_its, all_errors, label='Avg Rel. L2 Error (u,v)', color='green')
ax_tm.set_yscale('log')
ax_tm.set_title('Average Relative $L^2$ Error History (Log Scale)')
ax_tm.set_xlabel('Iterations')
ax_tm.set_ylabel('Avg Rel. $L^2$ Error (log scale)')
ax_tm.grid(True, which="both", ls="--")
ax_tm.legend()
ax_tr = axes[0, 2]
ax_tr.plot(all_its, all_Re_est, label='Estimated Re', color='purple')
ax_tr.axhline(y=Re_true, color='red', linestyle='--', label=f'True Re ({Re_true})')
ax_tr.set_xlabel('Iterations')
ax_tr.set_ylabel('Estimated Reynolds Number')
ax_tr.set_title('Reynolds Number Evolution')
ax_tr.grid(True)
ax_tr.legend()
# --- Row 2: U-velocity plots ---
ax_u_gt = axes[1, 0]
contour_u_gt = ax_u_gt.contourf(X_plot, Y_plot, u_plot_gt, cmap='jet', levels=levels)
fig.colorbar(contour_u_gt, ax=ax_u_gt)
ax_u_gt.set_title('Reference U-velocity')
ax_u_gt.set_xlabel('x')
ax_u_gt.set_ylabel('y')
ax_u_pred = axes[1, 1]
contour_u_pred = ax_u_pred.contourf(X_plot, Y_plot, u_pred_plot, cmap='jet', levels=levels)
fig.colorbar(contour_u_pred, ax=ax_u_pred)
ax_u_pred.set_title('Predicted U-velocity')
ax_u_pred.set_xlabel('x')
ax_u_pred.set_ylabel('y')
ax_u_err = axes[1, 2]
error_u_plot = np.abs(u_plot_gt - u_pred_plot)
err_u_log_norm = colors.LogNorm(vmin=error_u_plot.min()+1e-16, vmax=error_u_plot.max() + 1e-16)
mesh_u_err = ax_u_err.pcolormesh(X_plot, Y_plot, error_u_plot, cmap='jet', norm=err_u_log_norm, shading='gouraud')
fig.colorbar(mesh_u_err, ax=ax_u_err)
ax_u_err.set_title('Pointwise Error U (Log Scale)')
ax_u_err.set_xlabel('x')
ax_u_err.set_ylabel('y')
# --- Row 3: V-velocity plots ---
ax_v_gt = axes[2, 0]
contour_v_gt = ax_v_gt.contourf(X_plot, Y_plot, v_plot_gt, cmap='jet', levels=levels)
fig.colorbar(contour_v_gt, ax=ax_v_gt)
ax_v_gt.set_title('Reference V-velocity')
ax_v_gt.set_xlabel('x')
ax_v_gt.set_ylabel('y')
ax_v_pred = axes[2, 1]
contour_v_pred = ax_v_pred.contourf(X_plot, Y_plot, v_pred_plot, cmap='jet', levels=levels)
fig.colorbar(contour_v_pred, ax=ax_v_pred)
ax_v_pred.set_title('Predicted V-velocity')
ax_v_pred.set_xlabel('x')
ax_v_pred.set_ylabel('y')
ax_v_err = axes[2, 2]
error_v_plot = np.abs(v_plot_gt - v_pred_plot)
err_v_log_norm = colors.LogNorm(vmin=error_v_plot.min()+1e-16, vmax=error_v_plot.max() + 1e-16)
mesh_v_err = ax_v_err.pcolormesh(X_plot, Y_plot, error_v_plot, cmap='jet', norm=err_v_log_norm, shading='gouraud')
fig.colorbar(mesh_v_err, ax=ax_v_err)
ax_v_err.set_title('Pointwise Error V (Log Scale)')
ax_v_err.set_xlabel('x')
ax_v_err.set_ylabel('y')
plt.tight_layout()
# --- FIX: Only call savefig/show ONCE ---
plt.savefig('summary_all.png', dpi=150)
plt.show()
print('Minimum achieved average RL2 error (u,v):', np.nanmin(all_errors))
print(f'Final Estimated Re: {Re_est_final[0]:.3f}, True Re: {Re_true:.3f}, Relative Error Re: {jnp.abs(Re_est_final[0] - Re_true) / Re_true:.3e}')

# ## Cell 20: Save Experiment Results to .mat File
#
# Get predicted v at data points for saving
_, v_pred_at_data_points, _, _, _ = apply_fn(current_params_for_eval, x_data, y_data)
results_dict = {
    'model_name': args.Name,
    'all_errors': all_errors,
    'all_its': all_its,
    'all_loss': all_loss,
    'all_gamma': all_gamma,
    'all_lamB': all_lamB,
    'all_max_RBA': all_max_RBA,
    'all_L_infty': all_L_infty,
    'all_L_2': all_L_2,
    'all_Re_est': all_Re_est,
'Re': Re_est_final[0],
    'x_sensor': np.array(x_data),
    'y_sensor': np.array(y_data),
    'u_sensor': np.array(u_data_noisy),
    'v_hidden_at_sensor': np.array(v_pred_at_data_points),
}
# Save dictionary as a .npz file
np.savez('results.npz', **results_dict)