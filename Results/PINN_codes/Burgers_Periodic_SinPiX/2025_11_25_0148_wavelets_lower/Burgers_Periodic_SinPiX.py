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
parser.add_argument('--Equation', type=str, default='Burgers', help='Name of equation')
parser.add_argument('--Name', type=str, default='SS_Broyden', help='Name of the experiment')
parser.add_argument('--NC', type=int, default=200000, help='Number of samples for training')
parser.add_argument('--NI', type=int, default=512, help='Number of iterations')
parser.add_argument('--NB', type=int, default=512, help='Batch size')
parser.add_argument('--NC_TEST', type=int, default=100, help='Number of test samples')
parser.add_argument('--SEED', type=int, default=9999, help='Random seed')
parser.add_argument('--EPOCHS', type=int, default=10000, help='Number of training epochs')
parser.add_argument('--N_LAYERS', type=int, default=3, help='Number of layers in the network')
parser.add_argument('--HIDDEN', type=int, default=30, help='Number of hidden units per layer')
parser.add_argument('--FEATURES', type=int, default=1, help='Feature size')
parser.add_argument('--num_centers', type=int, default=15, help='Number of centers for wavelet embedding')
parser.add_argument('--kappa', type=float, default=0.75, help='Kappa parameter for wavelet embedding')
parser.add_argument('--lr_fact', type=float, default=0.2, help='Scale Lr')
parser.add_argument('--eta', type=float, default=0.01, help='Learning rate or step size for adaptive gamma')
parser.add_argument('--gamma', type=float, default=0.999, help='Decay rate for adaptive gamma')
parser.add_argument('--gamma_bfgs', type=float, default=0.1, help='Decay rate for adaptive gamma')
parser.add_argument('--gamma_grads', type=float, default=0.99, help='Decay rate for adaptive gamma')
parser.add_argument('--alpha', type=float, default=1.0, help='Decay rate for exponential moving average')
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
parser.add_argument('--nu_0', type=float, default=0.001, help='Initial viscosity for continuation')
parser.add_argument('--nu_final', type=float, default=5e-5, help='Final viscosity for continuation')
parser.add_argument('--N_nu_steps', type=int, default=25, help='Number of viscosity annealing steps')
parser.add_argument('--ssbroyden_iters_min', type=int, default=2500, help='Minimum iterations per SSBroyden viscosity step')
parser.add_argument('--ssbroyden_iters_max', type=int, default=35000, help='Maximum iterations for the final SSBroyden viscosity step')
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
num_centers = args.num_centers
kappa = args.kappa
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
##  We use a self scaling algoirhtm to automatically tune the global weights and to increase the cap of RBA to improve learning dunamics
max_RBA0 = args.max_RBA if args.max_RBA is not None else eta / (1 - gamma)
cap_RBA = args.cap_RBA
gamma_grads=args.gamma_grads
## vRBA Params Second order_ Sampling + SSBroycen
k_samp_bfgs = args.k_samp
c_samp_bfgs = args.c_samp
Nchange = args.N_change  # Number of SSBroyden steps before resampling
batch_size=args.batch_size
ssbroyden_iters_min = args.ssbroyden_iters_min
ssbroyden_iters_max = args.ssbroyden_iters_max
# Nbfgs=100000 # Number of SSBroyden BFGs iterations (REMOVED - replaced by adaptive schedule)
# Viscosity Continuation
nu_0 = args.nu_0
nu_final = args.nu_final
N_nu_steps = args.N_nu_steps
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
args.Name =project_root+'/Results/'+args.Equation+'/'+args.Name+'/'
os.makedirs(args.Name, exist_ok=True)


# ## Cell 3: Data Loading and Ground Truth Definition
# Path to the reference data for Burgers' equation
burgers_data_path = '/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes/Burgers_Periodic_SinPiX/2025_11_22_1730/burgers.npz'

# Load the data from the .npz file
data = np.load(burgers_data_path)
print(f"Loaded keys from NPZ file: {list(data.keys())}")

# Extract data and ensure correct types and shapes
t0 = data['t'].astype(jnp.float64)  # Shape (200,)
x0 = data['x'].astype(jnp.float64)  # Shape (256,)
Exact0 = data['usol'].T.astype(jnp.float64) # Shape (256, 200) after transpose

# Create meshgrid for plotting and evaluation
T, X = np.meshgrid(t0, x0)

# Ground Truth (to compute relative errors)
t = T.flatten()[:, None]
x = X.flatten()[:, None]
u_gt = Exact0.flatten()[:, None]

# ## Cell 4: Data Processing, Sampling, and Visualization
# Collocation points
lb_D = jnp.array([t0.min(), x0.min()])
ub_D = jnp.array([t0.max(), x0.max()])
# Generate collocation points and ensure they are float64
X_c = lb_D + (ub_D - lb_D) * lhs(2, NC)
X_c = X_c.astype(jnp.float64)
tc = X_c[:, 0:1]
xc = X_c[:, 1:2]
# Initial and boundary conditions are handled by the network architecture (ansatz)
# and periodic feature embedding. No separate data points are needed for the loss function.

# This data tuple now only contains collocation points for the PDE residual.
train_data = tc, xc
lambdas = tc * 0 + 0.1
# Plotting code
fig = plt.figure(figsize=(9, 3))
# Scatter plot for collocation, initial, and boundary points
ax1 = fig.add_subplot(121)
ax1.scatter(tc, xc, color='blue', label='Collocation Points',s=0.2)
# ax1.scatter(ti, xi, color='green', label='Initial Conditions') # This is now handled by the ansatz
ax1.set_xlabel('Time')
ax1.set_ylabel('Space')
ax1.set_title('Scatter Plot of Collocation Points')
ax1.legend()
# Surface plot for Exact solution
ax2 = fig.add_subplot(122)
contour = ax2.contourf(T, X, Exact0, cmap='jet', levels=50)  # Filled contour plot with 50 levels
fig.colorbar(contour)  # Add color bar to show scale
ax2.set_xlabel('t')
ax2.set_ylabel('x')
ax2.set_title('Exact Solution')
plt.savefig('Boundaries_data.png',dpi=300)
plt.show()

# ## Cell 5: Model Architecture Definition (The PINN Structure)
class Periodic_Wavelet_Embedding(nn.Module):
    """
    Implements a periodic wavelet-like embedding based on a von Mises-like function.
    This layer is designed to enforce periodicity on a given domain.
    
    The embedding for a single feature is defined by the formula:
    ψ(x) = exp(κ * cos(x_scaled - μ)) * [κ * cos(x_scaled - μ) + (κ * sin(x_scaled - μ))^2]
    
    where x_scaled is the input coordinate mapped to an angular domain [ -π, π ].
    """
    num_centers: int = 15
    kappa: float = 0.75
    period: float = 2.0  # The period of the input domain, e.g., for x in [-1, 1], period is 2.

    def setup(self):
        """Initializes the fixed centers (mu) for the basis functions."""
        # The centers μ are fixed, non-trainable, and distributed evenly over the angular domain [-π, π].
        self.mu = jnp.linspace(-jnp.pi, jnp.pi, self.num_centers, endpoint=False)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the periodic wavelet embedding.
        
        Args:
            x: Input tensor of shape (batch_size, 1).
            
        Returns:
            Output tensor of shape (batch_size, num_centers).
        """
        # Scale the input x from its domain (e.g., [-1, 1]) to an angular domain [-π, π]
        # to ensure the output is periodic with the specified `self.period`.
        x_scaled = 2 * jnp.pi * x / self.period

        # Reshape for broadcasting against the centers.
        # x_scaled becomes (batch, 1, 1)
        # self.mu becomes (1, num_centers, 1)
        x_reshaped = x_scaled[:, None, :]
        mu_reshaped = self.mu[None, :, None]

        # Calculate the argument for the trigonometric functions.
        arg = x_reshaped - mu_reshaped  # Shape: (batch, num_centers, 1)

        # Apply the specified mathematical formula.
        cos_arg = jnp.cos(arg)
        sin_arg = jnp.sin(arg)

        term1 = self.kappa * cos_arg
        term2 = (self.kappa * sin_arg)**2

        bracket_term = term1 + term2
        exp_term = jnp.exp(self.kappa * cos_arg)

        psi = exp_term * bracket_term  # Shape: (batch, num_centers, 1)

        # Squeeze the last dimension to get the final feature vector.
        # Final shape: (batch, num_centers)
        return jnp.squeeze(psi, axis=-1)
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
    num_centers: int = 15
    kappa: float = 0.75
    activation: Callable = nn.tanh

    def setup(self):
        self.embedding = Periodic_Wavelet_Embedding(num_centers=self.num_centers, kappa=self.kappa)
        self.MLP = MLP(features=self.features, activation=self.activation)

    @nn.compact
    def __call__(self, t: Array, x: Array) -> Array:
        x_embedded = self.embedding(x)
        X_concat = jnp.concatenate([t, x_embedded], axis=-1)
        NN_output = self.MLP(X_concat)

        # Hard constraint for initial condition u(0, x) = -sin(pi*x)
        u_initial = -jnp.sin(jnp.pi * x)
        u_pred = u_initial + t * NN_output
        return u_pred


# ## Cell 6: Model and Optimizer Initialization
feat_sizes = tuple([HIDDEN for _ in range(N_LAYERS)] + [FEATURES])
print(feat_sizes)
# make & init model
model = PINN(feat_sizes, activation=nn.tanh, num_centers=num_centers, kappa=kappa)
params = model.init(subkey, jnp.ones((NC, 1), dtype=jnp.float64), jnp.ones((NC, 1), dtype=jnp.float64))
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), params)
optimizers = {}
for key_params in params['params'].keys():
    if key_params=='g_fx':
        optimizers[key_params]=optax.adam(optax.exponential_decay(lr0*args.lr_fact, decay_step, decay_rate, staircase=False))
    else:
        optimizers[key_params]=optax.adam(optax.exponential_decay(lr0, decay_step, decay_rate, staircase=False))
# Initialize optimizer states for each parameter group
states = {key_params: optim.init(params['params'][key_params]) for key_params, optim in optimizers.items()}
# forward & loss function
apply_fn = jax.jit(model.apply)
total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(total_params )
# Global loss weights for Self-Balancing
lam_pde = 1.0
lam_ent = 1.0


# ## Cell 7: PDE Residual Definition
def Burgers_conservative(params, t, x, apply_fn, nu):
    u = apply_fn(params, t, x)
    
    # Tangent vectors for jvp
    v_t = jnp.ones_like(t)
    v_x = jnp.ones_like(x)
    
    # Functions for differentiation
    u_fn_t = lambda t_val: apply_fn(params, t_val, x)
    u_fn_x = lambda x_val: apply_fn(params, t, x_val)
    
    # First derivatives
    ut = jvp(u_fn_t, (t,), (v_t,))[1]
    
    # For the conservative term (0.5 * u^2)_x
    flux_fn = lambda x_val: 0.5 * (apply_fn(params, t, x_val)**2)
    flux_x = jvp(flux_fn, (x,), (v_x,))[1]
    
    # Second derivative u_xx
    g_ux = lambda x_val: jvp(u_fn_x, (x_val,), (v_x,))[1]
    uxx = jvp(g_ux, (x,), (v_x,))[1]
    
    return ut + flux_x - nu * uxx

def Burgers_entropy(params, t, x, apply_fn):
    # Entropy: E(u) = 0.5 * u^2
    # Entropy flux: F(u) = (1/3) * u^3
    
    v_t = jnp.ones_like(t)
    v_x = jnp.ones_like(x)

    # Time derivative of entropy: (0.5 * u^2)_t
    E_fn = lambda t_val: 0.5 * (apply_fn(params, t_val, x)**2)
    Et = jvp(E_fn, (t,), (v_t,))[1]
    
    # Space derivative of entropy flux: ((1/3) * u^3)_x
    F_fn = lambda x_val: (1/3) * (apply_fn(params, t, x_val)**3)
    Fx = jvp(F_fn, (x,), (v_x,))[1]
    
    return Et + Fx
PDE_residuals = Burgers_conservative

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
    params, lambdas,
    idx_train, it,
    hyperparams,
    nu,
    lam_pde, lam_ent, all_grads,
    *train_data
):
    gamma, eta, gamma_grads = hyperparams
    t, x = train_data
    lambdas_it = lambdas[idx_train]
    def loss_total(params, lambdas, lam_pde, lam_ent):
        # PDE Loss with vRBA
        residuals_pde = PDE_residuals(params, t, x, apply_fn, nu)
        r_i_norm = lax.stop_gradient(jnp.abs(residuals_pde))
        new_lambdas = update_weights_fn(r_i_norm, lambdas, it, gamma=gamma, eta=eta)
        pde_loss = jnp.mean((new_lambdas * residuals_pde) ** 2)
        # Entropy Loss
        residuals_ent = Burgers_entropy(params, t, x, apply_fn)
        ent_loss = jnp.mean(nn.relu(residuals_ent)**2)
        # Combine losses
        total_loss = lam_pde * pde_loss + lam_ent * ent_loss
        return total_loss, (pde_loss, ent_loss, new_lambdas)
    # Get gradients and auxiliary outputs
    (loss_val, (pde_loss, ent_loss, new_lambdas)), total_gradient = jax.value_and_grad(
        loss_total, has_aux=True
    )(params, lambdas_it, lam_pde, lam_ent)
    lambdas = lambdas.at[idx_train].set(new_lambdas)
    # Self-Balancing for global weights
    grad_pde_norm = jnp.linalg.norm(ravel_pytree(jax.tree.map(lambda g: g * lam_pde, total_gradient))[0])
    grad_ent_norm = jnp.linalg.norm(ravel_pytree(jax.tree.map(lambda g: g * lam_ent, total_gradient))[0])
    grad_bar_PDE = (1 - gamma_grads) * grad_pde_norm + gamma_grads * all_grads['grad_bar_PDE']
    grad_bar_ENT = (1 - gamma_grads) * grad_ent_norm + gamma_grads * all_grads['grad_bar_ENT']
    L_avg = (pde_loss + ent_loss) / 2.0
    lam_pde = (L_avg / (grad_bar_PDE + 1e-9))
    lam_ent = (L_avg / (grad_bar_ENT + 1e-9))
    all_grads_updated = {
        'grad_bar_PDE': grad_bar_PDE,
        'grad_bar_ENT': grad_bar_ENT
    }
    all_loss = {
        'loss_PDE': pde_loss,
        'loss_ENT': ent_loss,
        'Loss': loss_val,
    }
    return all_loss, total_gradient, lambdas, lam_pde, lam_ent, all_grads_updated
#

# ## Cell 12: vRBA Implementation via Importance Sampling
#
@partial(jax.jit, static_argnames=("batch_size", "k", "c"))
def sample_points_pdf(key, batch_size, lambdas, tc, xc, k=1, c=0.5):
    # Adjust lambdas with exponent k and normalize
    lambdas_k = lambdas**k
    lambdas_k = lambdas_k / (lambdas_k.mean() + 1e-20) + c # Added epsilon
    lambdas_k = lambdas_k.flatten()
    p = lambdas_k / (lambdas_k.sum() + 1e-20) # Added epsilon
    idx_train = jax.random.choice(
        key, len(lambdas), shape=(batch_size,), p=p
    )
    train_data = (tc[idx_train], xc[idx_train])
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
all_loss_pde = []
all_loss_ent = []
all_lam_pde = []
all_lam_ent = []
all_gamma = []
# all_lamB = [] # Removed as lamB is no longer used
all_max_RBA = []
all_lambdas=[]
all_idx=[]
start = time.time()
pbar = tqdm(range(1, EPOCHS + 1), desc='Training Progress')
gamma_it=get_gamma(eta, max_RBA0)
#Global weights
max_RBA=max_RBA0
# lamE,lamB=1,max_RBA**2 # lamB is removed
#RBA
# step_RBA=(cap_RBA-max_RBA0)/(EPOCHS/50000-1) # This dynamic update is removed for simplicity in Adam phase
# initialize grads container
all_grads={
    'grad_bar_PDE': 1.0,
    'grad_bar_ENT': 1.0,
}
for e in pbar:
    key, subkey = jax.random.split(key)
    idx_train, train_data = sample_points_pdf(
        subkey, batch_size, lambdas, tc, xc, k=args.k_samp, c=args.c_samp
    )
    # single run
    # single run
    hyperparams = (gamma_it, eta, gamma_grads) # Add gamma_grads for Self-Balancing
    all_loss_it, gradient, lambdas, lam_pde, lam_ent, all_grads = apply_model(
        apply_fn, update_weights_fn,
        params, lambdas,
        idx_train,
        e,
        hyperparams,
        nu_0, # Pass the fixed initial viscosity
        lam_pde, lam_ent, all_grads,
        *train_data
    )
    for key_params in params['params']:
        params, states[key_params] = update_model(key_params, optimizers[key_params], gradient, params, states[key_params])
    log_frequency = 1 if e < 500 else 100 if e <= 5000 else 500
    if e % log_frequency == 0:
        # Update RBA
        # max_RBA=max_RBA0+step_RBA*e//50000 # Dynamic update removed
        gamma_it=get_gamma(eta, max_RBA)
        all_lambdas.append(np.array(lambdas))  #
        all_idx.append(idx_train)
        #Compute errors
        u_pred_it=apply_fn(params, t, x)
        error = relative_l2(u_pred_it, u_gt)
        error_vec = u_pred_it - u_gt
        L_infty = scipy.linalg.norm(error_vec.flatten(), ord=np.inf)
        L_2 = scipy.linalg.norm(error_vec.flatten(), ord=2)
        # Updating the tqdm progress bar with loss and other metrics
        pbar.set_description(f"It: {e}/{EPOCHS} | Error: {error:.3e} | L_pde: {lam_pde:.2f} | L_ent: {lam_ent:.2f} | lam_max: {lambdas.max():.3f}| nu: {nu_0:.1e}|")
        all_errors.append(error)
        all_its.append(e)
        all_loss.append(all_loss_it['Loss'])
        all_loss_pde.append(all_loss_it['loss_PDE'])
        all_loss_ent.append(all_loss_it['loss_ENT'])
        all_lam_pde.append(lam_pde)
        all_lam_ent.append(lam_ent)
        all_gamma.append(gamma_it)
        all_max_RBA.append(max_RBA)
        # Error norms
        all_L_infty.append(L_infty)
        all_L_2.append(L_2)
end = time.time()
print(f'Runtime: {((end - start) / EPOCHS * 1000):.2f} ms/iter.')

# ## Cell 14: Visualization & Evaluation (Adam Phase)
#
# Adjusting the plot to use a log scale for both loss and error
plt.figure(figsize=(12, 6))
# Plotting loss history with log scale
plt.subplot(2, 2, 1)
plt.plot(all_its, all_errors, label='Loss', color='blue')
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Relative $L^2$ Error (log scale)')
plt.title('Error History (Log Scale)')
plt.grid(True)
plt.legend()
# Show the plots
plt.tight_layout()
plt.savefig('Loss_Adam.png')
plt.show()
error = relative_l2(apply_fn(params, t, x), u_gt)
print(f'RL2 error: {error:.8f}')
it=-1
print('Solution:')
u = apply_fn(params, t, x)
u = u.reshape(T.shape)
lambdas_grid=all_lambdas[it]
idx_it=all_idx[it]
# Plotting code
fig = plt.figure(figsize=(12, 3))
levels=100
# Scatter plot for collocation, initial, and boundary points
ax1 = fig.add_subplot(131)
contour = ax1.contourf(T, X, u, cmap='jet', levels=levels)  # Filled contour plot with 50 levels
fig.colorbar(contour)  # Add color bar to show scale
ax1.set_xlabel('t')
ax1.set_ylabel('x')
ax1.set_title('Prediction')
# Surface plot for Exact solution
ax2 = fig.add_subplot(132)
contour = ax2.contourf(T, X, Exact0, cmap='jet', levels=levels)  # Filled contour plot with 50 levels
fig.colorbar(contour)  # Add color bar to show scale
ax2.set_xlabel('t')
ax2.set_ylabel('x')
ax2.set_title('Reference')
# Surface plot for Exact solution
ax3 = fig.add_subplot(133)
contour = ax3.contourf(T, X, np.abs(Exact0-u), cmap='jet', levels=levels)  # Filled contour plot with 50 levels
fig.colorbar(contour)  # Add color bar to show scale
ax3.set_xlabel('t')
ax3.set_ylabel('x')
ax3.set_title('Error')
plt.tight_layout()
plt.savefig('Errorss_Adam.png',dpi=300)
plt.show()
#

# ## Cell 15: Preparation for Second-Order Optimization (SSBroyden)
gamma = args.gamma_bfgs
eta = 1-gamma
initial_weights, unflatten_func = ravel_pytree(params)
params_test=unflatten_func(initial_weights)
error = relative_l2(apply_fn(params, t, x), u_gt)
print(f'RL2 error: {error:.8f}') 
error = relative_l2(apply_fn(params_test, t, x), u_gt)
print(f'Reconstructed parameters: RL2 error: {error:.8f}') 
X=np.hstack([tc.astype(jnp.float64),xc.astype(jnp.float64)])
#

# ## Cell 16: Defining the Loss Functions for Second-Order Optimization
@partial(jax.jit, static_argnums=(0,))

def apply_model_2nd_Order(apply_fn, params, nu, lam_pde, lam_ent, *train_data):
    t, x = train_data
    
    # PDE Loss (no vRBA in 2nd order)
    residuals_pde = PDE_residuals(params, t, x, apply_fn, nu)
    pde_loss = jnp.mean(residuals_pde**2)

    # Entropy Loss
    residuals_ent = Burgers_entropy(params, t, x, apply_fn)
    ent_loss = jnp.mean(nn.relu(residuals_ent)**2)

    # Total loss with frozen weights
    loss_value = lam_pde * pde_loss + lam_ent * ent_loss
    return loss_value
@partial(jax.jit, static_argnums=(0,))
def update_lambdas(apply_fn, params, lambdas, gamma, eta, tc, xc, it, nu):
    t, x = tc, xc
    def get_lambdas(params, lambdas):
        residuals = PDE_residuals(params, t, x, apply_fn, nu)
        # Update lambdas
        r_i_norm = lax.stop_gradient(jnp.abs(residuals))
        new_lambdas = update_weights_fn(r_i_norm, lambdas, it,
                                          gamma=gamma,
                                          eta=eta)
        return new_lambdas
    new_lambdas = get_lambdas(params, lambdas)
    return new_lambdas
@partial(jax.jit, static_argnums=(1, 2))

def loss_and_gradient(weights, N_arg, unflatten_func_arg, nu, lam_pde, lam_ent, *train_data_tuple_arg):
    flat_jax_array = weights
    params_current = unflatten_func_arg(flat_jax_array)
    loss_val_jax = apply_model_2nd_Order(N_arg, params_current, nu, lam_pde, lam_ent, *train_data_tuple_arg)
    return loss_val_jax
#

# ## Cell 17: The SSBroyden Optimization Step Function
@partial(jax.jit, static_argnames=("apply_fn", "unflatten_func", "batch_size", "gamma", "eta", "k", "c", "static_options"))
def SSBroyden_step(
    initial_weights, H0, key,
    tc, xc, lambdas,
    apply_fn, unflatten_func, batch_size, gamma, eta, k, c, static_options, it, nu,
    lam_pde, lam_ent
):
    key, subkey = jax.random.split(key)
    # 2. Update Lambdas ...
    params_it = unflatten_func(initial_weights)
    updated_lambdas = update_lambdas(apply_fn, params_it, lambdas, gamma, eta, tc, xc, it, nu)
    _, current_train_data_tuple = sample_points_pdf(
        subkey, batch_size, updated_lambdas, tc, xc, k, c
    )
    # 4. Prepare the full options dictionary for minimize
    current_options = dict(static_options)
    current_options['initial_H'] = H0
    # 5. Run the SSBroyden optimization for Nchange steps
    result = minimize(
        fun=loss_and_gradient,
        x0=initial_weights,
args=(apply_fn, unflatten_func, nu, lam_pde, lam_ent, *current_train_data_tuple),
        method='BFGS', # SSBroyden version of BFGs
        options=current_options)
    # 6. Recycle the Hessian purely
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
    return result.x, final_H0, key, result.fun, result.nit, updated_lambdas
# Initialize the Hessian
H0 = jnp.eye(len(initial_weights), dtype=jnp.float64)
# SSBroyden Hyperparameters
static_options = static_options_SSBroyden
static_options['maxiter'] = Nchange
static_options_tuple = tuple(static_options.items())
# Create the viscosity annealing schedule (logarithmic spacing)
nu_schedule = jnp.geomspace(nu_0, nu_final, N_nu_steps)
all_nu = []  # To log the viscosity values
# Training
effective_steps = 0
Adam_Steps = all_its[-1] if all_its else 0
# Create the adaptive iteration schedule for the SSBroyden phase
ssbroyden_iters_schedule = np.linspace(ssbroyden_iters_min, ssbroyden_iters_max, N_nu_steps, dtype=int)

# Outer loop for viscosity continuation with adaptive dwell time
for i, nu_k in enumerate(nu_schedule):
    total_iters_for_nu = ssbroyden_iters_schedule[i]
    num_outer_iterations = total_iters_for_nu // Nchange

    print(f"\n--- Starting SSBroyden phase with nu = {nu_k:.2e} for {total_iters_for_nu} iterations ---")

    # Reset Hessian for each new viscosity value
    H0 = jnp.eye(len(initial_weights), dtype=jnp.float64)

    pbar = tqdm(range(num_outer_iterations), desc=f"SSBroyden (nu={nu_k:.1e})")
    # Inner loop for optimization at a fixed viscosity
    for it in pbar:
        initial_weights, H0, key, loss_val, nit, lambdas = SSBroyden_step(
            initial_weights, H0, key,
            tc, xc, lambdas,
            apply_fn, unflatten_func, batch_size, gamma, eta, k_samp_bfgs, c_samp_bfgs,
            static_options_tuple, it, nu_k, lam_pde, lam_ent
        )
        effective_steps += nit
        # --- Metrics Calculation ---
        current_params_for_eval = unflatten_func(initial_weights)
        # 1. Error metrics
        u_pred_it = apply_fn(current_params_for_eval, t, x)
        error = relative_l2(u_pred_it, u_gt)
        error_vec = u_pred_it - u_gt
        L_infty = scipy.linalg.norm(error_vec.flatten(), ord=np.inf)
        L_2 = scipy.linalg.norm(error_vec.flatten(), ord=2)
        # 2. Component Loss Calculation (MISSING IN ORIGINAL CODE)
        # We need to re-evaluate residuals to log the split between PDE and Entropy
        res_pde = PDE_residuals(current_params_for_eval, t, x, apply_fn, nu_k)
        loss_pde_val = jnp.mean(res_pde**2)
        res_ent = Burgers_entropy(current_params_for_eval, t, x, apply_fn)
        loss_ent_val = jnp.mean(nn.relu(res_ent)**2)
        # --- Logging ---
        all_errors.append(error)
        all_loss.append(loss_val)
        all_its.append(effective_steps + Adam_Steps)
        all_L_infty.append(L_infty)
        all_L_2.append(L_2)
        all_nu.append(nu_k)
        # Append the missing component losses
        all_loss_pde.append(loss_pde_val)
        all_loss_ent.append(loss_ent_val)
        # Append the weights (they remain fixed in this phase, but needed for plotting continuity)
        all_lam_pde.append(lam_pde)
        all_lam_ent.append(lam_ent)
        pbar.set_postfix({
            'It': effective_steps,
            'Loss': f'{loss_val:.3e}',
            'RL2': f'{float(error):.3e}',
            'nu': f'{nu_k:.2e}',
            'lam_max': f'{jnp.max(lambdas):.3e}',
        })
#


# ## Cell 19: Final Visualization & Error Reporting
T, X = np.meshgrid(t0, x0)
# Ground Truth (already float64 due to T, X)
t = T.flatten()[:,None]
x = X.flatten()[:,None]
u_gt = Exact0.flatten()[:,None]
it=-1
print('Solution:')
u = apply_fn(current_params_for_eval, t, x)
u = u.reshape(T.shape)

fig, axes = plt.subplots(3, 3, figsize=(18, 12))

# Plot individual loss histories
ax_loss_pde = axes[0, 0]
ax_loss_pde.plot(all_its[:len(all_loss_pde)], all_loss_pde, label='PDE Loss', color='blue')
ax_loss_pde.set_yscale('log')
ax_loss_pde.set_title('PDE Loss History (Log Scale)')
ax_loss_pde.set_xlabel('Iterations')
ax_loss_pde.set_ylabel('Loss (log scale)')
ax_loss_pde.grid(True, which="both", ls="--")
ax_loss_pde.legend()

ax_loss_ent = axes[0, 1]
ax_loss_ent.plot(all_its[:len(all_loss_ent)], all_loss_ent, label='Entropy Loss', color='orange')
ax_loss_ent.set_yscale('log')
ax_loss_ent.set_title('Entropy Loss History (Log Scale)')
ax_loss_ent.set_xlabel('Iterations')
ax_loss_ent.set_ylabel('Loss (log scale)')
ax_loss_ent.grid(True, which="both", ls="--")
ax_loss_ent.legend()

ax_tm = axes[0, 2]  # Re-assign error plot
ax_tm.plot(all_its, all_errors, label='Rel. L2 Error', color='green')
ax_tm.set_yscale('log')
ax_tm.set_title('Relative $L^2$ Error History (Log Scale)')
ax_tm.set_xlabel('Iterations')
ax_tm.set_ylabel('Rel. $L^2$ Error (log scale)')
ax_tm.grid(True, which="both", ls="--")
ax_tm.legend()

ax_tr = axes[1, 2]
print("Calculating PDE residuals for plotting...")
res = PDE_residuals(current_params_for_eval, t, x, apply_fn, nu_final)
res_grid = jnp.abs(res.reshape(T.shape))
print(f"Residuals calculated. Min: {res_grid.min()}, Max: {res_grid.max()}")
res_log_norm = colors.LogNorm(vmin=res_grid.min()+1e-16, vmax=res_grid.max() + 1e-16)
mesh_res = ax_tr.pcolormesh(T, X, res_grid, cmap='jet', norm=res_log_norm, shading='gouraud')
fig.colorbar(mesh_res, ax=ax_tr)
ax_tr.set_xlabel('t')
ax_tr.set_ylabel('x')
ax_tr.set_title('|PDE Residual| (Log Scale)')

# Plot for global weights and viscosity
ax_weights = axes[1, 0]
ax_weights.plot(all_its[:len(all_lam_pde)], all_lam_pde, label='$\\lambda_{PDE}$', color='cyan')
ax_weights.plot(all_its[:len(all_lam_ent)], all_lam_ent, label='$\\lambda_{ENT}$', color='magenta')
ax_weights.set_yscale('log')
ax_weights.set_title('Global Weights History (Log Scale)')
ax_weights.set_xlabel('Iterations')
ax_weights.set_ylabel('Weight Value')
ax_weights.grid(True, which="both", ls="--")
ax_weights.legend()

ax_nu = axes[1, 1]
# --- FIX IS HERE: Use negative indexing to get the last N iterations ---
ax_nu.plot(all_its[-len(all_nu):], all_nu, 'o-', label='Viscosity $\\nu$', color='red')
ax_nu.set_yscale('log')
ax_nu.set_title('Viscosity Annealing Schedule')
ax_nu.set_xlabel('Iterations')
ax_nu.set_ylabel('$\\nu$ (log scale)')
ax_nu.grid(True, which="both", ls="--")
ax_nu.legend()

# Placeholder for the third plot in the middle row if needed, or can be turned off
global_vmin = np.min([u, Exact0])
global_vmax = np.max([u, Exact0])
shared_levels = np.linspace(global_vmin, global_vmax, 500)

ax_bl = axes[2, 0]
contour1 = ax_bl.contourf(T, X, Exact0, cmap='jet', levels=shared_levels, extend='both')
fig.colorbar(contour1, ax=ax_bl)
ax_bl.set_xlabel('t')
ax_bl.set_ylabel('x')
ax_bl.set_title('Reference')

ax_bm = axes[2, 1]
contour2 = ax_bm.contourf(T, X, u, cmap='jet', levels=shared_levels, extend='both')
fig.colorbar(contour2, ax=ax_bm)
ax_bm.set_xlabel('t')
ax_bm.set_title('Prediction')

ax_br = axes[2, 2]
error_data = np.abs(Exact0 - u)
err_log_norm = colors.LogNorm(vmin=error_data.min()+1e-16, vmax=error_data.max() + 1e-16)
mesh_err = ax_br.pcolormesh(T, X, error_data, cmap='jet', norm=err_log_norm, shading='gouraud')
fig.colorbar(mesh_err, ax=ax_br)
ax_br.set_xlabel('t')
ax_br.set_title('Pointwise Error (Log Scale)')

plt.tight_layout()
plt.savefig('summary_all.png', dpi=150)
plt.show()
print('Minimum achieved  RL2 error:',np.nanmin(all_errors))

# ## Cell 20: Save Experiment Results to .mat File
#
results_dict = {
    'model_name': args.Name,
    'all_errors': all_errors,
    'all_its': all_its,
    'all_loss': all_loss,
    'all_gamma': all_gamma,
    'all_nu': all_nu,  # Add viscosity log
    'all_max_RBA': all_max_RBA,
    'all_L_infty': all_L_infty,
    'all_L_2': all_L_2,
}
# Save dictionary as a .mat file
scipy.io.savemat('Log_files_SSBroyden.mat', results_dict)