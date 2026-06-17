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
parser.add_argument('--Equation', type=str, default='KdV', help='Name of equation')
parser.add_argument('--Name', type=str, default='SS_Broyden', help='Name of the experiment')
parser.add_argument('--NC', type=int, default=200000, help='Number of samples for training')
parser.add_argument('--NI', type=int, default=512, help='Number of iterations')
parser.add_argument('--NB', type=int, default=512, help='Batch size')
parser.add_argument('--NC_TEST', type=int, default=100, help='Number of test samples')
parser.add_argument('--SEED', type=int, default=9999, help='Random seed')
parser.add_argument('--EPOCHS', type=int, default=5000, help='Number of training epochs')
parser.add_argument('--N_LAYERS', type=int, default=3, help='Number of layers in the network')
parser.add_argument('--HIDDEN', type=int, default=30, help='Number of hidden units per layer')
parser.add_argument('--FEATURES', type=int, default=1, help='Feature size')
parser.add_argument('--embedding_degree', type=int, default=5, help='Degree of features')
parser.add_argument('--rff_scale', type=float, default=3.0, help='Scale for Random Fourier Features')
parser.add_argument('--lr_fact', type=float, default=0.2, help='Scale Lr')
parser.add_argument('--eta', type=float, default=0.01, help='Learning rate or step size for adaptive gamma')
parser.add_argument('--gamma', type=float, default=0.999, help='Decay rate for adaptive gamma')
parser.add_argument('--gamma_bfgs', type=float, default=0.1, help='Decay rate for adaptive gamma')
parser.add_argument('--gamma_grads', type=float, default=0.99, help='Decay rate for adaptive gamma')
parser.add_argument('--alpha', type=float, default=0.999, help='Decay rate for exponential moving average')
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
parser.add_argument('--N_change', type=int, default=200, help='homogenize')
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
##  We use a self scaling algoirhtm to automatically tune the global weights and to increase the cap of RBA to improve learning dunamics
max_RBA0 = args.max_RBA if args.max_RBA is not None else eta / (1 - gamma)
cap_RBA = args.cap_RBA
gamma_grads=args.gamma_grads
## vRBA Params Second order_ Sampling + SSBroycen
k_samp_bfgs = args.k_samp
c_samp_bfgs = args.c_samp
Nchange = args.N_change  # Number of SSBroyden steps before resampling
batch_size=args.batch_size
Nbfgs=200000  # Number of SSBroyden BFGs iterations
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
# Define the analytical solution for the KdV two-soliton interaction
def get_exact_solution(x, t, c1=6.0, c2=2.0):
    """
    Computes the exact two-soliton solution for the KdV equation.
    """
    zeta1 = x - c1 * t + 2
    zeta2 = x - c2 * t - 2
    sqrt_c1 = jnp.sqrt(c1)
    sqrt_c2 = jnp.sqrt(c2)
    numerator_term1 = c1 * jnp.cosh(0.5 * sqrt_c2 * zeta2)**2
    numerator_term2 = c2 * jnp.sinh(0.5 * sqrt_c1 * zeta1)**2
    numerator = 2 * (c1 - c2) * (numerator_term1 + numerator_term2)
    cosh_arg_sum = 0.5 * (sqrt_c1 * zeta1 + sqrt_c2 * zeta2)
    cosh_arg_diff = 0.5 * (sqrt_c1 * zeta1 - sqrt_c2 * zeta2)
    denominator_term1 = (sqrt_c1 - sqrt_c2) * jnp.cosh(cosh_arg_sum)
    denominator_term2 = (sqrt_c1 + sqrt_c2) * jnp.cosh(cosh_arg_diff)
    denominator = (denominator_term1 + denominator_term2)**2
    return numerator / denominator
# Define the spatial derivative of the analytical solution using JAX's autodiff
get_exact_solution_x = jax.jit(jax.vmap(jax.grad(lambda x_val, t_val: get_exact_solution(x_val, t_val).squeeze(), argnums=0), in_axes=(0, 0)))
# Define the spatio-temporal domain
t_domain = [0., 5.]
x_domain = [0., 20.]
# Create a grid for ground truth evaluation
t0 = jnp.linspace(t_domain[0], t_domain[1], 101, dtype=jnp.float64)
x0 = jnp.linspace(x_domain[0], x_domain[1], 201, dtype=jnp.float64)
T, X = jnp.meshgrid(t0, x0)
# Compute the exact solution on the grid
Exact0 = get_exact_solution(X, T)
# Ground Truth (to compute relative errors)
t = T.flatten()[:, None]
x = X.flatten()[:, None]
u_gt = Exact0.flatten()[:, None]

# ## Cell 4: Data Processing, Sampling, and Visualization

# Define domain boundaries
lb_D = jnp.array([t_domain[0], x_domain[0]])
ub_D = jnp.array([t_domain[1], x_domain[1]])

# Generate collocation points using Latin Hypercube Sampling
X_c = lb_D + (ub_D - lb_D) * lhs(2, NC)
X_c = X_c.astype(jnp.float64)
tc = X_c[:, 0:1]
xc = X_c[:, 1:2]

# --- Initial Conditions (t=0) ---
x_ic = jnp.linspace(x_domain[0], x_domain[1], NI)[:, None].astype(jnp.float64)
t_ic = jnp.zeros_like(x_ic)
u_ic = get_exact_solution(x_ic, t_ic)

# --- Boundary Conditions ---
# Left Boundary (x=0) - Dirichlet
t_bc_left = jnp.linspace(t_domain[0], t_domain[1], NB)[:, None].astype(jnp.float64)
x_bc_left = jnp.zeros_like(t_bc_left)
u_bc_left = get_exact_solution(x_bc_left, t_bc_left)

# Right Boundary (x=20) - Dirichlet
t_bc_right = jnp.linspace(t_domain[0], t_domain[1], NB)[:, None].astype(jnp.float64)
x_bc_right = jnp.ones_like(t_bc_right) * x_domain[1]
u_bc_right = get_exact_solution(x_bc_right, t_bc_right)

# Right Boundary (x=20) - Neumann
t_bc_neumann = jnp.linspace(t_domain[0], t_domain[1], NB)[:, None].astype(jnp.float64)
x_bc_neumann = jnp.ones_like(t_bc_neumann) * x_domain[1]
u_x_bc_neumann = get_exact_solution_x(x_bc_neumann, t_bc_neumann)

# Consolidate all training data into a tuple
train_data = (
    tc, xc,
    t_ic, x_ic, u_ic,
    t_bc_left, x_bc_left, u_bc_left,
    t_bc_right, x_bc_right, u_bc_right,
    t_bc_neumann, x_bc_neumann, u_x_bc_neumann
)

# Initialize local weights (lambdas) for collocation points
lambdas = jnp.ones((NC, 1), dtype=jnp.float64) * 0.1

# --- Visualization ---
fig = plt.figure(figsize=(9, 3))
ax1 = fig.add_subplot(121)
ax1.scatter(tc, xc, color='blue', label='Collocation', s=0.2, alpha=0.1)
ax1.scatter(t_ic, x_ic, color='green', label='Initial (IC)', marker='x')
ax1.scatter(t_bc_left, x_bc_left, color='red', label='BC Left', marker='s')
ax1.scatter(t_bc_right, x_bc_right, color='orange', label='BC Right (D)', marker='o')
ax1.scatter(t_bc_neumann, x_bc_neumann, color='purple', label='BC Right (N)', marker='^')
ax1.set_xlabel('Time (t)')
ax1.set_ylabel('Space (x)')
ax1.set_title('Training Data Distribution')
ax1.legend()

ax2 = fig.add_subplot(122)
contour = ax2.contourf(T, X, Exact0, cmap='jet', levels=100)
fig.colorbar(contour)
ax2.set_xlabel('Time (t)')
ax2.set_ylabel('Space (x)')
ax2.set_title('Exact Solution')
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
        def b_init(key, shape, dtype=jnp.float64):
            return self.s * jax.random.normal(key, shape, dtype)
        B = self.param('B', b_init, (input_features, self.degree))
        X_proj = jnp.dot(X, B)
        # Do not concatenate original features, as per strategy
        return jnp.concatenate([jnp.sin(X_proj), jnp.cos(X_proj)], axis=-1)
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
    embedding_degree: int
    rff_scale: float
    lb: Array
    ub: Array
    activation: Callable = nn.tanh
    def setup(self):
        self.embedding = Random_Fourier_Embedding(degree=self.embedding_degree, s=self.rff_scale)
        self.MLP = MLP(features=self.features, activation=self.activation)
    @nn.compact
    def __call__(self, t: Array, x: Array) -> Array:
        # Combine t and x for normalization and embedding
        X_in = jnp.concatenate([t, x], axis=-1)
        # MinMax Normalization Layer to [0, 1]
        X_normalized = (X_in - self.lb) / (self.ub - self.lb)
        # Apply Random Fourier Features
        X_embedded = self.embedding(X_normalized)
        # Pass embedded features to the MLP
        H = self.MLP(X_embedded)
        return H

# ## Cell 6: Model and Optimizer Initialization
feat_sizes = tuple([HIDDEN for _ in range(N_LAYERS)] + [FEATURES])
print(feat_sizes)
# make & init model
model = PINN(
    features=feat_sizes,
    activation=nn.tanh,
    embedding_degree=embedding_degree,
    rff_scale=args.rff_scale,
    lb=lb_D,
    ub=ub_D
)
params = model.init(subkey, jnp.ones((NC, 1), dtype=jnp.float64), jnp.ones((NC, 1), dtype=jnp.float64))
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), params)
optimizers = {}
for key_params in params['params'].keys():
    optimizers[key_params] = optax.adam(optax.exponential_decay(lr0, decay_step, decay_rate, staircase=False))

# Initialize optimizer states for each parameter group
states = {key_params: optim.init(params['params'][key_params]) for key_params, optim in optimizers.items()}
# forward & loss function
apply_fn = jax.jit(model.apply)
total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(total_params )

# ## Cell 7: PDE Residual Definition
def KdV_residual(params, t, x, apply_fn):
    """Computes the KdV residual."""
    u = apply_fn(params, t, x)
    
    # Define tangent vectors for differentiation
    v_t = jnp.ones_like(t)
    v_x = jnp.ones_like(x)
    
    # Compute u_t using jvp
    u_t = jvp(lambda t_val: apply_fn(params, t_val, x), (t,), (v_t,))[1]
    
    # Compute u_x using jvp
    u_x = jvp(lambda x_val: apply_fn(params, t, x_val), (x,), (v_x,))[1]
    
    # To get u_xxx, we need to apply jvp three times.
    # This is equivalent to a third-order directional derivative.
    # 1. Define a function that computes the first derivative (gradient w.r.t. x)
    get_u_x = lambda x_val: jvp(lambda xv: apply_fn(params, t, xv), (x_val,), (v_x,))[1]
    # 2. Compute u_xx
    u_xx = jvp(get_u_x, (x,), (v_x,))[1]
    # 3. Compute u_xxx
    u_xxx = jvp(lambda x_val: jvp(get_u_x, (x_val,), (v_x,))[1], (x,), (v_x,))[1]
    
    # KdV residual
    return u_t + 6.0 * u * u_x + u_xxx

PDE_residuals = KdV_residual

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
    *train_data
):
    gamma, eta = hyperparams
    (tc, xc,
     t_ic, x_ic, u_ic,
     t_bc_left, x_bc_left, u_bc_left,
     t_bc_right, x_bc_right, u_bc_right,
     t_bc_neumann, x_bc_neumann, u_x_bc_neumann) = train_data

    lambdas_it = lambdas[idx_train]

    def loss_pde(params, lambdas_batch):
        residuals = PDE_residuals(params, tc, xc, apply_fn)
        r_i_norm = lax.stop_gradient(jnp.abs(residuals))
        new_lambdas = update_weights_fn(r_i_norm, lambdas_batch, it, gamma=gamma, eta=eta)
        pde_loss = jnp.mean((new_lambdas * residuals)**2)
        return pde_loss, new_lambdas

    def loss_ic(params):
        u_pred_ic = apply_fn(params, t_ic, x_ic)
        return jnp.mean((u_ic - u_pred_ic)**2)

    def loss_bc_left(params):
        u_pred_bc_left = apply_fn(params, t_bc_left, x_bc_left)
        return jnp.mean((u_bc_left - u_pred_bc_left)**2)

    def loss_bc_right_d(params):
        u_pred_bc_right = apply_fn(params, t_bc_right, x_bc_right)
        return jnp.mean((u_bc_right - u_pred_bc_right)**2)

    def loss_bc_right_n(params):
        v_x = jnp.ones_like(x_bc_neumann)
        u_x_pred = jvp(lambda x: apply_fn(params, t_bc_neumann, x), (x_bc_neumann,), (v_x,))[1]
        return jnp.mean((u_x_bc_neumann - u_x_pred)**2)

    # --- Compute losses and gradients for all components ---
    (pde_loss, new_lambdas), grad_pde = jax.value_and_grad(loss_pde, has_aux=True)(params, lambdas_it)
    ic_loss, grad_ic = jax.value_and_grad(loss_ic)(params)
    bc_left_loss, grad_bc_left = jax.value_and_grad(loss_bc_left)(params)
    bc_right_d_loss, grad_bc_right_d = jax.value_and_grad(loss_bc_right_d)(params)
    bc_right_n_loss, grad_bc_right_n = jax.value_and_grad(loss_bc_right_n)(params)

    # Update local weights
    lambdas = lambdas.at[idx_train].set(new_lambdas)

    # --- Apply fixed global weights and sum gradients ---
    lam_pde, lam_data = 1.0, 100.0
    total_gradient = jax.tree_util.tree_map(
        lambda g_pde, g_ic, g_bcl, g_bcrd, g_bcrn:
            lam_pde * g_pde +
            lam_data * (g_ic + g_bcl + g_bcrd + g_bcrn),
        grad_pde, grad_ic, grad_bc_left, grad_bc_right_d, grad_bc_right_n
    )

    total_loss = (lam_pde * pde_loss +
                  lam_data * (ic_loss + bc_left_loss + bc_right_d_loss + bc_right_n_loss))

    all_loss_terms = {
        'loss_PDE': pde_loss,
        'loss_IC': ic_loss,
        'loss_BC_Left': bc_left_loss,
        'loss_BC_Right_D': bc_right_d_loss,
        'loss_BC_Right_N': bc_right_n_loss,
        'Loss': total_loss,
    }

    return all_loss_terms, total_gradient, lambdas
#

# ## Cell 12: vRBA Implementation via Importance Sampling
#
@partial(jax.jit, static_argnames=("batch_size", "k", "c"))
def sample_points_pdf(key, batch_size, lambdas, full_train_data, k=1, c=0.5):
    # Unpack only the collocation points for sampling
    tc, xc = full_train_data[0], full_train_data[1]
    
    # Adjust lambdas with exponent k and normalize
    lambdas_k = lambdas**k
    lambdas_k = lambdas_k / (lambdas_k.mean() + 1e-20) + c
    lambdas_k = lambdas_k.flatten()
    p = lambdas_k / (lambdas_k.sum() + 1e-20)
    
    # Get indices for the collocation points
    idx_train = jax.random.choice(
        key, len(lambdas), shape=(batch_size,), p=p
    )
    
    # Create the new train_data tuple with sampled collocation points
    # and the original IC/BC data
    sampled_train_data = (
        tc[idx_train], xc[idx_train],
        *full_train_data[2:]  # Unpack the rest of the data
    )
    
    return idx_train, sampled_train_data

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
all_lambdas = []
all_idx = []
start = time.time()
pbar = tqdm(range(1, EPOCHS + 1), desc='Training Progress')
# Use a fixed gamma for vRBA as per strategy
gamma_it = gamma

for e in pbar:
    key, subkey = jax.random.split(key)

    # Pass the full train_data tuple to the sampling function
    idx_train, sampled_train_data = sample_points_pdf(
        subkey, batch_size, lambdas, train_data, k=args.k_samp, c=args.c_samp
    )

    # single run
    hyperparams = (gamma_it, eta)
    all_loss_it, gradient, lambdas = apply_model(
        apply_fn, update_weights_fn,
        params, lambdas,
        idx_train,
        e,
        hyperparams,
        *sampled_train_data
    )

    for key_params in params['params']:
        params, states[key_params] = update_model(key_params, optimizers[key_params], gradient, params, states[key_params])

    log_frequency = 1 if e < 500 else 100 if e <= 5000 else 500
    if e % log_frequency == 0:
        all_lambdas.append(np.array(lambdas))
        all_idx.append(idx_train)

        # Compute errors
        u_pred_it = apply_fn(params, t, x)
        error = relative_l2(u_pred_it, u_gt)
        error_vec = u_pred_it - u_gt
        L_infty = scipy.linalg.norm(error_vec.flatten(), ord=np.inf)
        L_2 = scipy.linalg.norm(error_vec.flatten(), ord=2)

        # Updating the tqdm progress bar
        pbar.set_description(f"It: {e}/{EPOCHS} | Error: {error:.3e} | Loss: {all_loss_it['Loss']:.3e} | lam_max: {lambdas.max():.3f}")

        all_errors.append(error)
        all_its.append(e)
        all_loss.append(all_loss_it['Loss'])
        all_gamma.append(gamma_it)

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
def apply_model_2nd_Order(apply_fn, params, *train_data):
    # Unpack all data
    (tc, xc,
     t_ic, x_ic, u_ic,
     t_bc_left, x_bc_left, u_bc_left,
     t_bc_right, x_bc_right, u_bc_right,
     t_bc_neumann, x_bc_neumann, u_x_bc_neumann) = train_data

    # --- Define individual loss components ---
    def loss_pde(params):
        residuals = PDE_residuals(params, tc, xc, apply_fn)
        return jnp.mean(residuals**2)

    def loss_ic(params):
        u_pred_ic = apply_fn(params, t_ic, x_ic)
        return jnp.mean((u_ic - u_pred_ic)**2)

    def loss_bc_left(params):
        u_pred_bc_left = apply_fn(params, t_bc_left, x_bc_left)
        return jnp.mean((u_bc_left - u_pred_bc_left)**2)

    def loss_bc_right_d(params):
        u_pred_bc_right = apply_fn(params, t_bc_right, x_bc_right)
        return jnp.mean((u_bc_right - u_pred_bc_right)**2)

    def loss_bc_right_n(params):
        v_x = jnp.ones_like(x_bc_neumann)
        u_x_pred = jvp(lambda x: apply_fn(params, t_bc_neumann, x), (x_bc_neumann,), (v_x,))[1]
        return jnp.mean((u_x_bc_neumann - u_x_pred)**2)

    # --- Calculate and combine losses with fixed weights ---
    lam_pde, lam_data = 1.0, 100.0
    total_loss = (lam_pde * loss_pde(params) +
                  lam_data * (loss_ic(params) + loss_bc_left(params) +
                              loss_bc_right_d(params) + loss_bc_right_n(params)))
    return total_loss
@partial(jax.jit, static_argnums=(0,))
def update_lambdas(apply_fn,params,lambdas,gamma,eta,tc,xc,it):
    t,x=tc,xc
    def get_lambdas(params, lambdas):
        residuals = PDE_residuals(params, t, x, apply_fn)
        # Update lambdas
        r_i_norm = lax.stop_gradient(jnp.abs(residuals))
        new_lambdas = update_weights_fn(r_i_norm,lambdas, it,
                                          gamma=gamma,
                                          eta=eta)
        return new_lambdas
    new_lambdas=get_lambdas(params, lambdas)
    return new_lambdas
@partial(jax.jit, static_argnums=(1, 2))
def loss_and_gradient(weights, N_arg, unflatten_func_arg, *train_data_tuple_arg):
    flat_jax_array = weights
    params_current = unflatten_func_arg(flat_jax_array)
    loss_val_jax = apply_model_2nd_Order(N_arg, params_current, *train_data_tuple_arg)
    return loss_val_jax
#

# ## Cell 17: The SSBroyden Optimization Step Function
@partial(jax.jit, static_argnames=("apply_fn", "unflatten_func", "batch_size", "gamma", "eta", "k", "c", "static_options"))
def SSBroyden_step(
    initial_weights, H0, key,
    lambdas,
    apply_fn, unflatten_func, batch_size, gamma, eta, k, c, static_options, it,
    full_train_data
):
    key, subkey = jax.random.split(key)
    
    # Unpack collocation points for lambda update
    tc, xc = full_train_data[0], full_train_data[1]
    
    # 2. Update Lambdas ...
    params_it = unflatten_func(initial_weights)
    updated_lambdas = update_lambdas(apply_fn, params_it, lambdas, gamma, eta, tc, xc, it)
    
    # 3. Sample points using the updated lambdas and full data tuple
    _, current_train_data_tuple = sample_points_pdf(
        subkey, batch_size, updated_lambdas, full_train_data, k, c
    )
    
    # 4. Prepare the full options dictionary for minimize
    current_options = dict(static_options)
    current_options['initial_H'] = H0
    
    # 5. Run the SSBroyden optimization for Nchange steps
    result = minimize(
        fun=loss_and_gradient,
        x0=initial_weights,
        args=(apply_fn, unflatten_func, *current_train_data_tuple),
        method='BFGS',  # SSBroyden version of BFGS
        options=current_options
    )
    
    # 6. Recycle the Hessian
    new_H0 = result.hess_inv
    new_H0 = (new_H0 + jnp.transpose(new_H0)) / 2
    try:
        L = jnp.linalg.cholesky(new_H0)
        is_failed = jnp.any(jnp.isnan(L))
        final_H0 = jax.lax.cond(
            is_failed,
            lambda op: jnp.eye(op.shape[0], dtype=op.dtype),
            lambda op: op,
            operand=new_H0
        )
    except LinAlgError:
        final_H0 = jnp.eye(new_H0.shape[0], dtype=new_H0.dtype)
        
    return result.x, final_H0, key, result.fun, result.nit, updated_lambdas

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
for it in pbar:
    initial_weights, H0, key, loss_val, nit, lambdas = SSBroyden_step(
        initial_weights, H0, key,
        lambdas,
        apply_fn, unflatten_func, batch_size, gamma, eta, k_samp_bfgs, c_samp_bfgs,
        static_options_tuple, it,
        train_data  # Pass the full data tuple
    )
    effective_steps=effective_steps+nit
    current_params_for_eval = unflatten_func(initial_weights)
    u_pred_it=apply_fn(current_params_for_eval, t, x)
    error = relative_l2(u_pred_it, u_gt)
    error_vec = u_pred_it - u_gt
    L_infty = scipy.linalg.norm(error_vec.flatten(), ord=np.inf)
    L_2 = scipy.linalg.norm(error_vec.flatten(), ord=2)
    all_errors.append(error)
    all_loss.append(loss_val)
    all_its.append(effective_steps+Adam_Steps)
    all_L_infty.append(L_infty)
    all_L_2.append(L_2)
    pbar.set_postfix({
        'It': effective_steps,
        'Loss': f'{loss_val:.3e}',
        'RL2': f'{float(error):.3e}',
        'lam_min': f'{jnp.min(lambdas):.3e}',
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
global_vmin = np.min([u, Exact0])
global_vmax = np.max([u, Exact0])
shared_levels = np.linspace(global_vmin, global_vmax, 500)
ax_bl = axes[1, 0]
contour1 = ax_bl.contourf(T, X, Exact0, cmap='jet', levels=shared_levels, extend='both')
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
error_data = np.abs(Exact0 - u)
err_log_norm = colors.LogNorm(vmin=error_data.min()+1e-16, vmax=error_data.max() + 1e-16)
mesh_err = ax_br.pcolormesh(T, X, error_data, cmap='jet', norm=err_log_norm, shading='gouraud')
fig.colorbar(mesh_err, ax=ax_br)
ax_br.set_xlabel('t')
ax_br.set_title('Pointwise Error (Log Scale)')
plt.tight_layout()
plt.savefig('summary_all.png', dpi=150) # 150 DPI is a good balance for token size
plt.show()
print('Minimum achieved  RL2 error:',np.nanmin(all_errors))
# --- Add 1D Slice Plots as per Strategy ---
fig_slices = plt.figure(figsize=(18, 5))
time_slices = [0.0, 2.5, 5.0]
time_indices = [np.argmin(np.abs(t0 - ts)) for ts in time_slices]

for i, (t_val, t_idx) in enumerate(zip(time_slices, time_indices)):
    ax = fig_slices.add_subplot(1, 3, i + 1)
    ax.plot(x0, Exact0[:, t_idx], 'b-', linewidth=2, label='Exact')
    ax.plot(x0, u[:, t_idx], 'r--', linewidth=2, label='Predicted')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.set_title(f'Solution at $t={t_val}$')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('summary_slices.png', dpi=150)
plt.show()

# ## Cell 20: Save Experiment Results to .npz File
# Retrieve predicted and exact solutions from the previous cell's workspace
u_pred = u
u_exact = Exact0
# Save the required data artifacts to a compressed NumPy .npz file
np.savez(
    'kdv_solution.npz',
    u_pred=u_pred,
    u_exact=u_exact,
    x=x0,  # 1D x-coordinate array
    t=t0   # 1D t-coordinate array
)
print("Results saved to kdv_solution.npz")
print("Keys in file: u_pred, u_exact, x, t")