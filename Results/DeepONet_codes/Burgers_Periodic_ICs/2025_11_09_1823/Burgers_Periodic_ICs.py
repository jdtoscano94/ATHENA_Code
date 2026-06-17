# ## Cell 1: Core Imports and Project Setup
import jax
import jax.numpy as jnp
from jax import random, grad, value_and_grad, vmap, jit, hessian, lax
from jax.lax.linalg import qr
from jax.nn import relu
from jax.numpy import index_exp as index
from jax.flatten_util import ravel_pytree
from jax import flatten_util
from flax import linen as nn
from typing import Sequence, Callable
import optax
from Crunch.Models.layers import *
from Crunch.Models.polynomials import *
from Crunch.Auxiliary.metrics import *
import numpy as np
import scipy
import scipy.io as sio
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import trange, tqdm
from functools import partial
import itertools
import sys
import os
from jax import config
config.update("jax_default_matmul_precision", "float32")
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
class Polynomial_Embedding(nn.Module):
    degree: int
    step: int = 1
    polynomial_type: str ='T'
    def setup(self):
        self.T_funcs = [globals()[f"{self.polynomial_type}{i}"] for i in range(0, self.degree+1, self.step)]
    @nn.compact
    def __call__(self, X):
        C_n = self.param('c_i', nn.initializers.ones, (len(self.T_funcs)))
        C_n_T_n = jnp.hstack([C_n[i] / (i + 1) * self.T_funcs[i](X) for i in range(len(self.T_funcs))])
        return C_n_T_n
class WN_layer(nn.Module):
    out_features: int
    kernel_init: nn.initializers.Initializer
    def setup(self):
        self.b = self.param('b', nn.initializers.zeros, (self.out_features,))
        self.g = self.param('g', nn.initializers.ones, (self.out_features,))
    @nn.compact
    def __call__(self, H):
        in_features = H.shape[-1]
        W = self.param('W', self.kernel_init, (in_features, self.out_features))
        V = W / jnp.linalg.norm(W, axis=0, keepdims=True)
        return self.g * jnp.dot(H, V) + self.b
class AdaptiveResNet(nn.Module):
    out_features: int
    def setup(self):
        self.alpha = self.param('alpha', nn.initializers.ones, ())
    @nn.compact
    def __call__(self, H):
        init = nn.initializers.glorot_normal()
        F = nn.activation.tanh(WN_layer(self.out_features, kernel_init=init)(H))
        G = WN_layer(self.out_features, kernel_init=init)(F)
        H = nn.activation.tanh(self.alpha * G + (1 - self.alpha) * H)
        return H
class RBF_KAN_layer(nn.Module):
    out_dim: int
    degree: int
    grid_min: float = -2.0
    grid_max: float = 2.0
    normalization: callable = jnp.tanh
    def setup(self):
        self.num_centers = self.degree
        self.sigma = (self.grid_max - self.grid_min) / (self.num_centers - 1)
        self.normalization_fn = self.normalization
    @nn.compact
    def __call__(self, X):
        X = X  # It is possible to add normalization self.normalization_fn(X) but is generally not needed
        batch_size, in_dim = X.shape
        centers = self.param(
            'centers',
            lambda rng, shape: jnp.linspace(self.grid_min, self.grid_max, self.num_centers),
            (self.num_centers,)
        )
        X_expanded = X[:, None, :]  # Shape: (batch_size, 1, in_dim)
        centers_expanded = centers[None, :, None]  # Shape: (1, num_centers, 1)
        diff = X_expanded - centers_expanded  # Shape: (batch_size, num_centers, in_dim)
        RBF_n = jnp.exp(- (diff ** 2) / (2 * self.sigma ** 2))
        RBF_n = jnp.transpose(RBF_n, (0, 2, 1))
        C_n = self.param(
            'C_n',
            nn.initializers.normal(1 / (in_dim * self.num_centers)),
            (in_dim, self.num_centers, self.out_dim)
        )
        output = jnp.einsum('bin,ino->bo', RBF_n, C_n)
        return output
class get_Psi(nn.Module):
    degree: int
    features: Sequence[int]
    M: int = 10
    def setup(self):
        self.T_funcs = [globals()[f"T{i}"] for i in range(self.degree + 1)]
    @nn.compact
    def __call__(self, inputs):
        init = nn.initializers.glorot_normal()
        sum_psi = 0
        for i, X in enumerate(inputs):
            X = Polynomial_Embedding(degree=self.degree)(X)
            H = nn.activation.tanh(WN_layer(self.features[0], kernel_init=init)(X))
            for fs in self.features[1:-1]:
                H = AdaptiveResNet(out_features=fs)(H)
            H = Polynomial_Embedding(degree=self.degree)(H)
            H = WN_layer(self.features[-1], kernel_init=init)(H)
            sum_psi += H
        return sum_psi
class eMLP(nn.Module):
    layers: Sequence[int]
    activation: Callable = nn.relu
    degree: int = 5
    @nn.compact
    def __call__(self, x):
        init = nn.initializers.glorot_normal()
        X = Polynomial_Embedding(degree=self.degree)(x)
        H = nn.activation.tanh(WN_layer(self.layers[0], kernel_init=init)(X))
        for feat in self.layers[1:-1]:
            H = AdaptiveResNet(out_features=feat)(H)
        H = Polynomial_Embedding(degree=self.degree)(H)
        H = WN_layer(self.layers[-1], kernel_init=init)(H)
        return H
class KKAN(nn.Module):
    features: Sequence[int]
    degree: int=5
    degree_T: int=5
    M: int = 10
    output_dim: int =1
    activation: Callable = nn.tanh
    basis_type: str = 'rbf'
    def setup(self):
        self.get_Psi = get_Psi(degree=self.degree_T, features=self.features, M=self.M)
        if self.basis_type.lower() == 'rbf':
            self.outer_kan_layer = RBF_KAN_layer
        elif self.basis_type.lower() == 'fourier':
            self.outer_kan_layer = Fourier_KAN_layer
        else:
            raise ValueError(f"Unsupported basis_type for KKAN: {self.basis_type}")
    @nn.compact
    def __call__(self, X):
        inputs = [X[:,0:1],X[:,1:2]]
        sum_psi = self.get_Psi(inputs)
        sum_Phi = self.outer_kan_layer(out_dim=self.output_dim, degree=self.degree)(sum_psi)
        return sum_Phi


# ## Cell 2: Load Data and Reshaping or Preprocessing
available_data_path='/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Data/Operator_Network_Examples/'
data=scipy.io.loadmat(available_data_path+'Burger_1000_5000.mat')
print(data.keys())
f_all = data['input']
print(f_all.shape)
s_all = data['output']
num_time_points_s = s_all.shape[1]
num_spatial_points_s = s_all.shape[2]
t_all = jnp.linspace(0, 1, num_time_points_s)
x_all = jnp.linspace(0, 1, num_spatial_points_s)
T, X = jnp.meshgrid(t_all, x_all, indexing='ij')
y_all = jnp.hstack([T.flatten()[:, None], X.flatten()[:, None]])
print(y_all.shape)
print(s_all.shape)
s_all = jnp.reshape(s_all, (s_all.shape[0], s_all.shape[1]*s_all.shape[2]))
print(s_all.shape)


# ## Cell 3: Train-Test Split, Type Conversion, and Data Structuring
train_size = 0.8
s_train, s_test, f_train, f_test = train_test_split(s_all, f_all, train_size=train_size, random_state=42)
s_train = jnp.float32(s_train)
y_train = jnp.float32(y_all)
f_train = jnp.float32(f_train)
s_test = jnp.float32(s_test)
f_test = jnp.float32(f_test)
y_train = jnp.float32(y_train)
print("s_train shape:", s_train.shape)
print("f_train shape:", f_train.shape)
print("s_test shape:", s_test.shape)
print("f_test shape:", f_test.shape)
W = 1.0 / jnp.max(jnp.abs(s_train))
W = float(W)
n_samples = f_train.shape[0]
print(f"Number of training samples (n_samples): {n_samples}")
t_min, t_max = jnp.min(y_train[:, 0]), jnp.max(y_train[:, 0])
x_min, x_max = jnp.min(y_train[:, 1]), jnp.max(y_train[:, 1])
u_min, u_max = jnp.min(f_train), jnp.max(f_train)
normalization_params = {
    't_min': t_min, 't_max': t_max,
    'x_min': x_min, 'x_max': x_max,
    'u_min': u_min, 'u_max': u_max
}
print(f"Normalization parameters: {normalization_params}")
X_DN={'F':f_train,
      'y':y_train,
      'S':s_train,
      'W':W,
      'normalization_params': normalization_params}
X_DN['F'].shape
X_Test={'F':f_test,
      'y':y_train,
      'S':s_test,}
X_DN['X_Test']=X_Test


# ## Cell 4: Hyperparameter Configuration
Run_type='Train'
N=840
m=101
num_layer=5
width_layer=100
embed_dim = 101
branch_layers = [32, 32, 32, embed_dim]
trunk_layers = [32, 32, 32, embed_dim]
print(branch_layers)
print(trunk_layers)
degree_branch = 7
degree_trunk_outer = 5
degree_trunk_inner = 5
ngsteps = 200000
batch_size = 3000
lr = 2e-3
lr_branch = lr
lr_trunk = lr / 3.0
decay_steps = 2500
decay_rate = 0.99
decay_steps = 2500
decay_rate = 0.99
rng_branch = random.PRNGKey(1234)
rng_trunk = random.PRNGKey(4321)
gamma=0.999
eta = 0.01
max_RBA0 = 10.0
phi = 0.5
c_log = 1.0
vRBA_hyperparams=phi, c_log, gamma, eta, max_RBA0
ngsteps_stage1 = ngsteps // 4


# ## Cell 5: . Architecture QR-DeepONet
class SVD_DeepONetModel(nn.Module):
    branch_layers: Sequence[int]
    trunk_layers: Sequence[int]
    N_fx: int
    normalization_params: dict
    degree_branch: int
    degree_trunk_outer: int
    degree_trunk_inner: int
    output_dim: int
    activation: Callable = nn.tanh
    def setup(self):
        self.branch_net = eMLP(layers=self.branch_layers,
                               activation=self.activation,
                               degree=self.degree_branch)
        self.trunk_net = KKAN(features=self.trunk_layers,
                               degree=self.degree_trunk_outer,
                               degree_T=self.degree_trunk_inner,
                               output_dim=self.output_dim,
                               activation=self.activation,
                               basis_type='rbf')
        self.t_min, self.t_max = self.normalization_params['t_min'], self.normalization_params['t_max']
        self.x_min, self.x_max = self.normalization_params['x_min'], self.normalization_params['x_max']
        self.u_min, self.u_max = self.normalization_params['u_min'], self.normalization_params['u_max']
        embed_dim = self.trunk_layers[-1]
    def apply_stage1(self, y):
        t_norm = 2 * (y[:, 0:1] - self.t_min) / (self.t_max - self.t_min) - 1
        x_norm = 2 * (y[:, 1:2] - self.x_min) / (self.x_max - self.x_min) - 1
        y_input_to_trunk = jnp.concatenate([t_norm, x_norm], axis=-1)
        Phi = self.trunk_net(y_input_to_trunk)
        return Phi
    def apply_stage2(self, F):
        F_normalized = 2 * (F - self.u_min) / (self.u_max - self.u_min) - 1
        C = self.branch_net(F_normalized)
        return C
    @nn.compact
    def __call__(self, F, y):
        """
        Defines the forward pass for PREDICTION/INFERENCE.
        This uses the SVD-based prediction: U_pred = C @ Phi.T
        """
        F_normalized = 2 * (F - self.u_min) / (self.u_max - self.u_min) - 1
        C = self.branch_net(F_normalized)
        t_norm = 2 * (y[:, 0:1] - self.t_min) / (self.t_max - self.t_min) - 1
        x_norm = 2 * (y[:, 1:2] - self.x_min) / (self.x_max - self.x_min) - 1
        y_input_to_trunk = jnp.concatenate([t_norm, x_norm], axis=-1)
        Phi = self.trunk_net(y_input_to_trunk)
        U_pred = C @ Phi.T
        return U_pred


# ## Cell 7: QR-DeepONet Initialization & Partitioning
print("Initializing model...")
model = SVD_DeepONetModel(branch_layers=branch_layers,
                         trunk_layers=trunk_layers,
                         N_fx=n_samples,
                         normalization_params=normalization_params,
                         degree_branch=degree_branch,
                         degree_trunk_outer=degree_trunk_outer,
                         degree_trunk_inner=degree_trunk_inner,
                         output_dim=embed_dim)
key = jax.random.PRNGKey(1234)
sample_F = X_DN['F']
sample_y = X_DN['y']
params = model.init(key, sample_F, sample_y)
_, unravel_fn = ravel_pytree(params)
print("Initializing partitioned optimizers (dynamic keys)...")
optimizers = {}
states = {}
schedule_branch = optax.exponential_decay(lr_branch, decay_steps, decay_rate)
schedule_trunk = optax.exponential_decay(lr_trunk, decay_steps, decay_rate)
for key in params['params'].keys():
    if key == 'branch_net':
        print(f"Assigning branch optimizer to key: {key}")
        optimizers[key] = optax.adam(learning_rate=schedule_branch)
    elif key == 'trunk_net':
        print(f"Assigning trunk optimizer to key: {key}")
        optimizers[key] = optax.adam(learning_rate=schedule_trunk)
    else:
        print(f"Assigning default optimizer to key: {key}")
        optimizers[key] = optax.adam(learning_rate=lr)
for key in optimizers.keys():
    states[key] = optimizers[key].init(params['params'][key])
itercount = itertools.count()
apply_fn = jax.jit(model.apply)
apply_stage1_fn = partial(model.apply, method='apply_stage1')
apply_stage1_fn = jax.jit(apply_stage1_fn)
apply_stage2_fn = partial(model.apply, method='apply_stage2')
apply_stage2_fn = jax.jit(apply_stage2_fn)


# ## Cell 8: vRBA Initialization for QR-DeepONet
def update_spatial_weights(R, Lambda_batch, it, gamma, eta,
                           phi=1.0, c_log=1.0, potential_type='exponential'):
    """
    Batch-compatible vRBA spatial weight update function.
    """
    r_i = jnp.abs(R)
    if potential_type == 'exponential':
        epsilon_q = c_log * jnp.max(r_i, axis=1, keepdims=True) / jnp.log(it + 2.0)
        beta_it = 1.0 / (epsilon_q + 1e-20)
        q_it = beta_it * jnp.exp(beta_it * r_i)
    elif potential_type == 'cubic':
        q_it = r_i**2
    elif potential_type == 'quadratic':
        q_it = r_i
    else:
        q_it = jnp.ones_like(r_i)
    lambda_it = phi * q_it / (jnp.max(q_it, axis=1, keepdims=True) + 1e-20) + (1.0 - phi)
    new_Lambda = gamma * Lambda_batch + eta * lambda_it
    return new_Lambda
@partial(jit, static_argnums=(2, 3, 4))
def update_function_pdf(Lambda_global, it,
                        phi=1.0, c_log=1.0, potential_type='exponential'):
    """
    vRBA function sampling PDF, driven by potential_type.
    """
    lambda_scores = jnp.sum(Lambda_global, axis=1)
    if potential_type == 'exponential':
        epsilon_q = c_log * jnp.max(lambda_scores) / jnp.log(it + 2.0)
        beta_it = 1.0 / (epsilon_q + 1e-20)
        q_it = beta_it * jnp.exp(beta_it * lambda_scores)
    elif potential_type == 'quadratic':
        q_it = lambda_scores**2
    elif potential_type == 'linear':
        q_it = lambda_scores
    else:
        q_it = jnp.ones_like(lambda_scores)
    lambda_it = phi * q_it / (jnp.max(q_it) + 1e-20) + (1.0 - phi)
    q_pdf = lambda_it / jnp.sum(lambda_it)
    return q_pdf
def get_gamma(eta, max_RBA):
    gamma_it = 1.0 - eta / max_RBA
    return gamma_it
phi, c_log, gamma, eta, max_RBA0 = vRBA_hyperparams
gamma_it = get_gamma(eta, max_RBA0)
Lambda_max = eta / (1.0 - gamma_it + 1e-16)
Lambda_spatial = jnp.ones_like(X_DN['S']) * Lambda_max / 2.0
print(f"Initialized Lambda_spatial (Stage 1) with shape: {Lambda_spatial.shape}")
n_samples = X_DN['S'].shape[0]
Lambda_latent = jnp.ones((n_samples, embed_dim)) * Lambda_max / 2.0
print(f"Initialized Lambda_latent (Stage 2) with shape: {Lambda_latent.shape}")
cap_RBA = 40
step_RBA = (cap_RBA - max_RBA0) / (ngsteps / 10.0 - 1.0)
weighting_potential= 'exponential'
sampling_potential= 'quadratic'
print(f"Baking generic vRBA functions with method {weighting_potential} potential in space/latent and {sampling_potential} in function space")
weight_update_fn_baked = partial(
    update_spatial_weights,
    phi=phi,
    c_log=c_log,
    potential_type=weighting_potential
)
function_pdf_fn_baked_and_jitted = partial(
    update_function_pdf,
    phi=phi,
    c_log=c_log,
    potential_type=sampling_potential
)


# ## Cell 9: QR-DeepONet Loss Functions
def loss_stage1(params, apply_stage1_fn, y, V_svd_target, Lambda_spatial_all,
                it, current_gamma, eta, weight_update_fn_baked, W):
    """
    Computes the vRBA-weighted loss for Stage 1 (Trunk) using the FULL dataset.
    """
    Phi = apply_stage1_fn(params, y)
    R1 = Phi - V_svd_target  # Residual for Phi (P, embed_dim)
    R1_T = R1
    new_Lambda_spatial = weight_update_fn_baked(R1_T, Lambda_spatial_all, it,
                                                current_gamma, eta)
    loss = jnp.mean((new_Lambda_spatial * R1 * W)**2)
    return loss, new_Lambda_spatial
def loss_stage2(params, apply_stage2_fn, F_all, US_svd_target, Lambda_latent_all,
                it, current_gamma, eta, weight_update_fn_baked, W):
    """
    Computes the vRBA-weighted loss for Stage 2 (Branch) using the FULL dataset.
    """
    C_all = apply_stage2_fn(params, F_all)
    R2 = C_all - US_svd_target
    new_Lambda_latent = weight_update_fn_baked(R2, Lambda_latent_all, it,
                                               current_gamma, eta)
    loss = jnp.mean((new_Lambda_latent * R2 * W)**2)
    return loss, new_Lambda_latent


# ## Cell 10: QR-DeepONet Gradient and Update Functions
@partial(jit, static_argnums=(1, 2, 8, 9))
def compute_grads_stage1(params, loss_fn, apply_fn,
                         y, V_svd_target, Lambda_spatial_all,
                         it, current_gamma,
                         weight_update_fn_baked, W):
    def loss_wrapper(p):
        return loss_fn(p, apply_fn, y, V_svd_target, Lambda_spatial_all, it,
                       current_gamma, eta, weight_update_fn_baked, W)
    (loss_value, new_Lambda_spatial), g = jax.value_and_grad(loss_wrapper, has_aux=True)(params)
    return loss_value, new_Lambda_spatial, g
@partial(jit, static_argnums=(1, 2, 8, 9))
def compute_grads_stage2(params, loss_fn, apply_fn,
                         F_all, US_svd_target, Lambda_latent_all,
                         it, current_gamma,
                         weight_update_fn_baked, W):
    def loss_wrapper(p):
        return loss_fn(p, apply_fn, F_all, US_svd_target, Lambda_latent_all, it,
                       current_gamma, eta, weight_update_fn_baked, W)
    (loss_value, new_Lambda_latent), g = jax.value_and_grad(loss_wrapper, has_aux=True)(params)
    return loss_value, new_Lambda_latent, g
@partial(jax.jit, static_argnums=(0, 1))
def update_model(key_params, optimizer, gradient, params, state):
    g_subset = gradient['params'][key_params]
    p_subset = params['params'][key_params]
    updates, new_state = optimizer.update(g_subset, state, p_subset)
    new_p_subset = optax.apply_updates(p_subset, updates)
    params['params'][key_params] = new_p_subset
    return params, new_state


# ## Cell 11: QR-DeepONet Two-Stage Training Loop
def test_error(apply_fn, params, X_data):
    """
    Calculates the relative L2 error for train and test sets.
    """
    u_pred = apply_fn(params, X_data['F'], X_data['y'])
    error_train = np.linalg.norm(X_data['S'] - u_pred) / np.linalg.norm(X_data['S'])
    X_Test = X_data['X_Test']
    u_pred_test = apply_fn(params, X_Test['F'], X_Test['y'])
    error_test = np.linalg.norm(X_Test['S'] - u_pred_test) / np.linalg.norm(X_Test['S'])
    return error_test, error_train
y = X_DN['y']
U0 = X_DN['S']
F0 = X_DN['F']
# U0_T is no longer needed for SVD-DeepONet
W = X_DN['W']

# --- SVD Decomposition for targets ---
# U0 shape: (N, P)
# N = n_samples, P = y.shape[0]
u_svd, s_svd, vh_svd = jnp.linalg.svd(U0, full_matrices=False)

# Truncate SVD components to embed_dim
U_svd_truncated = u_svd[:, :embed_dim] # (N, embed_dim)
S_svd_truncated = jnp.diag(s_svd[:embed_dim]) # (embed_dim, embed_dim)
V_svd_T_truncated = vh_svd[:embed_dim, :] # (embed_dim, P)

# Define targets for Stage 1 (Trunk) and Stage 2 (Branch)
V_svd_target = V_svd_T_truncated.T # (P, embed_dim) - target for Phi
US_svd_target = U_svd_truncated @ S_svd_truncated # (N, embed_dim) - target for C

# --- Lambda_spatial initialization (corrected shape for SVD-DeepONet) ---
Lambda_spatial = jnp.ones((y.shape[0], embed_dim)) * Lambda_max / 2.0 # Shape (P, embed_dim)
print(f"Initialized Lambda_spatial (Stage 1) with shape: {Lambda_spatial.shape}")
n_samples = X_DN['S'].shape[0]
Lambda_latent = jnp.ones((n_samples, embed_dim)) * Lambda_max / 2.0
print(f"Initialized Lambda_latent (Stage 2) with shape: {Lambda_latent.shape}")

print(f"Total steps: {ngsteps}. Stage 1 (Trunk) steps: {ngsteps_stage1}. Stage 2 (Branch) steps: {ngsteps - ngsteps_stage1}.")
if Run_type=='Train':
    loss_log_T = []
    it_log_T = []
    train_L2 = []
    test_L2 = []
    print("Starting 2-stage training (full-batch, vRBA weighting)...")
    pbar = trange(ngsteps + 1)
    # Target_all is no longer needed, replaced by US_svd_target
    stage2_ready = False
    for it in pbar:
        max_RBA = max_RBA0 + step_RBA * (it // 10)
        current_gamma_it = get_gamma(eta, max_RBA)
        if it < ngsteps_stage1:
            loss_value, new_Lambda_spatial, gradient = compute_grads_stage1(
                params, loss_stage1, apply_stage1_fn,
                y, V_svd_target, Lambda_spatial, # Changed U0_T to V_svd_target
                it, current_gamma_it,
                weight_update_fn_baked, W
            )
            for key in ['trunk_net']: # Removed 'A'
                params, states[key] = update_model(
                    key, optimizers[key], gradient, params, states[key]
                )
            Lambda_spatial = new_Lambda_spatial
            log_loss_value = loss_value
        else:
            if not stage2_ready:
                print(f"\n--- Iteration {it}: Stage 1 finished. ---")
                print("SVD targets (U_svd @ S_svd) are pre-computed for Stage 2.")
                # No need to compute Target_all here, it's already US_svd_target
                print(f"  ... US_svd_target shape: {US_svd_target.shape}")
                print("  ... Resetting optimizer state for 'branch_net'.")
                states['branch_net'] = optimizers['branch_net'].init(params['params']['branch_net'])
                stage2_ready = True
                print("--- Starting Stage 2 (Branch training) ---")
            loss_value, new_Lambda_latent, gradient = compute_grads_stage2(
                params, loss_stage2, apply_stage2_fn,
                F0, US_svd_target, Lambda_latent, # Changed Target_all to US_svd_target
                it, current_gamma_it,
                weight_update_fn_baked, W
            )
            for key in ['branch_net']:
                params, states[key] = update_model(
                    key, optimizers[key], gradient, params, states[key]
                )
            Lambda_latent = new_Lambda_latent
            log_loss_value = loss_value
        if it % 500 == 0:
            error_test, error_train = test_error(apply_fn, params, X_DN)
            loss_log_T.append(log_loss_value)
            it_log_T.append(it)
            train_L2.append(error_train)
            test_L2.append(error_test)
            pbar.set_postfix({'it': it,
                              'Stage': 1 if it < ngsteps_stage1 else 2,
                              'Loss': log_loss_value,
                              'Train RL2': error_train,
                              'Test RL2': error_test})
    print("Training finished.")


# ## Cell 12: Save and Load Utilities
if Run_type=='Train':
    print("Saving parameters...")
    flat_params_save, _ = ravel_pytree(params)
    np.save('params.npy', flat_params_save)
    print("Model parameters saved to 'params.npy'")
    print("Saving log files...")
    final_error_train = train_L2[-1]
    final_error_test = test_L2[-1]
    results_dict = {
        'all_its': it_log_T,
        'all_loss': loss_log_T,
        'all_train_L2': train_L2,
        'all_test_L2': test_L2,
        'error_train': final_error_train,
        'error_test': final_error_test,
    }
    scipy.io.savemat('Log_files.mat', results_dict)
    print("Log files saved to 'Log_files.mat'")
else:
    print("Loading parameters...")
    flat_params_load = np.load('params.npy')
    params = unravel_fn(flat_params_load)
    for key in optimizers.keys():
        states[key] = optimizers[key].init(params['params'][key])
    print("Model parameters loaded and optimizer state is reset.")
    print("Loading log files...")
    log_data = scipy.io.loadmat('Log_files.mat')
    it_log_T = log_data['all_its'].flatten()
    loss_log_T = log_data['all_loss'].flatten()
    train_L2 = log_data['all_train_L2'].flatten()
    test_L2 = log_data['all_test_L2'].flatten()
    error_train = log_data['error_train'][0][0]
    error_test = log_data['error_test'][0][0]
    print("Log files loaded.")
    print(f"Final Loaded Train Error: {error_train:.4e}")
    print(f"Final Loaded Test Error: {error_test:.4e}")


# ## Cell 13: Final Summary Plot and Worst-Case Analysis
print('Final average Relative L2 error for the functions on the training dataset',train_L2[-1])
print('Final average Relative L2 error for the functions on the testing dataset',test_L2[-1])
print("Generating advanced 3x3 summary plot...")
u_pred = apply_fn(params, X_Test['F'], X_Test['y'])
norm_diff = jnp.linalg.norm(X_Test['S'] - u_pred, axis=1)
norm_truth = jnp.linalg.norm(X_Test['S'], axis=1)
per_sample_error = norm_diff / norm_truth
worst_indices_sorted = jnp.argsort(per_sample_error)
worst_indices = worst_indices_sorted[-2:]
worst_indices = worst_indices[::-1]
print(f"Worst test sample index: {worst_indices[0]} (Error: {per_sample_error[worst_indices[0]]:.4e})")
print(f"Second worst test sample index: {worst_indices[1]} (Error: {per_sample_error[worst_indices[1]]:.4e})")
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))
ax = axes[0, 0]
ax.plot(it_log_T, loss_log_T,label='Unweighthed loss')
ax.set_title('Total Loss (Unweighted MSE)')
ax.set_xlabel('Iterations')
ax.set_ylabel('Loss')
ax.set_yscale('log')
ax.axvline(x=ngsteps_stage1, color='r', linestyle='--', label='Stage 1 End')
ax.legend()
ax.grid(True, which="both", ls="--")
ax = axes[0, 1]
ax.plot(it_log_T, train_L2, label='Train RL2')
ax.set_title('Train Relative Error (Mean)')
ax.set_xlabel('Iterations')
ax.set_ylabel('Error')
ax.set_yscale('log')
ax.axvline(x=ngsteps_stage1, color='r', linestyle='--', label='Stage 1 End')
ax.legend()
ax.grid(True, which="both", ls="--")
ax = axes[0, 2]
ax.plot(it_log_T, test_L2, label='Test RL2', color='orange')
ax.set_title('Test Relative Error (Mean)')
ax.set_xlabel('Iterations')
ax.set_ylabel('Error')
ax.set_yscale('log')
ax.axvline(x=ngsteps_stage1, color='r', linestyle='--', label='Stage 1 End')
ax.legend()
ax.grid(True, which="both", ls="--")
x_plot = np.linspace(0, 1, len(x_all))
t_plot = np.linspace(0, 1, len(t_all))
XX, TT = np.meshgrid(x_plot, t_plot)
plot_rows = [
    {'row_idx': 1, 'sample_idx': worst_indices[0], 'title_prefix': 'Worst Case'},
    {'row_idx': 2, 'sample_idx': worst_indices[1], 'title_prefix': '2nd Worst Case'}
]
for item in plot_rows:
    row_idx = item['row_idx']
    sample_idx = item['sample_idx']
    u_test_sample = np.array(X_Test['S'][sample_idx])
    u_pred_sample = np.array(u_pred[sample_idx])
    S_pred = griddata(y_all, u_pred_sample.flatten(), (XX, TT), method='cubic')
    S_test = griddata(y_all, u_test_sample.flatten(), (XX, TT), method='cubic')
    pointwise_error = np.abs(S_test - S_pred)
    ax = axes[row_idx, 0]
    contour1 = ax.contourf(XX, TT, S_pred, levels=100, cmap='jet')
    fig.colorbar(contour1, ax=ax)
    ax.set_title(f' Prediction (Sample {sample_idx})')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax = axes[row_idx, 1]
    contour2 = ax.contourf(XX, TT, S_test, levels=100, cmap='jet')
    fig.colorbar(contour2, ax=ax)
    ax.set_title(f' Reference (Sample {sample_idx})')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax = axes[row_idx, 2]
    contour3 = ax.contourf(XX, TT, pointwise_error, levels=100, cmap='jet')
    fig.colorbar(contour3, ax=ax)
    ax.set_title(f'Error (Sample {sample_idx})')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
plt.tight_layout()
plt.savefig('summary_all.png', dpi=150)
plt.show()
