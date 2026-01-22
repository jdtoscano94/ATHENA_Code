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


# ## Cell 2: Load Data and Reshaping or Preprocessing
available_data_path='/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Data/Operator_Network_Examples/'
data = scipy.io.loadmat(available_data_path + 'Burger_1000_5000.mat')
print(data.keys())
f_all=data['input']
print(f_all.shape)
s_all=data['output']
print(s_all.shape)
x_all=jnp.linspace(0, 1, f_all.shape[1])
t_all=jnp.linspace(0, 1, s_all.shape[1])
T, X = jnp.meshgrid(t_all, x_all)
y_all = jnp.hstack([T.flatten()[:,None], X.flatten()[:,None]])
print(y_all.shape)
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
print("Calculated W:", W)
X_DN={'F':f_train,
      'y':y_train,
      'S':s_train,
      'W':W}
X_DN['F'].shape
X_Test={'F':f_test,
      'y':y_train,
      'S':s_test,}
X_DN['X_Test']=X_Test


# ## Cell 4: Hyperparameter Configuration
Run_type='Train'
N=840
m=101
num_layer=6
width_layer=100
embed_dim = 120
branch_layers = [100, 100, 100, 100, 100, embed_dim]
trunk_layers = [50, 50, 50, 50, embed_dim]
degree = 5
print(trunk_layers)
kmax = 5
Lx_fourier = 2.0
t_min, t_max = jnp.min(t_all), jnp.max(t_all)
x_min, x_max = jnp.min(x_all), jnp.max(x_all)
u_min, u_max = jnp.min(f_all), jnp.max(f_all)
ngsteps = 50000
batch_size=3000
lr=1e-3
lr_trunk = lr / 3.0
decay_steps=2500
decay_rate=0.9
rng_branch = random.PRNGKey(1234)
rng_trunk = random.PRNGKey(4321)
gamma=0.999
eta = 0.01
max_RBA0 = 10.0
phi = 0.5
c_log = 1.0
vRBA_hyperparams=phi, c_log, gamma, eta, max_RBA0


# ## Cell 5: Model Architecture Definition
class DeepONetModel(nn.Module):
    branch_layers: Sequence[int]
    trunk_layers: Sequence[int]
    t_min: float
    t_max: float
    x_min: float
    x_max: float
    u_min: float
    u_max: float
    degree: int
    activation: Callable = nn.tanh
    def setup(self):
        self.branch_net = MLP(features=self.branch_layers,
                              activation=self.activation)
        self.trunk_net = KAN(layers=self.trunk_layers,
                             degree=self.degree,
                             basis_type='wavelet')
    @nn.compact
    def __call__(self, F, y):
        F_normalized = 2.0 * (F - self.u_min) / (self.u_max - self.u_min + 1e-8) - 1.0
        B = self.branch_net(F_normalized)
        t_coords = y[:, 0:1]
        x_coords = y[:, 1:2]
        t_normalized = 2.0 * (t_coords - self.t_min) / (self.t_max - self.t_min + 1e-8) - 1.0
        x_normalized = 2.0 * (x_coords - self.x_min) / (self.x_max - self.x_min + 1e-8) - 1.0
        y_processed = jnp.concatenate([t_normalized, x_normalized], axis=-1)
        T = self.trunk_net(y_processed).T
        U_pred = B @ T
        return U_pred


# ## Cell 6: Function Approximators (The "Swappable" Modules)
class MLP(nn.Module):
    features: Sequence[int]
    activation: Callable = nn.tanh
    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = self.activation(x)
        x = nn.Dense(self.features[-1])(x)
        return x
class Wavelet_KAN_layer(nn.Module):
    out_dim: int
    degree: int = 5
    @nn.compact
    def __call__(self, X):
        batch_size, in_dim = X.shape
        scales = self.param(
            'scales',
            nn.initializers.constant(1.0),
            (in_dim, self.degree),
            jnp.float32
        )
        translations = self.param(
            'translations',
            nn.initializers.uniform(-1.0, 1.0),
            (in_dim, self.degree),
            jnp.float32
        )
        X_reshaped = X[:, :, None]
        scales_reshaped = scales[None, :, :]
        translations_reshaped = translations[None, :, :]
        arg = (X_reshaped - translations_reshaped) / (scales_reshaped + 1e-8)
        basis_functions = (1 - arg**2) * jnp.exp(-arg**2 / 2)
        C_n = self.param(
            'C_n',
            nn.initializers.normal(1 / (in_dim * self.degree)),
            (in_dim, self.degree, self.out_dim),
            jnp.float32
        )
        output = jnp.einsum('bid,ido->bo', basis_functions, C_n)
        return output
class Sin_KAN_layer(nn.Module):
    out_dim: int
    degree: int = 5
    grid_min: float = -0.1
    grid_max: float = 0.1
    normalization: callable = jnp.sin
    def setup(self):
        self.num_centers = self.degree
        self.centers = self.param(
            'centers',
            lambda rng, shape: jnp.linspace(self.grid_min, self.grid_max, self.num_centers),
            (self.num_centers,)
        )
        self.normalization_fn = self.normalization
    @nn.compact
    def __call__(self, X):
        batch_size, in_dim = X.shape
        X = X[:, None, :]
        p_i = self.centers[None, :, None]
        W_i = self.param(
            'W_i',
            nn.initializers.normal(1),
            (p_i.shape)
        )
        mu = jnp.exp(-(W_i**2) / 2) * self.normalization_fn(p_i)
        sigma = jnp.sqrt(0.5 - 0.5 * jnp.exp(-(2 * W_i**2) * jnp.cos(2 * p_i) - mu**2)) + 1e-12
        b_i = (self.normalization_fn(W_i * X + p_i) - mu) / sigma
        b_i = jnp.transpose(b_i, (0, 2, 1))
        C_n = self.param(
            'C_n',
            nn.initializers.normal(1 / (in_dim * self.num_centers)),
            (in_dim, self.num_centers, self.out_dim)
        )
        output = jnp.einsum('bin,ino->bo', b_i, C_n)
        return output
class KAN(nn.Module):
    layers: Sequence[int]
    degree: int = 5
    basis_type: str = 'chebyshev'
    def setup(self):
        if self.basis_type.lower() == 'rbf':
            self.basis_layer_class = RBF_KAN_layer
            self.basis_kwargs = {'degree': self.degree}
        elif self.basis_type.lower() == 'chebyshev_grid':
            self.basis_layer_class = Polynomial_grid_layer
            self.basis_kwargs = {'degree': self.degree}
        elif self.basis_type.lower() == 'sin':
            self.basis_layer_class = Sin_KAN_layer
            self.basis_kwargs = {'degree': self.degree}
        elif self.basis_type.lower() == 'wavelet':
            self.basis_layer_class = Wavelet_KAN_layer
            self.basis_kwargs = {'degree': self.degree}
        elif self.basis_type.lower() == 'chebyshev':
            self.basis_layer_class = Polynomial_KAN_layer
            self.basis_kwargs = {'degree': self.degree, 'polynomial_type': 'T'}
        elif self.basis_type.lower() == 'legendre':
            self.basis_layer_class = Polynomial_grid_KAN_layer
            self.basis_kwargs = {'degree': self.degree, 'polynomial_type': 'L'}
        else:
            raise ValueError(f"Unknown basis_type: {self.basis_type}. Use 'chebyshev', 'legendre', 'rbf', 'sin', or 'wavelet'.")
    @nn.compact
    def __call__(self, x):
        for feat in self.layers:
            x = self.basis_layer_class(out_dim=feat, **self.basis_kwargs)(x)
        return x


# ## Cell 7: Initialize Model and Optimizer
print("Initializing model...")
model = DeepONetModel(branch_layers=branch_layers,
                      trunk_layers=trunk_layers,
                      t_min=t_min, t_max=t_max,
                      x_min=x_min, x_max=x_max,
                      u_min=u_min, u_max=u_max,
                      degree=degree)
key = jax.random.PRNGKey(1234)
sample_F = X_DN['F'][0:10]
sample_y = X_DN['y'][0:10]
params = model.init(key, sample_F, sample_y)
_, unravel_fn = ravel_pytree(params)
print("Initializing partitioned optimizers (dynamic keys)...")
optimizers = {}
states = {}
lr_branch = lr
lr_trunk = lr_trunk
schedule_branch = optax.exponential_decay(lr_branch, decay_steps, decay_rate)
schedule_trunk = optax.exponential_decay(lr_trunk, decay_steps, decay_rate)
for key in params['params'].keys():
    if key == 'branch_net':
        print(f"Assigning branch optimizer to key: {key}")
        optimizers[key] = optax.adam(learning_rate=schedule_branch)
    else:
        print(f"Assigning trunk optimizer to key: {key}")
        optimizers[key] = optax.adam(learning_rate=schedule_trunk)
for key in optimizers.keys():
    states[key] = optimizers[key].init(params['params'][key])
itercount = itertools.count()
apply_fn = jax.jit(model.apply)


# ## Cell 8: vRBA for Neural Operators (The Mixed Strategy)
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
Lambda_global = jnp.ones_like(X_DN['S']) * Lambda_max / 2.0
cap_RBA = 40
step_RBA = (cap_RBA - max_RBA0) / (ngsteps / 10.0 - 1.0)
weighting_potential= 'exponential'
sampling_potential= 'cubic'
print(f"Baking vRBA functions with metho {weighting_potential} potential in space and {sampling_potential} in function space")
spatial_update_fn_baked = partial(
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


# ## Cell 10: The vRBA-Weighted Loss Function
def loss_all(params, apply_fn, F, y, U, Lambda_batch, it, current_gamma, eta,
             spatial_update_fn_baked, W):
    outputs = apply_fn(params, F, y)
    R = outputs - U.T
    R_scaled = W * R
    new_Lambda_batch = spatial_update_fn_baked(R_scaled, Lambda_batch, it,
                                              current_gamma, eta)
    loss = jnp.mean((new_Lambda_batch * R_scaled)**2)
    return loss, new_Lambda_batch


# ## Cell 11: Gradient and Update Functions
@partial(jit, static_argnums=(1, 2, 9))
def compute_grads_and_loss(params, loss_fn, apply_fn,
                           F, y, U, Lambda_batch, it, current_gamma,
                           spatial_update_fn_baked, W):
    def loss_wrapper(p):
        return loss_fn(p, apply_fn, F, y, U, Lambda_batch, it,
                       current_gamma, eta, spatial_update_fn_baked, W)
    (loss_value, new_Lambda_batch), g = jax.value_and_grad(loss_wrapper, has_aux=True)(params)
    return loss_value, new_Lambda_batch, g
@partial(jax.jit, static_argnums=(0, 1))
def update_model(key_params, optimizer, gradient, params, state):
    g_subset = gradient['params'][key_params]
    p_subset = params['params'][key_params]
    updates, new_state = optimizer.update(g_subset, state, p_subset)
    new_p_subset = optax.apply_updates(p_subset, updates)
    params['params'][key_params] = new_p_subset
    return params, new_state


# ## Cell 12: The Main Training Loop
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
W = X_DN['W']
if Run_type=='Train':
    loss_log_T = []
    it_log_T = []
    train_L2 = []
    test_L2 = []
    print("Starting training (PINN-style loop with multiple optimizers)...")
    pbar = trange(ngsteps + 1)
    for it in pbar:
        max_RBA = max_RBA0 + step_RBA * (it // 10)
        current_gamma_it = get_gamma(eta, max_RBA)
        key = jax.random.PRNGKey(it)
        key, subkey = jax.random.split(key)
        q_pdf = function_pdf_fn_baked_and_jitted(Lambda_global, it)
        idx_train = jax.random.choice(
            subkey,
            a=jnp.arange(len(U0)),
            shape=(batch_size,),
            p=q_pdf
        )
        U_batch = U0[idx_train]
        F_batch = F0[idx_train]
        Lambda_batch = Lambda_global[idx_train]
        U_batch_T = U_batch.T
        loss_value, new_Lambda_batch, gradient = compute_grads_and_loss(
            params, loss_all, apply_fn,
            F_batch, y, U_batch_T, Lambda_batch,
            it, current_gamma_it,
            spatial_update_fn_baked, W
        )
        for key in optimizers.keys():
            params, states[key] = update_model(
                key, optimizers[key], gradient, params, states[key]
            )
        Lambda_global = Lambda_global.at[idx_train].set(new_Lambda_batch)
        if it % 500 == 0:
            U_pred_all = apply_fn(params, F0, y)
            R_all = U_pred_all - U0
            log_loss_value = jnp.mean(R_all**2)
            error_test, error_train = test_error(apply_fn, params, X_DN)
            loss_log_T.append(log_loss_value)
            it_log_T.append(it)
            train_L2.append(error_train)
            test_L2.append(error_test)
            pbar.set_postfix({'it': it,
                            'Loss': log_loss_value,
                            'Train RL2': error_train,
                            'Test RL2': error_test})
    print("Training finished.")


# ## Cell 13: Save and Load Utilities
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


# ## Cell 14: Final Summary Plot and Worst-Case Analysis
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
ax.plot(it_log_T, loss_log_T)
ax.set_title('Total Loss (Unweighted MSE)')
ax.set_xlabel('Iterations')
ax.set_ylabel('Loss')
ax.set_yscale('log')
ax.grid(True, which="both", ls="--")
ax = axes[0, 1]
ax.plot(it_log_T, train_L2, label='Train RL2')
ax.set_title('Train Relative Error (Mean)')
ax.set_xlabel('Iterations')
ax.set_ylabel('Error')
ax.set_yscale('log')
ax.grid(True, which="both", ls="--")
ax = axes[0, 2]
ax.plot(it_log_T, test_L2, label='Test RL2', color='orange')
ax.set_title('Test Relative Error (Mean)')
ax.set_xlabel('Iterations')
ax.set_ylabel('Error')
ax.set_yscale('log')
ax.grid(True, which="both", ls="--")
x = np.linspace(0, 1, m)
t = np.linspace(0, 1, m)
XX, TT = np.meshgrid(x, t)
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
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax = axes[row_idx, 1]
    contour2 = ax.contourf(XX, TT, S_test, levels=100, cmap='jet')
    fig.colorbar(contour2, ax=ax)
    ax.set_title(f' Reference (Sample {sample_idx})')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax = axes[row_idx, 2]
    contour3 = ax.contourf(XX, TT, pointwise_error, levels=100, cmap='jet')
    fig.colorbar(contour3, ax=ax)
    ax.set_title(f'Error (Sample {sample_idx})')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
plt.tight_layout()
plt.savefig('summary_all.png', dpi=150)
plt.show()
