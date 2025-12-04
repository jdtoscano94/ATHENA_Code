# %% [markdown]
# ## General Overview to identify the problem type
# 
# The fundamental goal of this framework is to learn a mapping, or an operator, which we will denote as $ \mathcal{G} $ that takes an input function $ f_i $ and produces another function $ s_i $ as its output.
# 
# We can formally write this relationship as:
# 
# $$
# s_i = \mathcal{G}[f_i]
# $$
# 
# Here, the input function $ f_i $ belongs to an input function space $ \mathcal{F} $, and the output function $ s_i $ belongs to an output function space $ \mathcal{S} $.
# 
# The relationship between these two function spaces defines the type of problem we are solving. We can broadly categorize these problems into two scenarios.
# 
# ### Case 1: Different Function Spaces ($ \mathcal{F} \neq \mathcal{S} $)
# 
# This case typically describes problems where an external input $ f_i $ is mapped to a solution function $ s $. The input $ f_i $ is not part of the solution $ s $ itself.This category includes two main scenarios:
# 
# 1. Mapping Parameter-to-FunctionThis is the classic parametric PDE problem. The input $ f_i $ is not a spatially-varying field, but a vector of scalar parameters $ \lambda $ that define the problem.
# Examples: Mapping different scalar values for Reynolds number ( $ \mathbb{R}e = 100, 200, \dots $ ) or viscosity ( $ \nu = 0.1, 0.01, \dots $ ) to their corresponding, unique solution functions $ s(x,t) $.
# 
# 2. Mapping Function-to-Function: This is the more general operator problem. The input $ f_i $ is itself a function defined over a domain. Examples: Mapping different source terms $ f_i(x,t) $ in reaction-diffusion equations to the system's response $ s_i(x,t) $. Mapping different forcing terms or pressure gradients $ f_i(x) $ to the resulting velocity field $ s_i(x) $ in the Navier-Stokes equations.
# 
# ### Case 2: Identical Function Spaces ($ \mathcal{F} = \mathcal{S} $)
# 
# This case typically describes transformation or evolution operators, where the input and output functions live in the same space and share the same domain.Examples include:Fundamental mathematical operators like differentiation or integration. Time-propagation operators. This maps the solution state at time $ t $, $ f_i(t,x) = s_i(t,x) $, to a future state $ s_i(t+\delta t, x) $.
# 

# %% [markdown]
# # Cell 1: Core Imports and Project Setup
# 

# %%
# --- JAX, Flax, and Core Engine ---
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

# --- Optimizer ---
import optax  

# --- Custom "Crunch" Library ---
from Crunch.Models.layers import *
from Crunch.Models.polynomials import *
from Crunch.Auxiliary.metrics import *

# --- Data Handling and Utilities ---
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

# %%
from jax import config
config.update("jax_default_matmul_precision", "float32")
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

# %% [markdown]
# ### Cell 2: Load Data and Reshaping or Preprocessing
# 
# A key challenge is that $ \mathcal{F} $ and $ \mathcal{S} $ could be infinite-dimensional function spaces. We cannot feed a continuous function $ f $, which is an infinite-dimensional object, into a neural network for training. To make this problem computationally tractable, we must create a finite-dimensional discretization. This involves a **double discretization**:
# - We first sample a finite set of $ N $ input functions $ \{f_i\}_{i=1}^N $ from the space $ \mathcal{F} $. This gives us our training dataset.
# - We then represent each of those functions by its values at a finite number of "sensor" locations.
# 
# ### Discretizing the Input
# 
# We represent the set of $ N $ input functions $ \{f_i\}_{i=1}^N \subset \mathcal{F} $ as a set of $ N $ finite-dimensional objects, $ \{\mathbf{f}_i\}_{i=1}^N $.
# 
# To define this object $ \mathbf{f}_i $, we must define two sets of dimensions:
# 
# - The input domain $ \Omega_X $ is $ d $-dimensional (e.g., $ x \in \mathbb{R}^d $).
# - The input function $ f_i $ itself is $ k $-dimensional (e.g., $ f_i: \mathbb{R}^d \to \mathbb{R}^k $).
# 
# We represent each function $ f_i $ using $ m $ fixed sensor locations $ \{x_j\}_{j=1}^m $ within the domain $ \Omega_X $. The discretized function $ \mathbf{f}_i $ is the collection of $ k $-dimensional vectors from these $ m $ sensor locations. Therefore, $ \mathbf{f}_i $ is an object that lies in $ \mathbb{R}^{m \times k} $.
# 
# $$
# \mathbf{f}_i = (f_i(x_1), f_i(x_2), \dots, f_i(x_m))
# $$
# 
# #### Input Data Shape and Pre-processing
# 
# The full training dataset for the $ N $ input functions is a 3D array, or tensor. The expected shape is:
# 
# $$
# (N, m, k)
# $$
# 
# Where:
# - $ N $ is the number of functions in the dataset (the batch size).
# - $ m $ is the number of sensor locations used to discretize each function.
# - $ k $ is the dimension of the function's value (the codomain $ \mathbb{R}^k $) at each sensor location.
# 
# #### Note on Network Input Shape
# 
# The raw $ (N, m, k) $ tensor is almost never fed directly to a network. The strategy for handling it depends on the chosen encoder architecture and the problem's complexity.
# 
# - **Strategy 1: MLP (Flattening)**
# For simple problems where $ m \times k $ is reasonably small, the standard approach is to flatten the last two dimensions. The array is reshaped from $ (N, m, k) $ to $ (N, m \times k) $. This 2D array is then fed into a standard network such as an MLP.
# 
# - **Strategy 2: CNN Feature Extraction (for 2D Spatial Grids)**
# This is a more advanced strategy, used when two conditions are met:
# 1. The $ m $ sensors represent a 2D spatial grid (i.e., $ m = m_x \times m_y $).
# 2. The codomain dimension $ k $ is small, acting as "channels" (e.g., $ k=2 $ for a $ (u, v) $ velocity field).
# In this case, a pure-MLP approach (Strategy 1) would destroy the valuable 2D spatial information.
# The array is first reshaped from $ (N, m \times k) $ or $ (N, m, k) $ into a 4D tensor of shape $ (N, m_x, m_y, k) $.
# This 4D tensor is fed into a 2D CNN, which acts as a powerful spatial feature extractor.
# The output feature map from the CNN is then flattened to a 2D tensor of shape $ (N, \text{features}) $.
# This final 2D tensor is fed into the model for processing.
# 
# - **Strategy 3: Latent Space Processing (Dimensionality Reduction)**
# This strategy is required for many real-world problems (e.g., high-fidelity CFD) where both $ m $ and $ k $ can be very large.
# In this case, both Strategy 1 (flattening) and Strategy 2 (CNN) are computationally intractable or ineffective.
# The standard approach is to apply a dimensionality reduction technique before the neural network. A common method is Proper Orthogonal Decomposition (POD), or to train an Autoencoder.
# In this case, the $ (N, m, k) $ data is first reduced to its principal components or latent representation. The input to the network is not the raw data, but its projection onto the $ r $ most important modes.
# The input array for the network must be adjusted based on the method, e.g., its shape would become $ (N, r) $, where $ r \ll m \times k $.
# """

# %%
available_data_path='/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Data/Operator_Network_Examples/'
data=scipy.io.loadmat(available_data_path+'Burger_100_5000.mat')
print(data.keys()) ## Keep this for debugging process
f_all=data['input']
print(f_all.shape)

# %% [markdown]
# ### Discretizing the Output
# 
# ### 1. Defining the Output Query (The "Where")
# 
# Before we can define the output data array, we must first answer the question: **"Where do we want our solution?"** The set of $ P $ **output sensor locations** $\{y_j\}_{j=1}^P$ dictates the structure of our target data.
# 
# The **Time Propagation** problem is the perfect illustration, as it highlights the two most common scenarios for the output data.
# 
# ### 2. Output Data Structures: Two Scenarios
# 
# Let's assume our input is the $ i $-th initial condition $\mathbf{f}_i$ (from a set of $N$).
# 
# #### Scenario A: Predicting the Final State ($\mathcal{F} \approx \mathcal{S}$)
# 
# This is a **state-to-state** mapping.
# * **Operator:** $\mathcal{G}$ maps the initial state $s_i(x, 0)$ to the *final* state $s_i(x, t_{final})$.
# * **Output Function:** The output $s_i$ is a 1D function of space.
# * **Sensors:** The sensors are $p_x$ points in the *spatial domain* $\{x_j\}_{j=1}^{p_x}$.
# * **Data Shape:** If the output solution is a $ q $-dimensional vector field (e.g., $q=2$ for velocity $(u, v)$), the data for $N$ functions is a 3D tensor of shape:
# 
# $$
# 
# (N, p_x, q)
# 
# $$
# 
# (If the output is scalar, $ q=1 $, and the shape is $(N, p_x)$.)
# 
# #### Scenario B: Predicting the Full Trajectory ($\mathcal{F} \neq \mathcal{S}$)
# 
# This is a **function-to-function** mapping.
# * **Operator:** $\mathcal{G}$ maps the 1D initial state $s_i(x, 0)$ to the full 2D spatio-temporal solution $s_i(x, t)$.
# * **Output Function:** The output $s_i$ is a 2D function of space and time.
# * **Sensors:** The sensors form a grid of $p_x$ spatial points and $p_t$ time points.
# * **Data Shape:** For a $ q $-dimensional vector field, the full target dataset for $N$ functions is a 4D tensor:
# 
# $$
# 
# (N, p_t, p_x, q)
# 
# $$
# 
# ### 3. The Output Query Array (Trunk Input)
# 
# The set of $ P $ output sensor locations $\{y_j\}_{j=1}^P$ is not just an abstract concept; it is the **input data for the Trunk network**.
# 
# * **Domain Dimension:** The output function $s_i$ lives on an output domain $\Omega_Y \subset \mathbb{R}^{d_y}$. The dimension of this domain is $d_y$ (e.g., $d_y=1$ for space $x$, $d_y=2$ for space-time $(x, t)$).
# * **Array Shape:** Each sensor location $y_j$ is a vector in $\mathbb{R}^{d_y}$. The full set of $P$ sensor locations is therefore a 2D array (matrix) of shape:
# 
#     $$
#     (P, d_y)
#     $$
# 
# * **Flattening:** This is where your point about flattening becomes critical. In **Scenario B**, the $P$ sensors form a 2D grid of shape $(p_t, p_x)$. The actual *coordinates* for this grid would be a tensor of shape $(p_t, p_x, d_y)$ (where $d_y=2$, for $t$ and $x$). To feed this into the Trunk network as a batch, this grid is **flattened** into the required $(P, d_y)$ shape, where $P = p_t \times p_x$.
# 
# ### 4. Generalizing the Target Data Shape
# 
# The key insight is that both scenarios can be generalized. The total number of output sensor locations is $P$.
# * In Scenario A, $P = p_x$.
# * In Scenario B, $P = p_t \times p_x$.
# 
# For practical implementation (e.g., calculating the loss), the spatio-temporal sensor grid is "flattened" into a single sensor dimension.
# 
# Therefore, the 4D tensor from Scenario B, $ (N, p_t, p_x, q) $, is **reshaped into a 3D tensor**:
# 
# $$
# 
# (N, P, q) \quad \text{where} \quad P = p_t \times p_x
# 
# $$
# 
# This $(N, P, q)$ shape is the **general form of the output data** (the **Target Data**). It cleanly handles Scenario A (where $P = p_x$) and Scenario B (where $P = p_t \times p_x$).
# 
# * $N$ is the number of output functions (the batch size).
# * $P$ is the total number of **output sensor locations**.
# * $q$ is the **dimension of the output function's value** at each sensor.
# 

# %%
x_all=jnp.linspace(0, 1, f_all.shape[1])
t_all=jnp.linspace(0, 1, f_all.shape[1])
T, X = jnp.meshgrid(t_all, x_all)
y_all = jnp.hstack([T.flatten()[:,None], X.flatten()[:,None]])
print(y_all.shape)
s_all=data['output']
print(s_all.shape)
s_all = jnp.reshape(s_all, (s_all.shape[0], s_all.shape[1]*s_all.shape[2]))
print(s_all.shape)

# %% [markdown]
# # Cell 3: Train-Test Split, Type Conversion, and Data Structuring
# 
# This code block performs the standard data pre-processing required before training. It takes the full datasets (`f_all`, `s_all`, `y_all`) loaded and defined in the previous steps and converts them into the final, packaged dictionaries required by the training loop. This process has three distinct steps.
# 
# ### 1. Data Structuring
# 
# The final step organizes all the processed arrays into two dictionaries, `X_DN` and `X_Test`. These dictionaries are the canonical "data objects" that will be passed to the model and training functions.
# 
# 
# 
# The keys in these dictionaries map directly to the theoretical components we have defined:
# * **`'F'`**: The **Branch Input Data**. This holds `f_train` (in `X_DN`) or `f_test` (in `X_Test`). Its expected shape is $(N_{train}, m, k)$ or $(N_{train}, m \times k)$.
# * **`'y'`**: The **Trunk Input Data**. This holds `y_train` (the converted `y_all`). Its shape is $(P, d_y)$ and it is shared by both `X_DN` and `X_Test`.
# * **`'S'`**: The **Target Data**. This holds `s_train` (in `X_DN`) or `s_test` (in `X_Test`). Its expected shape is $(N_{train}, P, q)$.
# 
# The `X_Test` dictionary is then nested inside the `X_DN` dictionary for convenience, creating a single object that contains all necessary data for a complete train-and-validate experiment.
# 

# %%
# Calculate the sizes for each split
train_size = 0.8

# First, split off the training data
s_train, s_test, f_train, f_test = train_test_split(s_all, f_all, train_size=train_size, random_state=42)

# Reshape the data 
s_train = jnp.float32(s_train)
y_train = jnp.float32(y_all)
f_train = jnp.float32(f_train)
# Reshape the data 
s_test = jnp.float32(s_test)
f_test = jnp.float32(f_test)
y_train = jnp.float32(y_train)

# Print shapes to verify
print("s_train shape:", s_train.shape)
print("f_train shape:", f_train.shape)
print("s_test shape:", s_test.shape)
print("f_test shape:", f_test.shape)

# --- ADD THIS LINE ---
n_samples = f_train.shape[0] 
print(f"Number of training samples (n_samples): {n_samples}")

X_DN={'F':f_train,
      'y':y_train,
      'S':s_train,}
X_DN['F'].shape

X_Test={'F':f_test,
      'y':y_train,
      'S':s_test,}
X_DN['X_Test']=X_Test

# %% [markdown]
# # Cell 4: Hyperparameter Configuration
# 
# This block defines all the tunable parameters for the experiment. It is the central location for modifying the model's architecture, the optimization process, and the adaptive loss (vRBA) settings.
# 
# ### 1. vRBA Parameters (Adaptive Loss)
# 
# This is a critical section. These parameters are imported directly from the **variational Residual-Based Attention (vRBA)** framework (as seen in the PINNs manual). This strongly implies that the DeepONet's standard $L^2$ loss will be *enhanced* with vRBA's adaptive weighting to target high-error regions.
# 
# * **`gamma = 0.999` / `eta = 0.01`**: These are the **Exponential Moving Average (EMA)** parameters. As in the PINNs framework, they will be used to create a stable, smoothed update for the adaptive vRBA weights (the $\lambda$ multipliers).
# * **`phi = 0.5`**: This parameter controls the blend between the fully adaptive vRBA weights (at $\phi=1$) and a uniform, standard loss (at $\phi=0$). A value of $0.5$ creates a balanced mix.
# * **`c_log = 1.0`**: In the vRBA framework (when `potential_type='exponential'`), this parameter controls the "annealing temperature" of the weights, which is essential for stable $L^\infty$ (uniform) error minimization.
# * **`max_RBA0 = 10.0`**: The initial **cap** on the vRBA weights. This prevents the weights from exploding early in training, ensuring stability.
# * **`vRBA_hyperparams`**: This tuple simply packages all the vRBA settings to be passed to the training functions.
# 

# %%
# Run_Type
Run_type='Train'
# Model Architecture
N=840
m=101
num_layer=6
width_layer=100
embed_dim=m
# Initialize model
branch_layers = num_layer*[width_layer] +[embed_dim]
trunk_layers =  num_layer*[width_layer] +[embed_dim]
print(branch_layers)
print(trunk_layers)

# Optimization
ngsteps=100000 # Number of gradient steps.
batch_size=3000
lr=1e-3
decay_steps=2500
decay_rate=0.9
# Random Seeds
rng_branch = random.PRNGKey(1234)
rng_trunk = random.PRNGKey(4321)

# --- vRBA PARAMETERS ---
gamma=0.999
eta = 0.01
max_RBA0 = 10.0
phi = 0.5  
c_log = 1.0 
vRBA_hyperparams=phi, c_log, gamma, eta, max_RBA0

###--- QR--DeepONet
ngsteps_stage1 = ngsteps // 4 


# %% [markdown]
# ## Cell 5. Architecture QR-DeepONet
# 
# This cell defines the `QR_DeepONetModel`, an advanced architecture that could replace the standard `DeepONetModel` from the previous cell. The motivation for this change is to solve a critical stability problem.
# 
# ---
# ### The "Why": Instability of the Standard DeepONet
# 
# The standard DeepONet (from Cell 5) learns two maps:
# 
# **Branch ($ \mathbf{C} $):** $ \mathbf{f}_i \mapsto \mathbf{c}_i $
# 
# **Trunk ($ \mathbf{\Phi} $):** $ y_j \mapsto \mathbf{\phi}(y_j) $
# 
# The final prediction is the inner product $ \mathbf{U}_{\text{pred}} = \mathbf{C} \mathbf{\Phi}^T $.
# 
# **The Problem:** The `trunk_net` learns a set of basis functions $ \{\phi_j\}_{j=1}^H $. During training, there is nothing to prevent these functions from becoming highly correlated or **linearly dependent** (collinear). This results in an ill-conditioned, or "unstable," basis $ \mathbf{\Phi} $, which can make training slow, difficult, or fail entirely.
# 
# ---
# ### The "Main Idea": A Stable Orthogonal Basis
# 
# The solution is to **enforce an orthogonal basis**. Instead of using the (potentially unstable) learned basis $ \mathbf{\Phi} $, we can use a new, perfectly stable basis $ \mathbf{Q} $ that is derived from it.
# 
# The goal is to compute the prediction as:
# 
# $$
# s_i(y) \approx \sum_{j=1}^H c_j q_j(y)
# $$
# 
# Where $ \{q_j\} $ is an **orthogonal basis** for the output space.
# 
# This `QR_DeepONetModel` class implements this idea using **QR Decomposition**.
# 
# ---
# ### The "How": Code Analysis of `QR_DeepONetModel`
# 
# This class is a direct implementation of this orthogonal strategy. It is designed specifically for the **two-step training procedure** outlined in the QR-DeepONet paper (which we will see in the training cells).
# 
# ### Analysis of the Architecture
# 
# **`setup(self)`**
# This is key. It creates the standard `branch_net` and `trunk_net`, but *also* initializes a new trainable parameter `self.A`. This $ \mathbf{A} $ is the **auxiliary matrix** from the theory, which is only used during the **Stage 1** training step.
# 
# **`apply_stage1(self, y)`**
# This is not a prediction function. It is a **utility function for training**. It exposes the two components, $ \mathbf{\Phi} $ (from the `trunk_net`) and $ \mathbf{A} $, that are required to solve the first optimization problem: $ \min_{\mu, A} \|\boldsymbol{\Phi}(\mu) A - \boldsymbol{U}\| $.
# 
# **`apply_stage2(self, F)`**
# This is the second **training utility**. It exposes the $ \mathbf{C} $ matrix from the `branch_net` to solve the *decoupled* second optimization problem (e.g., $ \min_{\theta} \|\boldsymbol{C}(\theta) - R^* A^*\| $).
# 
# **`__call__(self, F, y)`**
# This is the **inference/prediction function**. It defines the model's output *after* training is complete. It implements the stable prediction by:
# 1.  Getting the learned coefficients $ \mathbf{C} $ from the `branch_net`.
# 2.  Getting the learned basis $ \mathbf{\Phi} $ from the `trunk_net`.
# 3.  **Orthogonalizing the basis on-the-fly** using `jnp.linalg.qr(Phi)` to get $ \mathbf{Q} $.
# 4.  Calculating the final, stable prediction as the inner product $ \mathbf{U}_{\text{pred}} = \mathbf{C} \mathbf{Q}^T $.
# 
# ---
# ### New Possibilities (SVD, etc.)
# 
# The QR decomposition is an excellent and computationally efficient choice for orthogonalization, but it's not the only one. This architecture opens the door to several other powerful methods:
# 
# * **Singular Value Decomposition (SVD):** This is the "gold standard" of matrix decomposition. By computing the SVD of the learned basis $ \mathbf{\Phi} = \mathbf{U} \mathbf{S} \mathbf{V}^T $, the columns of $ \mathbf{U} $ (the left singular vectors) would form a perfect orthogonal basis. This is more computationally expensive (`jnp.linalg.svd`) but provides more insight, as the singular values $ \mathbf{S} $ would represent the "importance" of each basis function. The prediction would then be $ \mathbf{U}_{\text{pred}} = (\mathbf{C} \mathbf{V} \mathbf{S}^{-1}) \mathbf{U}^T $.
# * **Gram-Schmidt Orthogonalization:** This is the classical "by-hand" method of building an orthogonal basis. It's generally less numerically stable than a well-implemented QR or SVD and thus not preferred, but it's a valid theoretical alternative.
# * **Learned Orthogonal Bases:** More advanced methods could involve designing the network (e.g., the `trunk_net`) to *directly* output an orthogonal basis $ \mathbf{Q} $, rather than learning $ \mathbf{\Phi} $ and correcting it. This would involve a custom architecture.
# 
# **Agent Action:** The `planner_agent` should be aware that `QR_DeepONetModel` represents a *family* of solutions. If the standard QR-based approach fails, the agent could retrieve and implement an `SVD_DeepONetModel` as a more robust (though more expensive) alternative.
# """

# %%
class QR_DeepONetModel(nn.Module):
    branch_layers: Sequence[int]
    trunk_layers: Sequence[int]
    N_fx: int  
    activation: Callable = nn.tanh

    def setup(self):
        self.branch_net = MLP(features=self.branch_layers,
                              activation=self.activation)
        self.trunk_net = MLP(features=self.trunk_layers,
                             activation=self.activation)
        
        embed_dim = self.trunk_layers[-1]
        A_init = jax.nn.initializers.variance_scaling(
            scale=1.0, mode='fan_in', distribution='truncated_normal'
        )
        self.A = self.param('A', A_init, (embed_dim, self.N_fx))

    def apply_stage1(self, y):
        Phi = self.trunk_net(y)
        return Phi, self.A

    def apply_stage2(self, F):
        C = self.branch_net(F)
        return C

    @nn.compact
    def __call__(self, F, y):
        """
        Defines the forward pass for PREDICTION/INFERENCE.
        This uses the QR-based prediction: U_pred = C @ Q.T
        """
        C = self.branch_net(F)
        Phi = self.trunk_net(y)
        
        # Use JAX's QR decomposition
        Q, _ = jnp.linalg.qr(Phi, mode='reduced')
        U_pred = C @ Q.T
        
        return U_pred

# %% [markdown]
# # Cell 6: Function Approximators (The "Swappable" Modules)
# 
# This cell defines the actual **function approximators** that are used as the building blocks for the operator model (i.e., the **Branch** and **Trunk** networks in Cell 5).
# 
# The proposed architecture is general; it simply requires two function approximators. This notebook uses the standard `MLP` as the default, but this module is designed to be **swappable**.
# 
# ### Default Approximator: `MLP`
# 
# This is the standard, baseline Multi-Layer Perceptron used in the model. It is a simple stack of `Dense` (linear) layers separated by an activation function.
# 
# ### Basic Improvements & Patching Rules
# 
# When modifying this `MLP` (or any approximator), the agent should be aware of these common rules:
# 
# * **Stabilizing Layers**: For deeper networks, convergence can be accelerated and gradients stabilized by replacing `nn.Dense` with layers like `WN_layer()` (Weight-Normalized Layer) or by using `AdaptiveResNet()` connections.
# * **Activation Function**: For Physics-Informed models, `nn.relu` (ReLU) is often unacceptable because its second derivative is zero or undefined. All activations should be replaced with smooth, fully-differentiable functions, such as `nn.tanh` or `nn.swish` (SiLU).
# 
# ### Advanced Alternatives (Retrieve-to-Implement)
# 
# This is a critical "patching" location. The `planner_agent` must be aware that the `branch_net` and `trunk_net` are swappable. If a standard `MLP` is insufficient, the agent must retrieve the implementation details for more powerful architectures from the knowledge base.
# 
# Advanced, pre-defined alternatives in the `Crunch` library include:
# 
# * **`eMLP` (Enhanced MLP)**: A hybrid model that uses polynomial embeddings to improve expressivity while retaining an MLP-like structure.
# * **`KAN` (Kolmogorov-Arnold Network)**: Replaces linear layers with learnable 1D activation functions, often parameterized by splines or (in our library) Chebyshev polynomials.
# * **`KKANs` (Kürková-Kolmogorov-Arnold Networks)**: A powerful two-block architecture that separates the "inner" and "outer" functions of the K-A theorem, based on the `KKANs_Journal_Version.pdf` paper.
# 
# **Agent Action**: To use one of these, the `planner_agent` must issue a `retrieve` command to get the full JAX/Flax class definition from the relevant documentation and use it to replace the `MLP`.

# %%
class MLP(nn.Module):
    features: Sequence[int]
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = self.activation(x)
        x = nn.Dense(self.features[-1])(x)
        return x

# %% [markdown]
# # Cell 7 QR-DeepONet Initialization & Partitioning
# 
# This cell is the direct counterpart to the standard "Initialize Model and Optimizer" cell, but it is heavily modified to support the `QR_DeepONetModel` and its unique two-stage training.
# 
# ---
# ### 1. Model Initialization
# 
# The first part is similar but with one key difference. When `QR_DeepONetModel` (from Cell 5 - Alternative) is instantiated, it now requires the `N_fx` argument.
# 
# * **`N_fx`**: This is the number of functions in the training set (e.g., `n_samples`). This is required so that the model can correctly initialize the shape of the trainable auxiliary matrix `self.A` (from Cell 5 - Alt), which has a dimension that depends on the number of functions.
# 
# When `model.init` is called, the resulting `params` PyTree is now different. It contains **three** top-level trainable components:
# $$
# \text{params} = \{ \text{'params'}: \{ \text{'branch\_net'}: \{...\} , \text{'trunk\_net'}: \{...\}, \text{'A'}: \{...\} \} \}
# $$
# 
# ---
# ### 2. The Partitioned Optimizer (Now with `A`)
# 
# This is the most critical change. The "partitioned optimizer" concept (from the standard Cell 7) is now extended to handle the new `A` matrix.
# 
# * The code iterates through the keys of `params['params']`.
# * It finds `'branch_net'`, `'trunk_net'`, and now also `'A'`.
# * It creates a **third optimizer** (`optax.adam`) with its own learning rate (`lr_A`) and schedule (`schedule_A`), and initializes its own state (`states['A']`).
# 
# This is the core of the QR-DeepONet's training logic. It allows us to train the `branch_net`, `trunk_net`, and `A` matrix with completely different learning rates, which is essential for the stability of the two-stage process.
# 
# ---
# ### 3. JIT-Compiled Apply Functions
# 
# Since the new model has three distinct "modes" (`__call__`, `apply_stage1`, `apply_stage2`), we create JIT-compiled handles for them.
# 
# * **`apply_fn`**: This is the standard JIT-compiled `__call__` method. This is the **inference/prediction** function.
# * **`apply_stage1_fn`**: This is the JIT-compiled handle for the `apply_stage1` method. The training loop will call this during **Stage 1** to get the outputs of the `trunk_net` and `A`.
# * **`apply_stage2_fn`**: This is the JIT-compiled handle for the `apply_stage2` method. The training loop will call this during **Stage 2** to get the output of the `branch_net`.
# 
# ---
# ### Summary for Agent
# 
# * This cell is the **mandatory** initialization script for the `QR_DeepONetModel`.
# * The `planner_agent` must know that the `params` tree is now `branch_net`, `trunk_net`, and `A`.
# * The agent must use the JIT-compiled `apply_stage1_fn` and `apply_stage2_fn` handles in the new (upcoming) two-stage training loop.

# %%
print("Initializing model...")
# --- 1. Initialize New Model ---
model = QR_DeepONetModel(branch_layers=branch_layers,
                           trunk_layers=trunk_layers,
                           N_fx=n_samples)

key = jax.random.PRNGKey(1234)
sample_F = X_DN['F']
sample_y = X_DN['y']


params = model.init(key, sample_F, sample_y) # This is a FrozenDict
_, unravel_fn = ravel_pytree(params)

# --- 2. Initialize Partitioned Optimizers (PINN-style) ---
print("Initializing partitioned optimizers (dynamic keys)...")

# Create a dictionary for optimizers
optimizers = {}
# Create a dictionary for states
states = {}

# Define schedules (can be customized per key)
lr_branch = lr * 1.0
lr_trunk = lr * 0.8
lr_A = lr * 1.2  # Learning rate for the A matrix

schedule_branch = optax.exponential_decay(lr_branch, decay_steps, decay_rate)
schedule_trunk = optax.exponential_decay(lr_trunk, decay_steps, decay_rate)
schedule_A = optax.exponential_decay(lr_A, decay_steps, decay_rate)

# Iterate over the keys in the 'params' sub-tree
# We now have 'branch_net', 'trunk_net', and 'A'
for key in params['params'].keys():
    if key == 'branch_net':
        print(f"Assigning branch optimizer to key: {key}")
        optimizers[key] = optax.adam(learning_rate=schedule_branch)
    elif key == 'trunk_net':
        print(f"Assigning trunk optimizer to key: {key}")
        optimizers[key] = optax.adam(learning_rate=schedule_trunk)
    elif key == 'A':
        print(f"Assigning A optimizer to key: {key}")
        optimizers[key] = optax.adam(learning_rate=schedule_A)
    else:
        # Fallback for any other parameters (e.g., 'batch_stats' if using BatchNorm)
        print(f"Assigning default optimizer to key: {key}")
        optimizers[key] = optax.adam(learning_rate=lr) 

# Populate the states dictionary
for key in optimizers.keys():
    states[key] = optimizers[key].init(params['params'][key])

itercount = itertools.count()

# --- 3. Create JIT-compiled Apply Functions ---
apply_fn = jax.jit(model.apply) 

# apply_fn for Stage 1 Loss
apply_stage1_fn = partial(model.apply, method='apply_stage1')
apply_stage1_fn = jax.jit(apply_stage1_fn)

apply_stage2_fn = partial(model.apply, method='apply_stage2')
apply_stage2_fn = jax.jit(apply_stage2_fn)

# %% [markdown]
# ### Cell 8: vRBA Initialization for QR-DeepONet
# 
# This cell defines the vRBA components. The core vRBA function (`update_spatial_weights`) is a generic, reusable tool for updating an Exponential Moving Average (EMA) of weights based on a residual.
# 
# The critical change for the `QR-DeepONetModel` is that its two-stage training process creates **two distinct residual losses**. Therefore, we must initialize **two separate vRBA weight matrices** to manage them.
# 
# ---
# ### 1. The Two Residuals
# 
# 1.  **Stage 1 Loss (Spatial Domain):**
#     * **Goal:** Train the `trunk_net` ($\Phi$) and `A` matrix.
#     * **Residual:** $R_{\text{spatial}} = \Phi A - U$.
#     * **Domain:** This residual lives in the **spatial domain** (at the output points $y$). Its shape is `(N_func, N_points)`.
# 
# 2.  **Stage 2 Loss (Latent Domain):**
#     * **Goal:** Train the `branch_net` ($C$).
#     * **Residual:** $R_{\text{latent}} = C - (R^* A^*)$ (where $R^* A^*$ is the "target" coefficient matrix, $R^*$ being the pseudo-inverse of $Q$).
#     * **Domain:** This residual lives in the **latent/coefficient space**. Its shape is `(N_func, embed_dim)`.
# 
# ---
# ### 2. The Two vRBA Weight Matrices (`Lambda`)
# 
# To manage these two residuals, we create two separate `Lambda` state matrices:
# 
# * **`Lambda_spatial`**:
#     * Initialized with shape `(N_func, N_points)` (i.e., `jnp.ones_like(X_DN['S'])`).
#     * This matrix will be used in **Stage 1** to apply adaptive weights to the *spatial residual* ($R_{\text{spatial}}$).
# 
# * **`Lambda_latent`**:
#     * Initialized with shape `(N_func, embed_dim)`.
#     * This matrix will be used in **Stage 2** to apply adaptive weights to the *latent residual* ($R_{\text{latent}}$).
# 
# ---
# ### 3. "Baked" Functions
# 
# * **`weight_update_fn_baked`**: This is the `partial` function for the generic `update_spatial_weights`. It will be called by *both* training stages (once on `Lambda_spatial` and once on `Lambda_latent`).
# * **`function_pdf_fn_baked_and_jitted`**: (Note) This function is *defined* in the code, but for this simplified two-stage training, we are **not** using the complex "Importance Sampling (in Function-Space)". We will use a simple, uniform random sampling (i.e., `jax.random.choice` without the `p=q_pdf` argument).
# 
# ---
# ### Summary for Agent
# 
# * This cell sets up the vRBA states for the two-stage training.
# * **`Lambda_spatial`** is the state for the Stage 1 (Trunk+A) training.
# * **`Lambda_latent`** is the state for the Stage 2 (Branch) training.
# * The `planner_agent` must ensure that the new, two-stage training loop correctly uses `Lambda_spatial` with the Stage 1 loss and `Lambda_latent` with the Stage 2 loss.
# * The agent should be aware that the `function_pdf_fn_baked_and_jitted` is **not used** in this training loop, as we are only applying vRBA weighting, not function-space sampling.

# %%
def update_spatial_weights(R, Lambda_batch, it, gamma, eta,
                           phi=1.0, c_log=1.0, potential_type='exponential'):
    """
    Batch-compatible vRBA spatial weight update function.
    """
    r_i = jnp.abs(R) # Shape: (batch_size, n_points)
    
    if potential_type == 'exponential':
        # Per-sample max for normalization
        epsilon_q = c_log * jnp.max(r_i, axis=1, keepdims=True) / jnp.log(it + 2.0)
        beta_it = 1.0 / (epsilon_q + 1e-20)
        q_it = beta_it * jnp.exp(beta_it * r_i) 
    elif potential_type == 'cubic':
        q_it = r_i**2 # quadratic weights 
    elif potential_type == 'quadratic':
        q_it = r_i # linear weights # Standard RBA
    else: # 'linear'
        q_it = jnp.ones_like(r_i) # No weights
    
    lambda_it = phi * q_it / (jnp.max(q_it, axis=1, keepdims=True) + 1e-20) + (1.0 - phi)
    new_Lambda = gamma * Lambda_batch + eta * lambda_it
    return new_Lambda

@partial(jit, static_argnums=(2, 3, 4)) # JIT this function
def update_function_pdf(Lambda_global, it,
                        phi=1.0, c_log=1.0, potential_type='exponential'):
    """
    vRBA function sampling PDF, driven by potential_type.
    """
    lambda_scores = jnp.sum(Lambda_global, axis=1) # Shape (n_func,)
    
    if potential_type == 'exponential':
        epsilon_q = c_log * jnp.max(lambda_scores) / jnp.log(it + 2.0)
        beta_it = 1.0 / (epsilon_q + 1e-20)
        # Using your new PINN logic: beta * exp(beta * r)
        q_it = beta_it * jnp.exp(beta_it * lambda_scores)
    elif potential_type == 'quadratic':
        q_it = lambda_scores**2
    elif potential_type == 'linear':
        q_it = lambda_scores
    else: # 'no_weights'
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


cap_RBA = 40 # You can set this here
step_RBA = (cap_RBA - max_RBA0) / (ngsteps / 10.0 - 1.0)

weighting_potential= 'exponential' # 'exponential', 'quadratic', 'linear', or 'no_weights'
sampling_potential= 'quadratic' # 'exponential', 'quadratic', 'linear', or 'no_weights'

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

# %%
phi, c_log, gamma, eta, max_RBA0 = vRBA_hyperparams
gamma_it = get_gamma(eta, max_RBA0)
Lambda_max = eta / (1.0 - gamma_it + 1e-16)

Lambda_spatial = jnp.ones_like(X_DN['S']) * Lambda_max / 2.0
print(f"Initialized Lambda_spatial (Stage 1) with shape: {Lambda_spatial.shape}")

n_samples = X_DN['S'].shape[0]
Lambda_latent = jnp.ones((n_samples, embed_dim)) * Lambda_max / 2.0
print(f"Initialized Lambda_latent (Stage 2) with shape: {Lambda_latent.shape}")


cap_RBA = 40 # You can set this here
step_RBA = (cap_RBA - max_RBA0) / (ngsteps / 10.0 - 1.0)

# %%
weighting_potential= 'exponential' # 'exponential', 'quadratic', 'linear', or 'no_weights'
sampling_potential= 'quadratic' # 'exponential', 'quadratic', 'linear', or 'no_weights'

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

# %% [markdown]
# # Cell 9: QR-DeepONet Loss Functions
# 
# This cell defines the two separate loss functions required for the `QR_DeepONetModel`'s two-stage training. Each function calculates one of the two residuals we identified in Cell 8 (Alt).
# 
# ---
# ### 1. `loss_stage1` (Trunk + A)
# 
# This function computes the vRBA-weighted loss for the **spatial residual**.
# 
# * **Call:** It uses `apply_stage1_fn` to get the learned basis $ \mathbf{\Phi} $ (`Phi`) and the auxiliary matrix $ \mathbf{A} $ (`A`).
# * **Residual:** It computes the spatial residual $ R1 = \Phi A - U^T $.
# * **vRBA:** It calls the generic `weight_update_fn_baked` using this residual $ R1 $ and the `Lambda_spatial_all` state.
# * **Return:** It returns the vRBA-weighted loss `jnp.mean((new_Lambda_spatial * R1_T)**2)` and the updated state `new_Lambda_spatial`.
# 
# ---
# ### 2. `loss_stage2` (Branch)
# 
# This function computes the vRBA-weighted loss for the **latent residual**.
# 
# * **Call:** It uses `apply_stage2_fn` to get the coefficients $ \mathbf{C} $ (`C_all`) from the branch network.
# * **Residual:** It computes the latent residual $ R2 = C - \text{Target} $ (where `Target_all` is the pre-computed $ R^* A^* $).
# * **vRBA:** It calls the *same* generic `weight_update_fn_baked` using *this* residual $ R2 $ and the `Lambda_latent_all` state.
# * **Return:** It returns the vRBA-weighted loss `jnp.mean((new_Lambda_latent * R2)**2)` and the updated state `new_Lambda_latent`.
# 
# ---
# ### Summary for Agent
# 
# * This cell provides the two distinct loss functions for the two-stage training.
# * `loss_stage1` is used to train the `'trunk_net'` and `'A'` parameters.
# * `loss_stage2` is used to train the `'branch_net'` parameters.
# * The `planner_agent` must ensure that the main training loop calls these two functions *separately* and updates the correct parameters for each stage.
# """

# %%
def loss_stage1(params, apply_stage1_fn, y, U_all_T, Lambda_spatial_all, 
                it, current_gamma, eta, weight_update_fn_baked):
    """
    Computes the vRBA-weighted loss for Stage 1 (Trunk + A) using the FULL dataset.
    """
    Phi, A = apply_stage1_fn(params, y)
    Pred = Phi @ A # Shape: (N_points, N_samples)
    R1 = Pred - U_all_T    
    R1_T = R1.T
    new_Lambda_spatial = weight_update_fn_baked(R1_T, Lambda_spatial_all, it,
                                                current_gamma, eta)
    
    loss = jnp.mean((new_Lambda_spatial * R1_T)**2)
    
    return loss, new_Lambda_spatial

def loss_stage2(params, apply_stage2_fn, F_all, Target_all, Lambda_latent_all,
                it, current_gamma, eta, weight_update_fn_baked):
    """
    Computes the vRBA-weighted loss for Stage 2 (Branch) using the FULL dataset.
    """
    C_all = apply_stage2_fn(params, F_all)
    R2 = C_all - Target_all
    new_Lambda_latent = weight_update_fn_baked(R2, Lambda_latent_all, it,
                                               current_gamma, eta)
    
    loss = jnp.mean((new_Lambda_latent * R2)**2)    
    return loss, new_Lambda_latent

# %% [markdown]
# ### Cell 10: QR-DeepONet Gradient and Update Functions
# 
# This cell contains the JIT-compiled "engine" functions for the two-stage training loop.
# 
# ---
# ### 1. Two-Stage Gradient Computation
# 
# Instead of a single `compute_grads_and_loss` function, we now have two, each one tailored to a specific training stage.
# 
# * **`compute_grads_stage1`**:
#     * This is a JIT-compiled wrapper around `loss_stage1`.
#     * It uses `jax.value_and_grad` to get the loss value, the updated `new_Lambda_spatial`, and the gradient PyTree `g`.
#     * Crucially, this gradient `g` will have non-zero values only for the `'trunk_net'` and `'A'` parameters, as the `'branch_net'` is "invisible" to `loss_stage1`.
# 
# * **`compute_grads_stage2`**:
#     * This is a JIT-compiled wrapper around `loss_stage2`.
#     * It uses `jax.value_and_grad` to get the loss value, the updated `new_Lambda_latent`, and the gradient PyTree `g`.
#     * This gradient `g` will have non-zero values only for the `'branch_net'` parameters.
# 
# ---
# ### 2. `update_model` (Reusable)
# 
# This function is **identical** to the one used in the standard DeepONet notebook. This is a key advantage of the partitioned optimizer design (Cell 7, Alt).
# 
# Because `update_model` is designed to update any specific sub-tree (`key_params`) of the `params` PyTree, it can be reused perfectly here.
# 
# * In **Stage 1**, we will call it twice:
#     * `update_model('trunk_net', ...)`
#     * `update_model('A', ...)`
# * In **Stage 2**, we will call it once:
#     * `update_model('branch_net', ...)`
# 
# ---
# ### Summary for Agent
# 
# * This cell provides the JIT-compiled gradient functions for the two-stage loop.
# * The `planner_agent` must ensure the new training loop calls `compute_grads_stage1` for the Stage 1 update and `compute_grads_stage2` for the Stage 2 update.
# * The `update_model` function is generic and will be used by both stages to update their respective parameters.
# """

# %%
# --- 1. Compute Grads for STAGE 1 (Trunk + A) ---
@partial(jit, static_argnums=(1, 2, 8)) # loss_fn, apply_fn, baked_fn are static
def compute_grads_stage1(params, loss_fn, apply_fn,
                         y, U_all_T, Lambda_spatial_all,
                         it, current_gamma,
                         weight_update_fn_baked):
    
    # This wrapper passes the arguments needed by loss_stage1
    def loss_wrapper(p):
        return loss_fn(p, apply_fn, y, U_all_T, Lambda_spatial_all, it,
                       current_gamma, eta, weight_update_fn_baked)
    
    (loss_value, new_Lambda_spatial), g = jax.value_and_grad(loss_wrapper, has_aux=True)(params)
    
    return loss_value, new_Lambda_spatial, g

@partial(jit, static_argnums=(1, 2, 8)) # loss_fn, apply_fn, baked_fn are static
def compute_grads_stage2(params, loss_fn, apply_fn,
                         F_all, Target_all, Lambda_latent_all,
                         it, current_gamma,
                         weight_update_fn_baked):
    
    def loss_wrapper(p):
        return loss_fn(p, apply_fn, F_all, Target_all, Lambda_latent_all, it,
                       current_gamma, eta, weight_update_fn_baked)
    
    (loss_value, new_Lambda_latent), g = jax.value_and_grad(loss_wrapper, has_aux=True)(params)
    
    return loss_value, new_Lambda_latent, g


@partial(jax.jit, static_argnums=(0, 1))  # key_params (string) and optimizer are static
def update_model(key_params, optimizer, gradient, params, state):
    
    # 1. Get the sub-trees
    g_subset = gradient['params'][key_params]
    p_subset = params['params'][key_params]
    
    # 2. Perform the update
    updates, new_state = optimizer.update(g_subset, state, p_subset)
    new_p_subset = optax.apply_updates(p_subset, updates)
    
    # 3. "Mutate" the params tree
    params['params'][key_params] = new_p_subset    
    return params, new_state

# %% [markdown]
# ### Cell 11: QR-DeepONet Two-Stage Training Loop
# 
# This is the main orchestrator for the `QR_DeepONetModel`. It replaces the simple, single-stage loop with a more complex, two-stage procedure. This loop is designed to run on the **full batch** of data at each step, which is required for the linear algebra (LSTSQ, QR) to be stable and correct.
# 
# The loop is split into two phases by the `ngsteps_stage1` hyperparameter.
# 
# ---
# ### 1. Stage 1: Train `trunk_net` and `A`
# 
# This stage runs for the first `ngsteps_stage1` iterations. Its *only* goal is to train the parameters of the `trunk_net` (to learn the basis $\Phi$) and the auxiliary matrix `A`.
# 
# * **Gradient Step:** It calls `compute_grads_stage1`, which uses `loss_stage1`.
# * **Parameter Update:** It performs the "partitioned" update, but critically, it **only updates the `'trunk_net'` and `'A'` keys**. The `'branch_net'` parameters are "frozen" during this stage.
# * **vRBA Update:** It updates the `Lambda_spatial` state matrix.
# 
# ---
# ### 2. The "One-Time" Transition
# 
# When the loop crosses the `ngsteps_stage1` threshold, it executes a "one-time" setup block (the `if not stage2_ready:` block). This is the **most critical part of the QR-DeepONet algorithm**.
# 
# It "decouples" the problem by using the now-trained `trunk_net` to compute the target for the `branch_net`.
# 
# 1.  **Get Trained Basis:** It calls `apply_stage1_fn` one last time to get the final, trained basis $ \mathbf{\Phi} $ (`trained_Phi`).
# 2.  **Solve for $A_s$:** It solves the linear system $ \mathbf{\Phi} A_s = U^T $ for $ A_s $ (`A_s`) using `jnp.linalg.lstsq`. This $ A_s $ represents the *optimal* coefficients for the *trained basis* $ \mathbf{\Phi} $.
# 3.  **Get Orthogonal Basis:** It computes the $ R $ matrix from the QR decomposition of $ \mathbf{\Phi} $ (`_, R = jnp.linalg.qr(...)`).
# 4.  **Compute Target:** It computes the final **latent target** for the branch network: `Target_all = (R @ A_s).T`. This new matrix `Target_all` is the "ground truth" that the branch network will now try to predict.
# 5.  **Reset Optimizer:** It resets the optimizer state for the `'branch_net'`. This is a crucial step to ensure the branch network trains from a clean slate, as it was frozen and is now facing a brand-new objective.
# 
# ---
# ### 3. Stage 2: Train `branch_net`
# 
# For the rest of the training, the loop executes Stage 2. Its *only* goal is to train the `branch_net` to predict the `Target_all` matrix we just computed.
# 
# * **Gradient Step:** It calls `compute_grads_stage2`, which uses `loss_stage2` and the `Target_all`.
# * **Parameter Update:** It **only updates the `'branch_net'` key**. The `'trunk_net'` and `'A'` parameters are now frozen.
# * **vRBA Update:** It updates the *second* vRBA state, `Lambda_latent`.
# 
# ---
# ### Summary for Agent
# 
# * This cell implements the full two-stage QR-DeepONet training algorithm.
# * **Stage 1:** Trains `trunk_net` + `A` using `loss_stage1`.
# * **Transition:** Solves for `A_s` and `R` to compute `Target_all`.
# * **Stage 2:** Trains `branch_net` using `loss_stage2` and `Target_all`.
# * The `planner_agent` must understand this logic flow. Any patches (e.g., to the `loss` or `grad` functions) must be compatible with this two-stage process.
# """

# %%
def test_error(apply_fn, params, X_data):
    """
    Calculates the relative L2 error for train and test sets.
    """
    # Predict on the training set
    u_pred = apply_fn(params, X_data['F'], X_data['y'])
    error_train = np.linalg.norm(X_data['S'] - u_pred) / np.linalg.norm(X_data['S'])
    
    # Predict on the test set
    X_Test = X_data['X_Test']
    u_pred_test = apply_fn(params, X_Test['F'], X_Test['y'])
    error_test = np.linalg.norm(X_Test['S'] - u_pred_test) / np.linalg.norm(X_Test['S'])
    
    return error_test, error_train

# %%
# --- 5. Prepare Data, Loggers, and "Baked" Functions ---
y = X_DN['y']
U0 = X_DN['S']
F0 = X_DN['F']
U0_T = U0.T  # Transposed U, needed for Stage 1 loss (Shape: N_points, N_samples)

# --- Define the split point for training stages ---
print(f"Total steps: {ngsteps}. Stage 1 (Trunk+A) steps: {ngsteps_stage1}. Stage 2 (Branch) steps: {ngsteps - ngsteps_stage1}.")


if Run_type=='Train':
    # New loggers for the two different losses
    loss_log_T = [] 
    it_log_T = []
    train_L2 = []
    test_L2 = []

    
    # --- 6. Start 2-Stage Training Loop ---
    print("Starting 2-stage training (full-batch, vRBA weighting)...")
    pbar = trange(ngsteps + 1)
    
    Target_all = None  # This will hold the target for Stage 2
    stage2_ready = False

    for it in pbar:
        # --- vRBA: Get dynamic params (used by both stages) ---
        max_RBA = max_RBA0 + step_RBA * (it // 10)
        current_gamma_it = get_gamma(eta, max_RBA)

        # --- Check which stage we're in ---
        if it < ngsteps_stage1:
            # --- STAGE 1: Train Trunk + A ---
            # 1. Get grads (JIT-compiled)
            loss_value, new_Lambda_spatial, gradient = compute_grads_stage1(
                params, loss_stage1, apply_stage1_fn,
                y, U0_T, Lambda_spatial,  # Full data
                it, current_gamma_it,
                weight_update_fn_baked
            )
            
            # 2. Update params (Python loop over keys)
            for key in ['trunk_net', 'A']:
                params, states[key] = update_model(
                    key, optimizers[key], gradient, params, states[key]
                )
            
            # 3. Update vRBA state for next loop
            Lambda_spatial = new_Lambda_spatial
            
            log_loss_value = loss_value # For the pbar

        else:
            # --- STAGE 2: Train Branch ---
            
            # --- One-time setup for Stage 2 ---
            if not stage2_ready:
                print(f"\n--- Iteration {it}: Stage 1 finished. ---")
                print("Computing Stage 2 target (Phi, QR, A_s)...")
                
                # Get the trained Phi from the model
                trained_Phi, _ = apply_stage1_fn(params, y)
                
                # 1. Compute optimal A_s (A_solved) via LSTSQ
                # U0_T = Phi @ A_s
                print(f"  ... Solving LSTSQ for A_s (Phi shape: {trained_Phi.shape}, U0_T shape: {U0_T.shape})")
                A_s = jnp.linalg.lstsq(trained_Phi, U0_T, rcond=None)[0]
                print(f"  ... A_s solved, shape: {A_s.shape}")

                # 2. Compute QR
                print("  ... Computing QR for Phi...")
                _, R = jnp.linalg.qr(trained_Phi, mode='reduced')
                print(f"  ... R computed, shape: {R.shape}")

                # 3. Compute Target_all = (R @ A_s).T
                # A_s shape: (embed_dim, N_samples)
                # R shape: (embed_dim, embed_dim)
                Target_all = (R @ A_s).T
                print(f"  ... Target_all computed, shape: {Target_all.shape}")

                # 4. Reset optimizer state for 'branch_net' (good practice)
                print("  ... Resetting optimizer state for 'branch_net'.")
                states['branch_net'] = optimizers['branch_net'].init(params['params']['branch_net'])
                
                stage2_ready = True
                print("--- Starting Stage 2 (Branch training) ---")
            
            # --- Standard Stage 2 Training Step ---
            
            # 1. Get grads (JIT-compiled)
            loss_value, new_Lambda_latent, gradient = compute_grads_stage2(
                params, loss_stage2, apply_stage2_fn,
                F0, Target_all, Lambda_latent, # Full data
                it, current_gamma_it,
                weight_update_fn_baked
            )

            # 2. Update params (Python loop over keys)
            # ONLY update 'branch_net'. 'trunk_net' and 'A' are frozen.
            for key in ['branch_net']:
                params, states[key] = update_model(
                    key, optimizers[key], gradient, params, states[key]
                )

            # 3. Update vRBA state for next loop
            Lambda_latent = new_Lambda_latent

            log_loss_value = loss_value # For the pbar

        # --- Combined Logging (runs every 500 steps) ---
        if it % 500 == 0:            
            # Log test error (using the final prediction function)
            error_test, error_train = test_error(apply_fn, params, X_DN)
            
            # Store losses
            loss_log_T.append(log_loss_value)
            it_log_T.append(it)
            # Store L2 errors
            train_L2.append(error_train)
            test_L2.append(error_test)

            # Print losses
            pbar.set_postfix({'it': it,
                              'Stage': 1 if it < ngsteps_stage1 else 2,
                              'Loss': log_loss_value,
                              'Train RL2': error_train,
                              'Test RL2': error_test})

    print("Training finished.")

# %% [markdown]
# ### Cell 12: Save and Load Utilities
# 
# This is a standard utility cell that handles the persistence of the model's
# results. It has two modes, controlled by the `Run_type` hyperparameter
# (from Cell 4).

# %%
# --- 7. Save or Load ---
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
    # --- LOAD ---
    print("Loading parameters...")
    flat_params_load = np.load('params.npy')
    # Use 'unravel_fn' to load the params
    params = unravel_fn(flat_params_load) 
    
    # Re-initialize the opt_state with the loaded params
    opt_state = optimizer.init(params) 
    
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

# %% [markdown]
# ### Cell 13: Final Summary Plot and Worst-Case Analysis
# 
# This is the final "epilogue" cell of the notebook. Its purpose is to
# generate a single, comprehensive diagnostic image: `summary_all.png`. (DONT CHANGE THE FILE NAME)
# 

# %%
print('Final average Relative L2 error for the functions on the training dataset',train_L2[-1])
print('Final average Relative L2 error for the functions on the testing dataset',test_L2[-1])

print("Generating advanced 3x3 summary plot...")

# --- 1. Find the 2 Worst Test Cases ---
u_pred = apply_fn(params, X_Test['F'], X_Test['y'])

# Calculate per-sample relative L2 error
norm_diff = jnp.linalg.norm(X_Test['S'] - u_pred, axis=1)
norm_truth = jnp.linalg.norm(X_Test['S'], axis=1)
per_sample_error = norm_diff / norm_truth

worst_indices_sorted = jnp.argsort(per_sample_error)
worst_indices = worst_indices_sorted[-2:]

worst_indices = worst_indices[::-1] 

print(f"Worst test sample index: {worst_indices[0]} (Error: {per_sample_error[worst_indices[0]]:.4e})")
print(f"Second worst test sample index: {worst_indices[1]} (Error: {per_sample_error[worst_indices[1]]:.4e})")

# --- 2. Create the 3x3 Plot ---

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))

# --- Row 1: Training History ---
# Plot 1: Loss vs. Iterations
ax = axes[0, 0]
ax.plot(it_log_T, loss_log_T,label='Unweighthed loss')
ax.set_title('Total Loss (Unweighted MSE)')
ax.set_xlabel('Iterations')
ax.set_ylabel('Loss')
ax.set_yscale('log')
ax.axvline(x=ngsteps_stage1, color='r', linestyle='--', label='Stage 1 End')
ax.legend()
ax.grid(True, which="both", ls="--")

# Plot 2: Train Relative Error vs. Iterations
ax = axes[0, 1]
ax.plot(it_log_T, train_L2, label='Train RL2')
ax.set_title('Train Relative Error (Mean)')
ax.set_xlabel('Iterations')
ax.set_ylabel('Error')
ax.set_yscale('log')
ax.axvline(x=ngsteps_stage1, color='r', linestyle='--', label='Stage 1 End')
ax.legend()
ax.grid(True, which="both", ls="--")

# Plot 3: Test Relative Error vs. Iterations
ax = axes[0, 2]
ax.plot(it_log_T, test_L2, label='Test RL2', color='orange')
ax.set_title('Test Relative Error (Mean)')
ax.set_xlabel('Iterations')
ax.set_ylabel('Error')
ax.set_yscale('log')
ax.axvline(x=ngsteps_stage1, color='r', linestyle='--', label='Stage 1 End')
ax.legend()
# ------------------
ax.grid(True, which="both", ls="--")

# --- Rows 2 & 3: Worst Case Visualizations ---
# Helper meshgrid for plotting
x = np.linspace(0, 1, m) # Using m=101 from your data setup
t = np.linspace(0, 1, m)
XX, TT = np.meshgrid(x, t)

plot_rows = [
    {'row_idx': 1, 'sample_idx': worst_indices[0], 'title_prefix': 'Worst Case'},
    {'row_idx': 2, 'sample_idx': worst_indices[1], 'title_prefix': '2nd Worst Case'}
]

for item in plot_rows:
    row_idx = item['row_idx']
    sample_idx = item['sample_idx']
    
    # Retrieve data for this sample
    u_test_sample = np.array(X_Test['S'][sample_idx])
    u_pred_sample = np.array(u_pred[sample_idx])

    # Grid data for contour plotting
    S_pred = griddata(y_all, u_pred_sample.flatten(), (XX, TT), method='cubic')
    S_test = griddata(y_all, u_test_sample.flatten(), (XX, TT), method='cubic')
    pointwise_error = np.abs(S_test - S_pred)

    # Plot Predicted u
    ax = axes[row_idx, 0]
    contour1 = ax.contourf(XX, TT, S_pred, levels=100, cmap='jet')
    fig.colorbar(contour1, ax=ax)
    ax.set_title(f' Prediction (Sample {sample_idx})')
    ax.set_xlabel('x')
    ax.set_ylabel('t')

    # Plot Reference u
    ax = axes[row_idx, 1]
    contour2 = ax.contourf(XX, TT, S_test, levels=100, cmap='jet')
    fig.colorbar(contour2, ax=ax)
    ax.set_title(f' Reference (Sample {sample_idx})')
    ax.set_xlabel('x')
    ax.set_ylabel('t')

    # Plot Pointwise Error
    ax = axes[row_idx, 2]
    contour3 = ax.contourf(XX, TT, pointwise_error, levels=100, cmap='jet')
    fig.colorbar(contour3, ax=ax)
    ax.set_title(f'Error (Sample {sample_idx})')
    ax.set_xlabel('x')
    ax.set_ylabel('t')

# --- Finalize and Save ---
plt.tight_layout()
plt.savefig('summary_all.png', dpi=150)
plt.show()

# %%



