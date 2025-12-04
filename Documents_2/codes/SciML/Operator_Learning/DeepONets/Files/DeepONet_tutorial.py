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
# This block imports all necessary libraries, establishing the core technical stack for this DeepONet framework. The imports can be grouped into four categories.

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


# %% [markdown]
# # Cell 5: Model Architecture Definition
# 
# This cell defines the neural network architecture. The code provided is for a **DeepONet**, but it is best understood as one specific implementation of a more **general operator learning principle**.
# 
# ---
# ### 1. The General Theory of Operator Learning
# 
# As you pointed out, the general structure for most operator learning models (including DeepONet, FNO, and spectral methods) is to approximate the output function $s_i$ as a linear combination of basis functions:
# 
# $$
# 
# s_i(y) \approx \sum_{j=1}^H c_j \phi_j(y)
# 
# $$
# 
# Here, $s_i$ is the $i$-th output function, $y$ is a coordinate in the output domain $\Omega_Y$, and the model must learn two components:
# 
# 1.  **The Coefficients $\{c_j\}_{j=1}^H$:** These $H$ coefficients are unique to the input function $f_i$. They are *not* static. The model must learn a **map (or "Encoder")** that reads the entire discretized input function $\mathbf{f}_i \in \mathbb{R}^{m \times k}$ and "encodes" it into this $H$-dimensional coefficient vector $\mathbf{c}_i \in \mathbb{R}^H$.
# 
# 2.  **The Basis $\{\phi_j\}_{j=1}^H$:** These are $H$ basis functions that are defined over the output domain $\Omega_Y$. The model must learn these basis functions. This can be seen as a **map (or "Decoder")** that takes a single query coordinate $y \in \mathbb{R}^{d_y}$ and evaluates all $H$ basis functions at that point, producing a vector $\mathbf{t}(y) \in \mathbb{R}^H$.
# 
# The final prediction at any point $y$ is the inner product of these two learned maps.
# ### Case 1: The DeepONet Model Architecture
# 
# This class defines the "unstacked" DeepONet architecture, which is the core of our operator learning framework. It combines two sub-networks: a **Branch** network and a **Trunk** network.
# 
# ### 1. The `setup` Method
# 
# This method is called by Flax during initialization. Its job is to create the two sub-networks as class attributes:
# * **`self.branch_net`**: An instance of an `MLP` that will process the input functions $f_i$.
# * **`self.trunk_net`**: An instance of an `MLP` that will process the output coordinates $y_j$.
# 
# ### 2. The `__call__` Method (The Forward Pass)
# 
# This method defines the computation of the forward pass, which mathematically represents the operator $\mathcal{G}[f_i](y_j)$:
# 
# $$
# \mathcal{G}[f_i](y_j) \approx \sum_{k=1}^{p} B_k(f_i) \cdot T_k(y_j)
# $$
# 
# This is implemented in three steps:
# 1.  **`B = self.branch_net(F)`**: The entire batch of input functions $F$ (shape `(N, m*k)`) is passed through the `branch_net`. The output `B` (shape `(N, p)`) represents the "coefficients" for the basis functions.
# 2.  **`T = self.trunk_net(y).T`**: The batch of query coordinates $y$ (shape `(P, d_y)`) is passed through the `trunk_net`. The output `T` (shape `(P, p)`) represents the shared "basis functions." This is then transposed to `(p, P)`.
# 3.  **`U_pred = B @ T`**: A matrix multiplication (dot product) is performed between the coefficients `B` and the basis `T`. The resulting prediction `U_pred` has the shape `(N, P)`, which perfectly matches our target data $S$.
# 
# This standard architecture is a powerful and flexible baseline. Its performance can be significantly improved with more advanced structures, such as the **QR-DeepONet**, which uses a **QR decomposition** to ensure the Trunk network's basis functions remain orthogonal and linearly independent.
# 
# 

# %%
class DeepONetModel(nn.Module):
    branch_layers: Sequence[int] 
    trunk_layers: Sequence[int]  
    activation: Callable = nn.tanh
    def setup(self):
        self.branch_net = MLP(features=self.branch_layers, 
                                activation=self.activation)
        self.trunk_net = MLP(features=self.trunk_layers, 
                               activation=self.activation)
    @nn.compact
    def __call__(self, F, y):
        B = self.branch_net(F)
        T = self.trunk_net(y).T        
        U_pred = B @ T
        return U_pred

# %% [markdown]
# # Cell 6: Function Approximators (The "Swappable" Modules)
# 
# This cell defines the actual **function approximators** that are used as the building blocks for the operator model (i.e., the **Branch** and **Trunk** networks in Cell 5).
# 
# The `DeepONetModel` architecture is general; it simply requires two function approximators. This notebook uses the standard `MLP` as the default, but this module is designed to be **swappable**.
# 
# ### Default Approximator: `MLP`
# 
# This is the standard, baseline Multi-Layer Perceptron used in the `DeepONetModel`. It is a simple stack of `Dense` (linear) layers separated by an activation function.
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
# 
# ### Cell 7: Initialize Model and Optimizer
# 
# This cell performs two critical tasks:
# 1.  **Initialization:** It creates an instance of the `DeepONetModel` (from Cell 5) and initializes its parameters.
# 2.  **Optimizer Partitioning:** It sets up the `optax` optimizer in a flexible, "partitioned" structure, which is the key to training complex hybrid models.
# ---
# ### 1. Model Initialization
# 
# This section is standard. It uses the `branch_layers` and `trunk_layers` hyperparameters (from Cell 4) to instantiate the `DeepONetModel` (from Cell 5).
# 
# A key JAX/Flax step is `params = model.init(...)`. This runs one forward pass with "dummy" data (`sample_F`, `sample_y`) to determine the shapes of all weights and biases. The `params` object is a **PyTree** (a nested dictionary) with a structure that mirrors the model:
# 
# $$
# \text{params} = \{ \text{'params'}: \{ \text{'branch\_net'}: \{...\} , \text{'trunk\_net'}: \{...\} \} \}
# $$
# 
# ### 2. The Partitioned Optimizer (The "Why")
# 
# This section directly addresses a critical question: **"Why two optimizers?"**
# 
# **The Problem:** The **Branch network** and the **Trunk network** are fundamentally different components.
# * One might be a simple `MLP`, the other a complex `KAN` (from Cell 6).
# * They perform different "jobs" (one encodes a function, the other learns a basis).
# * They will almost certainly have different **convergence dynamics** and stability.
# 
# **Insight:** Architectures like `cKAN` (Chebyshev KAN) for the **Trunk** might require a smaller learning rate (e.g., `lr_trunk = 3e-3`), while the `MLP` **Branch** trains best at `lr_branch = 1e-3`. If we used a *single* optimizer for all `params`, we would be forced to compromise.
# 
# **The Solution:** This code implements a **partitioned optimizer**.
# * It iterates through the top-level keys of the `params` PyTree (i.e., `'branch_net'` and `'trunk\_net'`).
# * It assigns a *separate* `optax.adam` instance and a *separate* learning rate schedule (`schedule_branch`, `schedule_trunk`) to each key.
# * This creates the `optimizers` and `states` dictionaries, which allow the training loop to apply different updates to different parts of the model.
# 
# ### 3. Summary for Agent
# 
# This cell is a **critical patching location** for handling hybrid models.
# * The `planner_agent` must know that this is where the model is initialized and the optimizers are defined.
# * **If the agent swaps a model (e.g., to `cKAN` or `QR-DeepONet`), it MUST first use the `retrieve` tool to find the correct initialization procedure for that specific model.** This may involve different `init` functions or dummy data.
# * **If the agent swaps to `cKAN`**, it MUST also patch *this cell* to change `lr_trunk` to an appropriate value (e.g., `lr_trunk = lr * 0.1`) to ensure stable training.
# * **If the agent implements `QR-DeepONet`**, it MUST also patch *this cell* to add a separate optimizer for the new `A` parameters** from the QR decomposition, in addition to the branch and trunk networks.
# """

# %%

print("Initializing model...")
model = DeepONetModel(branch_layers=branch_layers, 
                      trunk_layers=trunk_layers)
key = jax.random.PRNGKey(1234)
sample_F = X_DN['F'][0:10]
sample_y = X_DN['y'][0:10]
params = model.init(key, sample_F, sample_y) 
_, unravel_fn = ravel_pytree(params)

print("Initializing partitioned optimizers (dynamic keys)...")

optimizers = {}
states = {}

lr_branch = lr * 1.0 
lr_trunk = lr * 0.5 
schedule_branch = optax.exponential_decay(lr_branch, decay_steps, decay_rate)
schedule_trunk = optax.exponential_decay(lr_trunk, decay_steps, decay_rate)

# Iterate over the keys in the 'params' sub-tree
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

# %% [markdown]
# ### Cell 8: vRBA for Neural Operators (The Mixed Strategy)
# 
# This cell defines the functions for applying the **variational Residual-Based Attention (vRBA)** framework to our DeepONet model.
# 
# This problem has **two distinct levels of discretization**:
# 1.  **Function Space:** The full dataset has $ N_{\text{func}} $ input-output function pairs, $ \{v_j, \mathcal G[v_j]\}_{j=1}^{N_{\text{func}}} $.
# 2.  **Spatial Domain:** Each output function $ \mathcal G[v_j] $ is evaluated at $ N $ discrete points, $ \{x_i\}_{i=1}^{N} $.
# 
# The standard loss is an average over *both* levels:
# 
# $$
# \mathcal{L}(\theta) = \frac{1}{N_{\text{func}}}\sum_{j=1}^{N_{\text{func}}} \frac{1}{N} \sum_{i=1}^{N}[R(v_j, x_i; \theta)]^2
# $$
# 
# A simple vRBA scheme is ill-suited for this. Therefore, this cell implements the **mixed strategy** described in the vRBA paper:
# * **Importance Weighting (in Space):** We apply weights $ \Lambda_{i,j} $ to the spatial points $ x_i $ for *each* function $ v_j $. This targets high-residual *regions* within a function.
# * **Importance Sampling (in Function-Space):** We use the aggregated $ \Lambda $ weights to build a sampling PDF $ \bar{q}_j $ to select the most "difficult" *functions* $ v_j $ from the dataset.
# 
# ---
# ### 1. `Lambda_global` Initialization
# 
# The code first initializes the global state for this entire process. This is the central "memory" of the vRBA system. It creates a matrix $ \mathbf{\Lambda} $ with the *exact same shape* as our target data, $ (N, P) $ or $ (N_{\text{func}}, N) $. Each entry $ \Lambda_{i,j} $ stores the current spatial weight for point $ i $ of function $ j $.
# 
# ---
# ### 2. Strategy 1: Importance Weighting (in Space)
# 
# This is implemented by the `update_spatial_weights` function. It is responsible for updating the $ \mathbf{\Lambda} $ matrix.
# 
# This function implements the core vRBA logic (from the PINNs framework) but does so in a *batched* way. It takes a batch of residuals $ R $ (shape $ (b_u, N) $) and the corresponding rows from $ \mathbf{\Lambda} $ (`Lambda_batch`) and computes their updated values via an **Exponential Moving Average (EMA)**, as described in the theory:
# 
# $$
# \Lambda_{i,j}^{k+1}=\gamma\Lambda_{i,j}^{k}+\eta^*Q_{i,j}^{k+1}(\theta^k)
# $$
# 
# The `spatial_update_fn_baked` function is a `partial` application that "bakes in" the chosen `weighting_potential` (e.g., 'exponential'). This is the function that will be called *during* the training step to apply the loss $ \mathcal{L}(\theta) = ... \sum [\Lambda_{i,j} R(v_j, x_i; \theta)]^2 $.
# 
# ---
# ### 3. Strategy 2: Importance Sampling (in Function-Space)
# 
# This is implemented by the `update_function_pdf` function. This function creates the sampling distribution $ \bar{q} $ that we will use to select our *next batch of functions*.
# 
# It executes the strategy from the theory precisely:
# 1.  **Aggregate Scores:** It first computes the aggregated importance score $ s_j $ for *each* function by summing its spatial weights. This is exactly the $ s_{j}=\sum_{i=1}^{N}\Lambda_{i,j} $ from the theory.
#     ```python
#     lambda_scores = jnp.sum(Lambda_global, axis=1) # Shape (N_func,)
#     ```
# 2.  **Create PDF:** It then applies a potential (e.g., 'cubic') to these scores and normalizes them to create the final p.m.f. $ \bar{q}_j $.
# 
# The `function_pdf_fn_baked_and_jitted` function is a `partial` application that bakes in the chosen `sampling_potential`. This function will be called *before* each training step to select the indices for the next batch.
# 
# ---
# ### Summary for Agent
# 
# This cell provides the two "engines" for the vRBA-Operator framework.
# * **`Lambda_global`** is the shared state matrix of shape $ (N_{\text{func}}, N) $.
# * **`spatial_update_fn_baked`** is called *inside* the training step. It updates the rows of $ \mathbf{\Lambda} $ corresponding to the current batch.
# * **`function_pdf_fn_baked_and_jitted`** is called *before* the training step. It reads the *entire* $ \mathbf{\Lambda} $ matrix to create a PDF for sampling the next batch.
# * The `planner_agent` can patch the `weighting_potential` and `sampling_potential` strings to test different vRBA strategies (e.g., 'exponential'/'exponential', 'exponential'/'cubic', etc.).
# * **If the agent implements `QR-DeepONet` (from Cell 5), the definition of the residual `R` that this function receives will be different.** The agent MUST use its `retrieve` function to find information on how to correctly apply vRBA weights for the `QR-DeepONet` architecture.
# """

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
Lambda_global = jnp.ones_like(X_DN['S']) * Lambda_max / 2.0
cap_RBA = 40 # You can set this here
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

# %% [markdown]
# 
# # Cell 10: The vRBA-Weighted Loss Function
# 
# This function, `loss_all`, defines the exact computation that will be differentiated at each training step. It brings together the model's prediction (Cell 5) and the vRBA weighting strategy (Cell 8).
# 
# The function executes a specific, critical sequence:
# 
# 1.  **Predict:** It calls the `apply_fn` to get the model's current prediction, `outputs`.
# 2.  **Get Residual:** It computes the raw error (the residual) `R` by subtracting the target `U.T` from the prediction.
# 3.  **Update Weights:** It passes this raw residual `R` and the old weights `Lambda_batch` into the `spatial_update_fn_baked` (from Cell 8). This function returns the *new* vRBA weights, `new_Lambda_batch`.
# 4.  **Compute Loss:** It computes the final loss value by taking the mean squared error of the residual *after* it has been multiplied by the `new_Lambda_batch`. This is the vRBA-weighted loss.

# %%
def loss_all(params, apply_fn, F, y, U, Lambda_batch, it, current_gamma, eta,
             spatial_update_fn_baked): 
    # Get model prediction using the apply_fn
    outputs = apply_fn(params, F, y)    
    R = outputs - U.T
    # Call the baked-in function. 
    new_Lambda_batch = spatial_update_fn_baked(R, Lambda_batch, it,
                                              current_gamma, eta)
    # Compute the weighted loss
    loss = jnp.mean((new_Lambda_batch * R)**2)
    return loss, new_Lambda_batch

# %% [markdown]
# ## Cell 11:Gradient and Update Functions
# 
# ---
# ### 1. `compute_grads_and_loss`
# 
# This is the JAX-jitted function for calculating the gradients. Its main job is to call the `loss_fn` (from Cell 10) and use `jax.value_and_grad`.
# ---
# ### 2. `update_model`
# 
# This function is the **engine** that executes the "partitioned" optimization strategy from Cell 7.
# 
# This is the critical part. The main training loop will *call this function multiple times*—once for each key in our `optimizers` dictionary (e.g., once for `'branch_net'` and once for `'trunk_net'`).
# 
# It takes the `key_params` (e.g., the string `"branch_net"`) and uses it to:
# 1.  **Get Sub-trees**: "Reach into" the full `gradient` and `params` PyTrees and pull out only the specific sub-trees (`g_subset`, `p_subset`) that correspond to that key.
# 2.  **Perform Update**: Apply the update using *only* the correct optimizer and state for that sub-tree.
# 3.  **"Mutate" Params**: Return a *new* `params` tree where only the relevant part (e.g., `'branch_net'`) has been updated.
# 
# This provides a highly efficient, JIT-compiled way to update different parts of a complex model with different optimizer states.
# 
# ---
# ### Summary for Agent
# 
# * This cell contains the core JIT-compiled update logic. The `update_model` function is the key to enabling the partitioned optimizer from Cell 7.
# * **If the agent implements `QR-DeepONet` (from Cell 5), this implementation will fail.** The `QR-DeepONet` requires **two loss functions** (a data loss and an orthogonality loss).
# * The agent **MUST use its `retrieve` function** to find the correct `compute_grads_and_loss` implementation for `QR-DeepONet` that correctly combines these two losses before differentiation.
# """

# %%

@partial(jit, static_argnums=(1, 2, 9)) # loss_fn, apply_fn, baked_fn are static
def compute_grads_and_loss(params, loss_fn, apply_fn, 
                           F, y, U, Lambda_batch, it, current_gamma, 
                           spatial_update_fn_baked):
    
    def loss_wrapper(p):
        return loss_fn(p, apply_fn, F, y, U, Lambda_batch, it, 
                       current_gamma, eta, spatial_update_fn_baked)            
    (loss_value, new_Lambda_batch), g = jax.value_and_grad(loss_wrapper, has_aux=True)(params)
    
    return loss_value, new_Lambda_batch, g

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
# ### Cell 12: The Main Training Loop
# 
# This cell is the "main" function that executes the entire training process. It brings together all the components defined in the previous cells: the vRBA functions (Cell 8), the loss (Cell 10), and the JIT-compiled update functions (Cell 11).
# 
# The loop follows a specific sequence at each iteration `it`:
# 
# 1.  **vRBA (Function Sampling):**
#     * It first calls `function_pdf_fn_baked_and_jitted` (from Cell 8). This function reads the *entire* `Lambda_global` state to compute the probability distribution `q_pdf`, which represents the "difficulty" of each function in the dataset.
#     * It then uses `jax.random.choice` to sample a batch of indices (`idx_train`) *using this `q_pdf`*. This is **Importance Sampling**.
# 
# 2.  **Batch Creation:**
#     * It uses `idx_train` to create the batches `U_batch`, `F_batch`, and `Lambda_batch` (the relevant rows from the global state).
# 
# 3.  **vRBA (Gradient Calculation):**
#     * It calls `compute_grads_and_loss` (from Cell 11). This single JIT-compiled call does all the work:
#         * Calculates the residual `R`.
#         * Calls `spatial_update_fn_baked` to get the `new_Lambda_batch`.
#         * Calculates the final vRBA-weighted loss.
#         * Returns the `loss_value`, the `new_Lambda_batch`, and the full `gradient` PyTree.
# 
# 4.  **Partitioned Optimizer Update (The "PINN-style" Loop):**
#     * This is the execution of the strategy from Cell 7. The code iterates through the `optimizers.keys()` (e.g., `'branch_net'`, `'trunk_net'`).
#     * For each key, it calls our JIT-compiled `update_model` function (from Cell 11). This efficiently updates only the corresponding part of the `params` PyTree.
# 
# 5.  **vRBA (State Update):**
#     * `Lambda_global = Lambda_global.at[idx_train].set(new_Lambda_batch)`
#     * This is the final, critical step. The `new_Lambda_batch` (returned from `compute_grads_and_loss`) is used to update the "memory" in the `Lambda_global` state, completing the vRBA cycle for the next iteration.
# 
# 6.  **Logging:**
#     * Every 500 steps, it calculates the *unweighted* mean squared error on the full dataset for logging, providing a true measure of performance.

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


if Run_type=='Train':
    loss_log_T = []
    it_log_T = []
    train_L2 = []
    test_L2 = []
    
    # --- 6. Start Training Loop ---
    print("Starting training (PINN-style loop with multiple optimizers)...")
    pbar = trange(ngsteps + 1)
    for it in pbar:
        # --- vRBA: Get dynamic params ---
        max_RBA = max_RBA0 + step_RBA * (it // 10)
        current_gamma_it = get_gamma(eta, max_RBA)

        # --- vRBA: Function Importance Sampling ---
        key = jax.random.PRNGKey(it)
        key, subkey = jax.random.split(key)
        
        q_pdf = function_pdf_fn_baked_and_jitted(Lambda_global, it)
        
        idx_train = jax.random.choice(
            subkey,
            a=jnp.arange(len(U0)),
            shape=(batch_size,),
            p=q_pdf
        )
        
        # --- Create Batches ---
        U_batch = U0[idx_train]
        F_batch = F0[idx_train]
        Lambda_batch = Lambda_global[idx_train]
        U_batch_T = U_batch.T 
        
        # --- Perform Training Step ---
        
        # 1. Get grads (JIT-compiled)
        loss_value, new_Lambda_batch, gradient = compute_grads_and_loss(
            params, loss_all, apply_fn,
            F_batch, y, U_batch_T, Lambda_batch, 
            it, current_gamma_it,     
            spatial_update_fn_baked   
        )
        
        # 2. Update params (Python loop over keys)
        # --- THIS IS THE PINN-STYLE LOOP ---
        for key in optimizers.keys(): 
            params, states[key] = update_model(
                key, optimizers[key], gradient, params, states[key]
            )

        
        # --- Update state for next loop ---
        Lambda_global = Lambda_global.at[idx_train].set(new_Lambda_batch)

        # --- Logging ---
        if it % 500 == 0:
            # 'params' is now the fully updated parameters
            
            # Log unweighted loss on full training set
            U_pred_all = apply_fn(params, F0, y)
            R_all = U_pred_all - U0
            log_loss_value = jnp.mean(R_all**2)
            
            # Log test error
            error_test, error_train = test_error(apply_fn, params, X_DN)
            
            # Store losses
            loss_log_T.append(log_loss_value)
            it_log_T.append(it)
            train_L2.append(error_train)
            test_L2.append(error_test)

            # Print losses
            pbar.set_postfix({'it': it,
                            'Loss': log_loss_value,
                            'Train RL2': error_train,
                            'Test RL2': error_test})

    print("Training finished.")

# %% [markdown]
# ### Cell 13: Save and Load Utilities
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
# ### Cell 14: Final Summary Plot and Worst-Case Analysis
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
ax.plot(it_log_T, loss_log_T)
ax.set_title('Total Loss (Unweighted MSE)')
ax.set_xlabel('Iterations')
ax.set_ylabel('Loss')
ax.set_yscale('log')
ax.grid(True, which="both", ls="--")

# Plot 2: Train Relative Error vs. Iterations
ax = axes[0, 1]
ax.plot(it_log_T, train_L2, label='Train RL2')
ax.set_title('Train Relative Error (Mean)')
ax.set_xlabel('Iterations')
ax.set_ylabel('Error')
ax.set_yscale('log')
ax.grid(True, which="both", ls="--")

# Plot 3: Test Relative Error vs. Iterations
ax = axes[0, 2]
ax.plot(it_log_T, test_L2, label='Test RL2', color='orange')
ax.set_title('Test Relative Error (Mean)')
ax.set_xlabel('Iterations')
ax.set_ylabel('Error')
ax.set_yscale('log')
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


