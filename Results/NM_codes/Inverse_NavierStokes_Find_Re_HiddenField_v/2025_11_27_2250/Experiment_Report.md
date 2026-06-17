# Experiment Report
*Last Updated: 2026-01-14 12:00:00 (America/New_York)*

## User Request

Let’s solve the **inverse** 2D **steady-state incompressible Navier–Stokes** equations (lid-driven cavity):

$$ \nabla \cdot \mathbf{u} = 0 $$
$$ (\mathbf{u}\cdot\nabla)\mathbf{u} = -\nabla p + \frac{1}{Re}\nabla^2\mathbf{u} $$

**Physics Parameter:**
- Reynolds number: **Re = 525**

**Domain:**
$(x,y) \in [0,1]\times[0,1]$

**Boundary Conditions:**
- Top lid ($y=1$):
  $$ u(x,1) = 1 - \frac{\cosh(C_0(x-0.5))}{\cosh(0.5C_0)}, \quad C_0=50 $$
  $$ v(x,1)=0 $$
- Bottom wall ($y=0$): $\mathbf{u}=(0,0)$ (no-slip)
- Left wall ($x=0$): $\mathbf{u}=(0,0)$ (no-slip)
- Right wall ($x=1$): $\mathbf{u}=(0,0)$ (no-slip)

**Cross Validation**

--- Sensors:

The data was saved as:  
path: `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Inverse_NavierStokes_Find_Re_HiddenField_v/2025_11_27_0058/`

details:  
`Saves 'results.npz' containing keys: 'model_name' (string), 'all_errors' (list of ~10000+ floats), 'all_its' (list), 'all_loss' (list), 'all_gamma' (list), 'all_lamB' (list), 'all_max_RBA' (list), 'all_L_infty' (list), 'all_L_2' (list), 'all_Re_est' (list of floats), 'Re' (scalar float), 'x_sensor' (500,), 'y_sensor' (500,), 'u_sensor' (500,), 'v_hidden_at_sensor' (500,).`

This file contains a list of sensors. Once the numerical solution is obtained, please evaluate it on the sensor points and report:
- Relative $L_2$ error of $u$ (numerical vs. PINN sensors)
- Relative $L_2$ error of $v$ (numerical vs. PINN sensors)

--- True Data:

path: `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Documents/codes/Data/ldc_Re500.npz`

The dataset is unstructured and was saved as an NPZ format containing keys: `x`, `y`, `u`, `v`, `p`
- `x`, `y`: node coordinates (1D arrays)
- `u`, `v`: velocity components at each node (1D arrays, same length as `x` and `y`)
- `p`: pressure at each node (1D array, same length as `x` and `y`)

The true solution is here; please report the $L_2$ error for $u$ and $v$ of the true solution against the new numerical solution.

**Output:**
The final plot must be saved as `summary_all.png` containing:
1. Magnitude of velocity field from the new numerical simulation.
2. Error maps comparing the numerical solution to the True Data (Re=500).
3. A scatter plot comparing the PINN predictions at sensors vs. the Numerical Solution at those same points.

---

## Latest Strategy Report
### 1. Problem Statement

**1. Mathematical Specification**

- **Governing Equations:**
  The 2D steady-state incompressible Navier–Stokes equations:
  $$
  \nabla\cdot\mathbf{u} = 0, \qquad
  (\mathbf{u}\cdot\nabla)\mathbf{u} = -\nabla p + \frac{1}{Re}\nabla^2\mathbf{u}.
  $$

- **Domain:**
  $(x,y)\in[0,1]\times[0,1]$

- **Boundary Conditions:**
  - Top lid ($y=1$):
    $$
    u(x,1) = 1 - \frac{\cosh(C_0(x-0.5))}{\cosh(0.5C_0)},\quad C_0=50,\qquad v(x,1)=0.
    $$
  - No-slip on $y=0$, $x=0$, $x=1$:
    $$ \mathbf{u}=(0,0). $$

- **Constants & Parameters:**
  - $Re=525$

- **Expected Physics:**
  Lid-driven cavity flow with a primary recirculation cell; at $Re\approx 525$ the vortex core shifts and boundary layers sharpen.

### 2. Strategic Methodology

- **Selected Main Method:** Steady Continuous Galerkin (FEM) in 2D.
- **Rationale:** The request is strictly steady-state; a direct nonlinear solve (Newton/Picard) is efficient and standard for incompressible flow in a cavity.

### 3. Implementation Requirements

- **Configuration:** Use template defaults, subject to the requirements below.

- **Continuation Strategy (Robustness Requirement):**
  Direct Newton at $Re=525$ from a zero initial guess may fail. Use **Reynolds-number continuation**, e.g.
  $$ Re: 100 \to 250 \to 525, $$
  using each converged solution as the initial guess for the next.

- **External Data Integration:**
  - Load **Sensors** from:
    `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Inverse_NavierStokes_Find_Re_HiddenField_v/2025_11_27_0058/results.npz`
  - Load **True Data** from:
    `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Documents/codes/Data/ldc_Re500.npz`

  **Action:**
  - Interpolate/evaluate the new numerical solution at:
    1) the sensor coordinates $(x_{\text{sensor}},y_{\text{sensor}})$, and
    2) the unstructured nodes $(x_{\text{true}},y_{\text{true}})$.

  **Metrics to Report:**
  - Relative $L_2$ error for $u$ and $v$ at sensors (numerical vs. PINN values).
  - Relative $L_2$ error for $u$ and $v$ against the true dataset (numerical vs. true).

### 4. Visualization Strategy

- **Task:** Generate a composite summary figure.
- **Panels (2×2 layout):**
  1. Magnitude of velocity field from the new numerical solution: $|\mathbf{u}|$.
  2. Absolute error map for $u$: $|u_{\text{num}}-u_{\text{true}}|$.
  3. Absolute error map for $v$: $|v_{\text{num}}-v_{\text{true}}|$.
  4. Scatter plot comparing PINN sensor values vs. numerical solution at sensor points
     (include identity line).

- **Constraints:**
  - Save as: `summary_all.png`
  - Style: automatic scaling.

---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Numerical implementation using steady Continuous Galerkin FEM in 2D for incompressible Navier–Stokes at $Re=525$ on $(x,y)\in[0,1]^2$ with regularized lid velocity ($C_0=50$). Nonlinear solve stabilized via Reynolds-number continuation (e.g., $100\to250\to525$). Postprocessing interpolates the FEM solution to (i) PINN sensor locations from `results.npz` and (ii) unstructured true-data nodes from `ldc_Re500.npz`, reporting relative $L_2$ errors for $(u,v)$ in both comparisons. Figure `summary_all.png` produced as a 2×2 panel: $|\mathbf{u}|$, $|u_{\text{num}}-u_{\text{true}}|$, $|v_{\text{num}}-v_{\text{true}}|$, and sensor scatter (PINN vs numerical). | N/A | 50 | 20 | 15 | 15 | **100** |

### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 0 | Saves `summary_all.png` (2×2 panel figure as specified). Optionally saves a `results_eval.npz` containing: interpolated numerical values at sensors and true nodes, plus reported relative $L_2$ errors for $(u,v)$. |

### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 0 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Inverse_NavierStokes_Find_Re_HiddenField_v/2025_11_27_0058/` |
