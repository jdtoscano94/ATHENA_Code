
# Experiment Report
*Last Updated: 2026-01-17 19:20:23*

## User Request

Let's solve the **Cahn-Hilliard Equation** in 2D.

**The Equation:**
$$ u_t = \Delta(u^3 - u) - \epsilon \Delta^2 u $$

**Physics Parameters:**
* $\epsilon = 0.01$.
* Diffusion coeff = 1.0.

**Domain:**
* $(x,y) \in [0, 2\pi] \times [0, 2\pi]$
* Time interval: $t \in [0, 1.0]$

**Resolution**
Lets use a grid of 128x128

**Boundary Conditions:**
* **Periodic** in both X and Y 

**Initial Conditions:**
* Start with a **small perturbation of the unstable steady state** $u(x) = 0$.
    $$ u(x,0) = 0.0 + 0.05 \cdot \text{noise}(x,y) $$

**Visualization:**
Save a **heatmap** as `summary_all.png`.

**Validation:**
Please print the Total Mass $\int u \, dA$ at the start and end. It should be conserved to machine precision.


---

## Latest Strategy Report
(This was the strategy for the *last successful run*)
**1. Problem Statement**

**1. Mathematical Specification (Code-Ready Extraction)**
* **Governing Equations:**
  $$ \frac{\partial u}{\partial t} = \nabla^2 (u^3 - u) - \epsilon \nabla^4 u $$
  (Note: $\nabla^4 u = \Delta^2 u$ is the bi-harmonic operator).
* **Domain:**
  $$ (x,y) \in [0, 2\pi] \times [0, 2\pi] $$
* **Boundary Conditions:**
  * x: Periodic
  * y: Periodic
* **Initial Conditions:**
  $$ u(x,y,0) = 0.05 \, \xi(x,y) $$
  where $\xi(x,y)$ is a random field with values drawn from a uniform distribution in $[-1, 1]$ or normal distribution $\mathcal{N}(0,1)$.
* **Constants & Parameters:**
  * $\epsilon = 0.01$
  * Diffusion Coefficient (implicit in equation scaling) = 1.0
* **Expected Physics:**
  Phase separation (Spinodal Decomposition). The random noise will coarsen into distinct domains of $u \approx 1$ and $u \approx -1$ over time $t \in [0, 1.0]$.

**2. Strategic Methodology**
* **Method Selection:** **Fourier Spectral Method**.
    * *Reasoning:* The domain is strictly periodic in both dimensions. Spectral methods provide exponential convergence for smooth solutions and handle high-order derivatives (like the bi-harmonic $\nabla^4$) with perfect accuracy and efficiency (simply multiplying by $k^4$ in frequency space), avoiding the stencil complexity of finite differences.
* **Exact Solution Status:** No analytical solution exists for random initial conditions in the Cahn-Hilliard equation. We rely on the spectral method's high accuracy.

**3. Implementation Requirements (CRITICAL)**
* **Configuration:** Use template defaults.

**3.0.1 De-aliasing & Bandwidth Analysis (Fourier Spectral Selected):**
* **Polynomial Degree Identification:** The nonlinear term is $\nabla^2(u^3)$. The core nonlinearity is the cubic term $u^3$. Thus, the polynomial degree is **$p=3$**.
* **Spectral Expansion Logic:** A cubic nonlinearity generates frequencies up to $3 \times k_{resolved}$.
* **Derivation Task:**
    * Let $N$ be the grid size. The Nyquist limit is $N/2$.
    * Let $k_{cut}$ be the maximum resolved wavenumber.
    * The cubic interaction generates wavenumbers up to $3k_{cut}$.
    * To prevent aliasing back into the resolved range $[-k_{cut}, k_{cut}]$, the aliased wave $3k_{cut} - N$ must fall outside the range, i.e., $3k_{cut} - N < -k_{cut}$.
    * $4k_{cut} < N \implies k_{cut} < N/4$.
    * The fraction of the spectrum to keep is $k_{cut} / (N/2) = (N/4) / (N/2) = 1/2$.
* **Constraint:** Set `dealiasing_mask_fraction = 0.5`. (Strictly filter the top 50% of frequencies to ensure stability of the cubic term).

**3.1 Advisor-Mandated Adjustments:**
* None.

**4. Visualization Strategy:**
* **Field Selection:**
    * Primary: $u$ (Concentration/Order Parameter). This is the only variable that matters for phase separation.
* **Task:** Plot $u$ at 4 distinct time steps (e.g., $t=0.0, 0.33, 0.66, 1.0$).
* **Layout:** 1 Row (Field $u$) $\times$ 4 Columns (Time).
* **Saving Data:** Save full state every `dt_save=0.1`.
* **Constraints:**
    * Save as `summary_all.png`.
    * **CRITICAL:** `colorbar=false` (Disable colorbars).
    * Style: Automatic scaling.

**5. Cross-Validation & Data Integration**
* **Metrics:**
    * Calculate and print the **Total Mass** $M(t) = \int_{\Omega} u(x,y,t) \, dA$ at $t=0$ and $t=t_{final}$.
    * Verify conservation (Difference should be near machine epsilon).

**REQUIRED COMPONENTS**

**Main Method:**
Fourier Spectral Method

**Enhancements:**
None

---

## Latest Outcome Report
(This is the advisor's report from the *last successful run*)
Advisor Analysis:
**Section 1: Status Decision**
*   **Status:** APPROVED

**Section 2: Quantitative Scorecard**
*   **Score Integrity:** **50** (Perfect execution, mass conserved to machine precision $10^{-16}$).
*   **Score Accuracy:** **20** (Physics of spinodal decomposition captured perfectly; expected phase separation and coarsening observed).
*   **Score Details:** **15** (Interfaces are smooth and well-resolved; no Gibbs phenomenon; grid resolution matches $\epsilon$ scale).
*   **Score Optimality:** **15** (Fourier Spectral + ETDRK4 is the theoretical "Gold Standard" for periodic Cahn-Hilliard).
*   **TOTAL SCORE:** **100**

**Section 3: Forensic Analysis & Feedback**

*   **Observation:**
    *   **Physics:** The simulation correctly exhibits **spinodal decomposition**. Starting from a randomized state ($t=0$), the system rapidly separates into two distinct phases (represented by yellow and purple regions) by $t=0.30$.
    *   **Dynamics:** The "coarsening" process (minimization of interfacial energy) is clearly visible as the domains grow larger and the total interface length decreases from $t=0.30$ to $t=1.00$.
    *   **Conservation:** The [STDOUT LOG] confirms **Mass Error: 8.77e-16**. This is machine precision, verifying that the conservative form of the equation was implemented correctly in the spectral domain (the $k=0$ mode remains constant).

*   **Diagnosis:**
    *   **Optimal Method Success:** The Cahn-Hilliard equation involves a bi-harmonic operator ($\nabla^4 u$), which is extremely stiff and numerically challenging for Finite Difference/Element methods (requiring $\Delta t \propto \Delta x^4$). The **Fourier Spectral Method** diagonalizes this operator, allowing for exact spatial differentiation, while the **ETDRK4** (Exponential Time Differencing) integrator handles the stiffness analytically. This combination is the mathematically optimal approach for this specific problem setup.
    *   **Resolution:** With $\epsilon=0.01$, the interface width is proportional to $\sqrt{\epsilon} = 0.1$. On a $2\pi$ domain with 128 points, $\Delta x \approx 0.05$. The interface is resolved by approximately 2-3 grid points, which is sufficient for high-order spectral methods to reconstruct the profile without oscillation.

*   **Targeted Improvements:**
    *   **None.** The solution achieved a perfect score. The methodology, resolution, and physics capture are ideal for generating Ground Truth data for PINN training. No further refinement is necessary.


---

## Full Trial History
| Run | Model Description | Error | Integrity | Accuracy | Details | Optimality | **Total** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Fourier Spectral Method using SplitODEProblem with ETDRK4 time integrator on a 128x128 grid over [0, 2π] x [0, 2π] periodic domain. Implements the Cahn-Hilliard equation ∂u/∂t = ∇²(u³ - u) - 0.01 ∇⁴u exactly via IMEX splitting: linear part L = k² - 0.01 k⁴ (implicit), nonlinear N = ∇²(u³) via pseudo-spectral (ifft -> u³ -> fft -> -k², with 0.5 dealiasing mask). Initial condition u=0.05*(2*rand-1). Time span [0.0, 1.0]. | 8.77e-16 | 50 | 20 | 15 | 15 | **100** |


### Data Output Reference
| Run | Data Saved |
| :--- | :--- |
| 1 | Saves cahn_hilliard_spectral.npz (NPZ format) containing keys: 't' (11,), 'x' (128,), 'y' (128,), 'usol' (11,128,128). Also saves summary_all.png (plot of u snapshots at 4 equidistant times). |


### Directory Reference
| Run | Project Directory |
| :--- | :--- |
| 1 | `/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/NM_codes//CahnHilliard_Periodic_Noise/2026_01_17_1918/` |


---
