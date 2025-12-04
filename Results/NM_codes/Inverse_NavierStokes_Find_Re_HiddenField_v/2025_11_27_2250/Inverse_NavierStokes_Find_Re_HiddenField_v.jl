# %% [markdown]
# # General Overview: Steady Incompressible Navier-Stokes
# 
# ## 1. Mathematical Formulation: The Saddle-Point Problem
# The objective is to find the stationary velocity field $\mathbf{u}$ and pressure field $p$ that satisfy the conservation of momentum and mass in a domain $\Omega \subset \mathbb{R}^d$.
# 
# This constitutes a **Boundary Value Problem (BVP)** for a system of non-linear elliptic PDEs:
# 
# $$
# \begin{aligned}
# -\nu \Delta \mathbf{u} + (\mathbf{u} \cdot \nabla) \mathbf{u} + \nabla p &= \mathbf{f} \quad \text{in } \Omega \\
# \nabla \cdot \mathbf{u} &= 0 \quad \text{in } \Omega
# \end{aligned}
# $$
# 
# Where:
# * $\nu = \mu / \rho$ is the kinematic viscosity ($Re^{-1}$).
# * The term $(\mathbf{u} \cdot \nabla) \mathbf{u}$ introduces **non-linearity** (convection).
# * The pressure $p$ acts as a Lagrange multiplier to enforce the incompressibility constraint $\nabla \cdot \mathbf{u} = 0$.
# 
# ## 2. Methodology: Galerkin Finite Element Method (FEM)
# Unlike the previous explicit time-marching (Hyperbolic), this problem requires solving a large, sparse, non-linear algebraic system $\mathcal{R}(\mathbf{U}) = \mathbf{0}$.
# 
# * **Weak Formulation**: We seek $(\mathbf{u}, p) \in V \times Q$ such that the residual vanishes for all test functions $(\mathbf{v}, q)$.
# * **LBB Condition**: Stability requires careful pairing of velocity/pressure spaces (e.g., Taylor-Hood elements) to prevent spurious pressure modes.
# * **Solver Strategy**: We employ a **Newton-Raphson** iterative solver.
#     $$ \mathbf{J}(\mathbf{U}^k) \delta \mathbf{U} = -\mathcal{R}(\mathbf{U}^k) $$
#     Where $\mathbf{J}$ is the Jacobian matrix (linearized stiffness + convection).
#

# ## Cell 0: Framework Initialization (Gridap.jl)
# 
# ## Core FEM Engine & Optimization
# Initializes the environment for monolithic Finite Element simulations.
# 
# 1.  **`Gridap` Ecosystem**:
#     * **`Gridap`**: The core generator for discrete models, reference elements, and FE spaces ($V_h, Q_h$). Handles the assembly of the global stiffness matrix $\mathbf{K}$ and residual vector $\mathbf{b}$.
#     * **`Gridap.Geometry`**: Utilities for constructing discrete domain topologies (Triangulations) and quadrature rules.
#     * **`Gridap.Adaptivity`**: Framework for $h$-refinement (mesh splitting) to resolve local features like boundary layers.
# 
# 2.  **Non-Linear Solver Utilities (`LineSearches`)**:
#     * **Role**: Enhances the robustness of the Newton-Raphson solver.
#     * **Mechanism**: If a full Newton step $\mathbf{U}^{k+1} = \mathbf{U}^k + \delta \mathbf{U}$ does not sufficiently reduce the residual, the **BackTracking** algorithm searches along the direction $\delta \mathbf{U}$ for a smaller step size $\alpha$ that guarantees descent (Globalized Newton).
# 
# 3.  **Diagnostics (`Plots`, `NPZ`)**:
#     * Standard I/O for visualizing the resulting vector fields and exporting nodal data.
# """

# %%
using Gridap
using Gridap.Geometry
using Gridap.Adaptivity
using LineSearches: BackTracking
using Plots
using DataStructures
using Printf
using NPZ
using Random
# %% [markdown]

# ## Cell 1: Computational Domain & Topology
# 
# ## Methodology: Simplicial Decomposition
# Defines the initial spatial discretization $\mathcal{T}_0$ of the domain $\Omega = [0, 1]^2$.
# 
# ### 1. Geometric Primitives (`CartesianDiscreteModel`)
# * **Base Grid**: We initialize a structured Cartesian grid with $N \times N$ cells (`n_initial`).
# * **Domain**: Unit square, representing the standard Lid-Driven Cavity benchmark.
# 
# ### 2. Topological Promotion (`simplexify`)
# * **Constraint**: The Newest Vertex Bisection (NVB) refinement algorithm (used in Cell 6) is topologically defined for **Simplices** (Triangles in 2D), not Hypercubes (Quads).
# * **Transformation**: We immediately convert the structured quad mesh into an **Unstructured Triangular Mesh**. This allows the mesh to support local $h$-refinement without hanging nodes, ensuring the resulting Finite Element spaces remain $H^1$-conforming throughout the adaptive process.
# """

# %%
function generate_initial_mesh(n_initial)
    # 1. Geometric Domain (Unit Square)
    domain = (0, 1, 0, 1)
    partition = (n_initial, n_initial)
    cart_model = CartesianDiscreteModel(domain, partition)
    model = simplexify(cart_model) 
    return model
end

# %% [markdown]

# ## Cell 2: Discretization & LBB-Stable Function Spaces
# 
# ## Methodology: Mixed Finite Elements (Taylor-Hood)
# Constructs the discrete functional framework for the saddle-point problem. To ensure numerical stability and prevent spurious pressure oscillations, we satisfy the **Ladyzhenskaya-Babuska-Brezzi (LBB)** or Inf-Sup condition.
# 
# ### 1. Element Selection: Taylor-Hood ($P_k / P_{k-1}$)
# We employ the gold-standard stable pair for incompressible flow:
# * **Velocity ($\mathbf{u} \in V_h$)**: Approximated using continuous Lagrangian polynomials of degree $k$ (Quadratic if order=2).
# * **Pressure ($p \in Q_h$)**: Approximated using continuous Lagrangian polynomials of degree $k-1$ (Linear if order=2).
# 
# ### 2. Function Space Topology (`setup_fe_spaces`)
# * **Test Spaces ($V, Q$)**:
#     * Defines the weighting functions $\mathbf{v}$ and $q$.
#     * **Velocity**: $H^1$-conforming vector space. The `dirichlet_tags="boundary"` ensures that test functions vanish on the boundary ($\mathbf{v}|_{\Gamma} = \mathbf{0}$), which is required for the weak form.
#     * **Pressure**: `constraint=:zeromean` enforces $\int_\Omega p \, d\Omega = 0$. Since only $\nabla p$ appears in the PDE, pressure is unique only up to a constant. This constraint fixes that hydrostatic mode.
# 
# * **Trial Spaces ($U, P$)**:
#     * Defines the ansatz for the solution $\mathbf{u}_h$ and $p_h$.
#     * **Dirichlet Lifting**: The `TrialFESpace` $U$ handles the non-homogeneous boundary values.
#         * It "lifts" the closure `u_bc(x)` onto the finite element basis.
#         * The closure defines the physical BCs: Tangential velocity $\mathbf{u}=(1,0)^T$ at the lid ($y \approx 1$) and No-Slip $\mathbf{u}=\mathbf{0}$ elsewhere.
# 
# * **Multifield Formulation**:
#     * Bundles the spaces into composite objects $X = U \times P$ and $Y = V \times Q$. This instructs the solver to assemble a single **Monolithic** block-structured Jacobian system, rather than solving the momentum and continuity equations sequentially.
# 

# %%
function u_bc(x)
        if abs(x[2] - 1.0) < 1e-8
            C0 = 50.0
            val = 1.0 - cosh(C0 * (x[1] - 0.5)) / cosh(0.5 * C0)
            return VectorValue(val, 0.0)
        else
            return VectorValue(0.0, 0.0)
        end
    end
function setup_fe_spaces(model, order)
    # 1. Define Reference Elements (Taylor-Hood Pk/Pk-1)
    reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
    reffe_p = ReferenceFE(lagrangian, Float64, order - 1)
    
    # 2. Test Spaces
    # V: H1-conforming vector space with Dirichlet tags on boundaries
    V = TestFESpace(model, reffe_u, conformity=:H1, dirichlet_tags="boundary")
    # Q: H1-conforming scalar space with zero-mean constraint (for pressure uniqueness)
    Q = TestFESpace(model, reffe_p, conformity=:H1, constraint=:zeromean)
    
    # 4. Trial Spaces
    U = TrialFESpace(V, u_bc)
    P = TrialFESpace(Q)
    
    # 5. Multifield Spaces (for coupled system)
    Y = MultiFieldFESpace([V, Q])
    X = MultiFieldFESpace([U, P])
    
    return X, Y
end

# %% [markdown]

# ## Cell 3: Physics Assembly & Weak Formulation
# 
# ## Derivation: From Strong to Weak Form
# To solve the PDE using Finite Elements, we must transform the **Strong Form** (point-wise PDE) into the **Weak Form** (integral equation).
# 
# ### 1. The Strong Form
# $$
# \begin{aligned}
# -\nu \Delta \mathbf{u} + (\mathbf{u} \cdot \nabla) \mathbf{u} + \nabla p &= \mathbf{0} \quad \text{in } \Omega \\
# \nabla \cdot \mathbf{u} &= 0 \quad \text{in } \Omega
# \end{aligned}
# $$
# 
# ### 2. Test Function Projection
# We multiply the momentum equation by a vector test function $\mathbf{v} \in V_0$ (which vanishes on Dirichlet boundaries) and the continuity equation by a scalar test function $q \in Q$. We then integrate over the domain $\Omega$:
# $$
# \int_\Omega (-\nu \Delta \mathbf{u}) \cdot \mathbf{v} \, d\Omega + \int_\Omega ((\mathbf{u} \cdot \nabla) \mathbf{u}) \cdot \mathbf{v} \, d\Omega + \int_\Omega \nabla p \cdot \mathbf{v} \, d\Omega + \int_\Omega q (\nabla \cdot \mathbf{u}) \, d\Omega = 0
# $$
# 
# ### 3. Integration by Parts (Green's Identity)
# We lower the derivative order on the viscous and pressure terms to allow for $C^0$ (continuous) basis functions.
# * **Viscous Term**: $-\int_\Omega \nu \Delta \mathbf{u} \cdot \mathbf{v} \longrightarrow \int_\Omega \nu \nabla \mathbf{u} : \nabla \mathbf{v} - \underbrace{\int_{\Gamma} \nu (\nabla \mathbf{u} \cdot \mathbf{n}) \cdot \mathbf{v}}_{\text{0 (v=0 on walls)}} $
# * **Pressure Term**: $\int_\Omega \nabla p \cdot \mathbf{v} \longrightarrow -\int_\Omega p (\nabla \cdot \mathbf{v}) + \underbrace{\int_{\Gamma} p (\mathbf{v} \cdot \mathbf{n})}_{\text{0 (v=0 on walls)}}$
# 
# ### 4. The Final Weak Form
# This yields the residual equation implemented in the code:
# $$
# \underbrace{\int_\Omega \nu (\nabla \mathbf{u} : \nabla \mathbf{v})}_{\text{Diffusive (a)}} + \underbrace{\int_\Omega ((\mathbf{u} \cdot \nabla) \mathbf{u}) \cdot \mathbf{v}}_{\text{Convective (c)}} - \underbrace{\int_\Omega p (\nabla \cdot \mathbf{v})}_{\text{Pressure (b)}} + \underbrace{\int_\Omega q (\nabla \cdot \mathbf{u})}_{\text{Continuity (b}^T\text{)}} = 0
# $$
# 
# ## Code Implementation (`setup_navier_stokes_operator`)
# Constructs the non-linear residual operator $\mathcal{R}(\mathbf{U})$ and its Jacobian.
# 
# * **`a((u,p),(v,q))`**: Implements the symmetric Stokes terms: Viscous $\nu (\nabla \mathbf{v} \odot \nabla \mathbf{u})$ and Pressure-Divergence coupling.
# * **`c(u,v)`**: Implements the non-linear Convection term $\mathbf{v} \odot (\mathbf{u} \cdot \nabla \mathbf{u})$.
# * **Automatic Differentiation**: `Gridap` automatically computes the Jacobian $\mathbf{J} = \delta \mathcal{R}$ (Gateaux derivative) required for the Newton solver, specifically linearizing the convection term to $dconv = \mathbf{u} \cdot \nabla \delta \mathbf{u} + \delta \mathbf{u} \cdot \nabla \mathbf{u}$.
# """

# %%
function setup_navier_stokes_operator(model, X, Y, Re, order)
    # 1. Geometry & Integration
    degree = 2*order
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    
    # 2. Convection Operators
    conv(u,∇u) = (∇u')⋅u
    dconv(du,∇du,u,∇u) = conv(u,∇du) + conv(du,∇u)
    
    # 3. Weak Form Terms (Residuals)
    # Viscous + Pressure + Convection
    a((u,p),(v,q)) = ∫((1/Re)*(∇(v)⊙∇(u)) - (∇⋅v)*p + q*(∇⋅u))dΩ
    c(u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
    
    # 4. Residual & Jacobian
    res((u,p),(v,q)) = a((u,p),(v,q)) + c(u,v)
    jac((u,p),(du,dp),(v,q)) = a((du,dp),(v,q)) + ∫(v⊙(dconv∘(du,∇(du),u,∇(u))))dΩ
    
    # 5. Construct FEOperator
    op = FEOperator(res, jac, X, Y)
    
    return op, Ω, dΩ
end

# %% [markdown]

# ## Cell 4: The Non-Linear Solver (Globalized Newton-Raphson)
# 
# ## Mathematical Formulation: Root Finding
# The discretization yields a large, sparse, non-linear algebraic system for the global coefficient vector $\mathbf{U} = [\mathbf{u}_h, p_h]^T$:
# $$
# \mathcal{R}(\mathbf{U}) = \mathbf{0}
# $$
# where $\mathcal{R}$ represents the residual of the weak form defined in Cell 3.
# 
# ## Solver Strategy: Newton-Raphson with Line Search
# We solve this system iteratively using a **Damped Newton Method**.
# $$
# \mathbf{J}(\mathbf{U}^k) \delta \mathbf{U}^k = -\mathcal{R}(\mathbf{U}^k)
# $$
# $$
# \mathbf{U}^{k+1} = \mathbf{U}^k + \alpha^k \delta \mathbf{U}^k
# $$
# 
# 1.  **The Jacobian ($\mathbf{J}$)**:
#     The matrix $\mathbf{J} = \partial \mathcal{R} / \partial \mathbf{U}$ includes contributions from the diffusive stiffness matrix ($A$), the pressure gradient ($B^T$), divergence ($B$), and the linearized convection term ($N(\mathbf{u})$). `Gridap` assembles this automatically via Automatic Differentiation (AD) of the `res` and `jac` closures.
# 
# 2.  **Globalization (`BackTracking`)**:
#     * **Problem**: Standard Newton ($ \alpha = 1 $) has a small radius of convergence. For high Reynolds numbers, the non-linearity (convection) is strong, and a poor initial guess leads to divergence.
#     * **Solution**: We employ a **Backtracking Line Search**. If the full step does not sufficiently reduce the residual norm, the step size $\alpha$ is iteratively reduced ($\alpha \to \rho \alpha$) until the **Armijo condition** (sufficient decrease) is satisfied.
# 
# 3.  **Solver Configuration (`NLSolver`)**:
#     * `ftol=1e-8`: Convergence criterion based on residual norm $\|\mathcal{R}\|$.
#     * `xtol=1e-8`: Convergence criterion based on increment norm $\|\delta \mathbf{U}\|$.
#     * `iterations=100`: Maximum Newton steps.

# %%

function solve_steady_system(op)
    nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking(), 
                   ftol=1e-8, xtol=1e-8, iterations=100)
    
    # 2. Wrap as FE Solver
    solver = FESolver(nls)
    
    # 3. Solve
    uh, ph = solve(solver, op)
    
    return uh, ph
end


# %% [markdown]

# ## Cell 5: Error Estimation
# 
# ## Methodology: A Posteriori Error Indication
# Computes a local error metric $\eta_K$ for each element $K$ in the mesh.
# * **Definition**: $h_K = \sqrt{\text{Area}(K)}$.
# * **Role**: Acts as a scaling factor for dimensional consistency. In Finite Element theory, the local error is bounded by powers of the mesh size multiplied by the residuals: $\|\mathbf{u} - \mathbf{u}_h\| \sim h_K^k$.
# 
# ### 2. The Error Functional ($\eta$)
# We define a composite indicator that targets three specific failure modes of the approximate solution:
# 
# $$
# \eta_K^2 = \underbrace{\| h_K (\nabla \cdot \mathbf{u}_h) \|^2_{L^2(K)}}_{\text{Mass Conservation}} + \underbrace{\| h_K \nabla p_h \|^2_{L^2(K)}}_{\text{Pressure Regularity}} + \underbrace{\| h_K \nabla \mathbf{u}_h \|^2_{L^2(K)}}_{\text{Shear Layers}}
# $$
# 
# * **Mass Conservation**: Measures the violation of the incompressibility constraint $\nabla \cdot \mathbf{u} = 0$. Large values indicate "mass leakage," common near boundaries.
# * **Pressure Regularity**: Detects sharp pressure gradients. In the Lid-Driven Cavity, these inevitably occur at the top corners (singularities).
# * **Shear Layers**: Measures the energy of the velocity gradient. This acts as a **Feature Detector**, prioritizing refinement in regions of high shear (the primary vortex and recirculation zones) rather than quiescent regions.
# 

# %%
function estimate_error(uh, ph, Ω, dΩ)
    hK = CellField(sqrt.(collect(get_array(∫(1)dΩ))), Ω)
    # 2. Define Residual-Based Error Indicator η
    ηh(x) = begin
        u, p = x
        ∫(hK*hK*(∇⋅u)⋅(∇⋅u))*dΩ +       # Mass conservation violation
        ∫(hK*hK*∇(p)⋅∇(p))*dΩ +         # Pressure regularity
        ∫(hK*hK*(∇(u)⊙∇(u)))*dΩ         # Velocity gradients (Shear)
    end 
    # 3. Compute element-wise error
    η = estimate(ηh, (uh, ph))
    return η
end

# %% [markdown]

# ## Cell 6: Mesh Adaptation 
# 
# ## Methodology: Feature-Driven $h$-Refinement
# This function executes the topological modification of the domain $\Omega_h \to \Omega_{h'}$ based on the error signal $\eta$. The goal is to equidistribute the error, refining regions where the sensor detects high residuals (singularities, boundary layers).
# 
# ### 1. Marking Strategy: Dorfler (Bulk) Marking
# We employ the **Dorfler Criterion** (Greedy Algorithm) to select the minimal subset of elements $\mathcal{M} \subset \Omega_h$ such that:
# $$
# \sum_{K \in \mathcal{M}} \eta_K^2 \ge \theta \sum_{K \in \Omega_h} \eta_K^2
# $$
# * **Parameter $\theta=0.8$**: We target the cells responsible for the top 80% of the total error. This is theoretically proven to guarantee contraction of the error in the AFEM loop.
# 
# ### 2. Topological Constraints
# * **Cartesian $\to$ Unstructured**: `Gridap`'s Cartesian models are rigid. To perform local operations, we must promote the mesh topology to a general **Unstructured Discrete Model**.
# 
# ### 3. Refinement Algorithm: Newest Vertex Bisection (NVB)
# * **Mechanism**: We use **NVB** (`Adaptivity.NVBRefinement`) rather than arbitrary splitting.
# * **Benefit**: NVB preserves the **Shape Regularity** (minimum angle condition) of the triangulation. It prevents the creation of "needle" elements which would degrade the conditioning of the stiffness matrix $\kappa(\mathbf{A})$.
# * **Conformity**: NVB automatically propagates refinement to neighbors to avoid "hanging nodes," ensuring the resulting space $V_{h'}$ remains $H^1$-conforming.
# 

# %%
function adapt_mesh(model, η, θ=0.8)
    # 1. Dorfler Marking
    # Selects the subset of cells that contribute to top θ (80%) of error
    m = DorflerMarking(θ)
    I = Adaptivity.mark(m, η)
    
    # 2. Convert to Unstructured if necessary
    # Gridap needs unstructured models to perform local refinement
    if model isa CartesianDiscreteModel
        model = UnstructuredDiscreteModel(model)
    end
    
    # 3. Perform Refinement (Newest Vertex Bisection)
    method = Adaptivity.NVBRefinement(model)
    amodel = refine(method, model; cells_to_refine=I)
    
    # 4. Extract new model
    fmodel = Adaptivity.get_model(amodel)
    
    return fmodel, I
end

# %% [markdown]

# ## Cell 7: The Adaptive Iteration Step
# 
# ## Methodology: The AFEM Cycle
# This function implements one iteration of the standard **Adaptive Finite Element Method (AFEM)** loop. The goal is to construct a sequence of nested meshes $\mathcal{T}_0 \subset \mathcal{T}_1 \subset \dots$ that minimizes the approximation error $\|\mathbf{u} - \mathbf{u}_h\|$ for a given degree of freedom count.
# 
# The cycle follows the standard paradigm:
# $$
# \text{SOLVE} \to \text{ESTIMATE} \to \text{MARK} \to \text{REFINE}
# $$
# 
# ### Execution Pipeline
# 1.  **Setup (`setup_fe_spaces`)**:
#     * Re-initializes the discrete function spaces ($V_h, Q_h$) on the current mesh topology $\mathcal{T}_k$.
# 
# 2.  **Assembly (`setup_navier_stokes_operator`)**:
#     * Constructs the non-linear residual operator $\mathcal{R}(\mathbf{U})$ and Jacobian $\mathbf{J}(\mathbf{U})$ for the target Reynolds number.
# 
# 3.  **Solve (`solve_steady_system`)**:
#     * Computes the discrete solution $\mathbf{U}_h$ satisfying $\mathcal{R}(\mathbf{U}_h) = \mathbf{0}$ via globalized Newton-Raphson.
# 
# 4.  **Estimate (`estimate_error`)**:
#     * Computes the element-wise a posteriori error indicators $\{\eta_K\}_{K \in \mathcal{T}_k}$ based on residual norms.
# 
# 5.  **Adapt (`adapt_mesh`)**:
#     * **Mark**: Selects a subset of elements $\mathcal{M}$ using the Dorfler criterion.
#     * **Refine**: Generates the new mesh $\mathcal{T}_{k+1}$ by bisecting marked elements, updating the topology for the next iteration.
# 
# ### Outputs
# Returns the evolved model state (`fmodel`, `uh`, `ph`) and the error metrics (`η`, `total_error`) required to assess convergence or trigger the next refinement step.
# 

# %%

function amr_ldc_step(model; order=2, Re=100.0, u_prev=nothing)
    
    # 1. Setup Function Spaces
    X, Y = setup_fe_spaces(model, order)
    
    # 2. Assemble Physics
    op, Ω, dΩ = setup_navier_stokes_operator(model, X, Y, Re, order)
    
    # 3. Solve System
    uh, ph = solve_steady_system(op)
    
    # 4. Estimate Error
    η = estimate_error(uh, ph, Ω, dΩ)
    
    # 5. Adapt Grid
    fmodel, refined_cells = adapt_mesh(model, η)
    
    # 6. Stats & Logging
    total_error = sum(η)
    n_cells = num_cells(model)
    n_refined = length(refined_cells)
    
    println("  Cells: $n_cells, Refined: $n_refined, Total error η: $(total_error)")
    
    return fmodel, uh, ph, η, refined_cells, total_error
end


# %% [markdown]

# ## Cell 8: Post-Processing & Diagnostic Visualization
#
# ## Methodology: Efficient Output & Field Probing
# This function handles the visualization and data export of the simulation results. It employs a dual-strategy to balance high-fidelity visualization with efficient data storage.
#
# ### 1. Visualization (Actual Mesh Nodes)
# * **Strategy**: To visualize the adaptive mesh structure, we plot the solution at the **actual mesh vertices**.
# * **Optimization**: We use **Fast Local Evaluation** (mapping physical points to cell-local coordinates) to avoid expensive global searches. If the mesh exceeds 40,000 nodes, we randomly subsample the vertices to prevent the plotting backend from hanging, ensuring the visualization pipeline remains robust for massive grids.
#
# ### 2. Data Export (Random Sampling / Virtual Sensors)
# * **Strategy**: For the final Reynolds number, we export a lightweight dataset by evaluating the solution at $N=20,000$ random points (Monte Carlo sampling).
# * **Application: Virtual Sensors**:
#     * This sampling demonstrates the capability to probe the Finite Element solution $\mathbf{u}_h(\mathbf{x})$ at **arbitrary coordinates**, independent of the mesh topology.
#     * **Validation**: This same mechanism is used to implement **Virtual Sensors**. You can define specific coordinate sets (e.g., matching experimental probe locations) to compute precise pointwise error metrics $\|\mathbf{u}_{exact}(\mathbf{x}_{sensor}) - \mathbf{u}_h(\mathbf{x}_{sensor})\|$ or to validate against benchmark data.
# """
# %%
function plot_and_save_final_solution(model, uh, ph, output_dir, Re_label, Re_final; save_data::Bool=true)
    println("Post-processing for Re=$Re_label...")
    flush(stdout)
    is_final_step = (Re_label == @sprintf("%.0f", Re_final))
    # Common plotting style for field plots
    common_style_fields = (
        marker = :circle,
        markersize = 1.5,
        markerstrokewidth = 0,
        aspect_ratio = :equal,
        axis = nothing,
        framestyle = :none,
        margin = 2Plots.mm,
        legend = false
    )
    # Get solution on mesh nodes for visualization
    Ω = get_triangulation(uh)
    cell_coords = get_cell_coordinates(Ω)
    q = Gridap.CellData.CellPoint(cell_coords, Ω, Gridap.CellData.PhysicalDomain())
    cell_u = uh(q)
    cell_p = ph(q)
    flat_x = vcat(collect(cell_coords)...)
    flat_u = vcat(collect(cell_u)...)
    flat_p = vcat(collect(cell_p)...)
    # Subsample for plotting if mesh is large
    n_total = length(flat_x)
    MAX_PLOT = 40000
    if n_total > MAX_PLOT
        indices = shuffle(1:n_total)[1:MAX_PLOT]
        points_to_plot = flat_x[indices]
        u_vals_plot = flat_u[indices]
        p_vals_plot = flat_p[indices]
    else
        points_to_plot = flat_x
        u_vals_plot = flat_u
        p_vals_plot = flat_p
    end
    x_plot = [p[1] for p in points_to_plot]
    y_plot = [p[2] for p in points_to_plot]
    u_plot = [v[1] for v in u_vals_plot]
    v_plot = [v[2] for v in u_vals_plot]
    u_mag_plot = sqrt.(u_plot.^2 .+ v_plot.^2)
    if is_final_step
        println("  Final Re reached. Performing detailed analysis...")
        # --- 1. Load External Data ---
        path_sensors = "/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes//Inverse_NavierStokes_Find_Re_HiddenField_v/2025_11_27_0058/results.npz"
        path_true = "/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Documents/codes/Data/ldc_Re500.npz"
        sensors_data = npzread(path_sensors)
        true_data = npzread(path_true)
        # --- 2. Interpolate Numerical Solution to External Coordinates ---
        pts_sensors = [VectorValue(sensors_data["x"][i], sensors_data["y"][i]) for i in 1:length(sensors_data["x"])]
        u_interp_sensors_vec = uh.(pts_sensors)
        u_interp_sensors = [v[1] for v in u_interp_sensors_vec]
        v_interp_sensors = [v[2] for v in u_interp_sensors_vec]
        pts_true = [VectorValue(true_data["x"][i], true_data["y"][i]) for i in 1:length(true_data["x"])]
        u_interp_true_vec = uh.(pts_true)
        u_interp_true = [v[1] for v in u_interp_true_vec]
        v_interp_true = [v[2] for v in u_interp_true_vec]
        # --- 3. Calculate and Print Relative L2 Errors ---
        function rel_l2_error(pred, true_vals)
            return sqrt(sum((pred .- true_vals).^2)) / sqrt(sum(true_vals.^2))
        end
        err_u_sensors = rel_l2_error(u_interp_sensors, sensors_data["u"])
        err_v_sensors = rel_l2_error(v_interp_sensors, sensors_data["v"])
        err_u_true = rel_l2_error(u_interp_true, true_data["u"])
        err_v_true = rel_l2_error(v_interp_true, true_data["v"])
        @printf("  Relative L2 Error (vs Sensors): u=%.4e, v=%.4e\n", err_u_sensors, err_v_sensors)
        @printf("  Relative L2 Error (vs True Data): u=%.4e, v=%.4e\n", err_u_true, err_v_true)
        # --- 4. Generate Composite Plot ---
        # Panel 1: Velocity Magnitude (already computed)
        p1 = scatter(x_plot, y_plot, zcolor=u_mag_plot; c=:turbo, title="Velocity Magnitude", common_style_fields...)
        # Panel 2: Absolute Error Map for u vs True Data
        p2 = scatter(true_data["x"], true_data["y"], zcolor=abs.(u_interp_true .- true_data["u"]);
                     c=:viridis, title="Abs Error: u_num - u_true", common_style_fields...)
        # Panel 3: Absolute Error Map for v vs True Data
        p3 = scatter(true_data["x"], true_data["y"], zcolor=abs.(v_interp_true .- true_data["v"]);
                     c=:viridis, title="Abs Error: v_num - v_true", common_style_fields...)
        # Panel 4: Scatter Plot of PINN Sensors vs Numerical Solution
        min_val = min(minimum(sensors_data["u"]), minimum(u_interp_sensors), minimum(sensors_data["v"]), minimum(v_interp_sensors))
        max_val = max(maximum(sensors_data["u"]), maximum(u_interp_sensors), maximum(sensors_data["v"]), maximum(v_interp_sensors))
        p4 = scatter(sensors_data["u"], u_interp_sensors,
                     label="u-component", markersize=3, markerstrokewidth=0,
                     aspect_ratio=:equal, xlabel="PINN Sensor Value", ylabel="Numerical Solution",
                     title="PINN vs Numerical @ Sensors", legend=:topleft)
        scatter!(p4, sensors_data["v"], v_interp_sensors,
                 label="v-component", markersize=3, markerstrokewidth=0)
        plot!(p4, [min_val, max_val], [min_val, max_val],
              color=:black, linestyle=:dash, label="Identity")
        final_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1200))
        savefig(final_plot, joinpath(output_dir, "summary_all.png"))
        # --- 5. Save Final Solution Data ---
        if save_data
            println("  Saving final solution data to NPZ...")
            # Use the subsampled points for a lightweight file
            npzwrite(joinpath(output_dir, "final_solution_Re$(Re_label).npz"), Dict(
                "x" => x_plot,
                "y" => y_plot,
                "u" => u_plot,
                "v" => v_plot,
                "p" => p_vals_plot
            ))
        end
    else
        p1 = scatter(x_plot, y_plot, zcolor=u_plot; c=:coolwarm, title="u-velocity (Re=$Re_label)", common_style_fields...)
        p2 = scatter(x_plot, y_plot, zcolor=v_plot; c=:coolwarm, title="v-velocity (Re=$Re_label)", common_style_fields...)
        p3 = scatter(x_plot, y_plot, zcolor=p_vals_plot; c=:viridis, title="Pressure (Re=$Re_label)", common_style_fields...)
        p4 = scatter(x_plot, y_plot, zcolor=u_mag_plot; c=:turbo, title="Magnitude (Re=$Re_label)", common_style_fields...)
        final_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1200))
        if Re_label == @sprintf("%.0f", Re_final)
            savefig(final_plot, joinpath(output_dir, "summary_all.png"))
        else
            savefig(final_plot, joinpath(output_dir, "summary_fields_Re$(Re_label).png"))
        end
    end
    println("  Plot saved (sampled $(length(x_plot)) points).")
    flush(stdout)
end
function plot_convergence(errors, n_cells_hist, output_dir, Re_label)
    # Ensure arrays are strictly Float64 for plotting
    errs = Float64.(errors)
    cells = Int.(n_cells_hist)
    p1 = plot(1:length(errs), errs,
              marker=:circle, linewidth=2,
              xlabel="AMR Iteration", ylabel="Total Error η",
              title="Error Convergence (Re=$Re_label)", legend=false,
              yscale=:log10)
    p2 = plot(1:length(cells), cells,
              marker=:square, linewidth=2,
              xlabel="AMR Iteration", ylabel="Number of Cells",
              title="Mesh Refinement (Re=$Re_label)", legend=false)
    p3 = plot(cells, errs,
              marker=:circle, linewidth=2,
              xlabel="Number of Cells", ylabel="Total Error η",
              title="Error vs DOF (Re=$Re_label)", legend=false,
              xscale=:log10, yscale=:log10)
    convergence_plot = plot(p1, p2, p3, layout=(1,3), size=(1500, 400))
    savefig(convergence_plot, joinpath(output_dir, "convergence_Re$(Re_label).png"))
end
function plot_convergence(errors, n_cells_hist, output_dir, Re_label)
    p1 = plot(1:length(errors), errors,
              marker=:circle, linewidth=2,
              xlabel="AMR Iteration", ylabel="Total Error η",
              title="Error Convergence (Re=$Re_label)", legend=false,
              yscale=:log10)
    p2 = plot(1:length(n_cells_hist), n_cells_hist,
              marker=:square, linewidth=2,
              xlabel="AMR Iteration", ylabel="Number of Cells",
              title="Mesh Refinement (Re=$Re_label)", legend=false)
    p3 = plot(n_cells_hist, errors,
              marker=:circle, linewidth=2,
              xlabel="Number of Cells", ylabel="Total Error η",
              title="Error vs DOF (Re=$Re_label)", legend=false,
              xscale=:log10, yscale=:log10)
    convergence_plot = plot(p1, p2, p3, layout=(1,3), size=(1500, 400))
    savefig(convergence_plot, joinpath(output_dir, "convergence_Re$(Re_label).png"))
end
function plot_continuation_summary(continuation_data, Re_values, output_dir)
    # Extract data
    final_cells = [continuation_data[Re]["final_cells"] for Re in Re_values]
    final_errors = [continuation_data[Re]["errors"][end] for Re in Re_values]
    # Plot 1: Final mesh size vs Re
    p1 = plot(Re_values, final_cells,
              marker=:circle, linewidth=2, markersize=6,
              xlabel="Reynolds Number", ylabel="Final Number of Cells",
              title="Mesh Growth with Re", legend=false,
              guidefontsize=10, tickfontsize=9)
    # Plot 2: Final error vs Re
    p2 = plot(Re_values, final_errors,
              marker=:square, linewidth=2, markersize=6,
              xlabel="Reynolds Number", ylabel="Final Total Error η",
              title="Error vs Re", legend=false,
              yscale=:log10,
              guidefontsize=10, tickfontsize=9)
    # Plot 3: Cells needed to achieve certain error level
    p3 = plot(final_cells, final_errors,
              marker=:diamond, linewidth=2, markersize=6,
              xlabel="Number of Cells", ylabel="Total Error η",
              title="Error vs Mesh Size", legend=false,
              xscale=:log10, yscale=:log10,
              guidefontsize=10, tickfontsize=9)
    summary_plot = plot(p1, p2, p3, layout=(1,3), size=(1500, 400))
    savefig(summary_plot, joinpath(output_dir, "continuation_summary.png"))
    println("\n✓ Continuation summary plots saved")
end
# %% [markdown]


# ## Cell 9: The AMR Iteration Controller (Single Re)
# 
# ## Methodology: Iterative Error Reduction
# This function orchestrates the inner refinement loop for a fixed Reynolds number. It generates a sequence of nested discrete models $\mathcal{T}_0 \subset \mathcal{T}_1 \subset \dots \subset \mathcal{T}_N$ by repeatedly invoking the AFEM step (Cell 7).
# 
# ### 1. The Refinement Loop
# * **Iterative Process**: For $i = 1 \dots N_{steps}$:
#     * **Step**: Call `amr_ldc_step` (Cell 7) to Solve $\to$ Estimate $\to$ Mark $\to$ Refine.
#     * **State Update**: The mesh model is updated in-place (`model = fmodel`). The solution $(\mathbf{u}_h, p_h)$ serves as the initial guess for the next iteration (optional continuation).
#     * **Metric Tracking**: We record the global error functional $\eta(\mathcal{T}_i)$ and the Degree of Freedom count (DOFs) to monitor the convergence rate.
# 
# ### 2. Visual Diagnostics (VTK)
# * **Trigger**: At the final iteration ($i = N_{steps}$).
# * **Artifact**: Writes a **Paraview (.vtu)** file containing the unstructured mesh and fields.
# * **Fields Exported**:
#     * `uh`, `ph`: Primal solution fields.
#     * `eta`: The element-wise error indicator map (visualizes *where* the solver is struggling).
# 
# ### 3. Convergence Analysis
# * **Objective**: We expect optimal convergence, where the error decays algebraically with the number of degrees of freedom: $\|\mathbf{u} - \mathbf{u}_h\| \sim N_{DOF}^{-1/2}$ (in 2D).
# * **Action**: Calls `plot_convergence` to visualize this decay rate, providing a "health check" for the adaptive strategy.

# %%
function run_amr_ldc_single_re(model, Re, nsteps, order, output_dir, Re_label,Re_final,save_vtk=true)
    println("\n" * "="^80)
    println("AMR for Re = $Re")
    println("Starting mesh: $(num_cells(model)) cells, AMR steps = $nsteps")
    println("="^80)
    
    # Storage for results
    errors = Float64[]
    n_cells_hist = Int[]
    
    # 1. Declare variables outside the loop so they persist!
    local uh, ph
    local fmodel = model # Initialize fmodel
    
    u_prev = nothing
    
    # 2. Run AMR Loop (if steps > 0)
    if nsteps > 0
        for i in 1:nsteps
            println("\n" * "-"^80)
            println("Re = $Re, AMR Iteration $i")
            println("-"^80)
            
            # Perform AMR step (defines uh, ph for this iteration)
            fmodel, uh, ph, η, I, total_error = amr_ldc_step(model; order=order, Re=Re, u_prev=u_prev)
            
            push!(errors, total_error)
            push!(n_cells_hist, num_cells(model))
            if i ==nsteps && Re==Re_final && save_vtk
            
                # Write output for visualization
                Ω = Triangulation(model)
                writevtk(
                    Ω, joinpath(output_dir, "ldc_Re$(Re_label)_step_$(i-1)"),
                    cellfields = [
                        "uh" => uh,
                        "ph" => ph,
                        "eta" => CellField(η, Ω),
                    ],
                )
            end
            # Update for next iteration
            u_prev = (uh, ph)
            model = fmodel
        end

    else
        # 3. Handle case where nsteps = 0 (Just solve, no refine loop)
        # We reuse amr_ldc_step but ignore the refinement output to get uh, ph
        println("No AMR steps. Solving on current mesh...")
        fmodel, uh, ph, η, I, total_error = amr_ldc_step(model; order=order, Re=Re, u_prev=nothing)
    end
    
    # 4. Plot convergence (only if we actually ran steps)
    if !isempty(errors)
        plot_convergence(errors, n_cells_hist, output_dir, Re_label)
    end
    
    println("\n" * "="^80)
    println("Re = $Re complete!")
    println("Final mesh: $(num_cells(model)) cells")
    println("="^80)
    
    return model, uh, ph, errors, n_cells_hist
end

# %% [markdown]

# ## Cell 10: The Outer Continuation Loop (Reynolds Stepping)
# 
# ## Methodology: Parameter Continuation (Homotopy)
# Solving the high-Reynolds Navier-Stokes equations directly is difficult because the non-linear convection term dominates the viscous stabilization, often causing Newton solver divergence.
# 
# To overcome this, we employ a **Reynolds Continuation Strategy**:
# $$
# Re_0 \to Re_1 \to \dots \to Re_{target}
# $$
# We solve a sequence of problems $\mathcal{P}(Re_i)$, passing the knowledge gained from the previous step to the next.
# 
# ### 1. Topological Inheritance (Mesh Continuation)
# * **Mechanism**: The optimized mesh $\mathcal{T}_{final}^{(Re_i)}$ generated for Reynolds number $Re_i$ is passed as the *initial* mesh for the next step $Re_{i+1}$.
# * **Physics Motivation**: Features like boundary layers and corner vortices develop gradually. By inheriting the refined mesh, the solver at $Re_{i+1}$ starts with a grid that already resolves the primary flow structures, significantly reducing the number of AMR cycles required (Warm Start).
# 
# ### 2. The  Integration
# * **Step Execution**: Calls `run_amr_ldc_single_re` (Cell 9) for each parameter step.
# * **Observation Generation**:
#     * **Intermediate**: Plots
#     * **Terminal**: Saves `final_data.npz` and `summary_all.png` at the final target Re.
# * **Aggregated Memory**: The `continuation_data` dictionary accumulates the history of error decay and mesh growth across the entire experiment, allowing the Advisor to evaluate the global efficiency of the strategy.
# 

# %%
function run_reynolds_continuation(;
    Re_values = [100, 300, 500, 700, 1000],
    n_initial=10,
    nsteps_per_Re=5,
    order=2,
    output_dir=nothing,
    save_vtk=true,
)
    println("="^80)
    println("REYNOLDS NUMBER CONTINUATION: LDC AMR")
    println("="^80)
    
    if output_dir === nothing
        output_dir = @__DIR__
    end
    println("Output directory: $output_dir")
    # Create initial coarse mesh
    model = generate_initial_mesh(n_initial)
    
    continuation_data = Dict{Float64, Any}()
    
    for (idx, Re) in enumerate(Re_values)
        Re_label = @sprintf("%.0f", Re)
        
        println("\n" * "█"^80)
        println("█  CONTINUATION STEP $idx/$(length(Re_values)): Re = $Re")
        println("█"^80)
        
        # Run AMR (uh and ph are now guaranteed to be defined)
        model, uh, ph, errors, n_cells_hist = run_amr_ldc_single_re(
            model, Re, nsteps_per_Re, order, output_dir, Re_label,Re_values[end],save_vtk
        )
        
        # --- INSERTED: Save the Field Plot for this Jump ---
        plot_and_save_final_solution(model, uh, ph, output_dir, Re_label,Re_values[end])  
        # Store results
        continuation_data[Re] = Dict(
            "errors" => errors,
            "n_cells" => n_cells_hist,
            "final_cells" => num_cells(model)
        )
        
        println("\n✓ Re = $Re complete. Mesh with $(num_cells(model)) cells passed to next step.\n")
    end
    
    plot_continuation_summary(continuation_data, Re_values, output_dir)    
    return model, continuation_data
end


# %% [markdown]

# ## Cell 11: Experiment Configuration & Execution
#
# ## Methodology: Discrete Homotopy Path Definition
# Defines the specific parameter trajectory $\Lambda = \{Re_0, Re_1, \dots, Re_{target}\}$ for the continuation solver.
#
# ### 1. Path Construction strategy
# Instead of uniform stepping, we define the Reynolds path using a coefficient vector $\mathbf{c} \in [0, 1]^N$:
# $$
# Re_i = Re_{start} + c_i (Re_{final} - Re_{start})
# $$
# * **Configuration**: `coefficients = [0.0, 0.5, 0.8, 1.0]`
# * **Physics Heuristic (Adaptive Stepping)**:
#     * **Low Re ($c=0.0 \to 0.5$)**: The flow is diffusion-dominated (stable). We can take large strides ($\Delta Re = 200$).
#     * **High Re ($c=0.8 \to 1.0$)**: The flow becomes convection-dominated, and the Jacobian condition number degrades. We reduce the step size ($\Delta Re = 80$) to ensure the Newton solver remains within its radius of convergence.
#
# ### 2. Execution
# Launches the `run_reynolds_continuation` orchestrator.
# * **`n_initial=10`**: Starts with a very coarse $10 \times 10$ grid (100 cells).
# * **`nsteps_per_Re=3`**: Allows 3 AMR adaptation cycles per Reynolds step to resolve new features before advancing the parameter.
# %%
Re_start=100.0
Re_final=525.0
coefficients = [0.0, 0.5,0.75, 1.0]
ΔR = Re_final - Re_start
Re_values = [Re_start + c * ΔR for c in coefficients]
model, data = run_reynolds_continuation(
    Re_values = Re_values,
    n_initial=10,
    nsteps_per_Re=3,
    order=2,
    save_vtk=true,
)