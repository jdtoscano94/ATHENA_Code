%% High-Accuracy Solution of the Allen-Cahn Equation with Dirichlet Boundary Conditions

% Parameters
L = 1;          % Domain from -L to L
N = 256;        % Number of Chebyshev points (increase for higher accuracy)
D_coef = 0.0001;% Diffusion coefficient
tspan = [0, 1]; % Time interval
steps = 200;    % Number of output time steps

% Generate Chebyshev differentiation matrices
[D_cheb, x_cheb] = cheb(N);   % x_cheb in [-1,1]
x = L * x_cheb;               % Scale x to [-L, L]
D_cheb = D_cheb / L;          % Adjust differentiation matrix for scaled x
D2 = D_cheb^2;                % Second derivative matrix

% Transformation: u(x,t) = (1 - x^2) * v(x,t) - 1
% Since (1 - x^2) = 0 at x = -1 and x = 1, u(-1,t) = u(1,t) = -1

% Compute the transformed initial condition v0
u0_full = x.^2 .* cos(pi * x / L);  % Original initial condition
u0 = (u0_full + 1) ./ (1 - x.^2);   % Compute v0 from u0

% Handle division by zero at x = ±1
u0(1) = 0;          % Since (1 - x^2) = 0 at x = -1, set v0(-1) = 0
u0(end) = 0;        % Similarly at x = 1

% Time vector for output
t_output = linspace(tspan(1), tspan(2), steps+1);

% Define the right-hand side function for ODE solver
rhs = @(t, v) compute_rhs(v, D_cheb, D2, x, D_coef);

% ODE solver options for high accuracy
opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-12);

% Solve the ODE system using ode15s
[t_sol, v_sol] = ode15s(rhs, t_output, u0, opts);

% Reconstruct the solution u(x,t) from v(x,t)
u_full = zeros(length(t_sol), N+1);
for k = 1:length(t_sol)
    v = v_sol(k, :)';
    u = (1 - x.^2) .* v - 1;
    u_full(k, :) = u';
end

% Plot the solution
figure;
[t_grid, x_grid] = meshgrid(t_sol, x);
pcolor(t_grid, x_grid, u_full');
shading interp; axis tight; colormap(jet);
xlabel('Time t');
ylabel('Spatial coordinate x');
title('High-Accuracy Solution of the Allen-Cahn Equation with Dirichlet Boundary Conditions');

% Save the solution
save('allen_cahn_dirichlet.mat', 't_sol', 'x', 'u_full');

%% Function to compute RHS
function dvdt = compute_rhs(v, D_cheb, D2, x, D_coef)
    N = length(x) - 1;
    % Compute u from v
    u = (1 - x.^2) .* v - 1;
    % Compute first and second derivatives of u
    dudx = D_cheb * u;
    d2udx2 = D2 * u;
    % Compute the RHS of the PDE
    dvdt = (1 ./ (1 - x.^2)) .* (D_coef * d2udx2 + 5 * u - 5 * u.^3);
    % Handle singularities at x = ±1
    dvdt(1) = 0;
    dvdt(end) = 0;
end

%% Chebyshev Differentiation Matrix Function
% Place this function at the end of your script or in a separate file named 'cheb.m'

function [D, x] = cheb(N)
    if N == 0
        D = 0; x = 1;
        return;
    end
    x = cos(pi * (0:N)' / N);
    c = [2; ones(N-1,1); 2] .* (-1).^(0:N)';
    X = repmat(x, 1, N+1);
    dX = X - X';
    D = (c * (1 ./ c)') ./ (dX + (eye(N+1)));
    D = D - diag(sum(D, 2));
end
