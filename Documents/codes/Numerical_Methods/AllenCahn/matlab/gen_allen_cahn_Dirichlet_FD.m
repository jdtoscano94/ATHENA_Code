%% Allen-Cahn Equation with Dirichlet Boundary Conditions using pdepe

% Parameters
D = 0.0001;
L = 1;
tspan = linspace(0, 1, 201);
x = linspace(-L, L, 512);

% PDE coefficients
m = 0;  % Symmetry parameter (0 for slab geometry)

% Solve the PDE
sol = pdepe(m, @pde_fun, @ic_fun, @bc_fun, x, tspan);

% Extract the solution
usol = sol;

% Plot the solution
[t_grid, x_grid] = meshgrid(tspan, x);
pcolor(t_grid, x_grid, usol');
shading interp; axis tight; colormap(jet);
xlabel('Time t');
ylabel('Spatial coordinate x');
title('Allen-Cahn Equation with Dirichlet Boundary Conditions');

% Save the solution
save('allen_cahn_dirichlet.mat', 'tspan', 'x', 'usol');

% Function definitions must appear at the end of the script

% PDE function
function [c, f, s] = pde_fun(x, t, u, DuDx)
    D = 0.0001;  % Ensure D is accessible inside the function
    c = 1;
    f = D * DuDx;
    s = 5 * u - 5 * u.^3;
end

% Initial condition function
function u0 = ic_fun(x)
    u0 = x.^2 .* cos(pi * x);
end

% Boundary conditions function
function [pl, ql, pr, qr] = bc_fun(xl, ul, xr, ur, t)
    % Left boundary (x = -L)
    pl = ul + 1;  % u(-L, t) = -1
    ql = 0;       % Dirichlet condition

    % Right boundary (x = L)
    pr = ur + 1;  % u(L, t) = -1
    qr = 0;       % Dirichlet condition
end
