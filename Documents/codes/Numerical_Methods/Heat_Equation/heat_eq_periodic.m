% --------------------------------------------------------------
%  Heat equation u_t = u_xx  on [0,1] with periodic BCs
%  Initial condition: u(x,0) = sin(2*pi*x)
%  Analytical solution: u(x,t) = exp(-(2*pi)^2*t) * sin(2*pi*x)
%  Time interval: t ∈ [0,1]
% --------------------------------------------------------------

% ----- discretisation parameters --------------------------------
nn    = 256;                 % number of spatial points (spectral grid)
steps = 400;                 % number of time snapshots (adjust as needed)

% ----- domain and grids -----------------------------------------
dom = [0 1];                 % spatial domain
x   = chebfun('x',dom);      % Chebfun variable on the domain
t   = linspace(0,1,steps+1); % time vector

% ----- spin operator set‑up -------------------------------------
S = spinop(dom,t);           % create spinop object (periodic by default)

% Linear part: pure diffusion
S.lin = @(u) diff(u,2);      % u_xx

% No nonlinear part for the heat equation
S.nonlin = [];               % (empty)

% Initial condition
S.init = sin(2*pi*x);        % u(x,0) = sin(2πx)

% ----- time integration -----------------------------------------
% The spin solver uses an adaptive time‑step internally; we only
% need to give a spatial resolution (nn) and a tolerance.
u = spin(S,nn,1e-5,'plot','off');   % u is a cell array of chebfuns

% ----- collect numerical solution into a matrix -----------------
usol_num = zeros(nn,steps+1);
for k = 1:steps+1
    usol_num(:,k) = u{k}.values;          % values on the spectral grid
end

% ----- analytical solution on the same grid --------------------
% Create spatial grid (including the endpoint for periodicity)
x_plot = linspace(0,1,nn+1);               % (nn+1) points, last = first
% Evaluate analytical solution at each (x,t)
usol_exact = zeros(nn+1,steps+1);
for k = 1:steps+1
    usol_exact(:,k) = exp(-(2*pi)^2 * t(k)) * sin(2*pi*x_plot).';
end

% ----- compute pointwise error ----------------------------------
err = abs(usol_exact - [usol_num; usol_num(1,:)]);   % (nn+1)×(steps+1)

% ----- visualisation of numerical solution ---------------------
figure;
pcolor(t,x_plot, [usol_num; usol_num(1,:)]); shading interp;
colormap(jet); colorbar;
xlabel('time t'); ylabel('space x');
title('Numerical solution of u_t = u_{xx} (periodic BCs)');
axis tight;

% ----- visualisation of analytical solution --------------------
figure;
pcolor(t,x_plot, usol_exact); shading interp;
colormap(jet); colorbar;
xlabel('time t'); ylabel('space x');
title('Analytical solution u(x,t) = e^{-(2\pi)^2 t} sin(2\pi x)');
axis tight;

% ----- visualisation of pointwise error ------------------------
figure;
pcolor(t,x_plot, err); shading interp;
colormap(parula); colorbar;
xlabel('time t'); ylabel('space x');
title('Pointwise absolute error |u_{num} - u_{exact}|');
caxis([0 max(err(:))]);   % scale colour axis to error range
axis tight;

% ----- optional: save data ---------------------------------------
save('heat_eq_periodic.mat','t','x_plot','usol_num','usol_exact','err','-double');