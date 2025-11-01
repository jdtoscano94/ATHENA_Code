nn = 511;
steps = 500;

dom = [0 1]; x = chebfun('x',dom); t = linspace(0,1,steps+1);
S = spinop(dom,t);
S.lin    = @(u) + 0.001*diff(u,2);          % ν u_xx
S.nonlin = @(u) - 0.5*diff(u.^2);  
S.init = sin(2*pi*x);
dt=1e-6;
u = spin(S,nn,1e-5,'plot','off');

usol = zeros(nn,steps+1);
for i = 1:steps+1
    usol(:,i) = u{i}.values;
end

x = linspace(-1,1,nn+1);
usol = [usol;usol(1,:)];
pcolor(t,x,usol); shading interp, axis tight, colormap(jet);
usol = usol'; % shape = (steps+1, nn+1)
save('Burgers.mat', 't', 'x', 'usol', '-double')
