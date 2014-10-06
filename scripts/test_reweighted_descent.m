% first make some data
n = 20;
m = 10;
r = 12;



% cost function parameters
lambda1 = 0.1;
lambda2 = 0.01;

lambda1gn = 0.1;
lambda2gn = 0.01;

%param = struct('mode',2,'lambda',lambda1,'lambda2',lambda2,'pos',true);
eps = 1e-6;

param0.posAlpha = 1;
param0.posD = 1;
param0.pos = 1;
param0.lambda = lambda1gn;
param0.lambda2 = lambda2gn;
param0.iter = 1000;


groupsize = 4;
t = 4;



D = rand(n,r);
V = rand(n,m);
% asumes overlap of half groupsize
Dgn = rand(r/(groupsize/2),t);

alpha0 = rand(r,m);
alpha = alpha0;
step = 0.1;

for j=1:200
    
    %f = betadiv(V,D*lassoRes,beta);
    [f_aux,dalpha] = measure_bilevel_cost(alpha, D, Dgn, V, lambda1,lambda2, lambda1gn, lambda2gn, groupsize, 'dX');
    
    f(j) = f_aux;
    
    alpha = alpha - step*dalpha;
    
    if~mod(j,10)
        disp(['Iter: ' num2str(j)])
        norm(dalpha,'fro')
    end

end