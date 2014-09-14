


% first make some data
n = 20;
m = 1;
r = 10;



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


groupsize = 2;
t = 4;

for j=1:10
    
    D = rand(n,r);
    V = rand(n,m);
    Dgn = rand(r/groupsize,t);
    
    alpha = rand(r,m);

    %f = betadiv(V,D*lassoRes,beta);
    [f,df] = measure_bilevel_cost_nonoverlap(alpha, D, Dgn, V, lambda1,lambda2, lambda1gn, lambda2gn, groupsize, 'dX');
    
    dalpha = eps*randn(size(alpha));
    alpha_ = alpha + dalpha;
    f_ = measure_bilevel_cost_nonoverlap(alpha_, D, Dgn, V, lambda1,lambda2, lambda1gn, lambda2gn, groupsize, 'dX');
    
    disp([f_ - f,df(:)'*dalpha(:)]/eps)
    
end

break




%% 

% first make some data
n = 20;
m = 1;
r = 10;



% cost function parameters
lambda1 = 0.1;
lambda2 = 0.01;

lambda1gn = 0.1;
lambda2gn = 0.01;

%param = struct('mode',2,'lambda',lambda1,'lambda2',lambda2,'pos',true);
eps = 1e-7;


param0.posAlpha = 1;
param0.posD = 1;
param0.pos = 1;
param0.lambda = lambda1gn;
param0.lambda2 = lambda2gn;
param0.iter = 1000;


groupsize = 2;
t = 4;

for j=1:10
    
    D = rand(n,r);
    V = rand(n,m);
    Dgn = rand(r/groupsize,t);
    
    alpha = rand(r,m);

    %f = betadiv(V,D*lassoRes,beta);
    [f,df] = measure_bilevel_cost(alpha, D, Dgn, V, lambda1,lambda2, lambda1gn, lambda2gn, groupsize, 'dX');
    
    dalpha = eps*rand(size(alpha));
    alpha_ = alpha + dalpha;
    f_ = measure_bilevel_cost(alpha_, D, Dgn, V, lambda1,lambda2, lambda1gn, lambda2gn, groupsize, 'dX');
    
    disp([f_ - f,df(:)'*dalpha(:)]/eps)
    
end

break

%%


% first make some data
n = 20;
m = 1;
r = 10;


eps = 1e-6;

D = rand(n,r)+0.1;
V = rand(n,m)  + eps;


lambda1gn = 0.1;
lambda2gn = 0.01;


param0.posAlpha = 1;
param0.posD = 1;
param0.pos = 1;
param0.lambda = lambda1gn;
param0.lambda2 = lambda2gn;
param0.iter = 1000;

alpha =  mexLasso(V,D,param0);

id = find(alpha>0);
Did = D(:,id);

alpha2 = (Did'*Did + lambda2gn*eye(length(id)))\(Did'*V - lambda1gn);


% change V
%dV = randn(size(V));
b = 1000*eps:100*eps:5e-1;

% finde a non-used index
for h = 1:size(D,2)
   if sum(id ==h) == 0
       ii = h;
       break
   end
end

clear H
H(1) = alpha(ii);

for i = 1:length(b)
dV = D(:,ii);
V_ = V + b(i)*dV;

alpha_ =  mexLasso(V_,D,param0);

H(i+1) = alpha_(ii);

end

break

