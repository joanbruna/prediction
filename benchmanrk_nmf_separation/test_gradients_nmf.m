% first make some data
n = 20;
m = 3;
r = 10;




tol = 0;
n_iter_max = 80000;


t = 2;

% Set a high-level cost function
loss_type = 'l2';
[lossFun,lossGrad_x,lossGrad_w] = GetLossFun_w(loss_type);


% choose divergence
beta = 1;

% cost function parameters
lambda_ast = 0;
lambda = 0;


eps = 1e-8;

for j=1:10
    
    D = abs(rand(n,r))+1;

    V = rand(n,m);
    M = eye(size(D,1));
    
    Y =  round(rand(t,m));
    
    % test function
    %[~, lassoRes] = nmf_beta(V, beta, lambda1, lambda2 , n_iter_max, tol, D, 0);
    H_ini = abs(randn(r,m)) + 1;
    E_ini = zeros(n,m);
    %[~,lassoRes]  = rnmf(V, beta, n_iter_max, tol, D, H_ini, E_ini, lambda_ast,lambda,0);
    rho = 1;
    [~, lassoRes] = nmf_admm(V, D, H_ini, beta, rho, 1:size(D,2));
    %lassoRes(lassoRes(:)<eps/2) = 0;
    
    V2 = rand(size(lassoRes));
    
    
    f = betadiv(V,D*lassoRes,beta);
    dG = betadiv_grad(V,D,lassoRes,beta,'dH');
    
    % Compute gradient
    dfD = nmf_beta_grads(lassoRes, V, dG, D, lambda_ast,beta,'dW');
    
    
    dfD = dfD + betadiv_grad(V,D,lassoRes,beta,'dW');

    dD = randn(size(D))*eps;
    D_ = D + dD;
    
    [~, lassoRes_] = nmf_admm(V, D, lassoRes, beta, rho, 1:size(D,2));

    f_ = betadiv(V,D_*lassoRes_,beta);
    
    
    [(f_-f) dfD(:)'*dD(:)] / eps
    pause
    
end
