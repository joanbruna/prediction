
function [Dout, Dgnout,Din,Dgnin] = twolevelDL_reweighted(X, options)


%D = double(D);
D = getoptions(options,'D',[]);
Dng = getoptions(options,'Dng',[]);



lambda = getoptions(options,'D',1e-3);
beta = getoptions(options,'beta',1e-2);
groupsize = getoptions(options,'groupsize',4);
time_groupsize = getoptions(options,'time_groupsize',2);
betagn = getoptions(options,'betagn',1e-1);
lambdagn = getoptions(options,'lambdagn',0.1);
nu = getoptions(options,'nu',0.2);

step = getoptions(options,'step',0.01);
iter = getoptions(options,'iter',1500);


[N,M]=size(X);
K = getoptions(options, 'K', 2*N);
Kgn = getoptions(options, 'Kgn', K/2);

f = zeros(iter,1);


param_nmf.pos = 1;
param_nmf.iter = 300;
param_nmf.numThreads=16;
param_nmf.batchsize=512;
param_nmf.pos = 1;
param_nmf.lambda = lambda;
param_nmf.lambda2 = beta;

if isempty( D )
    param_nmf.K = K;
    param_nmf.posAlpha = 1;
    param_nmf.posD = 1;
    
    Dnmf1 = mexTrainDL(X,param_nmf);
    alpha= mexLasso(X,Dnmf1,param_nmf);
    Dnmf1s = sortDZ(Dnmf1,full(alpha)');
    
    options_init = options;
    options_init.initD = Dnmf1;
    options_init.nepochs = 1;
    [D, Dgn] = twolevelDL_gpu(abs(X), options_init);
else
    Z =  full(mexLasso(X,D,param_nmf));
end

%param.nu=0;
%param.alpha_step=1;
%param.gradient_descent=0;
%param.itersout=800;
options_init.itersout=800;
%[Z1dm, Z1gn1dm, Z2dm, Zgn2dm] = twolevellasso_gpu(abs(Xr), D1, Dgn1, D2, Dgn2, param);
[Z, Zgn] = twolevellasso_gpu(X, D, Dgn, options_init);

rec = D * Z;
c1 = 0.5*norm(rec(:)-X(:))^2;
fprintf('Initial reconstruction error: %4.2f\n', cost.total, c1)


Zin = Z;
Din = D;
Dgnin = Dgn;
Xt = X;

for j=1:iter
    
    %f = betadiv(V,D*lassoRes,beta);
    [~,dZ,~,~,cost] = measure_bilevel_cost(Z, D, Dgn, Xt, lambda,beta, lambdagn, betagn, groupsize,time_groupsize, 'dX');
    
    
    if j<iter
        Z = max(Z - step*dZ,0);
    end
    
    if~mod(j,10)
        disp(['Iter: ' num2str(j)])
        fprintf('Z -> totcost %4.2f [ %4.2f %4.2f %4.2f %4.2f ] \n', cost.total, cost.c1, cost.c2, cost.c3, cost.grad_norm)
    end
    

    % Update D
    [~,dD,~,~,cost] = measure_bilevel_cost(Z, D, Dgn, Xt, lambda,beta, lambdagn, betagn, groupsize,time_groupsize, 'dD');

    
    if j<iter
        D = mexNormalize(max(D - step*dD,0));
    end
    
    if~mod(j,10)
        fprintf('D -> totcost %4.2f [ %4.2f %4.2f %4.2f %4.2f ] \n', cost.total, cost.c1, cost.c2, cost.c3, cost.grad_norm)
    end
    

    % Update Dgn
    [f,dDgn,~,~,cost] = measure_bilevel_cost(Z, D, Dgn, Xt, lambda,beta, lambdagn, betagn, groupsize,time_groupsize, 'dDgn');

    
    if j<iter
        Dgn_ = mexNormalize(max(Dgn - step*dDgn/norm(dDgn(:)),0));
        f_ = measure_bilevel_cost(Z, D, Dgn_, Xt, lambda,beta, lambdagn, betagn, groupsize,time_groupsize, 'dDgn');
        
        if f_ < f
            Dgn = Dgn_;
        else
            disp('No Change')
        end
    end
    
    if~mod(j,10)
        fprintf('Dgn ->totcost %4.2f [ %4.2f %4.2f %4.2f %4.2f ] \n', cost.total, cost.c1, cost.c2, cost.c3, cost.grad_norm)
    end
    

end


Dgnout = Dgn;
Dout = D;