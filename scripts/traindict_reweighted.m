
if ~exist('X1','var')
    load '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s31.mat'
    X1 = Xc;
    clear Xc;
    
    epsilon = 1;
    
    load '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s14.mat'
    X2 = Xc;
    
    X = [X1 X2];
    
    
    %renormalize data: whiten each frequency component.
    eps=4e-1;
    stds = std(X,0,2) + eps;
    avenorm = mean(sqrt(sum(X.^2)));
    clear X
    
    
    X1 = X1./repmat(stds,1,size(X1,2));
    X1 = X1/avenorm;
    
    
    X2 = X2./repmat(stds,1,size(X2,2));
    X2 = X2/avenorm;
    
    
    
end



%%

load('/misc/vlgscratch3/LecunGroup/bruna/grid_data/pooled_dictionaries_speaker31.mat');
D = double(D);
Dgn=double(Dbis(1:end-1,:));
K = size(D,2);


lambda=1e-3;
beta=2e-2;
groupsize=4;
time_groupsize=4;
nu=0.2;
betagn=0.1;
lambdagn = 0.1;


step = 0.01;

iter = 1500;

f = zeros(iter,1);

param_nmf.pos = 1;
param_nmf.lambda = 0.01;
param_nmf.lambda2 = 0;
param_nmf.iter = 1000;

Xt = X1(:,1:5000);

Z =  full(mexLasso(Xt,D,param_nmf));

Zin = Z;
Din = D;
Dgnin = Dgn;


for j=1:iter
    
    %f = betadiv(V,D*lassoRes,beta);
    [f_aux,dZ,Zgnout,pZout,cost] = measure_bilevel_cost(Z, D, Dgn, Xt, lambda,beta, lambdagn, betagn, groupsize,time_groupsize, 'dX');

    
    if j<iter
        Z = max(Z - step*dZ,0);
    end
    
    if~mod(j,10)
        disp(['Iter: ' num2str(j)])
        fprintf('Z -> totcost %4.2f [ %4.2f %4.2f %4.2f %4.2f ] \n', cost.total, cost.c1, cost.c2, cost.c3, cost.grad_norm)
    end
    

    % Update D
    [f_aux,dD,Zgnout,pZout,cost] = measure_bilevel_cost(Z, D, Dgn, Xt, lambda,beta, lambdagn, betagn, groupsize,time_groupsize, 'dD');

    
    if j<iter
        D = mexNormalize(max(D - step*dD,0));
    end
    
    if~mod(j,10)
        fprintf('D -> totcost %4.2f [ %4.2f %4.2f %4.2f %4.2f ] \n', cost.total, cost.c1, cost.c2, cost.c3, cost.grad_norm)
    end
    

    % Update Dgn
    [f,dDgn,Zgnout,pZout,cost] = measure_bilevel_cost(Z, D, Dgn, Xt, lambda,beta, lambdagn, betagn, groupsize,time_groupsize, 'dDgn');

    
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


D1 = D;
D1gn = Dgn;


%%

load('/misc/vlgscratch3/LecunGroup/bruna/grid_data/pooled_dictionaries_speaker14.mat');
D = double(D);
Dgn=double(Dbis(1:end-1,:));
K = size(D,2);

Xt = X2(:,1:5000);

Z =  full(mexLasso(Xt,D,param_nmf));

Zin = Z;
D2in = D;
D2gnin = Dgn;



for j=1:iter
    
    %f = betadiv(V,D*lassoRes,beta);
    [f_aux,dZ,Zgnout,pZout,cost] = measure_bilevel_cost(Z, D, Dgn, Xt, lambda,beta, lambdagn, betagn, groupsize,time_groupsize, 'dX');

    
    if j<iter
        Z = max(Z - step*dZ,0);
    end
    
    if~mod(j,10)
        disp(['Iter: ' num2str(j)])
        fprintf('Z -> totcost %4.2f [ %4.2f %4.2f %4.2f %4.2f ] \n', cost.total, cost.c1, cost.c2, cost.c3, cost.grad_norm)
    end
    

    % Update D
    [f_aux,dD,Zgnout,pZout,cost] = measure_bilevel_cost(Z, D, Dgn, Xt, lambda,beta, lambdagn, betagn, groupsize,time_groupsize, 'dD');

    
    if j<iter
        D = mexNormalize(max(D - step*dD,0));
    end
    
    if~mod(j,10)
        fprintf('D -> totcost %4.2f [ %4.2f %4.2f %4.2f %4.2f ] \n', cost.total, cost.c1, cost.c2, cost.c3, cost.grad_norm)
    end
    

    % Update Dgn
    [f,dDgn,Zgnout,pZout,cost] = measure_bilevel_cost(Z, D, Dgn, Xt, lambda,beta, lambdagn, betagn, groupsize,time_groupsize, 'dDgn');

    
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






