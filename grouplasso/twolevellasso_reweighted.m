

function [Zout, Zgnout] = twolevellasso_reweighted(Xin, Din, Dgnin,Zin, options)


     
lambda = getoptions(options,'lambda',0.1);
lambdagn = getoptions(options,'lambdagn',0.1);

groupsize = getoptions(options,'groupsize',2);
time_groupsize = getoptions(options,'time_groupsize',2);

beta = getoptions(options,'beta',2e-1);
betagn = getoptions(options,'betagn',2e-1);

step = 0.01;

iter = 500;

f = zeros(iter,1);

Z = Zin;

for j=1:iter
    
    %f = betadiv(V,D*lassoRes,beta);
    [f_aux,dZ,Zgnout,pZout,cost] = measure_bilevel_cost(Z, Din, Dgnin, Xin, lambda,beta, lambdagn, betagn, groupsize,time_groupsize, 'dX');
    
    f(j) = f_aux;
    
    if j<iter
        Z = max(Z - step*dZ,0);
    end
    
    if~mod(j,10)
        disp(['Iter: ' num2str(j)])
        fprintf('totcost %4.2f [ %4.2f %4.2f %4.2f %4.2f ] \n', cost.total, cost.c1, cost.c2, cost.c3, cost.grad_norm)
    end

end

Zout = Z;