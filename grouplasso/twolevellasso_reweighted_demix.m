
function [Z1out, Z1gnout,Z2out, Z2gnout,fp1,fp2] = twolevellasso_reweighted_demix(Xin, D1in, D1gnin,D2in, D2gnin,Z1in,Z2in, options)


     
lambda = getoptions(options,'lambda',0.1);
lambdagn = getoptions(options,'lambdagn',0.1);

groupsize = getoptions(options,'groupsize',2);
time_groupsize = getoptions(options,'time_groupsize',2);

beta = getoptions(options,'beta',2e-1);
betagn = getoptions(options,'betagn',2e-1);

step = 0.01;

iter = 500;

f = zeros(iter,1);

Z1 = Z1in;
Z2 = Z2in;


%L = time_groupsize - (size(Xin,2) - round(size(Xin,2)/time_groupsize)*time_groupsize);
%Xin = [Xin zeros(size(Xin,1),L)];

for j=1:iter
    
    %f = betadiv(V,D*lassoRes,beta);
    [~,dZ1,Z1gnout,fp1,cost1] = measure_bilevel_cost(Z1, D1in, D1gnin, Xin - D2in*Z2, lambda,beta, lambdagn, betagn, groupsize,time_groupsize, 'dX');
    
    if j<iter
        Z1 = max(Z1 - step*dZ1,0);
    end
    
        [~,dZ2,Z2gnout,fp2,cost2] = measure_bilevel_cost(Z2, D2in, D2gnin, Xin - D1in*Z1, lambda,beta, lambdagn, betagn, groupsize,time_groupsize, 'dX');
    

    if j<iter
        Z2 = max(Z2 - step*dZ2,0);
    end
    
    aux = Xin - D1in*Z1 - D2in*Z2;
    rec = 0.5*sum(aux(:).^2);
    
    c1 =cost1.c1+cost2.c1;
    c2 =cost1.c2+cost2.c2;
    c3 = cost1.c3+cost2.c3;
    
    f_aux = rec + c1 +c2+c3;
    f(j) = f_aux;
    
    
    
    if~mod(j,10)
        fprintf('Iter: %d, totcost %4.2f [ %4.2f %4.2f %4.2f %4.2f ] \n',j, f_aux, c1,c2, c3, cost1.grad_norm+cost2.grad_norm)
    end

end

Z1out = Z1;
Z2out = Z2;