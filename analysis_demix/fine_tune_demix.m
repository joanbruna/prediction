
function [x1,x2] = fine_tune_demix(x,D1,D2,fprop,bprop,options)


% Initial conditions
x1 = getoptions(options,'x1_init',zeros(size(x)));
x2 = getoptions(options,'x2_init',zeros(size(x)));


for i=1:niter
    
    % gradient descent on x1
    x1_aux = x1 - rho*bprop(x,D1*z1,x1);
    
    % gradient descent on x2
    x2_aux = x2 - rho*bprop(x,D2*z2,x2);
    
    P1 = fprop(x1);

end




