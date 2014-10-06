function grad = nmf_beta_grads(Y, X, dY, D, lambda2,beta,gradName)


if ~exist('gradName','var')
    gradName = 'dW';
end



if size(Y,2) > 1
    
    % Initialize grads
    switch gradName
        case 'dW'
            grad = zeros(size(D));
        case 'dM'
            grad = zeros(size(M));
        otherwise
            error('unknown gradient')
    end
    
    % compute the gradient for each data vector
    for k=1:size(Y,2)
        grad = grad + nmf_beta_grads(Y(:,k), X(:,k), dY(:,k), D, lambda2,beta,gradName);
    end
    
    return
end

if beta==2
eps = 0;
else
eps = 0.001;
end

% Find active set
id = find(Y ~=0);
lact = length(id);


Dd = D(:,id);

DY = Dd*Y(id)+eps;

phi = (DY - X).*(DY.^(beta-2));
if beta ==2
    A = eye(size(X,1));
else
    A = diag( (beta-1).* (DY.^(beta-2)) -  (beta-2)*(X .* (DY.^(beta-3))) );
end


bd = (Dd'*A*Dd + lambda2*eye(lact))\dY(id);

if isnan(sum(bd(:)))
   disp('a') 
   keyboard
end
% Gradient
switch gradName
    case 'dW'
        
        dD = zeros(size(D));
        dD(:,id) = -A*Dd*bd*Y(id)' -  phi*bd';
        grad = dD;
        
    otherwise
        error('unknown gradient')
end



