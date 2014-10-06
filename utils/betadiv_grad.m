function d = betadiv_grad(V,W,H,beta,gradName)

% beta-divergence
%
%   d = betadiv(A,B,beta)
% 
% - a \= 0,1
%   d(x|y) = ( x^a + (a-1)*y^a - a*x*y^(a-1) ) / (a*(a-1))
% - a = 1
%   d(x|y) = x(log(x)-log(y)) + (y-x)  KULLBACK-LEIBLER
% - a = 0
%   d(x|y) = x/y - log(x/y) -1         ITAKURA-SAITO


Vap = W*H+0.001;
switch gradName
    case 'dW'
        d = ((Vap - V).*(Vap.^(beta-2)))*H';
    case 'dH'
        d = W'*((Vap - V).*(Vap.^(beta-2)));
    otherwise
        error('unknown gradient')
end