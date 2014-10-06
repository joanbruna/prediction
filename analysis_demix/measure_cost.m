
function [obj,c] = measure_cost(D1,D2,Z1,Z2,X1,X2,fprop,lambda,gamma,Xaux)


if nargin<9
    gamma = 1;
end
if nargin<10
    Xaux = X1;
end

NFFT = 1024;
hop = NFFT/2;


rec1 = fprop(X1) - D1*Z1;
rec2 = fprop(X2) - D2*Z2;

c(1) = 0.5*norm(rec1(:),'fro').^2;
c(2) = 0.5*norm(rec2(:),'fro').^2;

c(3) = sum(lambda.*Z1(:));
c(4) = sum(lambda.*Z2(:));


c(5) = gamma*0.5*norm(Xaux - X1,'fro')^2;

obj = sum(c);


