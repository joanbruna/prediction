function [out1,out2,costout]= group_pooling_graph( D, T, X, options,t0)
%this function estimates two codes out1  out2 that minimize a group tree norm
%using spatio-temporal group lasso

costout=0;

fista=getoptions(options,'fista',1);
iters=getoptions(options,'iters',150);


[N,M]=size(X);
K=size(D,2);
Dsq=D'*D;
DX = D'*X;

if ~exist('t0','var')
    t0 = getoptions(options,'alpha_step',0.25);
        t0 = t0 * (1/max(svd(D))^2);
end
t0 = t0 / options.time_groupsize;
epsi = getoptions(options,'initmixnorm',1e-5);

y1 = epsi*abs(randn(K,M));
out1 = y1;
y2 = epsi*abs(randn(K,M));%zeros(K,M);
out2 = y2;

nmf=getoptions(options,'nmf', 0);
lambda = getoptions(options,'lambda',0.1);
tparam.lambda = t0 * lambda;% * (size(D,2)/K);
t=1;


for i=1:iters

	aux1 = y1 - t0*(Dsq * (y1+y2) - DX);
	aux2 = aux1 - y1 + y2;
    
    if nmf
        aux1 = max(0,aux1);
        aux2 = max(0,aux2);
    end
    
    % compute proximal gradient
    newout1 = Proximal_tree(aux1,options.indexes,options.indexes_inv, tparam.lambda);
    newout2 = Proximal_tree(aux2,options.indexes,options.indexes_inv, tparam.lambda);
    
    if fista
        newt = (1+ sqrt(1+4*t^2))/2;
        y1 = newout1 + ((t-1)/newt)*(newout1-out1);
        y2 = newout2 + ((t-1)/newt)*(newout2-out2);
        t=newt;
    end
	out1=newout1;
	out2=newout2;
end


if nargout>1
    costout = cost(X,D,out1,out2,options.indexes, lambda);
end

end


function [obj,c1,c2] = cost(X,D,out1,out2,indexes,lambda)

K=size(D,2);

rec = D*(out1+out2);
c1 = .5*norm(X(:) - rec(:)).^2;

S = length(indexes);
J = length(indexes{1});

c21 = zeros(S,J);
c22 = zeros(S,J);

for i=1:S
    for j=1:J
    aux = out1(indexes{i}{j});
    c21(i,j) = sum(sqrt(sum(aux.^2)));
    aux = out2(indexes{i}{j});
    c22(i,j) = sum(sqrt(sum(aux.^2)));
    end
end


obj.tot = (c1 + lambda *sum(c21(:)) + lambda * sum(c22(:)) )/size(X,2);
obj.c1 = sqrt(2*c1)/norm(X(:));
obj.c2 = (sum(c21(:))+sum(c22(:)))/size(X,2);


end

