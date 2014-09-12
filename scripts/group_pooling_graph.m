function [out,costout]= group_pooling_graph( D, T, X, options,t0)
%this function does the sparse inference on a collection of trees 
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

y = zeros(K,M);
out = y;

nmf=getoptions(options,'nmf', 0);
lambda = getoptions(options,'lambda',0.1);
tparam.lambda = t0 * lambda;% * (size(D,2)/K);
t=1;


for i=1:iters

	aux = y - t0*(Dsq * y - DX);
    
    if nmf
        aux = max(0,aux);
    end
    
    % compute proximal gradient
    newout = Proximal_tree(aux,options.indexes,options.indexes_inv, tparam.lambda);
    
    if fista
        newt = (1+ sqrt(1+4*t^2))/2;
        y = newout + ((t-1)/newt)*(newout-out);
        t=newt;
    end
	out=newout;
end


if nargout>1
    costout = cost(X,D,out,options.indexes, lambda);
end

end


function [obj,c1,c2] = cost(X,D,out,indexes,lambda)

K=size(D,2);

rec = D*out;
c1 = .5*norm(X(:) - rec(:)).^2;

S = length(indexes);
J = length(indexes{1});

c2 = zeros(S,J);

for i=1:S
    for j=1:J
    aux = out(indexes{i}{j});
    c2(i,j) = sum(sqrt(sum(aux.^2)));
    end
end


obj.tot = (c1 + lambda *sum(c2(:)) )/size(X,2);
obj.c1 = sqrt(2*c1)/norm(X(:));
obj.c2 = sum(c2(:))/size(X,2);


end

