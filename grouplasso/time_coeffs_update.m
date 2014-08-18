function [out,costout]= time_coeffs_update( D, X, options, t0)

%this is where I need to do all the changes
%reshape input, redefine the groups, apply the FISTA algo, 
%and then reshape again to produce the corresponding Aout,Bouts, alphas


costout=0;

iters=getoptions(options,'alpha_iters',50);
iters_encoder=getoptions(options,'alpha_iters_encoder',60);
overlapping=getoptions(options,'overlapping',1);
nmf=getoptions(options,'nmf', 0);
lambda = getoptions(options,'lambda',0.1);
groupsize = getoptions(options,'groupsize',2);
time_groupsize = getoptions(options,'time_groupsize',2);

if ~exist('t0','var')
    t0 = getoptions(options,'alpha_step',0.25);
    t0 = t0 * (1/max(svd(D))^2);
end
t0 = t0 / time_groupsize;


[~,M]=size(X);
K=size(D,2);
Dsq=D'*D;
DX = D'*X;
y = zeros(K,M);

out = y;


tparam.regul='group-lasso-l2';
%keyboard
tparam.lambda = t0 * lambda;% * (size(D,2)/K);
t=1;

v = getoptions(options,'v',[0,0]);
[indexes,indexes_inv] = getGroupIndexes(K,M,groupsize,time_groupsize,v);


if overlapping


count = 1;
for i=1:iters
%	if verb
%	fprintf('it %d \n',i)
%	end
	aux = y - t0*(Dsq * y - DX);
	if nmf
	aux = max(0,aux);
    end
    

    newout = Proximal_group(aux,indexes,indexes_inv, 2*tparam.lambda);

	%newout = newout2;
	newt = (1+ sqrt(1+4*t^2))/2;
	y = newout + ((t-1)/newt)*(newout-out);
	out=newout;
	t=newt;
    
%     if ~mod(i,5)
%     obj(count) = cost(X,D,out,indexes, tparam.lambda);
%     count = count+1;
%     end

    
end

if nargout>1
    costout = cost(X,D,out,indexes, tparam.lambda);
end

else


for i=1:iters
	tempo = reshape(Dsq * reshape(y,K,M),KK,MM);
	aux = y - t0*(tempo - DX);
	newout = ProximalFlat(aux, I0, I1, tparam.lambda,ss,rr);
	newt = (1+ sqrt(1+4*t^2))/2;
	y = newout + ((t-1)/newt)*(newout-out);
	out=newout;
	t=newt;
end

end



end


function [obj,c1,c2] = cost(X,D,out,indexes, lambda)

K=size(D,2);

rec = D * reshape(out,K,numel(out)/K);
c1 = .5*norm(X(:)-rec(:)).^2;

S = length(indexes);

c2 = zeros(1,S);

for i=1:S
    aux = out(indexes{i});
    c2(i) = sum(sqrt(sum(aux.^2)));
end

obj = (c1 + lambda *sum(c2))/size(X,2);
end

