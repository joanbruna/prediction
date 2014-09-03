function [out,costout,z]= group_pooling_semisup( D, X, options,t0)

%this is where I need to do all the changes
%reshape input, redefine the groups, apply the FISTA algo, 
%and then reshape again to produce the corresponding Aout,Bouts, alphas


costout=0;

fista=getoptions(options,'fista',1);
iters=getoptions(options,'iters',50);
tau = getoptions(options,'tau',0.1);


[N,M]=size(X);
K=size(D,2);
Dsq=D'*D;
DX = D'*X;

semisup=getoptions(options,'semisup',0);
if semisup
    W=getoptions(options,'W',zeros(N,1));
    Kw = size(W,2);
    z = zeros(Kw,M);
    WX = W'*X;
    DW = D'*W;
    WD = W'*D;
    Wsq = W'*W;
else
    z = [];
    Kw = 0;
end


if ~exist('t0','var')
    t0 = getoptions(options,'alpha_step',0.25);
    if semisup
        t0 = t0 * (1/(norm(D,2)^2 +norm(W,2)^2 + tau^2));
    else
        t0 = t0 * (1/max(svd(D))^2);
    end
end
t0 = t0 / options.time_groupsize;


% Use seed for the coefficients
H = getoptions(options,'H',[]);
if isempty(H)
    y = zeros(K,M);
else
    if size(H,1)~= K+Kw
       error('Size of H do not match size of D and Wn');
    end
    y = H(1:K,:);
    z = H(K+1:end,:);
end
out = y;

nmf=getoptions(options,'nmf', 0);

tparam.regul='group-lasso-l2';
lambda = getoptions(options,'lambda',0.1);
%tparam.regul='l1';
groupsize = getoptions(options,'groupsize',2);
time_groupsize = getoptions(options,'time_groupsize',2);
%keyboard
tparam.lambda = t0 * lambda;% * (size(D,2)/K);
t=1;

v = getoptions(options,'v',[0,0]);
[indexes,indexes_inv] = getGroupIndexes(K,M,groupsize,time_groupsize,v);



for i=1:iters

	aux = y - t0*(Dsq * y - DX);
    
    if semisup
        aux = aux - t0*DW*z;
        z = z - t0*(Wsq * z + WD*y - WX + tau*z);
    end
    
    if nmf
        aux = max(0,aux);
        if semisup
            z = max(0,z);
        end
    end
    
    % compute proximal gradient
    newout = Proximal_group(aux,indexes,indexes_inv, tparam.lambda);
    
    if fista
        newt = (1+ sqrt(1+4*t^2))/2;
        y = newout + ((t-1)/newt)*(newout-out);
        t=newt;
    end
	out=newout;
    
    %c(i) = cost(X,D,out,indexes, lambda,W,z,tau);
    
end


if nargout>1

if ~exist('W','var')
   W = 0;
   z = 0;
   tau =0;
end

    costout = cost(X,D,out,indexes, lambda,W,z,tau);
end

end


function [obj,c1,c2] = cost(X,D,out,indexes, lambda,W,z,tau)


aux = W*z;

K=size(D,2);

rec = D*out;
c1 = .5*norm(X(:) - aux(:) - rec(:)).^2;

S = length(indexes);

c2 = zeros(1,S);

for i=1:S
    aux = out(indexes{i});
    c2(i) = sum(sqrt(sum(aux.^2)));
end

c3 =  tau*0.5*sum(z(:).*z(:));

obj.tot = (c1 + lambda *sum(c2) + c3)/size(X,2);
obj.c1 = c1/size(X,2);
obj.c2 = sum(c2)/size(X,2);
obj.c3 = c3/size(X,2);


end

