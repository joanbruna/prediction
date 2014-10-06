function [out,dout] = measure_bilevel_cost_nonoverlap(alpha, D, Dgn, data, lambda1,lambda2, lambda1gn, lambda2gn, groupsize,grad_type)

batchsize = size(data,2);
N = size(alpha,1);
rec = D * alpha;
c1 = 0.5*norm(rec(:)-data(:))^2;

%pZ = modphas_decomp(alpha,groupsize);
pZ = down_sample(alpha,groupsize,1);


param0.posAlpha = 1;
param0.posD = 1;
param0.pos = 1;
param0.lambda = lambda1gn;
param0.lambda2 = lambda2gn;
param0.iter = 1000;
Z2 =  mexLasso(pZ,Dgn,param0);
p = Dgn * Z2;

% reweighting function
eps = 1e-2;
fp = 1./(p + eps);


% compute the multiplicative term
c2 = lambda1*sum(fp(:).*pZ(:));

c3 = 0.5*lambda2*sum(alpha(:).^2);

out=(c1+c2+c3)/batchsize;

if nargout>1
    
    switch grad_type
        case 'dX'
            
            dc1 = D'*(D*alpha - data);
            dc3 = lambda2* alpha;
            
            aux = zeros(size(fp));
            jj = pZ ~=0;
            aux(jj) = fp(jj).*(1./pZ(jj));
            dc2_1 = lambda1*up_sample(aux,N,groupsize).*alpha;
            
            %dfp = 1./(p+eps);
            dfp = -1./((p+eps).^2);
            
            aux2 = nmf_grad(Z2,pZ,dfp,Dgn,lambda2gn);  
            dc2_2 = lambda1*alpha.*up_sample(aux2,N,groupsize);

            dout = (dc1 + dc2_1 + dc2_2 + dc3)/batchsize;
            
        otherwise
            
            error(['Gradient ' grad_tpe ' is not a valid option'])
               
    end
end

end


function Zp = down_sample(Z,groupsize,time_groupsize)

box=ones(groupsize,time_groupsize);

Zp = sqrt(conv2(Z.^2,box,'same'));
Zp=Zp(groupsize/2:groupsize:end,1:time_groupsize:end);

end

function V = up_sample(p,M,groupsize)

if size(p,2)>1
    V = zeros(M,size(p,2));
    for i=1:size(p,2)
         V(:,i) = up_sample(p(:,i),M,groupsize);
    end
    return
end

T = repmat(p',groupsize,1);
V = reshape(T,M,1);
end



function out = nmf_grad(Z2,pZ,dfp,Dgn,lambda2gn)

N = size(Dgn,1);

if size(Z2,2)>1
    out = zeros(N,size(Z2,2));
    for i=1:size(Z2,2)
         out(:,i) = nmf_grad(Z2(:,i),pZ(:,i),dfp(:,i),Dgn,lambda2gn);
    end
    return
end


id = find(Z2>0);
Did = Dgn(:,id);

A = Did*((Did'*Did + lambda2gn*eye(length(id)))\Did');


B0 = A.*repmat(1./pZ',N,1);
out = B0'*(pZ.*dfp);

end
