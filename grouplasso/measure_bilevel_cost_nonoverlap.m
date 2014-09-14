function [out,dout] = measure_bilevel_cost_nonoverlap(alpha, D, Dgn, data, lambda1,lambda2, lambda1gn, lambda2gn, groupsize,grad_type)

[~,batchsize] = size(data);
N = size(alpha,1);
rec = D * alpha;
modulus = modphas_decomp(alpha,groupsize);
c1 = 0.5*norm(rec(:)-data(:))^2/batchsize;
pZ = modulus;


param0.posAlpha = 1;
param0.posD = 1;
param0.pos = 1;
param0.lambda = lambda1gn;
param0.lambda2 = lambda2gn;
param0.iter = 1000;
Z2 =  mexLasso(pZ,Dgn,param0);
p = Dgn * Z2;

% reweighting function
eps = 1e-4;
fp = 1./(p + eps);


% compute the multiplicative term
c2 = lambda1*sum(fp.*pZ);

c3 = 0.5*lambda2*sum(alpha(:).^2)/batchsize;


out=c1+c2+c3;

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
            
            id = find(Z2>0);
            Did = Dgn(:,id);
            
            A = Did*((Did'*Did + lambda2gn*eye(length(id)))\Did');

            
            B0 = A.*repmat(1./pZ',size(A,1),1);
            aux2 = B0'*(pZ.*dfp);
            
            
            dc2_2 = lambda1*alpha.*up_sample(aux2,N,groupsize);

            
            dout = dc1 + dc2_1 + dc2_2 + dc3;
            
        otherwise
            
            error(['Gradient ' grad_tpe ' is not a valid option'])
               
    end
end

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