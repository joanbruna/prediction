function [out,dout] = measure_bilevel_cost(alpha, D, Dgn, data, lambda1,lambda2, lambda1gn, lambda2gn, groupsize,grad_type)

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
fp = log( p + eps);
%fp = 1./(p + eps);

c2 = lambda1*sum(fp.*pZ);
c3 = 0.5*lambda2*sum(alpha(:).^2)/batchsize;

sp = sum(modulus(:)>0)/numel(modulus);
out=c1+c2+c3;

if nargout>1
    
    switch grad_type
        case 'dX'
            
            dc1 = D'*(D*alpha - data);
            dc3 = lambda2* alpha;
            
            aux = fp.*(1./pZ);
            dc2_1 = lambda1*up_sample(aux,N,groupsize).*alpha;
            
            dfp = 1./(p+eps);
            %dfp = (p+eps).^(-2);
            
            id = find(Z2>0);
            Did = Dgn(:,id);
            
            A = (Did'*Did + lambda2gn*eye(length(id)))\Did';
            
            %Pr = eye(N/groupsize);
            
 
            B = up_sample(Did*A,N,groupsize);
            B = up_sample(B',N,groupsize)';
            
            dc2_2 = lambda1*up_sample(dfp,N,groupsize).*(B*alpha);
            
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


