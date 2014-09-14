function [out,dout] = measure_bilevel_cost(alpha, D, Dgn, data, lambda1,lambda2, lambda1gn, lambda2gn, groupsize,grad_type)

batchsize = size(data,2);
N = size(alpha,1);
rec = D * alpha;
c1 = 0.5*norm(rec(:)-data(:))^2;

%modulus = modphas_decomp(alpha,groupsize);
pZ = down_sample(alpha,groupsize,1);

keyboard
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
c2 = lambda1*sum(fp(:).*pZ(:));

c3 = 0.5*lambda2*sum(alpha(:).^2);

c1 = 0;
out=(c1+c2+c3)/batchsize;

if nargout>1
    
    switch grad_type
        case 'dX'
            
            dc1 = D'*(D*alpha - data);
            dc3 = lambda2* alpha;
            
            % we compute separate for each set of groups
            aux = zeros(size(fp));
            jj = pZ ~=0;
            aux(jj) = fp(jj).*(1./pZ(jj));
            daux = up_sample(aux(1:2:end,:),N,groupsize)+ circshift(up_sample(aux(2:2:end,:),N,groupsize),[groupsize/2,0]);
            dc2_1 = lambda1*daux.*alpha;
            
            %dfp = 1./(p+eps);
            dfp = -1./((p+eps).^2);
            
            aux2 = nmf_grad(Z2,pZ,dfp,Dgn,lambda2gn,groupsize);  
            dc2_2 = lambda1*alpha.*aux2;

            dc1 = 0;
            dout = (dc1 + dc2_1 + dc2_2 + dc3)/batchsize;
            
        otherwise
            
            error(['Gradient ' grad_tpe ' is not a valid option'])
               
    end
end

end


function Zp = down_sample(Z,groupsize,time_groupsize)

box=ones(groupsize,time_groupsize);
M = size(Z,1);

%Zp = sqrt(cconv(Z.^2,box,M));

Zpad = [Z(end-groupsize/2:end,:);Z];

if time_groupsize>1
   Zpad = [Zpad(:, end),Zpad];
end

Zp = sqrt(conv2(Zpad.^2,box,'valid'));

Zp=Zp(groupsize/2:groupsize/2:end,1:max(time_groupsize/2,1):end);
Zp = circshift(Zp,[-1,0]);

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



function out = nmf_grad(Z2,pZ,dfp,Dgn,lambda2gn,groupsize)

N = size(Dgn,1);
M = groupsize*N/2;

if size(Z2,2)>1
    out = zeros(M,size(Z2,2));
    for i=1:size(Z2,2)
         out(:,i) = nmf_grad(Z2(:,i),pZ(:,i),dfp(:,i),Dgn,lambda2gn,groupsize);
    end
    return
end


id = find(Z2>0);
Did = Dgn(:,id);

A = Did*((Did'*Did + lambda2gn*eye(length(id)))\Did');


pZ1 = pZ(1:2:end);
pZ2 = pZ(2:2:end);

A1 = A(:,1:2:end);
A2 = A(:,2:2:end);

B1 = A1.*repmat(1./pZ1',N,1);
B2 = A2.*repmat(1./pZ2',N,1);

aux1 = B1'*(pZ.*dfp);
aux2 = B2'*(pZ.*dfp);

out = up_sample(aux1,M,groupsize) + circshift(up_sample(aux2,M,groupsize),[groupsize/2,0]);

end