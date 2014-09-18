function [out,dout,Z2,fp,cost] = measure_bilevel_cost(alpha, D, Dgn, data, lambda1,lambda2, lambda1gn, lambda2gn, groupsize,time_groupsize,grad_type,Pool)

verbose = 0;

N = size(alpha,1);
rec = D * alpha;
c1 = 0.5*norm(rec(:)-data(:))^2;

%modulus = modphas_decomp(alpha,groupsize);
pZ = down_sample(alpha,groupsize,time_groupsize);

%V0 = up_sample(pZ(1:2:end,1:2:end),groupsize,time_groupsize);

param0.posAlpha = 1;
param0.posD = 1;
param0.pos = 1;
param0.lambda = lambda1gn;
param0.lambda2 = lambda2gn;
Z2 =  full(mexLasso(pZ,Dgn,param0));
p = Dgn * Z2;

% reweighting function
eps = 1e-2;
fp = 1./(p + eps);

% reweighted group-lasso
c2 = lambda1*sum(fp(:).*pZ(:));


% ridge term
c3 = 0.5*lambda2*sum(alpha(:).^2);

out=(c1+c2+c3);

cost.total = out;
cost.c1 = c1;
cost.c2 = c2;
cost.c3 = c3;

if nargout>1 && nargin > 10
    
    switch grad_type
        
        case 'dDgn'
            
            W = -(pZ./(p+eps).^2);
            dcp = Dgn'*W;
            
            dDgn = weight_grad(Z2,Dgn,dcp,pZ,lambda2gn);
            
            dout = lambda1*(dDgn + W*Z2');
            
        case 'dD'
            
            dout = (D*alpha - data)*alpha';
            
        case 'dX'
            
            dc1 = D'*(D*alpha - data);
            dc3 = lambda2* alpha;
            
            % we compute separate for each set of groups
            paux = zeros(size(fp));
            jj = pZ ~=0;
            paux(jj) = fp(jj).*(1./pZ(jj));
            
            
            %aux1 = up_sample(paux(1:2:end,1:2:end),groupsize,time_groupsize)+ circshift(up_sample(paux(2:2:end,1:2:end),groupsize,time_groupsize),[groupsize/2,0]);
            %aux2 = up_sample(paux(1:2:end,2:2:end),groupsize,time_groupsize)+ circshift(up_sample(paux(2:2:end,2:2:end),groupsize,time_groupsize),[groupsize/2,0]);
            daux = up_sample2(paux,groupsize,time_groupsize);
            
            
            %daux = aux1; 
            %daux(:,time_groupsize/2+1:end-time_groupsize/2)  = daux(:,time_groupsize/2+1:end-time_groupsize/2) + aux2;

            dc2_1 = lambda1*daux.*alpha;
            
            %dfp = 1./(p+eps);
            dfp = -1./((p+eps).^2);

            
            aux2 = nmf_grad(Z2,pZ,dfp,Dgn,lambda2gn,groupsize,time_groupsize);
            daux2 = up_sample2(aux2,groupsize,time_groupsize);
%            aux2 = nmf_grad(Z2(:,2:2:end),pZ(:,2:2:end),dfp(:,2:2:end),Dgn,lambda2gn,groupsize,time_groupsize);
                
            
            dc2_2 = lambda1*alpha.*daux2;
            

            dout = (dc1 + dc2_1 + dc2_2 + dc3);
            
            
            if verbose
                fprintf('totcost %4.2f [ %4.2f %4.2f %4.2f %4.2f ] \n', sum(out), c1, c2, c3, cost.grad_norm)
            end
            
        otherwise
            
            error(['Gradient ' grad_tpe ' is not a valid option'])
               
    end
else
    dout = 0;
end

cost.grad_norm = norm(dout,'fro');

end


function Zp = down_sample(Z,groupsize,time_groupsize)

box=ones(groupsize,time_groupsize);

Zp = sqrt(conv2(Z.^2,box,'valid'));
Zp=Zp(1:groupsize/2:end,1:time_groupsize/2:end);

end

% function Zp = down_sample(Z,groupsize,time_groupsize)
% 
% box=ones(groupsize,time_groupsize);
% M = size(Z,1);
% 
% %Zp = sqrt(cconv(Z.^2,box,M));
% 
% Zpad = [Z(end-groupsize/2:end,:);Z];
% 
% eps =1e-4;
% Zp = sqrt(conv2(Zpad.^2+eps,box,'valid'));
% 
% Zp=Zp(groupsize/2:groupsize/2:end,1:max(time_groupsize/2,1):end);
% Zp = circshift(Zp,[-1,0]);
% 
% end


function V = up_sample2(p,groupsize,time_groupsize)

f1=round(groupsize/2);
f2=round(time_groupsize/2);
off1=[1 1 f1+1 f1+1];
off2=[1 f2+1 1 f2+1];
box=ones(groupsize,time_groupsize);

aux = zeros(groupsize/2*(size(p,1)-1)+1,(size(p,2)-1)*time_groupsize/2+1);

aux(1:groupsize/2:end,1:time_groupsize/2:end)=p;
V = conv2(aux,box,'full');


end

function V = up_sample(p,groupsize,time_groupsize)

M = groupsize*size(p,1);

if size(p,2)>1
    V = zeros(M,size(p,2)*time_groupsize);
    for i=1:size(p,2)
         V(:,((i-1)*time_groupsize+1):i*time_groupsize) = up_sample(p(:,i),groupsize,time_groupsize);
    end
    return
end

T = repmat(p',groupsize,1);
V = reshape(T,M,1);
V = repmat(V,1,time_groupsize);

end



function out = nmf_grad(Z2,pZ,dfp,Dgn,lambda2gn,groupsize,time_groupsize)

N = size(Dgn,1);
M = groupsize*N/2;

R = size(Z2,2);

if size(Z2,2)>1
    out = zeros(N,R);
    for i=1:size(Z2,2)
         out(:,i) = nmf_grad(Z2(:,i),pZ(:,i),dfp(:,i),Dgn,lambda2gn,groupsize,time_groupsize);
    end
    return
end


id = find(Z2>0);
if ~isempty(id)
Did = Dgn(:,id);

A = Did*((Did'*Did + lambda2gn*eye(length(id)))\Did');

pZi = zeros(size(pZ));
ii = (pZ~=0);
pZi(ii) = 1./pZ(ii);

B = A.*repmat(pZi',N,1);
out = B'*(pZ.*dfp);

% pZ1 = pZ(1:2:end);
% pZ2 = pZ(2:2:end);
% 
% pZ1i = zeros(size(pZ1));
% ii = (pZ1~=0);
% pZ1i(ii) = 1./pZ1(ii);
% 
% pZ2i = zeros(size(pZ2));
% ii = (pZ2~=0);
% pZ2i(ii) = 1./pZ2(ii);
% 
% A1 = A(:,1:2:end);
% A2 = A(:,2:2:end);
% 
% B1 = A1.*repmat(pZ1i',N,1);
% B2 = A2.*repmat(pZ2i',N,1);
% 
% aux1 = B1'*(pZ.*dfp);
% aux2 = B2'*(pZ.*dfp);
% 
% out = up_sample(aux1,groupsize,time_groupsize) + circshift(up_sample(aux2,groupsize,time_groupsize),[groupsize/2,0]);

else
    out = zeros(N,1);
end

end


function dDgn = weight_grad(Z2,Dgn,dc2,pZ,lambda2gn)

if size(Z2,2)>1
    dDgn = zeros(size(Dgn));
    for i=1:size(Z2,2)
         dDgn = dDgn + weight_grad(Z2(:,i),Dgn,dc2(:,i),pZ(:,i),lambda2gn);
    end
    return
end

dDgn = zeros(size(Dgn));
id = find(Z2>0);
if ~isempty(id)
    Did = Dgn(:,id);
    
    bd = (Did'*Did + lambda2gn*eye(length(id)))\dc2(id);
    
    fit = pZ-Did*Z2(id);
    

    dDgn(:,id) = -Did*bd*Z2(id)'+fit*bd';
    
end

end