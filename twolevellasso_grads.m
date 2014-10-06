function [dD,dZgnout] = twolevellasso_grads(X,D,Dgn,Z,Zgn,G, options,PP)

%box=ones(groupsize,time_groupsize);
%Z = gpuArray(Zout);
f=0;
groupsize = getoptions(options,'groupsize',2);
time_groupsize = getoptions(options,'time_groupsize',2);
overlapping = getoptions(options, 'overlapping', 1);
nu = getoptions(options,'nu',1);
beta = getoptions(options,'beta',2e-1);
betagn = getoptions(options,'betagn',2e-1);
lambda = getoptions(options,'lambda',0.1);
lambdagn = getoptions(options,'lambdagn',0.1);


if overlapping
    f1 = groupsize/2;
    f2 = time_groupsize/2;
    off1=[1 1 f1+1 f1+1];
    off2=[1 f2+1 1 f2+1];
else
    f1 = groupsize;
    f2 = time_groupsize;
end

box = ones(groupsize,time_groupsize);

[ma,na] = size(Z); [mb,nb] = size(box);
sizeAux = max([ma-max(0,mb-1),na-max(0,nb-1)],0);
aux = zeros(sizeAux);

eps = getoptions(options,'eps',0.01);
Poole = sqrt(conv2(Z.^2+eps,box,'valid'));
Pool = sqrt(conv2(Z.^2,box,'valid'));

% Compute grads
Q = zeros(size(Z));
if overlapping
    
    
for k=1:4
    
    
    tPoole=Poole(off1(k):groupsize:end,off2(k):time_groupsize:end);
    tPool=Pool(off1(k):groupsize:end,off2(k):time_groupsize:end);
    
    fp = zeros(size(tPoole));
    id = find(tPool>0);
    fp(id) = 1./tPoole(id);
    
    aux = 0*aux;
    aux(off1(k):groupsize:end,off2(k):time_groupsize:end)=fp;
    T1 = conv2(aux,box,'full');
    
    
    Zp = conv2(Z,box,'valid');
    Zp=Zp(off1(k):groupsize:end,off2(k):time_groupsize:end);
    
    aux = 0*aux;
    aux(off1(k):groupsize:end,off2(k):time_groupsize:end)=(fp.^3).*Zp;
    T2 = (conv2(aux,box,'full')).*Z;
    
    Q = Q + T1-T2;
    
end
else
    
    Poole=Poole(1:f1:end,1:f2:end);
    Pool=Pool(1:f1:end,1:f2:end);
    
    fp = zeros(size(Poole));
    id = find(Pool>0);
    fp(id) = 1./Poole(id);
    
    aux = 0*aux;
    aux(1:f1:end,1:f2:end)=fp;
    T1 = conv2(aux,box,'full');
    
    Zp = conv2(Z,box,'valid');
    Zp=Zp(1:f1:end,1:f2:end);
    
    aux = 0*aux;
    aux(1:f1:end,1:f2:end)=(fp.^3).*Zp;
    T2 = (conv2(aux,box,'full')).*Z;
    
    Q = T1 - T2;
    
end

dD = nmf_grad(X,Z,D,G,Q,lambda , beta);

%dZout = 0;
dZgnout = 0;

end


function A = nmf_grad(X,Z,D,G,J,lambda , beta)


Zg = Z(:);

N = length(Zg);

A = zeros(N,N);

for i=1:size(Z,2)
    
    id = find(Z(:,i)>0);
    lact = length(id);
    
    A(count:count+lact,count:count+lact) = D(:,id)'*D(:,id);
    
end

A = A + lambda*J + beta*eye(N,N);

end

% function dD = nmf_grad(X,Z,D,G,Q,lambda , beta)
% 
% 
% if size(Z,2)>1
%     dD = zeros(size(D));
%     for i=1:size(Z,2)
%         dD = dD + nmf_grad(X(:,i),Z(:,i),D,G(:,i),Q(:,i),lambda , beta);
%     end
%     return
% end
% 
% 
% id = find(Z>0);
% if ~isempty(id)
%     
%     lact = length(id);
%     
%     Dd = D(:,id);
%     
%     b = 0*Z;
%     b(id) = (Dd'*Dd + lambda*diag(Q(id)) + beta*eye(lact))\G(id);
%     
%     % Gradient
%     dD = -D*b*Z'+(X-D*Z)*b';
%     
% else
%     dD = zeros(size(D));
% end
% 
% end


