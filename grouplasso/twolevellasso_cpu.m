function [Zout, Zgnout,Pool] = twolevellasso_cpu(X, D, Dgn, options)
%this function performs a dictionary learning using 
%the proximal toolbox and iterated gradient descent
%from Mairal et Al (2010)
%requires the spams proximal operator toolbox 

%we learn a dictionary which maximizes group sparsity,
%where groups are bi-clusters.

%Joan Bruna 2014 Courant Institute


[N,M]=size(X);

% X=gpuArray(single(Xin));
% D=gpuArray(single(Din));
% Dgn=gpuArray(single(Dgnin));

groupsize = getoptions(options,'groupsize',2);
time_groupsize = getoptions(options,'time_groupsize',2);
overlapping = getoptions(options, 'overlapping', 1);
nu = getoptions(options,'nu',1);
beta = getoptions(options,'beta',2e-1);
betagn = getoptions(options,'betagn',2e-1);
lambda = getoptions(options,'lambda',0.1);
lambdagn = getoptions(options,'lambdagn',0.1);

K = size(D,2);
Kgn = size(Dgn,2);

t0 = getoptions(options,'alpha_step',1);
if overlapping
t0 = t0 * (1/(4*nu+beta+max(svd(D))^2))
else
t0 = t0 * (1/(nu+beta+max(svd(D))^2))
end
itersout=getoptions(options,'itersout',300);
nmf=getoptions(options,'nmf', 0);
tlambda = t0 * lambda;% * (size(D,2)/K);

t0gn = getoptions(options,'alpha_step',0.5);
t0gn = t0gn * (1/(betagn+max(svd(Dgn))^2));

tlambdagn = t0gn * lambdagn;% * (size(D,2)/K);


Mf= time_groupsize*floor(M/time_groupsize);
X=X(:,1:Mf);
Dsq=D'*D;
DX = D'*X;

t=1;
Ksm=K/groupsize;
Msm=Mf/time_groupsize;
if overlapping
f1=round(groupsize/2);
f2=round(time_groupsize/2);
KK=2*Ksm-1;
MM=2*Msm-1;
else
f1=groupsize;%round(groupsize/2);
f2=time_groupsize;%round(time_groupsize/2);
KK=Ksm;%2*Ksm-1;
MM=Msm;%2*Msm-1;
end

y = zeros(K,Mf);
yo = zeros(K,Mf,4);
aux = zeros(K,Mf);
Pool = zeros(KK,MM);
Poole = zeros(KK,MM);
Rgn = zeros(KK,MM);
dPool = zeros(KK,MM);
tPool = zeros(KK,MM);
lI = zeros(KK,MM);


dy = zeros(K,Mf);
tmp = zeros(K,Mf);
new = zeros(K,Mf);
lout = zeros(K,Mf);
oldZgn = zeros(Kgn, MM);
Zgn = zeros(Kgn, MM);
Dgnsq = zeros(Kgn, Kgn);
Dgnsq = Dgn'*Dgn;
tmpgn = zeros(Kgn, MM);
loutgn = zeros(Kgn, MM);


box = ones(groupsize,time_groupsize);
cost= zeros(6,1);

off1=[1 1 f1+1 f1+1];
off2=[1 f2+1 1 f2+1];

%eps = 0.01;
eps = getoptions(options,'eps',0.01);

for i=1:itersout

	%compute pooling of current z and gradient 
	aux=sqrt(conv2(y.^2,box,'valid'));
	Pool=aux(1:f1:end,1:f2:end);	

	aux=sqrt(conv2(y.^2+eps,box,'valid'));
	Poole=aux(1:f1:end,1:f2:end);	

	dPool = Poole - Rgn;
	aux=0*aux;
	aux(1:f1:end,1:f2:end)=dPool./(Poole);
	dy=conv2(aux,box,'full').*y;

	if overlapping
		for k=1:4
		tmp = 2*y - yo(:,:,k) - t0*(Dsq * y - DX + beta*y + nu*dy);
		if nmf
		tmp = max(0,tmp);
		end
		aux=sqrt(conv2(tmp.^2+eps,box,'valid'));
		tPool=aux(off1(k):groupsize:end,off2(k):time_groupsize:end);	
		dPool = max(0,tPool-4*tlambda);
		lI=find(tPool>0);
		dPool(lI)=dPool(lI)./tPool(lI);
		aux=0*aux;
		aux(off1(k):groupsize:end,off2(k):time_groupsize:end)=dPool;
		yo(:,:,k) = yo(:,:,k) + conv2(aux,box,'full').*tmp - y;
		end
		lout=y;
		y=mean(yo,3);
		newt = (1+ sqrt(1+4*t^2))/2;
	else

		tmp = y - t0*(Dsq * y - DX + beta*y + nu*dy);
		if nmf
		tmp = max(0,tmp);
		end
		aux=sqrt(conv2(tmp.^2+eps,box,'valid'));
		tPool=aux(1:f1:end,1:f2:end);	
		dPool = max(0,tPool-tlambda);
		aux=0*aux;
		%aux(1:f1:end,1:f2:end)=dPool./(eps+tPool);
        aux(1:f1:end,1:f2:end)=dPool./(tPool);
		new=conv2(aux,box,'full').*tmp;
	
		newt = (1+ sqrt(1+4*t^2))/2;
		y = new + ((t-1)/newt)*(new-lout);
        
        ee(i) = norm(lout-new,'fro');
		lout=new;
	end
	
	tmpgn = Zgn  - t0gn *(Dgnsq * Zgn - Dgn'*Poole + betagn * Zgn);	
	tmpgn = (tmpgn > tlambdagn).*(tmpgn - tlambdagn);
	%Zgn = tmpgn + ((t-1)/newt)*(tmpgn - loutgn);
	%loutgn = tmpgn;
	
	Rgn = Dgn * Zgn;
	oldZgn = Zgn;
	Zgn = tmpgn;	

	t=newt;

	if mod(i,10)==9
	cost(1) = .5*norm(D*lout-X,'fro')^2;
	cost(2) =  lambda * (sum(Pool(:)));
	cost(3) = .5*nu*norm(Poole-Rgn,'fro')^2;
	cost(4) = nu*lambdagn*sum(oldZgn(:));
	cost(5) = .5*beta*norm(lout,'fro')^2;
	cost(6) = .5*nu*betagn*norm(oldZgn,'fro')^2;
    fprintf('it %d totcost %4.2f [ %4.2f %4.2f %4.2f %4.2f ] \n',i+1, sum(cost), cost(1),cost(2), cost(3), cost(4))

	end


end


llout=reshape(lout,K,Mf);

Zout=llout;
Zgnout = gather(Zgn);

aux=sqrt(conv2(Zout.^2+eps,box,'valid'));
Pool=aux(1:f1:end,1:f2:end);

Z = Zout;
AA = zeros(size(Zout));

if overlapping
for k=1:4
    
    
    tPoole=aux(off1(k):groupsize:end,off2(k):time_groupsize:end);
    
    fp = zeros(size(tPoole));
    id = find(tPoole>0);
    fp(id) = 1./tPoole(id);
    
    aux = 0*aux;
    aux(off1(k):groupsize:end,off2(k):time_groupsize:end)=fp;
    uPoole = (conv2(aux,box,'full')).*Z;
    
    AA = AA + uPoole;
    
end

else
    fp = zeros(size(Pool));
    id = find(Pool>0);
    fp(id) = 1./Pool(id);
    
    aux = 0*aux;
    aux(1:f1:end,1:f2:end)= fp;
    uPoole = (conv2(aux,box,'full')).*Z;
    
    AA = uPoole;
end

id = find(Z(:,10)>0);
Dd = D(:,id);

v = Dd'*(Dd*Z(id,10) - X(:,10)) + lambda*AA(id,10)+beta*Z(id,10);
v
max(abs(v(:)))



end







