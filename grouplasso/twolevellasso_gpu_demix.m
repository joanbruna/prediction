function [Zout1, Zgnout1, Zout2, Zgnout2] = twolevellasso_gpu_demix(Xin, Din1, Dgnin1, Din2, Dgnin2, options)
%we learn a dictionary which maximizes group sparsity,
%where groups are bi-clusters.

%Joan Bruna 2014 Courant Institute


[N,M]=size(Xin);

X=gpuArray(single(Xin));
D1=gpuArray(single(Din1));
Dgn1=gpuArray(single(Dgnin1));
D2=gpuArray(single(Din2));
Dgn2=gpuArray(single(Dgnin2));

groupsize = getoptions(options,'groupsize',2);
time_groupsize = getoptions(options,'time_groupsize',2);
overlapping = getoptions(options, 'overlapping', 1);
nu = getoptions(options,'nu',1);
beta = getoptions(options,'beta',2e-1);
betagn = getoptions(options,'betagn',2e-1);
lambda = getoptions(options,'lambda',0.1);
lambdagn = getoptions(options,'lambdagn',0.1);
itersout=getoptions(options,'itersout',300);
nmf=getoptions(options,'nmf', 0);

K = size(D1,2);
Kgn = size(Dgn1,2);

t00 = getoptions(options,'alpha_step',1);
if overlapping
t01 = t00 * (1/(4*nu+beta+max(svd(D1))^2))
t02 = t00 * (1/(4*nu+beta+max(svd(D2))^2))
else
t01 = t00 * (1/(nu+beta+max(svd(D1))^2))
t02 = t00 * (1/(nu+beta+max(svd(D2))^2))
end

tlambda1 = t01 * lambda;% * (size(D,2)/K);
tlambda2 = t02 * lambda;% * (size(D,2)/K);

t01gn = t00 * (1/(betagn+max(svd(Dgn1))^2));
tlambdagn1 = t01gn * lambdagn;% * (size(D,2)/K);
t02gn = t00 * (1/(betagn+max(svd(Dgn2))^2));
tlambdagn2 = t02gn * lambdagn;% * (size(D,2)/K);


Mf= time_groupsize*floor(M/time_groupsize);
X=X(:,1:Mf);
DX1 = zeros(K,Mf,'single','gpuArray');
Dsq1 = zeros(K, K, 'single', 'gpuArray');
Dsq1=D1'*D1;
DX1 = D1'*X;
DX2 = zeros(K,Mf,'single','gpuArray');
Dsq2 = zeros(K, K, 'single', 'gpuArray');
Dsq2=D2'*D2;
DX2 = D2'*X;
D12 = zeros(K, K, 'single', 'gpuArray');
D12 = D1'*D2;

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

y1 = zeros(K,Mf,'single','gpuArray');
yo1 = zeros(K,Mf,4,'single','gpuArray');
y2 = zeros(K,Mf,'single','gpuArray');
yo2 = zeros(K,Mf,4,'single','gpuArray');
aux = zeros(K,Mf,'single','gpuArray');
Pool1 = zeros(KK,MM, 'single', 'gpuArray');
Poole1 = zeros(KK,MM, 'single', 'gpuArray');
Rgn1 = zeros(KK,MM, 'single', 'gpuArray');
Pool2 = zeros(KK,MM, 'single', 'gpuArray');
Poole2 = zeros(KK,MM, 'single', 'gpuArray');
Rgn2 = zeros(KK,MM, 'single', 'gpuArray');
dPool = zeros(KK,MM, 'single', 'gpuArray');
tPool = zeros(KK,MM, 'single', 'gpuArray');
lI = zeros(KK,MM, 'single', 'gpuArray');
dy1 = zeros(K,Mf,'single','gpuArray');
dy2 = zeros(K,Mf,'single','gpuArray');
tmp = zeros(K,Mf,'single','gpuArray');
%new = zeros(K,Mf,'single','gpuArray');
lout1 = zeros(K,Mf,'single','gpuArray');
lout2 = zeros(K,Mf,'single','gpuArray');
oldZgn1 = zeros(Kgn, MM, 'single', 'gpuArray');
oldZgn2 = zeros(Kgn, MM, 'single', 'gpuArray');
Zgn1 = zeros(Kgn, MM, 'single', 'gpuArray');
Zgn2 = zeros(Kgn, MM, 'single', 'gpuArray');
Dgnsq1 = zeros(Kgn, Kgn,'single','gpuArray');
Dgnsq1 = Dgn1'*Dgn1;
Dgnsq2 = zeros(Kgn, Kgn,'single','gpuArray');
Dgnsq2 = Dgn2'*Dgn2;
tmpgn = zeros(Kgn, MM, 'single', 'gpuArray');
loutgn = zeros(Kgn, MM,'single','gpuArray');

gradient_descent = getoptions(options,'gradient_descent',0);
if gradient_descent
dz1 = zeros(K,Mf,'single','gpuArray');
dz2 = zeros(K,Mf,'single','gpuArray');
st1 = zeros(K,Mf,'single','gpuArray');
st2 = zeros(K,Mf,'single','gpuArray');
lr = getoptions(options,'lrate',6e-2);
rho = getoptions(options,'momentum', 0.9);
epsi = getoptions(options,'eps',1e-8);
end

box = ones(groupsize,time_groupsize,'single','gpuArray');
cost= zeros(6,1,'single','gpuArray');

if isfield(options,'Z1in')
y1=gpuArray(single(options.Z1in));
y2=gpuArray(single(options.Z2in));
end

off1=[1 1 f1+1 f1+1];
off2=[1 f2+1 1 f2+1];

for i=1:itersout


	%compute pooling of current z and gradient 
	aux=sqrt(conv2(y1.^2,box,'valid'));
	Pool1=aux(1:f1:end,1:f2:end);	
	aux=sqrt(conv2(y1.^2+eps,box,'valid'));
	Poole1=aux(1:f1:end,1:f2:end);	
	dPool1 = Poole1 - Rgn1;
	aux=0*aux;
	aux(1:f1:end,1:f2:end)=dPool1./(Poole1);
	dy1=conv2(aux,box,'full').*y1;

	aux=sqrt(conv2(y2.^2,box,'valid'));
	Pool2=aux(1:f1:end,1:f2:end);	
	aux=sqrt(conv2(y2.^2+eps,box,'valid'));
	Poole2=aux(1:f1:end,1:f2:end);	
	dPool2 = Poole2 - Rgn2;
	aux=0*aux;
	aux(1:f1:end,1:f2:end)=dPool2./(Poole2);
	dy2=conv2(aux,box,'full').*y2;

	if gradient_descent

	%simply compute the gradient wrt z1 and z2 and use momentum
	dz1 = Dsq1 * y1 - DX1 + beta*y1 + nu*dy1 + D12*y2;	
	dz2 = Dsq2 * y2 - DX2 + beta*y2 + nu*dy2 + D12'*y1;	

	aux=0*aux;
	aux(1:f1:end,1:f2:end)=1./(Poole1);
	dy1=conv2(aux,box,'full').*y1;
	aux=0*aux;
	aux(1:f1:end,1:f2:end)=1./(Poole2);
	dy2=conv2(aux,box,'full').*y2;
	
	dz1 = dz1 + lambda*dy1;
	dz2 = dz2 + lambda*dy2;

	st1 = rho * st1 - lr * dz1;
	st2 = rho * st2 - lr * dz2;
	lout1=y1;
	lout2=y2;
	y1 = y1 + st1;
	y2 = y2 + st2;
	if nmf 
	y1=max(0,y1);
	y2=max(0,y2);
	end


	else

	if overlapping
		for k=1:4
		tmp = 2*y1 - yo1(:,:,k) - t01*(Dsq1 * y1 - DX1 + beta*y1 + nu*dy1 + D12*y2);
		if nmf
		tmp = max(0,tmp);
		end
		aux=sqrt(conv2(tmp.^2,box,'valid'));
		tPool=aux(off1(k):groupsize:end,off2(k):time_groupsize:end);	
		dPool = max(0,tPool-4*tlambda1);
		lI=find(tPool>0);
		dPool(lI)=dPool(lI)./tPool(lI);
		aux=0*aux;
		aux(off1(k):groupsize:end,off2(k):time_groupsize:end)=dPool;
		yo1(:,:,k) = yo1(:,:,k) + conv2(aux,box,'full').*tmp - y1;
		end

		for k=1:4
		tmp = 2*y2 - yo2(:,:,k) - t02*(Dsq2 * y2 - DX2 + beta*y2 + nu*dy2 + D12'*y1);
		if nmf
		tmp = max(0,tmp);
		end
		aux=sqrt(conv2(tmp.^2,box,'valid'));
		tPool=aux(off1(k):groupsize:end,off2(k):time_groupsize:end);	
		dPool = max(0,tPool-4*tlambda2);
		lI=find(tPool>0);
		dPool(lI)=dPool(lI)./tPool(lI);
		aux=0*aux;
		aux(off1(k):groupsize:end,off2(k):time_groupsize:end)=dPool;
		yo2(:,:,k) = yo2(:,:,k) + conv2(aux,box,'full').*tmp - y2;
		end

		lout1=y1;
		y1=mean(yo1,3);
		lout2=y2;
		y2=mean(yo2,3);

	else

        %tmp = y - t0*(Dsq * y - DX + beta*y + nu*dy);
% 	%	tmp = 2*y1 - yo1(:,:,k) - t01*(Dsq1 * y1 - DX1 + beta*y1 + nu*dy1 + D12*y2);
%         if nmf
%             tmp = max(0,tmp);
%         end
%         aux=sqrt(conv2(tmp.^2,box,'valid'));
%         tPool=aux(1:f1:end,1:f2:end);
%         dPool = max(0,tPool-tlambda);
%         aux=0*aux;
%         aux(1:f1:end,1:f2:end)=dPool./(eps+tPool);
%         new=conv2(aux,box,'full').*tmp;
%         
%         newt = (1+ sqrt(1+4*t^2))/2;
%         y = new + ((t-1)/newt)*(new-lout);
%         lout=new;
	end
	
	end
	newt = (1+ sqrt(1+4*t^2))/2;

	tmpgn = Zgn1  - t01gn *(Dgnsq1 * Zgn1 - Dgn1'*Poole1 + betagn * Zgn1);	
	tmpgn = (tmpgn > tlambdagn1).*(tmpgn - tlambdagn1);
	
	Rgn1 = Dgn1 * Zgn1;
	oldZgn1 = Zgn1;
	Zgn1 = tmpgn;	

	tmpgn = Zgn2  - t02gn *(Dgnsq2 * Zgn2 - Dgn2'*Poole2 + betagn * Zgn2);	
	tmpgn = (tmpgn > tlambdagn2).*(tmpgn - tlambdagn2);
	
	Rgn2 = Dgn2 * Zgn2;
	oldZgn2 = Zgn2;
	Zgn2 = tmpgn;	

	t=newt;

	if mod(i,10)==9
	cost(1) = .5*norm(D1*lout1 + D2*lout2-X,'fro')^2;
	cost(2) =  lambda * (sum(Pool1(:))) + lambda*sum(Pool2(:));
	cost(3) = .5*nu*(norm(Poole1-Rgn1,'fro')^2 + norm(Poole2-Rgn2,'fro')^2);
	cost(4) = nu*lambdagn*(sum(oldZgn1(:)) + sum(oldZgn2(:)));
	cost(5) = .5*beta*(norm(lout1,'fro')^2 + norm(lout2,'fro')^2);
	cost(6) = .5*nu*betagn*(norm(oldZgn1,'fro')^2+norm(oldZgn2,'fro')^2);
	fprintf('it %d totcost %4.2f [ %4.2f %4.2f %4.2f %4.2f ] \n',i+1, sum(cost), cost(1),cost(2), cost(3), cost(4))

	end


end

Zout1=gather(lout1);
if size(Zout1,2)<M
Zout1(:,end:M) = 0;
end
Zgnout1 = gather(Zgn1);
Zout2=gather(lout2);
if size(Zout2,2)<M
Zout2(:,end:M) = 0;
end
Zgnout2 = gather(Zgn2);

end








