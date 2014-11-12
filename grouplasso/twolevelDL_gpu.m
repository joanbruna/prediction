function [Dout, Dgnout] = twolevelDL_gpu(Xin, options)
%this function performs a dictionary learning using 
%the proximal toolbox and iterated gradient descent
%from Mairal et Al (2010)
%requires the spams proximal operator toolbox 

%we learn a dictionary which maximizes group sparsity,
%where groups are bi-clusters.

%Joan Bruna 2014 Courant Institute

Xin = Xin(:,1:end-1);

[N,M]=size(Xin);

X=gpuArray(single(Xin));

K = getoptions(options, 'K', 2*N);
Kgn = getoptions(options, 'Kgn', K/2);
groupsize = getoptions(options,'groupsize',2);
time_groupsize = getoptions(options,'time_groupsize',2);
overlapping=getoptions(options,'overlapping',1);
batchsize=getoptions(options,'batchsize',1024);
diters=getoptions(options,'dict_iters',2);
lambda = getoptions(options,'lambda',0.1);
nepochs=getoptions(options,'epochs',4);
itersout=getoptions(options,'itersout',300);
nmf=getoptions(options,'nmf', 0);
beta = getoptions(options,'beta',2e-1);
betagn = getoptions(options,'betagn',2e-1);
lambdagn = getoptions(options,'lambdagn',0.1);
t00 = getoptions(options,'alpha_step',1);
rho=getoptions(options,'rho',5);
nu = getoptions(options,'nu',0.5);
plotstuff=getoptions(options,'plotstuff',0);
if plotstuff
caccum=[];
end

%initial dictionaries
D=zeros(N,K,'single','gpuArray');
II=randperm(M-1);
D=X(:,II(1:K));
norms = sqrt(sum(D.^2));
D=D ./ repmat(norms,[N 1]);

Mf= time_groupsize*floor(batchsize/time_groupsize);
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


if isfield(options,'initD')
D=gpuArray(single(options.initD));
end

%analysis coeffs
Ztmp = max(0,D'*X);
box = ones(groupsize,time_groupsize,'single','gpuArray');
Ztmp=sqrt(conv2(Ztmp.^2,box,'valid'));
Ptmp=Ztmp(1:f1:end,1:f2:end);
II=randperm(size(Ptmp,2));
Dgn=Ptmp(:,II(1:Kgn));
norms = sqrt(sum(Dgn.^2));
Dgn=Dgn ./ repmat(norms,[size(Dgn,1) 1]);

if isfield(options,'initDgn')
Dgn=gpuArray(single(options.initDgn));
end


B=zeros(N,K,'single','gpuArray');
A=zeros(K,K,'single','gpuArray');
Bgn=zeros(size(Dgn,1),Kgn,'single','gpuArray');
Agn=zeros(Kgn,Kgn,'single','gpuArray');
Dsq = zeros(K,K,'single','gpuArray');
DX = zeros(K,Mf,'single','gpuArray');


if overlapping
t0 = t00 * (1/(4*nu+beta+max(svd(D))^2))
else
t0 = t00 * (1/(nu+beta+max(svd(D))^2))
end
tlambda = t0 * lambda;

t0gn = t00 * (1/(betagn+(max(svd(Dgn))^2)));
tlambdagn = t0gn * lambdagn;

niters=round(nepochs*M/(batchsize));
epochn=round(niters/nepochs);

nepochs
batchsize
niters
epochn


t=1;

data = zeros(N, batchsize,'single','gpuArray');
y = zeros(K,batchsize,'single','gpuArray');
yo = zeros(K,batchsize,4,'single','gpuArray');
aux = zeros(K,batchsize,'single','gpuArray');
Pool = zeros(KK,MM, 'single', 'gpuArray');
Poole = zeros(KK,MM, 'single', 'gpuArray');
Rgn = zeros(KK,MM, 'single', 'gpuArray');
dPool = zeros(KK,MM, 'single', 'gpuArray');
tPool = zeros(KK,MM, 'single', 'gpuArray');
lI = zeros(KK,MM, 'single', 'gpuArray');
dy = zeros(K,Mf,'single','gpuArray');
tmp = zeros(K,Mf,'single','gpuArray');
new = zeros(K,Mf,'single','gpuArray');
lout = zeros(K,Mf,'single','gpuArray');
oldZgn = zeros(Kgn, MM, 'single', 'gpuArray');
Zgn = zeros(Kgn, MM, 'single', 'gpuArray');
Dgnsq = zeros(Kgn, Kgn,'single','gpuArray');
tmpgn = zeros(Kgn, MM, 'single', 'gpuArray');
loutgn = zeros(Kgn, MM,'single','gpuArray');
cost= zeros(6,1,'single','gpuArray');

off1=[1 1 f1+1 f1+1];
off2=[1 f2+1 1 f2+1];

IItmp=randperm(floor(M/batchsize));
II=gpuArray(IItmp);

for ep=1:nepochs
for n=1:epochn
	fprintf('batch %d of %d \n',n,niters)
	init= mod( n-1, length(II)); 
	data = X(:,1+batchsize*init:(init+1)*batchsize); 

	Dsq=D'*D;
	DX = D'*data;
	Dgnsq = Dgn'*Dgn;
	y=0*y;
	yo=0*yo;
	lout=0*lout;
	Zgn=0*Zgn;
	Rgn=0*Rgn;
	aux=0*aux;
	tmp=0*tmp;
	Pool=0*Pool;
	Poole=0*Poole;
	dPool=0*dPool;
	tPool=0*tPool;
	lI=0*lI;
	dy=0*dy;
	new=0*new;
	tmpgn=0*tmpgn;
	loutgn=0*loutgn;
	t=1;
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
				tmp=max(0,tmp);
			end			
			aux=sqrt(conv2(tmp.^2,box,'valid'));
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
			%if nmf
			%y = max(0,y);
			%end
			newt = (1+ sqrt(1+4*t^2))/2;
		else

			tmp = y - t0*(Dsq * y - DX + beta*y + nu*dy);
			if nmf
			tmp = max(0,tmp);
			end
			aux=sqrt(conv2(tmp.^2,box,'valid'));
			tPool=aux(1:f1:end,1:f2:end);	
			dPool = max(0,tPool-tlambda);
			aux=0*aux;
			aux(1:f1:end,1:f2:end)=dPool./(eps+tPool);
			new=conv2(aux,box,'full').*tmp;
		
			newt = (1+ sqrt(1+4*t^2))/2;
			y = new + ((t-1)/newt)*(new-lout);
			lout=new;
		end
		
		tmpgn = Zgn  - t0gn *(Dgnsq * Zgn - Dgn'*Poole + betagn * Zgn);	
		tmpgn = (tmpgn > tlambdagn).*(tmpgn - tlambdagn);
		
		Rgn = Dgn * Zgn;
		oldZgn = Zgn;
		Zgn = tmpgn;	
		t=newt;
		if mod(i,itersout/2)==itersout/2-1
		%if mod(i,20)==19
		cost(1) = .5*norm(D*lout-data,'fro')^2;
		cost(2) =  lambda * (sum(Pool(:)));
		cost(3) = .5*nu*norm(Poole-Rgn,'fro')^2;
		cost(4) = nu*lambdagn*sum(Zgn(:));
		cost(5) = .5*beta*norm(lout,'fro')^2;
		cost(6) = .5*nu*betagn*norm(Zgn,'fro')^2;
		fprintf(' -- it %d totcost %4.2f [ %4.2f %4.2f %4.2f %4.2f ] \n',i+1, sum(cost), cost(1)/(.5*norm(data,'fro')^2),cost(2), cost(3), cost(4))
		
        end
	end

	%%%%%%%update D%%%%%%%
	A = (((n-1)/n)^rho)*A + lout*lout';
	B = (((n-1)/n)^rho)*B + data*lout';
	if 1 
	dia = diag(A).^(-1);
	Ip = gpuArray(randperm(K));
	for i=1:diters
	for j=1:K
	u = D(:,Ip(j)) + (B(:,Ip(j)) - D*(A(:,Ip(j))))*dia(Ip(j));
	if nmf
	u = max(0,u);
	end
	D(:,Ip(j)) = u / max(1, norm(u));
	end
	end
	end
	%%%%%%%%%%%%%%%

	%%%%%update Dgn%%%%%%
    if nu > 0
	Agn = (((n-1)/n)^rho)*Agn + Zgn*Zgn';
	Bgn = (((n-1)/n)^rho)*Bgn + Poole*Zgn';
	if 1 
	dia = diag(Agn).^(-1);
	Ipg = gpuArray(randperm(Kgn));
	for i=1:diters
	for j=1:Kgn
	u = Dgn(:,Ipg(j)) + (Bgn(:,Ipg(j)) - Dgn*(Agn(:,Ipg(j))))*dia(Ipg(j));
	if 1
	u = max(0,u);
	end
	Dgn(:,Ipg(j)) = u / max(1, norm(u));
	end
	end
    end
    end
	%%%%%%
	
	if 1
	%update proximal splitting parameters
	if overlapping
	t0 = t00 * (1/(4*nu+beta+max(svd(D))^2));
	else
	t0 = t00 * (1/(nu+beta+max(svd(D))^2));
	end
	%t0 = 4e-3;
	tlambda = t0 * lambda;
	t0gn = t00 * (1/(betagn+(max(svd(Dgn))^2)));
	tlambdagn = t0gn * lambdagn;
	%%%%%
    end

    plotstuff = 0;
	if plotstuff
	caccum=[caccum sum(cost)];
	if mod(n,16)==1
 	figure(1);
 	plot(caccum);drawnow;
	figure(2);imagesc(D);drawnow;
	figure(3);imagesc(Dgn);drawnow;
	end
	end

end
A=0*A;
B=0*B;
Agn=0*Agn;
Bgn=0*Bgn;
end

Dout=gather(D);
Dgnout = gather(Dgn);

end








