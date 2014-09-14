function [Zout, Zgnout] = twolevellasso_gpu(Xin, Din, Dgnin, options)
%this function performs a dictionary learning using 
%the proximal toolbox and iterated gradient descent
%from Mairal et Al (2010)
%requires the spams proximal operator toolbox 

%we learn a dictionary which maximizes group sparsity,
%where groups are bi-clusters.

%Joan Bruna 2013 Courant Institute


[N,M]=size(Xin);
X=gpuArray(single(Xin));
D=gpuArray(single(Din));
Dgn=gpuArray(single(Dgnin));

groupsize = getoptions(options,'groupsize',2);
time_groupsize = getoptions(options,'time_groupsize',2);

K = size(D,2);
Kgn = size(Dgn,2);
KK=K * time_groupsize;

t0 = getoptions(options,'alpha_step',0.5);
t0 = t0 * (1/(1+max(svd(D))^2))
lambda = getoptions(options,'lambda',0.1);
itersout=getoptions(options,'alpha_itersout',20);
nmf=getoptions(options,'nmf', 0);
tlambda = t0 * lambda;% * (size(D,2)/K);

t0gn = getoptions(options,'alpha_step',0.5);
t0gn = t0gn * (1/max(svd(Dgn))^2);
lambdagn = getoptions(options,'lambdagn',0.1);
tlambdagn = t0gn * lambdagn;% * (size(D,2)/K);

nu = getoptions(options,'nu',1);
beta = getoptions(options,'beta',2e-1);
betagn = getoptions(options,'betagn',2e-1);


%%%produce final synthesis coefficient too
fprintf('producing synthesis coeffs\n ')
MM=floor(M/time_groupsize);
%data = zeros(N, M,'single','gpuArray');
Mf = MM*time_groupsize;
y = zeros(KK,MM,'single','gpuArray');
yr = zeros(KK,MM,'single','gpuArray');
tmp = zeros(KK,MM,'single','gpuArray');
lout = zeros(KK,MM,'single','gpuArray');
llout = zeros(K,Mf,'single','gpuArray');
ss=groupsize*time_groupsize;
rr=KK*MM/ss;
newout = zeros(KK,MM,'single','gpuArray');
newouto = zeros(KK,MM,'single','gpuArray');
newout1 = zeros(KK,MM,'single', 'gpuArray');
normes=zeros(ss,rr,'single','gpuArray');
DX = zeros(K,Mf,'single','gpuArray');
%DX = zeros(KK,MM,'single','gpuArray');
%DXo = zeros(KK,MM,'single','gpuArray');
Dsq=D'*D;
DX = D'*X;
%DX = reshape(D'*X(:,1:Mf),KK,MM);
%DXo = reshape(circshift(D'*X(:,1:Mf),[0 round(time_groupsize/2)]),KK,MM);

ss=groupsize*time_groupsize;
rr=KK*MM/ss;
lI=zeros(ss,rr,'uint8','gpuArray');
dia=zeros(K,1,'single','gpuArray');

%even groups
groups=(mod(floor([0:KK-1]/groupsize),K/groupsize) )+1;
[~,tI0]=sort(groups);
tI1=invperm(tI0);
%odd groups
rien = circshift(reshape(groups,K,time_groupsize),[ceil(groupsize/2) 0]);
groupso=rien(:);
[~,tI00]=sort(groupso);
tI11=invperm(tI00);

I0=gpuArray(tI0);
I1=gpuArray(tI1);
I00=gpuArray(tI00);
I11=gpuArray(tI11);

t=1;
%we start by alternating between one update on z and one update on zgn

phas1 = zeros(K,Mf,'single','gpuArray');
phas2 = zeros(K,Mf,'single','gpuArray');
phas3 = zeros(K,Mf,'single','gpuArray');
phas4 = zeros(K,Mf,'single','gpuArray');

cost= zeros(4,1,'single','gpuArray');
PP=zeros(4,rr,'single','gpuArray');
Pool = zeros(2*K/groupsize, 2*M/time_groupsize, 'single', 'gpuArray');
Zgn = zeros(Kgn, 2*M/time_groupsize, 'single', 'gpuArray');
Dgnsq = zeros(Kgn, Kgn,'single','gpuArray');
Dgnsq = Dgn'*Dgn;
loutgn = zeros(Kgn, 2*M/time_groupsize,'single','gpuArray');
Rgn=zeros(4,rr,'single','gpuArray');


box = zeros(2*groupsize-1,2*time_groupsize-1,'single','gpuArray');
box(groupsize:end,time_groupsize:end)=1;
dbox = zeros(2*groupsize-1,2*time_groupsize-1,'single','gpuArray');
dbox(1:groupsize,1:time_groupsize)=1;


f1=round(groupsize/2);
f2=round(time_groupsize/2);


for i=1:itersout

	%compute pooling of current z and gradient 
	yp=0;yp(groupsize:end,time_groupsize:end)=y;
	aux=sqrt(conv2(yp.^2,box,'valid'));
	Pool=aux(1:f1:end,1:f2:end);	
	
	dPool = Pool - Rgn;
	aux=0;aux(1:f1:end,1:f2:end)=dPool./(eps+Pool);
	dyp=conv2(aux,dbox,'full').*yp;
	dy=dyp(groupsize:end,time_groupsize:end);
	
	tmp = y - t0*(Dsq * y - DX + beta*y + nu*dy):
	if nmf
	tmp = max(0,tmp);
	end
	yp=0;yp(groupsize:end,time_groupsize:end)=tmp;
	aux=sqrt(conv2(yp.^2,box,'valid'));
	tPool=aux(1:f1:end,1:f2:end);	
	dPool = max(0,tPool-tlambda);
	aux=0;aux(1:f1:end,1:f2:end)=dPool./(eps+tPool);
	yp=conv2(aux,dbox,'full').*yp;
	new=yp(groupsize:end,time_groupsize:end);
	
	newt = (1+ sqrt(1+4*t^2))/2;
	y = new + ((t-1)/newt)*(new-lout);
	lout=new;
	
	tmp = Zgn  - t0gn *(Dgnsq * Zgn - Dgn'*Pool + betagn * Zgn);	
	tmp = (tmp > tlambdagn).*tmp;
	Zgn = tmp + ((t-1)/newt)*(tmp - loutgn);
	loutgn = tmp;
	Rgn = Dgn * Zgn;

	t=newt;
	if mod(i,10)==9
	cost(1) = .5*norm(D*lout-X,'fro')^2;
	cost(2) =  lambda * double(sum(Pool(:)));
	cost(3) = .5*nu*norm(Pool-Rgn,'fro')^2;
	cost(4) = lambdagn*sum(Zgn(:));
	cost(5) = .5*beta*norm(lout,'fro')^2;
	cost(6) = .5*betagn*norm(Zgn,'fro')^2;
	fprintf('it %d totcost %4.2f [ %4.2f %4.2f %4.2f %4.2f ] \n',i+1, sum(cost), cost(1),cost(2), cost(3), cost(4))

	end

%
%	tmp = yr - t0*(Dsq * yr - DX + beta*yr + nu*phas1);
%	newout = reshape(tmp,KK,MM);
%	newouto= reshape(circshift(tmp,[0 round(time_groupsize/2)]),KK,MM);
%	%newout = y - t0*(Dsq * y- DX);
%	if nmf
%	newout = max(0,newout);
%        newouto = max(0,newouto);
%	end
%
%	newout1=reshape(newout(I0,:),ss,rr);
%	phas1=0*newout1;
%	PP(1,:)=sqrt(sum(newout1.^2));
%	normes=repmat(PP(1,:),[ss 1]);
%	lI=find(normes>0);
%	phas1(lI) = newout1(lI)./normes(lI);
%	phas1 = phas1 .* repmat(PP(1,:)-Rgn(1,:),[ss 1]);
%	newout1(lI) = newout1(lI).*(max(0,normes(lI)-tlambda)./normes(lI));
%	newout1 = reshape(newout1, KK,MM);
%	newout1= newout1(I1,:);
%	phas1 = reshape(phas1,KK,MM);
%	phas1 = phas1(I1,:);
%
%	newout=reshape(newout(I00,:),ss,rr);
%	phas2=0*newout;
%	PP(2,:)=sqrt(sum(newout.^2));
%	normes=repmat(PP(2,:),[ss 1]);
%	lI=find(normes>0);
%	phas2(lI) = newout(lI)./normes(lI);
%	phas2 = phas2 .* repmat(PP(2,:)-Rgn(2,:),[ss 1]);
%	newout(lI) = newout(lI).*(max(0,normes(lI)-tlambda)./normes(lI));
%	newout = reshape(newout, KK,MM);
%	newout= newout(I11,:);
%	phas2 = reshape(phas2,KK,MM);
%	phas2 = phas2(I11,:);
%	newout = .5*(newout1+newout);
%	phas1 = .5*(phas1+phas2);
%
%	newout1=reshape(newouto(I0,:),ss,rr);
%	phas3=0*newout1;
%	PP(3,:)=sqrt(sum(newout1.^2));
%	normes=repmat(PP(3,:),[ss 1]);
%	lI=find(normes>0);
%	phas3(lI) = newout1(lI)./normes(lI);
%	phas3 = phas3 .* repmat(PP(3,:)-Rgn(3,:),[ss 1]);
%	newout1(lI) = newout1(lI).*(max(0,normes(lI)-tlambda)./normes(lI));
%	newout1 = reshape(newout1, KK,MM);
%	newout1= newout1(I1,:);
%	phas3 = reshape(phas3,KK,MM);
%	phas3 = phas3(I1,:);
%	
%        newouto=reshape(newouto(I00,:),ss,rr);
%	phas4=0*newouto;
%	PP(4,:)=sqrt(sum(newouto.^2));
%	normes=repmat(PP(4,:),[ss 1]);
%	lI=find(normes>0);
%	phas4(lI) = newouto(lI)./normes(lI);
%	phas4 = phas4 .* repmat(PP(4,:)-Rgn(4,:),[ss 1]);
%	newouto(lI) = newouto(lI).*(max(0,normes(lI)-tlambda)./normes(lI));
%	newouto = reshape(newouto, KK,MM);
%	newouto= newouto(I11,:);
%	phas4 = reshape(phas4,KK,MM);
%	phas4 = phas4(I11,:);
%	newouto = .5*(newout1+newouto);
%	phas3 = .5*(phas3+phas4);
%
%	tmp = reshape(circshift(reshape(newouto,K,Mf),[0 -round(time_groupsize/2)]),KK,MM) ;
%	%tmp(:,1:time_window/time_groupsize:end)=newout(:,1:time_window/time_groupsize:end);
%	newout = .5*(tmp+newout);
%	phas3 = reshape(circshift(reshape(phas3,K,Mf),[0 -round(time_groupsize/2)]),KK,MM) ;
%	phas1 = reshape(.5*(phas1+phas3), K, Mf);
%
%	newt = (1+ sqrt(1+4*t^2))/2;
%	y = newout + ((t-1)/newt)*(newout-lout);
%	lout=newout;
%	
%	%update the coefficients from the Dgn dictionary.
%	%and produce Rgn(i,:), 
%	%PP(1,:) and PP(2,:) form a vector of pooled coefficients
%	%PP(3,:) and PP(4,:) form the other one		
%	Pool(1:2:end,1:2:end)=reshape(PP(1,:),K/groupsize,MM);
%	Pool(2:2:end,1:2:end)=reshape(PP(2,:),K/groupsize,MM);
%	Pool(1:2:end,2:2:end)=reshape(PP(3,:),K/groupsize,MM);
%	Pool(2:2:end,2:2:end)=reshape(PP(4,:),K/groupsize,MM);
%
%	tmp = Zgn  - t0gn *(Dgnsq * Zgn - Dgn'*Pool + betagn * Zgn);	
%	tmp = (tmp > tlambdagn).*tmp;
%	Zgn = tmp + ((t-1)/newt)*(tmp - loutgn);
%	loutgn = tmp;
%	Pool=Dgn * Zgn;
%	chunk=Pool(1:2:end,1:2:end);
%	Rgn(1,:)=chunk(:);
%	chunk=Pool(2:2:end,1:2:end);
%	Rgn(2,:)=chunk(:);
%	chunk=Pool(1:2:end,2:2:end);
%	Rgn(3,:)=chunk(:);
%	chunk=Pool(2:2:end,2:2:end);
%	Rgn(4,:)=chunk(:);
%
%	t=newt;
%
%	if mod(i,10)==9
%	cost(1) = .5*norm(D*reshape(lout,K,Mf)-X,'fro')^2;
%	cost(2) =  lambda * double(sum(PP(:)));
%	cost(3) = .5*nu*norm(PP-Rgn,'fro')^2;
%	cost(4) = lambdagn*sum(Zgn(:));
%	cost(5) = .5*beta*norm(lout,'fro')^2;
%	cost(6) = .5*betagn*norm(Zgn,'fro')^2;
%	fprintf('it %d totcost %4.2f [ %4.2f %4.2f %4.2f %4.2f ] \n',i+1, sum(cost), cost(1),cost(2), cost(3), cost(4))
%
%	end

end
llout=reshape(lout,K,Mf);

Zout=gather(llout);
Zgnout = gather(Zgn);

end








