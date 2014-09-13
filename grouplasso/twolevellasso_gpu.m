function [Z, Zgn] = twolevellasso_gpu(Xin, Din, Dgnin, param)
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
KK=K * time_groupsize;

t0 = getoptions(options,'alpha_step',0.5);
t0 = t0 * (1/max(svd(D))^2)
lambda = getoptions(options,'lambda',0.1);
itersout=getoptions(options,'alpha_itersout',20);
nmf=getoptions(options,'nmf', 0);
tlambda = t0 * lambda;% * (size(D,2)/K);

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
normes1=zeros(ss,rr,'single','gpuArray');
normes2=zeros(ss,rr,'single','gpuArray');
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

for i=1:itersout
	yr = reshape(y,K,Mf);
	
	tmp = yr - t0*(Dsq * yr - DX + phas1);
	newout = reshape(tmp,KK,MM);
	newouto= reshape(circshift(tmp,[0 round(time_groupsize/2)]),KK,MM);
	%newout = y - t0*(Dsq * y- DX);
	if nmf
	newout = max(0,newout);
        newouto = max(0,newouto);
	end

	newout1=reshape(newout(I0,:),ss,rr);
	normes1=repmat(sqrt(sum(newout1.^2)),[ss 1]);
	lI=find(normes1>0);
	phas1(lI) = newout1(lI)./normes1(lI);
	phas1 = phas1 .* repmat(normes1(1,:)-Rgn(1,:),[ss 1]);
	newout1(lI) = newout1(lI).*(max(0,normes1(lI)-tlambda)./normes1(lI));
	newout1 = reshape(newout1, KK,MM);
	newout1= newout1(I1,:);
	phas1 = reshape(phas1,KK,MM);
	phas1 = phas1(I1,:);

	newout=reshape(newout(I00,:),ss,rr);
	normes2=repmat(sqrt(sum(newout.^2)),[ss 1]);
	lI=find(normes2>0);
	phas2(lI) = newout(lI)./normes2(lI);
	phas2 = phas2 .* repmat(normes2(1,:)-Rgn(2,:),[ss 1]);
	newout(lI) = newout(lI).*(max(0,normes2(lI)-tlambda)./normes2(lI));
	newout = reshape(newout, KK,MM);
	newout= newout(I11,:);
	phas2 = reshape(phas2,KK,MM);
	phas2 = phas2(I11,:);
	newout = .5*(newout1+newout);
	phas1 = .5*(phas1+phas2);

	newout1=reshape(newouto(I0,:),ss,rr);
	normes1=repmat(sqrt(sum(newout1.^2)),[ss 1]);
	lI=find(normes1>0);
	phas3(lI) = newout1(lI)./normes1(lI);
	phas3 = phas3 .* repmat(normes1(1,:)-Rgn(3,:),[ss 1]);
	newout1(lI) = newout1(lI).*(max(0,normes1(lI)-tlambda)./normes1(lI));
	newout1 = reshape(newout1, KK,MM);
	newout1= newout1(I1,:);
	phas3 = reshape(phas3,KK,MM);
	phas3 = phas3(I1,:);
	
        newouto=reshape(newouto(I00,:),ss,rr);
	normes2=repmat(sqrt(sum(newouto.^2)),[ss 1]);
	lI=find(normes2>0);
	phas4(lI) = newouto(lI)./normes2(lI);
	phas4 = phas4 .* repmat(normes2(1,:)-Rgn(4,:),[ss 1]);
	newouto(lI) = newouto(lI).*(max(0,normes2(lI)-tlambda)./normes2(lI));
	newouto = reshape(newouto, KK,MM);
	newouto= newouto(I11,:);
	phas4 = reshape(phas4,KK,MM);
	phas4 = phas4(I11,:);
	newouto = .5*(newout1+newouto);
	phas3 = .5*(phas3+phas4);

	tmp = reshape(circshift(reshape(newouto,K,Mf),[0 -round(time_groupsize/2)]),KK,MM) ;
	%tmp(:,1:time_window/time_groupsize:end)=newout(:,1:time_window/time_groupsize:end);
	newout = .5*(tmp+newout);
	phas3 = reshape(circshift(reshape(phas3,K,Mf),[0 -round(time_groupsize/2)]),KK,MM) ;
	phas1 = reshape(.5*(phas1+phas3), K, Mf);

	newt = (1+ sqrt(1+4*t^2))/2;
	y = newout + ((t-1)/newt)*(newout-lout);
	lout=newout;
	
	


	t=newt;
end
llout=reshape(lout,K,Mf);

lastzout=gather(llout);

end








