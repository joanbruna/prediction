function [Dout,lastzout] = group_pooling_st_gpu(Xin, options)
%this function performs a dictionary learning using 
%the proximal toolbox and iterated gradient descent
%from Mairal et Al (2010)
%requires the spams proximal operator toolbox 

%we learn a dictionary which maximizes group sparsity,
%where groups are bi-clusters.

%Joan Bruna 2013 Courant Institute


[N,M]=size(Xin);
X=gpuArray(single(Xin));

renorm=getoptions(options,'renorm_input', 0);
if renorm
norms=zeros(1,M,'single','gpuArray');
norms = sqrt(sum(X.^2));
I0=find(norms>0);
norms=norms(I0);
X(:,I0)=X(:,I0) ./ repmat(norms,[size(X,1) 1]);
end

produce_synthesis=getoptions(options,'produce_synthesis',0);
nepochs=getoptions(options,'epochs',4);
batchsize=getoptions(options,'batchsize',256);
groupsize = getoptions(options,'groupsize',2);
time_groupsize = getoptions(options,'time_groupsize',2);
time_window = getoptions(options,'time_window',8);

niters=round(nepochs*M/(batchsize*time_window));
K = getoptions(options, 'K', 2*N);
KK=K * time_groupsize;
MM=batchsize*time_window/time_groupsize;
Mf = MM*time_groupsize;
%N: input dimension
%M: number of examples
%K output dimension


%initial dictionary
D=zeros(N,K,'single','gpuArray');
II=randperm(M-1);
D=X(:,II(1:K));
if isfield(options,'initD')
D=gpuArray(single(options.initD));
end

B=zeros(N,K,'single','gpuArray');
A=zeros(K,K,'single','gpuArray');
Dsq = zeros(K,K,'single','gpuArray');
DX = zeros(KK,MM,'single','gpuArray');
DXo = zeros(KK,MM,'single','gpuArray');


%verbose variables
chunks=100;
ch = ceil(niters/chunks);
chunks = ceil(niters/ch);

t0 = getoptions(options,'alpha_step',0.5);
t0 = t0 * (1/max(svd(D))^2)
tot_tested=0;
options.batchsize=batchsize;
lambda = getoptions(options,'lambda',0.1);
beta=getoptions(options,'beta',5e-1);
iters=getoptions(options,'alpha_iters',200);
itersout=getoptions(options,'alpha_itersout',100);
diters=getoptions(options,'dict_iters',2);
nmf=getoptions(options,'nmf', 0);
overlapping=getoptions(options,'overlapping',1);
tlambda = t0 * lambda;% * (size(D,2)/K);
rho=getoptions(options,'rho',5);

%D0=D;
rast=1;
c1=0;
c2=0;

data = zeros(N, batchsize*time_window,'single','gpuArray');
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

II=(randperm(M-time_window));

niters

for n=1:niters
init= mod( (n-1)*batchsize, M-batchsize-time_window+0); 
BI0 = II(1+init:batchsize+init);
for tt=1:time_window
data(:,tt:time_window:end)=X(:,BI0+tt-1);
end

%[A,B,alpha] = time_coeffs_update22( D, data, options,A,B,t0, n);
%%%%%%%%

Dsq=D'*D;
DX = reshape(D'*data,KK,MM);
DXo = reshape(circshift(D'*data,[0 round(time_groupsize/2)]),KK,MM);
t=1;
y=0*y;
lout=0*lout;
for i=1:iters
	%newout = y - t0*(reshape(Dsq * reshape(y,K,Mf),KK,MM)- DX);
	yr = reshape(y,K,Mf);
	tmp = yr - t0*(Dsq * yr + beta*yr);
	newout = reshape(tmp,KK,MM)+t0*DX;
	newouto= reshape(circshift(tmp,[0 round(time_groupsize/2)]),KK,MM)+t0*DXo;
	%newout = y - t0*(Dsq * y- DX);
	if nmf
	newout = max(0,newout);
        newouto = max(0,newouto);
	end

	newout1=reshape(newout(I0,:),ss,rr);
	normes1=repmat(sqrt(sum(newout1.^2)),[ss 1]);
	lI=find(normes1>0);
	newout1(lI) = newout1(lI).*(max(0,normes1(lI)-tlambda)./normes1(lI));
	newout1 = reshape(newout1, KK,MM);
	newout1= newout1(I1,:);
	newout=reshape(newout(I00,:),ss,rr);
	normes2=repmat(sqrt(sum(newout.^2)),[ss 1]);
	lI=find(normes2>0);
	newout(lI) = newout(lI).*(max(0,normes2(lI)-tlambda)./normes2(lI));
	newout = reshape(newout, KK,MM);
	newout= newout(I11,:);
	newout = .5*(newout1+newout);

	newout1=reshape(newouto(I0,:),ss,rr);
	normes1=repmat(sqrt(sum(newout1.^2)),[ss 1]);
	lI=find(normes1>0);
	newout1(lI) = newout1(lI).*(max(0,normes1(lI)-tlambda)./normes1(lI));
	newout1 = reshape(newout1, KK,MM);
	newout1= newout1(I1,:);
	newouto=reshape(newouto(I00,:),ss,rr);
	normes2=repmat(sqrt(sum(newouto.^2)),[ss 1]);
	lI=find(normes2>0);
	newouto(lI) = newouto(lI).*(max(0,normes2(lI)-tlambda)./normes2(lI));
	newouto = reshape(newouto, KK,MM);
	newouto= newouto(I11,:);
	newouto = .5*(newout1+newouto);
	tmp = reshape(circshift(reshape(newouto,K,Mf),[0 -round(time_groupsize/2)]),KK,MM) ;
	tmp(:,1:time_window/time_groupsize:end)=newout(:,1:time_window/time_groupsize:end);
	newout = .5*(tmp+newout);

	newt = (1+ sqrt(1+4*t^2))/2;
	y = newout + ((t-1)/newt)*(newout-lout);
	lout=newout;
	t=newt;
end
llout=reshape(lout,K,Mf);

A = (((n-1)/n)^rho)*A + llout*llout';
B = (((n-1)/n)^rho)*B + data*llout';

%%%%%%%%
cost1 = norm(D*llout-data,'fro')^2 / norm(data,'fro')^2;
cost2 =  lambda * (sum(normes1(:))+sum(normes2(:)))/ss;
fprintf('it %d %f %f\n',n,cost1, cost2)
%measure_cost(alpha, D, data, lambda, groupsize, 'after lasso');

%%dictionary update
%D = dictionary_update( D,  A,B,options);
%%%%%
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
%t0 = .5 * (1/max(svd(D))^2)
%tlambda = t0 * lambda;% * (size(D,2)/K);
end

%%%%%%%%%
%verbo(rast)=measure_cost(alpha, D, data, lambda, groupsize, 'after dict update');
%rast=rast+1;

if mod(n,ch)==ch-1
%fprintf('done chunk %d of %d\n',ceil(n/ch),chunks )
end
end

if produce_synthesis
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
lI=zeros(ss,rr,'uint8','gpuArray');
newout = zeros(KK,MM,'single','gpuArray');
newouto = zeros(KK,MM,'single','gpuArray');
newout1 = zeros(KK,MM,'single', 'gpuArray');
normes1=zeros(ss,rr,'single','gpuArray');
normes2=zeros(ss,rr,'single','gpuArray');
DX = zeros(KK,MM,'single','gpuArray');
DXo = zeros(KK,MM,'single','gpuArray');
Dsq=D'*D;
DX = reshape(D'*X(:,1:Mf),KK,MM);
DXo = reshape(circshift(D'*X(:,1:Mf),[0 round(time_groupsize/2)]),KK,MM);
t=1;
for i=1:itersout
	%newout = y - t0*(reshape(Dsq * reshape(y,K,Mf),KK,MM)- DX);
	yr = reshape(y,K,Mf);
	tmp = yr - t0*(Dsq * yr + beta*yr);
	newout = reshape(tmp,KK,MM)+t0*DX;
	newouto= reshape(circshift(tmp,[0 round(time_groupsize/2)]),KK,MM)+t0*DXo;
	%newout = y - t0*(Dsq * y- DX);
	if nmf
	newout = max(0,newout);
        newouto = max(0,newouto);
	end

	newout1=reshape(newout(I0,:),ss,rr);
	normes1=repmat(sqrt(sum(newout1.^2)),[ss 1]);
	lI=find(normes1>0);
	newout1(lI) = newout1(lI).*(max(0,normes1(lI)-tlambda)./normes1(lI));
	newout1 = reshape(newout1, KK,MM);
	newout1= newout1(I1,:);
	newout=reshape(newout(I00,:),ss,rr);
	normes2=repmat(sqrt(sum(newout.^2)),[ss 1]);
	lI=find(normes2>0);
	newout(lI) = newout(lI).*(max(0,normes2(lI)-tlambda)./normes2(lI));
	newout = reshape(newout, KK,MM);
	newout= newout(I11,:);
	newout = .5*(newout1+newout);

	newout1=reshape(newouto(I0,:),ss,rr);
	normes1=repmat(sqrt(sum(newout1.^2)),[ss 1]);
	lI=find(normes1>0);
	newout1(lI) = newout1(lI).*(max(0,normes1(lI)-tlambda)./normes1(lI));
	newout1 = reshape(newout1, KK,MM);
	newout1= newout1(I1,:);
	newouto=reshape(newouto(I00,:),ss,rr);
	normes2=repmat(sqrt(sum(newouto.^2)),[ss 1]);
	lI=find(normes2>0);
	newouto(lI) = newouto(lI).*(max(0,normes2(lI)-tlambda)./normes2(lI));
	newouto = reshape(newouto, KK,MM);
	newouto= newouto(I11,:);
	newouto = .5*(newout1+newouto);
	tmp = reshape(circshift(reshape(newouto,K,Mf),[0 -round(time_groupsize/2)]),KK,MM) ;
	%tmp(:,1:time_window/time_groupsize:end)=newout(:,1:time_window/time_groupsize:end);
	newout = .5*(tmp+newout);

	newt = (1+ sqrt(1+4*t^2))/2;
	y = newout + ((t-1)/newt)*(newout-lout);
	lout=newout;
	t=newt;
end
llout=reshape(lout,K,Mf);

%%%
end


Dout=gather(D);
lastzout=gather(llout);

end


function D= dictionary_update(Din, A,B,options)

iters=getoptions(options,'dict_iters',2);
nmf=getoptions(options,'nmf', 0);


D=Din;

N=size(B,1);
dia = diag(A)';

%lr=1e-2;
tol=1e-8;
I=find(dia>tol);
fix=0;

if length(I) < length(dia)

dia=dia(I);
D0=D(:,I);
B=B(:,I);
A=A(I,I);
fix=1;
else
D0=D;
end

At=(dia.^(-1));
Att=repmat(At,[size(B,1) 1]);

K=size(D0,2);
Ip = randperm(K);

for i=1:iters

for j=1:K

u = D0(:,Ip(j)) + (B(:,Ip(j)) - D0*(A(:,Ip(j))))*At(Ip(j));
if nmf
u = max(0,u);
end
D0(:,Ip(j)) = u / max(1, norm(u));

end
end

if fix
D(:,I)=D0;
else
D=D0;
end
 
%D = ortho_pools(D',2)';
Ds1 = D(:,1:2:end);
Ds2 = D(:,2:2:end);
corrs = abs(sum(Ds1.*Ds2));
Dtmp = circshift(D,[0 -1]);
Ds1b = Dtmp(:,1:2:end);
Ds2b = Dtmp(:,2:2:end);
corrsb = abs(sum(Ds1b.*Ds2b));
fprintf('dictionary group coherence (even): %f %f %f \n',min(corrs), max(corrs), median(corrs))
fprintf('dictionary group coherence (odd): %f %f %f \n',min(corrsb), max(corrsb), median(corrsb))

end


function out=measure_cost(alpha, D, data, lambda, groupsize, str)

batchsize = size(data,2);
rec = D * alpha;
modulus = modphas_decomp(alpha,groupsize);
c1 = norm(rec(:)-data(:))^2/batchsize;
c2 = lambda * sum(modulus(:))/batchsize;
c3 = sum(modulus(:)>0)/numel(modulus);
out=c1+c2;
fprintf( '%s...%f (%f %f) nonzeros %f \n', str, c1+c2,c1,c2,c3)

end





