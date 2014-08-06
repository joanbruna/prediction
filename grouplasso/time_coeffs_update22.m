function [Aout,Bout,zout]= time_coeffs_update22( D, X, options, Ain,Bin, t0, iter)

%this is where I need to do all the changes
%reshape input, redefine the groups, apply the FISTA algo, 
%and then reshape again to produce the corresponding Aout,Bouts, alphas

verb=0;
if nargin < 6
Ain=0;
Bin=0;
t0 = .5 * (1/max(svd(D))^2);
iter=1;
verb=1;
end

%we assume groupsize=time_groupsize=2

%X=single(X);we assume single from the input
%D=single(D);

Xg=gpuArray(X);
Dg=gpuArray(D);
Aing=gpuArray(single(Ain));
Bing=gpuArray(single(Bin));

iters=getoptions(options,'alpha_iters',50);

t0 = t0 / 2 ; %options.time_groupsize;


[N,M]=size(X);
K=size(D,2);
KK=K * 2;%options.time_groupsize;
MM=M/2;%options.time_groupsize;
Dsq=Dg'*Dg;
DX = reshape(Dg'*Xg,KK,MM);
y = zeros(KK,MM,'single','gpuArray');
out = zeros(KK,MM,'single','gpuArray');

nmf=getoptions(options,'nmf', 0);
lambda = getoptions(options,'lambda',0.1);
groupsize=2;%getoptions(options,'groupsize',2);
lambda = t0 * lambda;% * (size(D,2)/K);
t=1;

ss=4;%groupsize*options.time_groupsize;
rr=KK*MM/ss;

%even groups
groups=(mod(floor([0:KK-1]/groupsize),K/groupsize) )+1;
[~,I0]=sort(groups);
I1=invperm(I0);

overlapping=getoptions(options,'overlapping',1);

%if overlapping

%odd groups
rien = circshift(reshape(groups,K,options.time_groupsize),[ceil(groupsize/2) 0]);
groupso=rien(:);
[~,I00]=sort(groupso);
I11=invperm(I00);


Dpinv = inv(3*t0*Dsq+eye(K,'single','gpuArray'));
DX0 = Dg'*Xg;


newout = zeros(KK,MM,'single','gpuArray');
newout1 = zeros(KK,MM,'single', 'gpuArray');
normes=zeros(ss,rr,'single','gpuArray');
I=zeros(ss,rr,'uint8','gpuArray');

for i=1:iters
	%if verb
	%fprintf('it %d \n',i)
	%end
	newout = y - t0*(reshape(Dsq * reshape(y,K,M),KK,MM)- DX);
	if nmf
	newout = max(0,newout);
	end
	%newout1 = ProximalFlat(aux, I0, I1, tparam.lambda,ss,rr);
	newout1=reshape(newout(I0,:),ss,rr);
	normes=repmat(sqrt(sum(newout1.^2)),[ss 1]);
	I=find(normes>0);
	newout1(I) = newout1(I).*(max(0,normes(I)-lambda)./normes(I));
	newout1 = reshape(newout1, KK,MM);
	newout1= newout1(I1,:);
	
	%newout2 = ProximalFlat(aux, I00, I11, tparam.lambda,ss,rr);
	newout=reshape(newout(I00,:),ss,rr);
	normes=repmat(sqrt(sum(newout.^2)),[ss 1]);
	I=find(normes>0);
	newout(I) = newout(I).*(max(0,normes(I)-lambda)./normes(I));
	newout = reshape(newout, KK,MM);
	newout= newout(I11,:);

	newout = .5*(newout1+newout);
	newt = (1+ sqrt(1+4*t^2))/2;
	y = newout + ((t-1)/newt)*(newout-out);
	out=newout;
	t=newt;
end

out=reshape(out,K,M);
zout=gather(out);
rho=5;

Aout = gather((((iter-1)/iter)^rho)*Aing + out*out');
Bout = gather((((iter-1)/iter)^rho)*Bing + Xg*out');

end

