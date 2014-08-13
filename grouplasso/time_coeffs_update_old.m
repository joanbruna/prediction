function [Aout,Bout,out,costout]= time_coeffs_update( D, X, options, Ain,Bin, t0, iter)

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

costout=0;

iters=getoptions(options,'alpha_iters',50);
iters_encoder=getoptions(options,'alpha_iters_encoder',60);

t0 = t0 / options.time_groupsize;


[~,M]=size(X);
K=size(D,2);
KK=K * options.time_groupsize;
MM=M/options.time_groupsize;
Dsq=D'*D;
DX = reshape(D'*X,KK,MM);
y = zeros(KK,MM);

out = y;

tparam.regul='group-lasso-l2';
lambda = getoptions(options,'lambda',0.1);
%tparam.regul='l1';
groupsize=getoptions(options,'groupsize',2);
%keyboard
tparam.lambda = t0 * lambda;% * (size(D,2)/K);
t=1;

ss=groupsize*options.time_groupsize;
rr=KK*MM/ss;

%even groups
groups=(mod(floor([0:KK-1]/groupsize),K/groupsize) )+1;
[~,I0]=sort(groups);
I1=invperm(I0);

overlapping=getoptions(options,'overlapping',1);
nmf=getoptions(options,'nmf', 0);

if overlapping
%odd groups
rien = circshift(reshape(groups,K,options.time_groupsize),[ceil(groupsize/2) 0]);
groupso=rien(:);
[~,I00]=sort(groupso);
I11=invperm(I00);

%keyboard;

if 0
Dpinv = pinv(3*t0*Dsq+eye(K));
DX0 = D'*X;
%this is wrong! I am using a hacked version of fista, but the PPXA code below does not work so far. TODO fix it
%y1=y;
%y2=y;
%y3=y;
%for i=1:iters
%
%	new1=reshape(Dpinv*(reshape(y1,K,M)+DX0),KK,MM);
%	new2 = ProximalFlat(y2,I0, I1, 3*tparam.lambda,ss,rr);
%	new3 = ProximalFlat(y3,I00,I11,3*tparam.lambda,ss,rr);
%
%	new=(new1+new2+new3)/3;
%
%	y1=y1+new-new1;
%	y2=y2+new-new2;
%	y3=y3+new-new3;
%
%
%end
%out=new;
y1=y;
y2=y;
out1=y;
out2=y;
for i=1:iters
	%y is x
	tempo = reshape(Dsq * reshape(y, K, M),KK,MM);
	aux1 = 2*y - y1 - t0*(tempo - DX);
	aux2 = 2*y - y2 - t0*(tempo - DX);
	y1 = y1 + ProximalFlat(aux1, I0, I1, 2*tparam.lambda, ss, rr) - y;
	y2 = y2 + ProximalFlat(aux2, I00, I11, 2*tparam.lambda, ss, rr) - y;
	%y = .5*(y1+y2);
	y = y2;
	disp('heree')
%	tempo = reshape(Dsq * reshape(y,K,M),KK,MM);
%	aux1 = 2*y -y1 - t0*(tempo - DX);
%	aux2 = 2*y -y2 - t0*(tempo - DX);
%	newout1 = ProximalFlat(aux1, I0, I1, 2*tparam.lambda,ss,rr);
%	newout2 = ProximalFlat(aux2, I00, I11, 2*tparam.lambda,ss,rr);
%	%newout = .5*(newout1+newout2);
%	%newout = newout2;
%	newt = (1+ sqrt(1+4*t^2))/2;
%	y1 = newout1 + ((t-1)/newt)*(newout1-out1);
%	y2 = newout2 + ((t-1)/newt)*(newout2-out2);
%	y=.5*(y1+y2);
%	out1=newout1;
%	out2=newout2;
%	t=newt;

end
out=y;
else
newout1=y;
newout2=y;
for i=1:iters
%	if verb
%	fprintf('it %d \n',i)
%	end
	tempo = reshape(Dsq * reshape(y,K,M),KK,MM);
	aux = y - t0*(tempo - DX);
	if nmf
	aux = max(0,aux);
	end
	newout1 = ProximalFlat(aux, I0, I1, 2*tparam.lambda,ss,rr);
	newout2 = ProximalFlat(aux, I00, I11, 2*tparam.lambda,ss,rr);
	newout = .5*(newout1+newout2);
	%newout = newout2;
	newt = (1+ sqrt(1+4*t^2))/2;
	y = newout + ((t-1)/newt)*(newout-out);
	out=newout;
	t=newt;
end
end

if getoptions(options,'measure_cost',0);
	rec = D * reshape(out,K,numel(out)/K);
	c1 = .5*norm(X(:)-rec(:)).^2;
	[~,c21]=ProximalFlat(out,I0,I1,tparam.lambda,ss,rr);	
	[~,c22]=ProximalFlat(out,I00,I11,tparam.lambda,ss,rr);	
	costout = c1 + lambda * (sum(c21) + sum(c22));
	fprintf('costs are %f %f total %f \n', c1, lambda*(sum(c21)+sum(c22)),costout );
end



else


for i=1:iters
	tempo = reshape(Dsq * reshape(y,K,M),KK,MM);
	aux = y - t0*(tempo - DX);
	newout = ProximalFlat(aux, I0, I1, tparam.lambda,ss,rr);
	newt = (1+ sqrt(1+4*t^2))/2;
	y = newout + ((t-1)/newt)*(newout-out);
	out=newout;
	t=newt;
end

end

out=reshape(out,K,numel(out)/K);

rho=5;

Aout = (((iter-1)/iter)^rho)*Ain + out*out';
Bout = (((iter-1)/iter)^rho)*Bin + X*out';
%Aout = alpha * Ain + (1-alpha)*(out*out');
%Bout = alpha * Bin + (1-alpha)*(X*out');

end



