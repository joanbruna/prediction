function [Aout,Bout,out]= nn_time_coeffs_update( D, X, options, Ain,Bin, t0)

%this is where I need to do all the changes
%reshape input, redefine the groups, apply the FISTA algo, 
%and then reshape again to produce the corresponding Aout,Bouts, alphas


if nargin < 6
Ain=0;
Bin=0;
t0 = .5 * (1/max(svd(D))^2);
end


iters=getoptions(options,'alpha_iters',50);
iters_encoder=getoptions(options,'alpha_iters_encoder',60);

t0 = t0 / options.time_groupsize;

%alpha=getoptions(options,'iir_param',(.02)^(1/size(X,2)));
alpha = 0.9;

[N,M]=size(X);
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
tparam.groups=int32(mod(floor([0:KK-1]/groupsize),K/groupsize) )+1;
%keyboard
tparam.lambda = t0 * lambda;% * (size(D,2)/K);
t=1;

%for i=min(tparam.groups):max(tparam.groups)
%	II{i}=find(tparam.groups==i);
%end
[~,I0]=sort(tparam.groups);
I1=invperm(I0);


ss=groupsize*options.time_groupsize;
rr=KK*MM/ss;

for i=1:iters
	%aux=y;
	tempo = reshape(Dsq * reshape(y,K,M),KK,MM);
	aux = y - t0*(tempo - DX);
	%newout = mexProximalFlat(aux, tparam);
	newout = ProximalFlat(aux, I0, I1, tparam.lambda,ss,rr);
	%non-negative factorization
	newout = max(0, newout);
	newt = (1+ sqrt(1+4*t^2))/2;
	y = newout + ((t-1)/newt)*(newout-out);
	out=newout;
	t=newt;
end

out=reshape(out,K,numel(out)/K);

%Aout = ((iter-1)*Ain + out*out')/iter;
%Bout = ((iter-1)*Bin + ref*out')/iter;
Aout = alpha * Ain + (1-alpha)*(out*out');
Bout = alpha * Bin + (1-alpha)*(X*out');

end


%else
% y = zeros(size(D,2),M);
%end
%if iter > 0
%D=repmat(D,[1 options.time_groupsize]);
%if ~fista
%out = mexProximalFlat(out-t0*(Dsq*out-DX), tparam);
%else

%groupind = mod([0:M-1],MM);
%batchind = floor([0:M-1]/MM);
%permindex = 1 + batchind + groupind*options.time_groupsize;
%ipermindex = invperm(permindex);
	%tempo = tempo(:,permindex);
	%tempo = reshape(tempo,MM*K,options.time_groupsize);
	%tempo = repmat(sum(tempo,2),[1, options.time_groupsize]);
	%tempo = reshape(tempo,K,M);
	%tempo = reshape(tempo(:,ipermindex),KK,MM);
%else
%Aout=out*out';
%Bout=X*out';
%end

