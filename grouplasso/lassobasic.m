function out= lassobasic( X,D, options)

%this is where I need to do all the changes
%reshape input, redefine the groups, apply the FISTA algo, 
%and then reshape again to produce the corresponding Aout,Bouts, alphas

iters=120;
% iters_encoder=getoptions(options,'alpha_iters_encoder',60);


%alpha=getoptions(options,'iir_param',(.02)^(1/size(X,2)));
% alpha = 0.9;

[N,M]=size(X);
K=size(D,2);
% KK=K * options.time_groupsize;
% MM=M/options.time_groupsize;
DX = D'*X;
Dsq = D'*D;
y = zeros(K,M);


t0 = .5 * (1/(norm(D,2)^2 )) ;

out = y;

tparam.regul='l1';
lambda = getoptions(options,'lambda',0.1);

tparam.lambda = t0 * lambda;% * (size(D,2)/K);
t=1;

for i=1:iters
    
	aux = y - t0*(Dsq * y - DX);
	newout = mexProximalFlat(aux, tparam);

	newt = (1+ sqrt(1+4*t^2))/2;
	y = newout + ((t-1)/newt)*(newout-out);
	out=newout;
	t=newt;
end


end


