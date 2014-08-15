function A = nmf_optflow( X, D, A, options,t0)
%function out= nmf_linear_dynamic_pursuit( X,D,A, options)

%this is where I need to do all the changes
%reshape input, redefine the groups, apply the FISTA algo, 
%and then reshape again to produce the corresponding Aout,Bouts, alphas

iters=200;
% iters_encoder=getoptions(options,'alpha_iters_encoder',60);


%alpha=getoptions(options,'iir_param',(.02)^(1/size(X,2)));
% alpha = 0.9;

[~,M]=size(X);
K=size(D,2);
% KK=K * options.time_groupsize;
% MM=M/options.time_groupsize;
DX = D'*X;
Dsq = D'*D;
Asq = A'*A;
y = zeros(K,M);

mu = getoptions(options,'mu',0.5);

t0 = .5 * (1/(norm(D,2)^2 + mu^2*norm(A,2)^2 + mu^2 )) ;

out = y;

tparam.regul='l1';
lambda = getoptions(options,'lambda',0.1);


tparam.lambda = t0 * lambda;% * (size(D,2)/K);
tparam.pos = 'true'; % impose non-negativity
t=1;


for i=1:iters
    
     yt = y; yt(:,end) = 0;
     yt1 = [y(:,2:end) zeros(K,1)];
     ym = [zeros(K,1) y(:,1:end-1)];
    
	aux = y - t0*(Dsq * y - DX) - t0*mu*(Asq*yt - A'*yt1) - t0*mu*(y - A*ym );
	newout = mexProximalFlat(aux, tparam);

	newt = (1+ sqrt(1+4*t^2))/2;
	y = newout + ((t-1)/newt)*(newout-out);
	out=newout;
	t=newt;
end


end