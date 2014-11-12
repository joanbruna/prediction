function [y,z,obj] = nmf_semisup(X,D,W,A,options)

%tau =0.5;
tau = getoptions(options,'tau',0);
%iters=400;
iters = getoptions(options,'iter',400);

%alpha=getoptions(options,'iir_param',(.02)^(1/size(X,2)));
% alpha = 0.9;

if isempty(A)
   A = eye(size(D,2)); 
end

[N,M]=size(X);
K=size(D,2);
Kw=size(W,2);
% KK=K * options.time_groupsize;
% MM=M/options.time_groupsize;
DX = D'*X;
WX = W'*X;

Dsq = getoptions(options,'Dsq',0);
if Dsq == 0
Dsq = D'*D;
end
DW = D'*W;
WD = W'*D;

Wsq = W'*W;

mu = getoptions(options,'mu',0);
if mu>0
    Asq = A'*A;
end


H = getoptions(options,'H',-1);

if H==-1
    y = zeros(K,M);
    z = zeros(Kw,M);
else
    if size(H,1)~= K+Kw
       error('Size of H do not match size of D and Wn');
    end
    y = H(1:K,:);
    z = H(K+1:end,:);
end



t0 = getoptions(options,'t0',0);
if t0==0
    t0 = .5 * (1/(norm(D,2)^2 +norm(W,2)^2 + mu^2*norm(A,2)^2 + mu^2 + tau^2)) ;
end

out = y;

tparam.regul='l1';
lambda = getoptions(options,'lambda',0.1);


tparam.lambda = t0 * lambda;% * (size(D,2)/K);
tparam.pos = true; % impose non-negativity
t=1;


%obj0 = compute_obj(X,[y;z],D,W,options);


for i=1:iters
        
	aux_y = y - t0*(Dsq * y + DW*z - DX);
    if mu~=0
        yt = y; yt(:,end) = 0;
        yt1 = [y(:,2:end) zeros(K,1)];
        ym = [zeros(K,1) y(:,1:end-1)];
        aux_y = aux_y - t0*mu*(Asq*yt - A'*yt1) - t0*mu*(y - A*ym );
    end
    z = max(z - t0*(Wsq * z + WD*y - WX + tau*z),0);
	
    % proximal projection on y
    newout = mexProximalFlat(aux_y, tparam);
    
    y = newout;
    
    %obj(i) = compute_obj(X,[y;z],D,W,options);
    
%     compute_obj(X,[y;z],D,W,options,A)
    
end
%keyboard

end


    