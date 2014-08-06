function [y,z] = nmf_semisup(X,D,W,A,options)

iters=10;

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
Dsq = D'*D;
DW = D'*W;
WD = W'*D;
Asq = A'*A;
Wsq = W'*W;


H = getoptions(options,'H',[]);

if isempty(H)
    y = zeros(K,M);
    z = zeros(Kw,M);
else
    if size(H,1)~= K+Kw
       error('Size of H do not match size of D and Wn');
    end
    y = H(1:K,:);
    z = H(K+1:end,:);
end

mu = getoptions(options,'mu',0);

t0 = .5 * (1/(norm(D,2)^2 +norm(W,2)^2 + mu^2*norm(A,2)^2 + mu^2 )) ;
tz = .5 *( 1/norm(W,2)^2 ) ;

out = y;

tparam.regul='l1';
lambda = getoptions(options,'lambda',0.1);



%tparam.regul='l1';
% tparam.groups=int32(mod(floor([0:KK-1]/groupsize),K/groupsize) )+1;
%keyboard
tparam.lambda = t0 * lambda;% * (size(D,2)/K);
tparam.pos = true; % impose non-negativity
t=1;


obj0 = compute_obj(X,[y;z],D,W,options);


for i=1:iters
    
     yt = y; yt(:,end) = 0;
     yt1 = [y(:,2:end) zeros(K,1)];
     ym = [zeros(K,1) y(:,1:end-1)];
    
	aux_y = y - t0*(Dsq * y + DW*z - DX) - t0*mu*(Asq*yt - A'*yt1) - t0*mu*(y - A*ym );
    
    z = max(z - tz*(Wsq * z + WD*y - WX),0);
	
    % proximal projection on y
    newout = mexProximalFlat(aux_y, tparam);
    
    y = newout;
% 	newt = (1+ sqrt(1+4*t^2))/2;
% 	y = newout + ((t-1)/newt)*(newout-out);
% 	out=newout;
% 	t=newt;
    
    %obj(i) = compute_obj(X,[y;z],D,W,options);
    
%     compute_obj(X,[y;z],D,W,options,A)
    
end



end

    