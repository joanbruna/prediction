function [out,z,costout]= group_pooling_semisup( X, D, W, options)

%this is where I need to do all the changes
%reshape input, redefine the groups, apply the FISTA algo, 
%and then reshape again to produce the corresponding Aout,Bouts, alphas


costout=0;

iters=getoptions(options,'alpha_iters',50);


t0 = .5 * (1/(norm(D,2)^2 +norm(W,2)^2));
tz = .5 *( 1/norm(W,2)^2 );
t0 = t0 / options.time_groupsize;
tz = t0;

[N,M]=size(X);
K=size(D,2);
KK=K * options.time_groupsize;
MM=M/options.time_groupsize;
Dsq=D'*D;
DX = reshape(D'*X,KK,MM);
y = zeros(KK,MM);

Kw = size(W,2);
z = zeros(Kw,M);

WX = W'*X;
DW = D'*W;
WD = W'*D;
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


iters = 10;

for i=1:iters
  
    
	tempo = reshape(Dsq * reshape(y,K,M),KK,MM);
	aux = y - t0*(tempo + DW*z - DX);
    
    z = z - t0*(Wsq * z + WD*y - WX);
    
	if nmf
	aux = max(0,aux);
    z = max(0,z);
	end
	newout1 = ProximalFlat(aux, I0, I1, 2*tparam.lambda,ss,rr);
	newout2 = ProximalFlat(aux, I00, I11, 2*tparam.lambda,ss,rr);
	newout = .5*(newout1+newout2);
	
% 	newt = (1+ sqrt(1+4*t^2))/2;
% 	y = newout + ((t-1)/newt)*(newout-out);
 	y=newout;
    out=newout;
	%t=newt;
    
    %obj(i) = cost(X,D,W,out,z,I0,I00,I1,I11,tparam,rr,ss);
    
end

if getoptions(options,'measure_cost',0);
% 	rec = D * reshape(out,K,numel(out)/K) + W*z;
% 	c1 = .5*norm(X(:)-rec(:)).^2;
% 	[~,c21]=ProximalFlat(out,I0,I1,tparam.lambda,ss,rr);	
% 	[~,c22]=ProximalFlat(out,I00,I11,tparam.lambda,ss,rr);	
% 	costout = c1 + lambda * (sum(c21) + sum(c22));
    [costout,c1,c21,c22] = cost(X,D,W,out,z,I0,I00,I1,I11,tparam,rr,ss);
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


end


function [obj,c1,c21,c22] = cost(X,D,W,out,z,I0,I00,I1,I11,tparam,rr,ss)

K=size(D,2);

rec = D * reshape(out,K,numel(out)/K) + W*z;
c1 = .5*norm(X(:)-rec(:)).^2;
[~,c21]=ProximalFlat(out,I0,I1,tparam.lambda,ss,rr);
[~,c22]=ProximalFlat(out,I00,I11,tparam.lambda,ss,rr);
obj = c1 + tparam.lambda * (sum(c21) + sum(c22));
end



