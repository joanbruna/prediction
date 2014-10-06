function [theta,estim] = optflow_taylor_temp(z, options , theta)
%this computes optical flow using simple taylor expansion

[N, L] = size(z);
h=zeros(N,1);
h(1)=1;
h(end)=-1;

itersflow=getoptions(options,'flow_iters',20);
lambda=getoptions(options,'lambda',0.1);
lambdar=getoptions(options,'lambdar',0);
lambdat=getoptions(options,'lambdat',0.1);
options.lambdat = lambdat;

% Smoothing filter
hn = getoptions(options,'hn',5);
sigma = getoptions(options,'sigma',0.1);
h = fspecial('gaussian',[hn;1],sigma);
S = imfilter(eye(N),h,'circ');

% Gradient
Gx=eye(N);
Gx=Gx - circshift(Gx,[-1 0]);
%G=G(1:end-1,:);
Gx2 = Gx'*Gx;
zb = S*z;
gradz=S*Gx*z;

zbis=0*zb;
zbis(:,1:end-1)=zb(:,2:end);
zb(:,end)=0;
zdif = zbis - zb;

Gt=eye(N);
Gt=Gt - circshift(Gt,[0 1]);
Gt=Gt(1:end-1,:);
%G(1,end)=0;
Gt2=Gt'*Gt;

if nargin < 3
theta = zeros(size(z));
end

for j=1:itersflow

for l=1:L
    if l==1
        thetab = zeros(N,1);
    else
        thetab = theta(:,l-1);
    end
    if l==L
        thetaf = zeros(N,1);
    else
        thetaf = theta(:,l+1);
    end
    theta(:,l) = (diag(gradz(:,l).^2) + lambda *Gt2  + (lambdar + 2*lambdat) * eye(N))\(gradz(:,l).*zdif(:,l) + lambdat*(thetaf + thetab));
    
end

end

%gradz=real(ifft(repmat(hf,1,L).*zf));
estim = zb + gradz.*theta;

[c,rec,dt2,t2] = getCost(zdif,Gt, gradz, theta,options);
c
% [c,rec,dt2,t2]
% [c,rec,dt2,t2] = getCost(zdif,Gt, gradz, theta0,options);
% [c,rec,dt2,t2]

end


function [c,rec,t2,dt2] = getCost(zdiff,Gt, gradz, theta,param)



dt2 = Gt*theta;
dt2 = 0.5*sum(dt2(:).^2);
t2 = 0.5*sum(theta(:).^2);

rec = 0.5*(zdiff - gradz.*theta).^2;

rec = sum(rec(:));

thetaf = 0*theta;
thetaf(:,1:end-1) = theta(:,2:end);
df = (thetaf - theta).^2;

thetab = 0*theta;
thetab(:,2:end) = theta(:,1:end-1);
db = (thetab - theta).^2;

d = 0.5*(sum(df(:))+sum(db(:)));

c = rec + param.lambda*dt2 + param.lambdar*t2 + param.lambdat*d;

end
