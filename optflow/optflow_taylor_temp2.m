function [Theta,estim] = optflow_taylor_temp2(z, options,Theta)
%this computes optical flow using simple taylor expansion

[N, L] = size(z);
h=zeros(N,1);
h(1)=1;
h(end)=-1;

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

if ~exist('Theta','var')
    iters = 100;
else
    Theta = zeros(size(z));
    iters = 500;
end

K = size(z,1);

GZzdif = gradz.*zdif;
GZsq = gradz.^2;

t0 = .5 * (1/(max(gradz(:))^2 + 2*lambdat^2 + lambdar^2 + lambda^2*norm(Gt)^2 ))/5;



for i=1:iters
    
%    c(i) = getCost(zdif,Gt, gradz, Theta,options);
    
    Theta2 = Theta; Theta2(:,end) = 0;
    Thetaf = [Theta(:,2:end) zeros(K,1)];
    Thetab = [zeros(K,1) Theta(:,1:end-1)];
    Theta1 = Theta; Theta1(:,1) = 0;
    
    % reconstruction term
    g_rec =  GZsq.*Theta - GZzdif;

    % ridge
    g_ridge = Theta;
    
    % spatial smoothness
    g_sp_smooth = Gt2*Theta;
    
    % temporal smoothness
    g_temp_smooth = (Theta1 - Thetab) + (Theta2 - Thetaf);
    
    % do gradient descent
    Theta = Theta - t0*(g_rec + lambdar*g_ridge + lambdat*g_temp_smooth + lambda*g_sp_smooth);
    
    
	 
end


estim = zb + gradz.*Theta;


end


function [c,rec,t2,dt2] = getCost(zdiff,Gt, gradz, theta,param)



dt2 = Gt*theta;
dt2 = 0.5*sum(dt2(:).^2);

t2 = 0.5*sum(theta(:).^2);

rec = (zdiff - gradz.*theta).^2;

rec = 0.5*sum(rec(:));

thetaf = 0*theta;
thetaf(:,1:end-1) = theta(:,2:end);
thetam = theta; thetam(:,end)= 0;
df = (thetaf - thetam).^2;

thetab = 0*theta;
thetab(:,2:end) = theta(:,1:end-1);
thetam = theta; thetam(:,1)= 0;
db = (thetab - thetam).^2;

d = 0.5*(sum(df(:))+sum(db(:)));

c = rec + param.lambda*dt2 + param.lambdar*t2 + param.lambdat*d;

end
