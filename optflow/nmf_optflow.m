function [out,cout] = nmf_optflow( X, D, Theta, options,y)
%function out= nmf_linear_dynamic_pursuit( X,D,A, options)

%this is where I need to do all the changes
%reshape input, redefine the groups, apply the FISTA algo, 
%and then reshape again to produce the corresponding Aout,Bouts, alphas

iters=500;
% iters_encoder=getoptions(options,'alpha_iters_encoder',60);


%alpha=getoptions(options,'iir_param',(.02)^(1/size(X,2)));
% alpha = 0.9;

[N,M]=size(X);
K=size(D,2);


fista = getoptions(options,'fista',1);


lambda_t = getoptions(options,'lambda_t',0.1);
lambda_tr = getoptions(options,'lambda_tr',0.1);

% Smoothing filter
hn = getoptions(options,'hn',5);
sigma = getoptions(options,'sigma',0.1);
h = fspecial('gaussian',[hn;1],sigma);
S = imfilter(eye(K),h,'circ');

% Gradient
G=eye(K);
G=G - circshift(G,[-1 0]);
%G=G(1:end-1,:);

A=S*G;

% KK=K * options.time_groupsize;
% MM=M/options.time_groupsize;
DX = D'*X;
Dsq = D'*D;
Asq = A'*A;


if ~exist('y','var')
y = zeros(K,M);
end

Theta2 = Theta.^2;
%Thetat1 = [Theta(:,2:end) zeros(K,1)];
Thetam = [zeros(K,1) Theta(:,1:end-1)]; 


mu = getoptions(options,'mu',0.5);
Q = eye(K)+max(Theta(:))*A;
t0 = .5 * (1/(norm(D,2)^2 + mu^2*norm(Q)^2 + mu^2 )) ;

out = y;

% Lasso param
tparam.regul='l1';
lambda = getoptions(options,'lambda',0.1);


tparam.lambda = t0 * lambda;% * (size(D,2)/K);
tparam.pos = 'true'; % impose non-negativity
t=1;


%disp('initial')
%[a,b,c,d,e,f] = getCost(X,D,Theta,A,S,y,lambda,lambda_t,lambda_tr,mu);
%[a,b,c,d,e,f]

for i=1:iters
    
    yt = y; yt(:,end) = 0;
    yt1 = [y(:,2:end) zeros(K,1)];
    ym = [zeros(K,1) y(:,1:end-1)];
    ym1 = y; ym1(:,1) = 0;
    
    % reconstruction term
    g_rec = Dsq * y - DX; 

    
    % optical flow
    g_of = - (S'*S*yt1 + Theta.*(A'*S*yt1)) + S'*(S*yt + Theta.*(A*yt) ) ...
        + (Theta.*(A'*S*yt) + Theta2.*(Asq*yt) ) + S'*(S*ym1 - S*ym - Thetam.*(A*ym));

    
    % do gradient descent
    aux = y - t0*g_rec - t0*mu*g_of;
    
    % Proximal operator
    newout = mexProximalFlat(aux, tparam);

    
    if fista
        newt = (1+ sqrt(1+4*t^2))/2;
        y = newout + ((t-1)/newt)*(newout-out);
        t=newt;
    else
        y = newout;
    end

    out=newout;
	
    
    %[obj(i),r(i),opf(i),s(i)] = getCost(X,D,Theta,A,S,y,lambda,lambda_t,lambda_tr,mu);
    
    
end

%[obj(end) r(end) opf(end) s(i)]
[objf,rf,opff,sf,t2,dt2] = getCost(X,D,Theta,A,S,y,lambda,lambda_t,lambda_tr,mu);
fprintf('Total cost: %1.4f, rec: %1.4f, opt-flow %1.4f,spar: %1.4f, reg-theta: %1.4f\n',objf,rf,opff,sf,t2+dt2)
cout = objf;


end


function [obj,r,opf,s,t2,dt2] = getCost(X,D,Theta,G,S,y,lambda,lambda_t,lambda_tr,mu)

K = size(D,2);
ym = y;
ym(:,end) = 0;
ym1 = 0*y; ym1(:,1:end-1) = y(:,2:end);

rec = 0.5*(X - D*y).^2;
opflow = 0.5*(S*ym1 - S*ym -Theta.*(S*G*ym)).^2;

r = sum(rec(:));
opf = sum(opflow(:));


Gt=eye(K);
Gt=Gt - circshift(Gt,[0 1]);
Gt=Gt(1:end-1,:);


dt2 = Gt*Theta;
dt2 = 0.5*sum(dt2(:).^2);
t2 = 0.5*sum(Theta(:).^2);

s = sum(y(:));

obj = sum(rec(:)) +mu*opf + lambda*s + mu*( lambda_t*dt2 + lambda_tr*t2);


end