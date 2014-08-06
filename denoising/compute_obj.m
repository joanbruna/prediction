function obj = compute_obj(V,H,W,Wn,param,A)


if nargin == 5
    A = eye(size(W,2));
end

beta = getoptions(param,'beta',2);
mu = getoptions(param,'mu',0);
[F,N] = size(V);
[~,K] = size(W);
[~,Kn] = size(Wn);


lambda1 = param.lambda*[ones(K,1);zeros(Kn,1)];
Lambda1 = repmat(lambda1,1,N);


V_ap = [W,Wn]*H;


Haux = Lambda1.*H;
Hs = H(1:end-Kn,:);

obj = betadiv(V,V_ap,beta)+ sum(Haux(:)) + mu*0.5*norm(Hs(:,2:end) - A*Hs(:,1:end-1),'fro')^2;