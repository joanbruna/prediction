function [H,W] = nmf_beta(S,W,params) 

% beta, lambda1, lambda_ast, n_iter_max, tol, W , switch_W,H)

V = abs(S);

beta = getoptions(params,'beta',2);
Kn = getoptions(params,'Kn',2);
lambda1 = getoptions(params,'lambda1',0.1);
lambda_ast = getoptions(params,'lambda_ast',0);
tol = getoptions(params,'tol',1e-6);
n_iter_max = getoptions(params,'max_iter',200);

% SCALE = 0;
switch_H = 1;
verbo = 1;
switch_W = 1;

[F,N] = size(V);
[~,K] = size(W);

if ~any(strcmp('H',fieldnames(params)))
    H = abs(randn(K+Kn,N)) + 1;
end


Wn = mexNormalize(abs(randn(F,Kn)));

V_ap = [W,Wn]*H;

lambda1 = lambda1*[ones(K,1);zeros(Kn,1)];



Lambda1 = repmat(lambda1,1,N);


eps = 1e-9;

iter = 1;
if tol>0
    fit = zeros(1,n_iter_max);
    obj = zeros(1,n_iter_max);
%     fit(iter) = betadiv(V,V_ap,beta); % Fit (beta-div)
     Haux = Lambda1.*H;
    obj(iter) = betadiv(V,V_ap,beta) + sum(Haux(:));;%+ lambda_ast*sum(trace(H'*H))  + sum(Haux(:)); % Objective (penalized fit)
    err = Inf;
end


if (verbo == 1)
fprintf('iter = %4i | obj = %+5.2E | err = %4.2E (target is %4.2E) \n',iter,obj(iter),err,tol)
end

% while (err >= tol ) && (iter < n_iter_max)
while (iter < n_iter_max)    
    
    if switch_W
        V_ap_n = Wn*H((K+1):end,:);
        Wn = Wn .* (((V - W*H(1:K,:)).*V_ap_n.^(beta-2))*H((K+1):end,:)')./(V_ap_n.^(beta-1)*H((K+1):end,:)'+eps);
        % normalize columns
        Wn = mexNormalize(Wn);
        V_ap = [W,Wn]*H;
    end
    
    if switch_H
        H = H .* ([W,Wn]'*(V.*V_ap.^(beta-2)))./([W,Wn]'*V_ap.^(beta-1)+lambda_ast*H + Lambda1 + eps);
        V_ap = [W,Wn]*H;
        H(H(:)<eps) = 0;
    end

%     if SCALE
%         scale = sum(W,1);
%         W = W .* repmat(scale.^-1,F,1);
%         H = H .* repmat(scale',1,N);
%     end
    
    iter = iter + 1;
    if tol>0
%     fit(iter) = betadiv(V,V_ap,beta);
    Haux = Lambda1.*H;
    obj(iter) = betadiv(V,V_ap,beta)+ sum(Haux(:));% + lambda_ast*sum(trace(H'*H))  + sum(Haux(:)); % Objective (penalized fit)
    
    err = (obj(iter-1)-obj(iter))/obj(iter);
    end
    
    if rem(iter,2)==0 && (verbo == 1)
        fprintf('iter = %4i | obj = %+5.2E | err = %4.2E (target is %4.2E) \n',iter,obj(iter),err,tol)
    end
        
end

if (verbo == 1)
fprintf('iter = %4i | obj = %+5.2E | err = %4.2E (target is %4.2E) \n',iter,obj(iter),err,tol)
end
if tol>0
obj = obj(1:iter); fit = fit(1:iter);
end