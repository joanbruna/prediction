function [H,W] = nmf_beta(S,params) 

% beta, lambda1, lambda_ast, n_iter_max, tol, W , switch_W,H)

V = abs(S).^params.p;

beta = params.beta;
lambda1 = params.lambda1;
lambda_ast = params.lambda_ast;
tol = params.tol;
W = params.W;
n_iter_max = params.max_iter;

% SCALE = 0;
switch_H = 1;
verbo = 0;
if ~any(strcmp('switch_W',fieldnames(params)))
    switch_W = 1;
    
end


[F,N] = size(V);
[~,K] = size(W);

if ~any(strcmp('H',fieldnames(params)))
    H = abs(randn(K,N)) + 1;
end


V_ap = W*H;

if length(lambda1)==1
    lambda1 = lambda1*ones(K,1);
end


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
        W = W .* ((V.*V_ap.^(beta-2))*H')./(V_ap.^(beta-1)*H'+lambda_ast*W+eps);
        W(W(:)<eps) = 0;
        % normalize columns
        W = projectW(W);
        V_ap = W*H;
    end
    
    if switch_H
        H = H .* (W'*(V.*V_ap.^(beta-2)))./(W'*V_ap.^(beta-1)+lambda_ast*H + Lambda1 + eps);
        V_ap = W*H;
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
    
    if rem(iter,50)==0 && (verbo == 1)
        fprintf('iter = %4i | obj = %+5.2E | err = %4.2E (target is %4.2E) \n',iter,obj(iter),err,tol)
    end
        
end

if (verbo == 1)
fprintf('iter = %4i | obj = %+5.2E | err = %4.2E (target is %4.2E) \n',iter,obj(iter),err,tol)
end
if tol>0
obj = obj(1:iter); fit = fit(1:iter);
end