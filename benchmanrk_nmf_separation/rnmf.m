function [W, H, E, obj, fit, V_ap] = rnmf(V, beta, n_iter_max, tol, W, H, E, lambda_ast,lambda,switch_W)

% Robust NMF with the beta-divergence
%switch_W = 1;
switch_H = 1;
SCALE = and(switch_W,switch_H) & 1;
%SCALE = 0;
switch_E = 0;

[F,N] = size(V);

V_ap = W*H + E;


fit = zeros(1,n_iter_max);
obj = zeros(1,n_iter_max);

iter = 1;
fit(iter) = betadiv(V,V_ap,beta); % Fit (beta-div)
obj(iter) = betadiv(V,V_ap,beta) + lambda*sum(E(:)); % Objective (penalized fit)
err = Inf;

fprintf('iter = %4i | obj = %+5.2E | err = %4.2E (target is %4.2E) \n',iter,obj(iter),err,tol)

while (err >= tol ) && (iter < n_iter_max)
    
    if switch_W
        W = W .* ((V.*V_ap.^(beta-2))*H')./(V_ap.^(beta-1)*H'+lambda_ast*W);
        V_ap = W*H + E;
    end
    
    if switch_H
        H = H .* (W'*(V.*V_ap.^(beta-2)))./(W'*V_ap.^(beta-1)+lambda_ast*H);
        V_ap = W*H + E;
    end
    
    if switch_E
        E = E .* (V.*V_ap.^(beta-2))./(V_ap.^(beta-1) + lambda);
        V_ap = W*H + E;
    end
    
    if SCALE
        scale = sum(W,1);
        %scale = sqrt(sum(W.^2,1));
        W = W .* repmat(scale.^-1,F,1);
        H = H .* repmat(scale',1,N);
    end
    
    iter = iter + 1;
    fit(iter) = betadiv(V,V_ap,beta);
    obj(iter) = betadiv(V,V_ap,beta) +0.5*lambda_ast*(sum(trace(W'*W))+sum(trace(H'*H))) + lambda*sum(E(:));
    
    err = (obj(iter-1)-obj(iter))/obj(iter);
    
    if rem(iter,50)==0
        fprintf('iter = %4i | obj = %+5.2E | err = %4.2E (target is %4.2E) \n',iter,obj(iter),err,tol)
    end
        
end

fprintf('iter = %4i | obj = %+5.2E | err = %4.2E (target is %4.2E) \n',iter,obj(iter),err,tol)
obj = obj(1:iter); fit = fit(1:iter);
