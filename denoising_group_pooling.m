function [alpha,alphan,W,obj,Wo] = denoising_group_pooling(D,V,options,Px,Pn)
%
%  0.5* || V -Ds*alphas - Dn*alphan ||^2 + lambda || alphas ||_1
%
% Ds is sparse non-negative dictionary for speech
% V is the magnitude spectrogram of the mixed signal

% number of atoms for noise dictionary


Kn = getoptions(options,'Kn',5);

[N,M] = size(V);

Wo = mexNormalize(max(0.1+rand(N,Kn),0));
W = getoptions(options,'W',Wo);


niter = getoptions(options,'iter',20);

eps = 1e-9;

in_iter = 1;

for i=1:niter
    
    % minimize over [alphas, alphan]
    % [alpha,alphan] = nmf_semisup(V,D,W,A,options);
    options.W = W;
    options.fista = 1;
    [alpha,cost,alphan] = group_pooling_semisup( D,V, options);
    
    c(i) = cost;
    
    options.H = [alpha;alphan];

    
    % minimize Dn
    for j=1:in_iter
        V_ap = [D,W]*options.H;
        W = W .* ( V*alphan')./(V_ap*alphan');

        W = mexNormalize(W);
        W(W(:)<eps) = 0;
    end
    
    
end

obj = cost;


