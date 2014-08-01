function [alphas,alphan,W] = denoising_nmf(V,D,options)
%
%  0.5* || V -Ds*alphas - Dn*alphan ||^2 + lambda || alphas ||_1
%
% Ds is sparse non-negative dictionary for speech
% V is the magnitude spectrogram of the mixed signal

% number of atoms for noise dictionary
Kn = getoptions(options,'Kn',5);

[N,M] = size(V);

W = mexNormalize(max(0.1+rand(N,Kn),0));
W = getoptions(options,'W',W);


niter = 20;

for i=1:niter
    i
    
    % minimize over [alphas, alphan]
    [alphas,alphan] = nmf_semisup(V,D,W,[],options);

    % minimize Dn
    for j=1:10
    V_ap = W*alphan;
    W = W .* ( (V-D*alphas)*alphan')./(V_ap*alphan');
    end
    
end






