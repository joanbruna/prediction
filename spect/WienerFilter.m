function [y_out,S_new,y_residual,Mask] = WienerFilter(R,S,NFFT,step,CS)
% R = cell con los espectros estimados
% S = espectrograma (complejo) de la mexcla


% Number of signals
N_sources = length(R);

% Check configuration
if ~exist('NFFT','var')
set_configuration

NFFT = config.NFFT;
step = config.step;
end

% Define mask's denominator
if ~exist('CS','var')
CS = eps*ones(size(R{1}));
for h = 1:N_sources
    CS = CS + R{h}.^2;
end
end


S_new = cell(1,N_sources);
Mask = cell(1,N_sources);
y_out = cell(1,N_sources);
S_diff = S;
S = S(1:end/2,:);
for h = 1:N_sources

    % Create mask
    Mask{h} = R{h}.^2./CS;

    % Filter
%     S_new{h} = zeros(size(S_diff));
%     S_new{h}(1:NFFT/2,:) = Mask{h}.*S;
%     S_new{h}(NFFT/2+2:end,:) = flipud(conj(S_new{h}(2:NFFT/2,:)));
    S_new{h} = Mask{h}.*S;

    % Reconstruct signals
    y_out{h} = invert_spectrum(S_new{h},NFFT,step);
   
    
    % Compute difference
    if nargout>2
    S_diff = S_diff -S_new{h};
    end
    
end



% Reconstruct difference
if nargout>2
y_residual = invert_spectrum(S_diff,NFFT,step);
end



