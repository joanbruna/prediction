function y_out = wienerFilter2(R,S,params)
% R = cell con los espectros estimados
% S = espectrograma (complejo) de la mexcla

if ~exist('params','var')
    params = audio_config();
end

% Number of signals
N_sources = length(R);


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
for h = 1:N_sources

    % Create mask
    Mask{h} = R{h}.^2./CS;

    % Filter
    S_new{h} = Mask{h}.*S;

    % Reconstruct signals
%     y_out{h} = istft(S_new{h}, params.NFFT , params.winsize, params.hop);
    y_out{h} = invert_spectrum(S_new{h},params.NFFT , params.hop);

end