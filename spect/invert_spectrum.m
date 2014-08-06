
function    y = invert_spectrum(spectrogram,NFFT,step,pad)

% y = istft(spectrogram, NFFT, NFFT, step);
% 
% 
% return

% SET BASIC CONFIGURATION
% =========================================================================

if ~exist('NFFT','var')
set_configuration

NFFT = config.NFFT;
step = config.step;
pad = 0;
end

if ~exist('pad','var')
pad=0;
end

% Duplicate the spectrum
[M,N] = size(spectrogram);
S = zeros(2*(M-1),N);
S(1:(NFFT/2+1),:) = spectrogram;
S(NFFT/2+2:end,:) = flipud(conj(spectrogram(2:end-1,:)));
spectrogram = S;

 
% INITIALIZATION
% =========================================================================


% Define the window centers for the analysis
pos = (1+NFFT/2):step:(N*step+NFFT/2);
range = (0:((1+pad)*NFFT-1))-(1+pad)*NFFT/2;
hannWin = hanning(length(range));

% define output signal
y = zeros(pos(end)+range(end),1);
win = zeros(pos(end)+range(end),1);

% OVERLAP-ADD
% =========================================================================


for i_t = 1:1:size(spectrogram,2)
    

    pos_win = pos(i_t);

    y(pos_win+range) = y(pos_win+range)+real(ifft(spectrogram(:,i_t)));
    
    win(pos_win+range) = win(pos_win+range)+hannWin;

    
end


y(NFFT:end-NFFT) = y(NFFT:end-NFFT)./win(NFFT:end-NFFT);