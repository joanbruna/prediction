
function    y = invert_spectrum(spectrogram,NFFT,step,T)

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
end


% Duplicate the spectrum
[M,N] = size(spectrogram);
S = zeros(2*(M-1),N);
S(1:(NFFT/2+1),:) = spectrogram;
S(NFFT/2+2:end,:) = flipud(conj(spectrogram(2:end-1,:)));
spectrogram = S;

 
% INITIALIZATION
% =========================================================================
% Ly = length(y);
% n_frames = ceil((Ly+overlap)/step);
% Ly_pad = overlap + n_frames*step;
% 
% pos = 1:step:(n_frames)*step;
% range = 0:(NFFT-1);

overlap = NFFT - step;
n_frames = size(spectrogram,2);

Tpad = overlap + n_frames*step;

% define output signal
y = zeros(1,Tpad);
win = zeros(1,Tpad);

% Define the window centers for the analysis
pos = 1:step:(n_frames)*step;
range = 0:(NFFT-1)';

hannWin = hanning(length(range));


% OVERLAP-ADD
% =========================================================================


for i_t = 1:1:size(spectrogram,2)
    
    pos_win = pos(i_t);
    y(pos_win+range) = y(pos_win+range)+real(ifft(spectrogram(:,i_t)))';
    win(pos_win+range) = win(pos_win+range)+hannWin';
    
end

if nargin == 4
y = y(overlap+1:overlap+T)./win(overlap+1:overlap+T);
else
y = y(overlap+1:Tpad-overlap)./win(overlap+1:Tpad-overlap);    
end
%y(NFFT:end-NFFT) = y(NFFT:end-NFFT)./win(NFFT:end-NFFT);