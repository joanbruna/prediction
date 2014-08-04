

function    [y,pspectrogram,spectrogram,pos] = compute_spectrum(y,NFFT,step,pad)


if ~exist('pad','var')
    pad = 0;
end

% SET BASIC CONFIGURATION
% =========================================================================


if ~exist('NFFT','var')
    set_configuration
    NFFT = config.NFFT;
end

if ~exist('step','var')
    step = config.step;
end


% INITIALIZATION
% =========================================================================


% Define the window centers for the analysis
pos = (1+NFFT/2):step:(length(y)-NFFT/2+1);
range = (0:(NFFT-1))-NFFT/2;


% Correct the length of the signal to be exact
y = y(1:(pos(end)+range(end)));


% Compute the spectrogram of the input audio
spectrogram = zeros((1+pad)*NFFT,length(pos));
pspectrogram = zeros((1+pad)*NFFT/2,length(pos));


% START MAIN LOOP
% =========================================================================


for i_t = 1:1:size(spectrogram,2)
    
    pos_win = pos(i_t);
    x_t = [zeros(pad/2*length(range),1); y(pos_win+range); zeros(pad/2*length(range),1)];
    spectrogram(:,i_t) = fft(x_t.*hanning(length(x_t)));
    
    % power spectrum
    pspectrogram(:,i_t) = abs(spectrogram(1:end/2,i_t));
    
        
    end
     
end