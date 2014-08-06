

function    [spectrogram,y] = compute_spectrum(y,NFFT,step,pad)


if ~exist('pad','var')
    pad = 0;
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


% START MAIN LOOP
% =========================================================================


for i_t = 1:1:size(spectrogram,2)
    
    pos_win = pos(i_t);
    x_t = [zeros(pad/2*length(range),1); y(pos_win+range); zeros(pad/2*length(range),1)];
    spectrogram(:,i_t) = fft(x_t.*hanning(length(x_t)));
    
end

spectrogram = spectrogram(1:(end/2+1),:);

end