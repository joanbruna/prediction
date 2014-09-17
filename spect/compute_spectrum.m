

function  [spectrogram,y_frames] = compute_spectrum(y,NFFT,step)


y = y(:)';

% INITIALIZATION
% =========================================================================


overlap = NFFT-step;

% Define the window centers for the analysis
Ly = length(y);
n_frames = ceil((Ly+overlap)/step);
Ly_pad = overlap + n_frames*step;

pos = 1:step:(n_frames)*step;
range = 0:(NFFT-1);

% Correct the length of the signal to be exact
y = [zeros(1,overlap), y, zeros(1,Ly_pad-Ly-overlap)];
%y = y(1:(pos(end)+range(end)));


% Compute the spectrogram of the input audio
spectrogram = zeros(NFFT/2+1,length(pos));


% START MAIN LOOP
% =========================================================================
flag = 0;
if nargout >1
    y_frames = zeros(n_frames,NFFT);
    flag = 1;
end

for i_t = 1:1:size(spectrogram,2)
    
    pos_win = pos(i_t);
    x_t = y(pos_win+range);
    aux = fft(x_t'.*hanning(length(x_t)));
    spectrogram(:,i_t) = aux(1:end/2+1);
    
    if flag
        y_frames(i_t,:) = x_t;
    end
    
end


end