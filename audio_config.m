function params=audio_config()
% function param=audio_config()

params=struct;
params.fs=16000;
params.window_type='sinebell';
params.NFFT = 640;
params.winsize=params.NFFT;
params.hop=params.NFFT/4;
params.scf = 2/3;


end