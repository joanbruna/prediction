function param=audio_config()
% function param=audio_config()

param=struct;
param.fs=16000;
param.window_type='sinebell';
params.NFFT = 1024;
param.winsize=params.NFFT;
param.hop=params.NFFT/4;


end