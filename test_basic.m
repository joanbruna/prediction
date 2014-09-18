
addpath utils
addpath stft

% parameters
fs = 16000;
NFFT = 1024;
winsize = NFFT;
hop = winsize/4;
scf = 2/3;
p = 1;

% load audio
[x,Fs] = audioread('data/bbbj8n.wav');

% resample to 16K 
x = resample(x,fs,Fs);
x = x(:); 

% Compute spectral representation
X = scf * stft(x, NFFT ,winsize, hop);
V = abs(X).^p;
[F,N] = size(V);

dbimagesc(V);

% test

% load ../../../Multimodal' Sparsity'/code/audio_outliers/supervised_nmf/final_results_paper_KL_dics