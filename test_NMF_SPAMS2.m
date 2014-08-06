% addpath bss_eval_3/
% addpath utils/
% addpath denoising/
% addpath stft/
% addpath ../spams-matlab/build/


%%

% Load data for single speaker

%load class_s4.mat
load ../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s4.mat
X = Xc;
clear Xc;

epsilon = 0.1;
%X = X ./ repmat(sqrt(epsilon^2+sum(X.^2)),size(X,1),1) ;
X = mexNormalize(X);


%%

% Train dictionary for single speaker


param.K=50; % learns a dictionary with 100 elements 
param.lambda=0.1; 
%param.numThreads=12;	%	number	of	threads 
param.batchsize =1000;
param.iter=200; % let us see what happens after 1000 iterations .
param.posD=1;
param.posAlpha=1;
param.pos=1;


D=mexTrainDL(X, param);


% a=mexLasso(X,D, param);

%% 

% train noise dictionary
%noise = '../../instrument_separation/data/noise/train/noise/noise_sample_04.wav';
noise = '../../../../misc/vlgscratch3/LecunGroup/bruna/noise_data/train/noise_sample_04.wav';

nparam = audio_config();

[n,Fs] = audioread(noise);
n = resample(n,nparam.fs,Fs);
n = n(:);

Sn = nparam.scf * stft(n, nparam.NFFT , nparam.winsize, nparam.hop);
Xn = abs(Sn);

epsilon = 1;
% Xn = Xn ./ repmat(sqrt(epsilon^2+sum(Xn.^2)),size(Xn,1),1) ;


nparam.K=10; % learns a dictionary with 100 elements 
nparam.lambda=0; 
%param.numThreads=12;	%	number	of	threads 
nparam.iter=100; % let us see what happens after 1000 iterations .
nparam.posD=1;
nparam.posAlpha=1;
nparam.pos=1;

Wn=mexTrainDL(Xn, nparam);

clear Xn

param.Wn = Wn;



%% 



%speech = '../data/grid/s4/lrak4s.wav';
%noise = '../../instrument_separation/data/noise/train/noise/noise_sample_08.wav';
speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s4/lrak4s.wav';
noise = '../../../../misc/vlgscratch3/LecunGroup/bruna/noise_data/train/noise_sample_08.wav';


params = audio_config();

SNR_dB = 0;

[x,Fs] = audioread(speech);
x = resample(x,params.fs,Fs);
x = x(:);


[n,Fs] = audioread(noise);
n = resample(n,params.fs,Fs);
n = n(:);



% adjust the size
m = min(length(x),length(n));

x = x(1:m);
n = n(1:m);

% adjust SNR
x = x/sqrt(sum(power(x,2)));
if sum(power(n,2))>0
    n = n/sqrt( sum(power(n,2)));
    n = n*power(10,(-SNR_dB)/20);
end

% compute noisy signal
mix = x + n;

Smix = compute_spectrum(mix,params.NFFT, params.hop);
%Smix = Smix(1:(end/2+1),:);

%Smix = params.scf * stft(mix, params.NFFT , params.NFFT , params.hop);
% Smix = Smix(1:end,:);
% Vmix = abs(Smix);

Vmix = abs(Smix);


 Sx = params.scf * stft(x, params.NFFT , params.winsize, params.hop);
 Vx = abs(Sx);
% 
% Sn = params.scf * stft(n, params.NFFT , params.winsize, params.hop);
% Vn = abs(Sn);

% compute spectral representation
%  Smix = params.scf * stft(mix, params.NFFT , params.winsize, params.hop);
%  Vmix = abs(Smix);

[N,K] = size(D);


%% 

% Compute unmixing
param.lambda = 0;
% param.lambda =0;
%param.Kn = 5;


Pmix = mexNormalize(Vmix);
%     Pmix = Vmix ./ repmat(sqrt(epsilon^2+sum(Vmix.^2)),size(Vmix,1),1) ;
%  [~,H] = nmf_beta(Pmix,2,0.1,0,100,0,[D,Wn],0);
%     param.iter = 1;
H = mexLasso(Pmix,[D,Wn],param);

Hs = H(1:K,:);
Hn = H((K+1):end,:);



%%
Vs2 = D* Hs;
Vn2 = Wn* Hn;


R = {};
R{1} = Vs2;
R{2} = Vn2;

y_out = wienerFilter2(R,Smix);



m = length(y_out{1});
x = x(1:m);
n = n(1:m);

[SDR,SIR,SAR,perm] = bss_eval_sources( [y_out{1},y_out{2}]',[x,n]');
[SDR,SIR,SAR]