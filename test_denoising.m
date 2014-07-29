
addpath ../../../Multimodal' Sparsity'/code/spams-matlab/build/
speech = 'data/swin7p.wav';
noise = 'data/noise_sample_21.wav';


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


Sx = params.scf * stft(x, params.NFFT , params.winsize, params.hop);
Vx = abs(Sx);

Sn = params.scf * stft(n, params.NFFT , params.winsize, params.hop);
Vn = abs(Sn);

% compute noisy signal
mix = x+ n;

% compute spectral representation
Smix = params.scf * stft(mix, params.NFFT , params.winsize, params.hop);
Vmix = abs(Smix);


% Compute unmixing
param.lambda = 0.05;
Hmix = mexLasso(Pmix,[Ds,Dn],param);


n1 = size(Ds , 2);

Vs = Ds* Hmix(1:n1,:);
Vn = Dn* Hmix((n1+1):end,:);


R = {};
R{1} = Vs;
R{2} = Vn;

y = WienerFilter(R,Smix,params.NFFT,params.hop);




