

if ~exist('D','var')
    load dict_2_speakers 
end

epsilon = 1;

%% 

%speech ='/misc/vlgscratch3/LecunGroup/bruna/grid_data/s31/pwag9a.wav';
%speech = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/s14/prin8s.wav';

%speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s4/lrak4s.wav'; % same as training
%speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s18/sram2s.wav'; % different woman;
speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s1/lrbr4n.wav';% man

% Noise
noise = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/s14/prin8s.wav';

SNR_dB = 0;


params_aux = audio_config();

fs = params_aux.fs;
NFFT = params_aux.NFFT;
hop = params_aux.hop;


[x,Fs] = audioread(speech);
x = resample(x,fs,Fs);
x = x(:);


[n,Fs] = audioread(noise);
n = resample(n,fs,Fs);
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

Smix = compute_spectrum(mix,NFFT, hop);
Vmix = abs(Smix);
Pmix = softNormalize(Vmix,epsilon);

Sx = compute_spectrum(x,NFFT, hop);
Vx = abs(Sx);
Px  = softNormalize(Vx,epsilon);

Sn = compute_spectrum(n,NFFT, hop);
Vn = abs(Sn);
Pn  = softNormalize(Vn,epsilon);


%%

options.K=100;
options.epochs=2;
options.nmf = 1;
options.alpha_iters=80;
options.batchsize=256;
voptions.sort_dict = 1;
options.plot_dict = 0;
options.lambda = 0.1;
options.mu = 0.5;


ptheta = struct;
ptheta.sigma = 1;
ptheta.hn = 11;
ptheta.lambda = 0.1;
ptheta.lambdar = 0.00001;


%%

%[Hs,Hn] = testFun(Pmix,nparam);
[A,theta,SA] = nmf_optflow_smooth(Pmix,D,options,ptheta);


[A1,theta1,SA1] = nmf_optflow_smooth(Px,D,options,ptheta);

[A2,theta2,SA2] = nmf_optflow_smooth(Pn,D,options,ptheta);


figure(2)
subplot(311)
imagesc(SA)
subplot(312)
imagesc(SA1)
subplot(313)
imagesc(SA2)


