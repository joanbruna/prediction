SNR_dB = 0;


params_aux = audio_config();

fs = params_aux.fs;
NFFT = params_aux.NFFT;
hop = params_aux.hop;

epsilon = 1;


speech = '../../../../misc/vlgscratch3/LecunGroup/pablo/mallat_bss/mix_3.wav'; 



[x,Fs] = audioread(speech);
x = resample(x,fs,Fs);
x = x(:);
x = x/sqrt( sum(power(x,2)));


Sx = compute_spectrum(x,NFFT, hop);
Vx = abs(Sx);
[Px,norms]  = softNormalize(Vx,epsilon);

%% Code


options.lambda = 0.2;

options_nmf = options;
%options_nmf.lambda = 2*options.lambda;
options_nmf.mu = 0;
options_nmf.total_iter = 0;

ptheta_nmf = struct;
ptheta_nmf.sigma = 1;
ptheta_nmf.hn = 11;
ptheta_nmf.lambda = 0;
ptheta_nmf.lambdar = 0;

%[y,z] = nmf_semisup(Pmix,D,W,[],options_nmf);
[y,~,Sy,z] = nmf_optflow_smooth(Px,D,options_nmf,ptheta_nmf);


%%

ptheta = struct;
ptheta.sigma = 1;
ptheta.hn = 11;
ptheta.lambda = 0.001;
ptheta.lambdat = ptheta.lambda;
ptheta.lambdar = 0.00001;


[A,theta,SA,An] = nmf_optflow_smooth(Px,D,options,ptheta);



