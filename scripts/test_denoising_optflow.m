


%%
if ~exist('W','var')
params_aux = audio_config();

fs = params_aux.fs;
NFFT = params_aux.NFFT;
hop = params_aux.hop;

%noise = '../../../../misc/vlgscratch3/LecunGroup/bruna/noise_data/train/noise_sample_02.wav'; % easy
noise = '../../../../misc/vlgscratch3/LecunGroup/bruna/noise_data/babble/noise_sample_02.wav'; 


epsilon = 1;
Kn = 20;


[n,Fs] = audioread(noise);
n = resample(n,fs,Fs);
n = n(:);

Sn = compute_spectrum(n,NFFT, hop);
Vn = abs(Sn);
Pn  = softNormalize(Vn,epsilon);

param0 = struct;
param0.K = Kn;
param0.lambda = 0;
param0.posD = 1;
param0.posAlpha = 1;
param0.iter = 200;
W = mexTrainDL(Pn, param0);


optionsx = options;

options.W = W;
end

%%

epsilon = 1;

%% 

speech ='/misc/vlgscratch3/LecunGroup/bruna/grid_data/s31/pwag9a.wav';
%speech = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/s14/prin8s.wav';

%speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s4/lrak4s.wav'; % same as training
%speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s18/sram2s.wav'; % different woman;
%speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s1/lrbr4n.wav';% man

% Noise
%noise = '../../../../misc/vlgscratch3/LecunGroup/bruna/noise_data/train/noise_sample_08.wav'; % easy
%noise = '../../../../misc/vlgscratch3/LecunGroup/bruna/noise_data/babble/noise_sample_10.wav'; % hard
%noise = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/s14/prin8s.wav';

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
[Pmix,norms] = softNormalize(Vmix,epsilon);

Sx = compute_spectrum(x,NFFT, hop);
Vx = abs(Sx);
[Px,norms]  = softNormalize(Vx,epsilon);

Sn = compute_spectrum(n,NFFT, hop);
Vn = abs(Sn);
Pn  = softNormalize(Vn,epsilon);


%%
if ~exist('options','var')
options.K=100;
options.epochs=2;
options.nmf = 1;
options.alpha_iters=80;
options.batchsize=256;
voptions.sort_dict = 1;
options.plot_dict = 0;
options.lambda = 0.1;
options.mu = 0.5;
end

ptheta = struct;
ptheta.sigma = 1;
ptheta.hn = 11;
ptheta.lambda = 0.001;
ptheta.lambdat = ptheta.lambda;
ptheta.lambdar = 0.00001;


%%
options.tau = 0.5;
options.mu = 1;
options.fista = 0;

%[Hs,Hn] = testFun(Pmix,nparam);
[A,theta,SA,An] = nmf_optflow_smooth(Pmix,D,options,ptheta);


R = {};
R{1} = D* A;
R{2} = W* An;

y_out = wienerFilter2(R,Smix);
y_out_of = y_out;

m = length(y_out{1});
x2 = x(1:m);
n2 = n(1:m);

[SDR,SIR,SAR,perm] = bss_eval_sources( [y_out{1},y_out{2}]',[x2,n2]');

[SDR,SIR,SAR]

figure(1)
subplot(311)
dbimagesc(Vx+0.001);
subplot(312)
dbimagesc(D*A+0.001);
subplot(313)
imagesc(SA)



%% NMF

options_nmf = options;
%options_nmf.lambda = 2*options.lambda;
options_nmf.mu = 0;
options_nmf.total_iter = 0;
options_nmf.tau = options.tau;

ptheta_nmf = struct;
ptheta_nmf.sigma = 1;
ptheta_nmf.hn = 11;
ptheta_nmf.lambda = 0;
ptheta_nmf.lambdar = 0;

%[y,z] = nmf_semisup(Pmix,D,W,[],options_nmf);
[y,~,Sy,z] = nmf_optflow_smooth(Pmix,D,options_nmf,ptheta_nmf);



R = {};
R{1} = D* y;
R{2} = W* z;

y_out = wienerFilter2(R,Smix);
y_out_nmf = y_out;

m = length(y_out{1});
x2 = x(1:m);
n2 = n(1:m);

[SDR_nmf,SIR_nmf,SAR_nmf,perm] = bss_eval_sources( [y_out{1},y_out{2}]',[x2,n2]');


[SDR_nmf,SIR_nmf,SAR_nmf]


%%

[Ax,thetax,SAx] = nmf_optflow_smooth(Px,D,optionsx,ptheta);


R = {};
R{1} = D* Ax;
R{2} = W* z;

y_out = wienerFilter2(R,Smix);

m = length(y_out{1});
x2 = x(1:m);
n2 = n(1:m);

[SDR_nmf,SIR_nmf,SAR_nmf,perm] = bss_eval_sources( [y_out{1},y_out{2}]',[x2,n2]');


[SDR_nmf,SIR_nmf,SAR_nmf]


figure(2)
subplot(411)
dbimagesc(D*Ax+0.001);
subplot(412)
imagesc(Ax);
subplot(413)
imagesc(A);
subplot(414)
imagesc(y);




%==============

% Ideal

R = {};
R{1} = Pn;
R{2} = Px;

y_out = wienerFilter2(R,Smix);

m = length(y_out{1});
x2 = x(1:m);
n2 = n(1:m);

[SDR_nmf,SIR_nmf,SAR_nmf,perm] = bss_eval_sources( [y_out{1},y_out{2}]',[x2,n2]');

[SDR_nmf,SIR_nmf,SAR_nmf,perm] 
