


%%

if 1

load dictionary_s4_sort
D = DD;

param.K=100; % learns a dictionary with 100 elements 
param.lambda=0.1; 
%param.numThreads=12;	%	number	of	threads 
param.batchsize =1000;
param.iter=200; % let us see what happens after 1000 iterations .
param.posD=1;
param.posAlpha=1;
param.pos=1;

testFun    = @(Pmix,param,Px,Pn) denoising_nmf(Pmix,D,param,Px,Pn);
nparam = param;

end


%%

if 0

%load denoising/dics/group_pooling_v1
%load denoising/dics/group_pooling_s1_k100

load dictionary_s4_sort

D = DD;

options.iter = 10;
options.semisup = 1;

testFun    = @(Pmix,param) denoising_group_pooling(D,Pmix,param);
nparam = options;

end




%% 

speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s4/lrak4s.wav'; % same as training
%speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s18/sram2s.wav'; % different woman;
%speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s1/lrbr4n.wav';% man

% Noise
noise = '../../../../misc/vlgscratch3/LecunGroup/bruna/noise_data/train/noise_sample_08.wav'; % easy
%noise = '../../../../misc/vlgscratch3/LecunGroup/bruna/noise_data/babble/noise_sample_08.wav'; % hard


SNR_dB = 0;


%%


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


Sn = compute_spectrum(n,NFFT, hop);
Vn = abs(Sn);
%Pn = mexNormalize(Vn);


Sx = compute_spectrum(x,NFFT, hop);
Vx = abs(Sx);
%Px = mexNormalize(Vx);


% Impose rank Kn solution
Kn = 2;
if 0

[U,S,V] = svds(Vn,2);
Vn2 = U*S*V'; 

nparam.W = U;

Sn2 = Vn2.*angle(Sn);

n = invert_spectrum(Sn2,NFFT , hop);

m = min(length(x),length(n));
x = x(1:m);
n = n(1:m);

end


mix = n + x;

Smix = compute_spectrum(mix,NFFT, hop);
Vmix = abs(Smix);


%%

rep = 1;
rates = zeros(rep,3);
obj = zeros(rep,1);
rec = obj;

% semi-sup denoising parameters
nparam.iter = 100;

nparam.Kn = Kn;


norms = sqrt(sum(Vmix.^2));
M = repmat(norms,[size(Vmix,1) 1]);
Vmix=Vmix ./ M;
Px=Vx ./ M;
Pn=Vn ./ M;
Pmix = abs(Vmix);


for i=1:rep

[Hs,Hn,Wn] = testFun(Pmix,nparam,Px,Pn);

R = {};
R{1} = D* Hs;
R{2} = Wn* Hn;

y_out = wienerFilter2(R,Smix);


m = length(y_out{1});
x2 = x(1:m);
n2 = n(1:m);

[SDR,SIR,SAR,perm] = bss_eval_sources( [y_out{1},y_out{2}]',[x2,n2]');


rates(i,:) = [SDR(1) SIR(1) SAR(1)];

obj(i) = compute_obj(Pmix,[Hs;Hn],D,Wn,nparam);

rec(i) = norm(Px - R{1},'fro');

end


