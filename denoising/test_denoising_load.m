


%%

% multi speaker
if 1
load denoising/dics/LD_dic_k500_l01_mu1
% load denoising/dics/LD_dic_k500_l01_mu5

testFun    = @(Pmix,param) denoising_nmf(Pmix,D_ld,param,A_ld);
nparam = ld_param;
D = D_ld;
end

%%

if 0
%load denoising/dics/NMF_dic_k500_l01
load denoising/dics/NMF_dic_k100_l01

testFun    = @(Pmix,param) denoising_nmf(Pmix,D,param);
nparam = param;
end


%% 

%speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s4/lrak4s.wav'; % same as training
%speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s18/sram2s.wav'; % different woman;
speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s1/lrbr4n.wav';% man

% Noise
%noise = '../../../../misc/vlgscratch3/LecunGroup/bruna/noise_data/train/noise_sample_08.wav'; % easy
noise = '../../../../misc/vlgscratch3/LecunGroup/bruna/noise_data/babble/noise_sample_08.wav'; % hard


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

% compute noisy signal
mix = x + n;

Smix = compute_spectrum(mix,NFFT, hop);
Vmix = abs(Smix);

Sx = compute_spectrum(x,NFFT, hop);
Vx = abs(Sx);
Px = mexNormalize(Vx);


%%

rep = 10;
rates = zeros(rep,3);
obj = zeros(rep,1);
rec = obj;

% semi-sup denoising parameters
nparam.iter = 60;
nparam.Kn = 2;

for i=1:rep


Pmix = mexNormalize(Vmix);

[Hs,Hn,Wn] = testFun(Pmix,nparam);


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




