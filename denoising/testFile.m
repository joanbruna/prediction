
function output = testFile(speech,noise,fun,options)


% load file


% Load files
params_aux = audio_config();


fs = getoptions(options,'fs',params_aux.fs);
NFFT = getoptions(options,'fs',params_aux.NFFT);
hop = getoptions(options,'fs',params_aux.hop);


SNR_dB = 0;

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



%%

ld_nparam = ld_param;

rep = 10;
rates = zeros(rep,3);
obj = zeros(rep,1);
ld_nparam.iter = 100;

ld_nparam.Kn = 2;

for i=1:rep


Pmix = mexNormalize(Vmix);

[Hs,Hn,Wn] = denoising_nmf(Pmix,D_ld,ld_nparam,A_ld);


R = {};
R{1} = D_ld* Hs;
R{2} = Wn* Hn;

y_out = wienerFilter2(R,Smix);


m = length(y_out{1});
x2 = x(1:m);
n2 = n(1:m);

[SDR,SIR,SAR,perm] = bss_eval_sources( [y_out{1},y_out{2}]',[x2,n2]');

if isnan(SDR(1))
    keyboard
end

rates(i,:) = [SDR(1) SIR(1) SAR(1)];

obj(i) = compute_obj(Pmix,[Hs;Hn],D_ld,Wn,ld_nparam);

end

output.rates = rates;
output.obj = obj;

