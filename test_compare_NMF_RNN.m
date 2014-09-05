
params_aux = audio_config();

fs = params_aux.fs;
NFFT = params_aux.NFFT;
hop = params_aux.hop;
epsilon = 1;

speech = '../external/deeplearningsourceseparation-master/codes/timit/Data_with_dev/female_train.wav';
noise = '../external/deeplearningsourceseparation-master/codes/timit/Data_with_dev/male_train.wav';


K = 100;
param0.K = K;
param0.posAlpha = 1;
param0.pos = 1;
param0.lambda = 0.1;
param0.iter = 1000;

% Train female

[x,Fs] = audioread(speech);
x = resample(x,fs,Fs);
x = x(:);

Sx = compute_spectrum(x,NFFT, hop);
Vx = abs(Sx);
[Px,norms]  = softNormalize(Vx,epsilon);

Ds = mexTrainDL(Px, param0);


% Train male

[n,Fs] = audioread(noise);
n = resample(n,fs,Fs);
n = n(:);

Sn = compute_spectrum(n,NFFT, hop);
Vn = abs(Sn);
Pn  = softNormalize(Vn,epsilon);

Dn = mexTrainDL(Pn, param0);



%% Testing

speech = '../external/deeplearningsourceseparation-master/codes/timit/Data_with_dev/female_test.wav';
noise = '../external/deeplearningsourceseparation-master/codes/timit/Data_with_dev/male_test.wav';


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
[Pmix,norms] = softNormalize(Vmix,epsilon);


% code
alpha =  mexLasso(Pmix,[Ds,Dn],param0);


R = {};
R{1} = Ds* alpha(1:K,:);
R{2} = Dn* alpha((K+1):end,:);

y_out = wienerFilter2(R,Smix);

m = length(y_out{1});
x2 = x(1:m);
n2 = n(1:m);
mix = mix(1:m);

Parms =  BSS_EVAL(x2, n2, y_out{1}, y_out{2}, mix);


