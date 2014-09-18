
%% Get data

params_aux = audio_config();

fs = params_aux.fs;
NFFT = params_aux.NFFT;
hop = params_aux.hop;
epsilon = 0;

speech = '../external/deeplearningsourceseparation-master/codes/timit/Data_with_dev/female_train.wav';
noise = '../external/deeplearningsourceseparation-master/codes/timit/Data_with_dev/male_train.wav';


K = 50;
param0.K = K;
param0.posAlpha = 1;
param0.posD = 1;
param0.pos = 1;
param0.lambda = 0.1;
param0.lambda2 = 0.001;
param0.iter = 100;

RN = 0;


% Train female

[x,Fs] = audioread(speech);
x = resample(x,fs,Fs);
x = x(:);
%x = x/sqrt(sum(power(x,2)));

Sx = compute_spectrum(x,NFFT, hop);
Vx = abs(Sx);


% Train male

[n,Fs] = audioread(noise);
n = resample(n,fs,Fs);
n = n(:);
%n = n/sqrt( sum(power(n,2)));

Sn = compute_spectrum(n,NFFT, hop);
Vn = abs(Sn);

if RN

X = [Vn,Vx];
eps=1e-2;
stds = std(X,0,2) + eps;

Vx = Vx./repmat(stds,1,size(Vx,2));
Vn = Vn./repmat(stds,1,size(Vn,2));

end

Px  = softNormalize(Vx,epsilon);
Pn  = softNormalize(Vn,epsilon);


%% Train dicts

Dn_init = mexTrainDL(Pn, param0);


Dx_init = mexTrainDL(Px, param0);


Dx = Dx_init;
Dn = Dn_init;



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

if RN
Vmix = Vmix./repmat(stds,1,size(Vmix,2));
end

[Pmix,norms] = softNormalize(Vmix,epsilon);


% code
alpha =  mexLasso(Pmix,[Dx,Dn],param0);


R = {};
R{1} = Dx* alpha(1:K,:);
R{2} = Dn* alpha((K+1):end,:);

y_out = wienerFilter2(R,Smix);

m = length(y_out{1});
x2 = x(1:m);
n2 = n(1:m);
mix = mix(1:m);

Parms =  BSS_EVAL(x2, n2, y_out{1}, y_out{2}, mix);


Parms


audiowrite('../../public_html/speech/speech1.wav',y_out{1},fs);
unix(['chmod 777 * ../../public_html/speech/speech1.wav']);
audiowrite('../../public_html/speech/speech2.wav',y_out{2},fs);
unix(['chmod 777 * ../../public_html/speech/speech2.wav']);


%% Train supervised NMF
options.Dx = Dx_init;
options.Dn = Dn_init;
options.Kx = K;
options.Kn = K;

options.params = param0;

[Ds,Dn,out] = train_supervisedNMF(Sx,Sn , options);
