
if ~exist('X1','var')
    load '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s31.mat'
    X1 = Xc;
    clear Xc;
    
    epsilon = 1;
    
    load '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s14.mat'
    X2 = Xc;
    
    X = [X1 X2];
    
    
    %renormalize data: whiten each frequency component.
    eps=4e-1;
    stds = std(X,0,2) + eps;
    avenorm = mean(sqrt(sum(X.^2)));
    clear X
    
    
    X1 = X1./repmat(stds,1,size(X1,2));
    X1 = X1/avenorm;
    X2 = X2./repmat(stds,1,size(X2,2));
    X2 = X2/avenorm;
    
    param_nmf.K = 96;
    param_nmf.posAlpha = 1;
    param_nmf.posD = 1;
    param_nmf.pos = 1;
    param_nmf.lambda = 0.15;
    param_nmf.lambda2 = 0.02;
    param_nmf.iter = 1000;

    
    D1_nmf = mexTrainDL(X1, param_nmf);
    
    D2_nmf = mexTrainDL(X2, param_nmf);
    
end



%%

load('/misc/vlgscratch3/LecunGroup/bruna/grid_data/pooled_dictionaries_speaker31.mat');
D1 = double(D);
D1gn=double(Dbis);
K = size(D1,2);

load('/misc/vlgscratch3/LecunGroup/bruna/grid_data/pooled_dictionaries_speaker14.mat');
D2 = double(D);
D2gn=double(Dbis);

clear D Dbis

%%

speech1 ='/misc/vlgscratch3/LecunGroup/bruna/grid_data/s31/pwag9a.wav';
speech2 = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/s14/prin8s.wav';

%speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s4/lrak4s.wav'; % same as training
%speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s18/sram2s.wav'; % different woman;
%speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s1/lrbr4n.wav';% man

% Noise
noise = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/s14/prin8s.wav';

SNR_dB = 0;


params_aux = audio_config();

fs = params_aux.fs;
NFFT = params_aux.NFFT;
hop = params_aux.hop;


[x,Fs] = audioread(speech1);
x = resample(x,fs,Fs);
x = x(:);


[n,Fs] = audioread(speech2);
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

Pmix = Vmix./repmat(stds,1,size(Smix,2));
Pmix = Pmix/avenorm;


Sx = compute_spectrum(x,NFFT, hop);
P1 = abs(Sx);
P1  = P1/avenorm;

Sn = compute_spectrum(n,NFFT, hop);
P2 = abs(Sn);
P2  = P2/avenorm;


%% coding NMF

H =  full(mexLasso(Pmix,[D1_nmf,D2_nmf],param_nmf));

Z1in = H(1:K,:);
Z2in = H(K+1:end,:);

R = {};
R{1} = D1_nmf* Z1in;
R{2} = D2_nmf* Z2in;

y_out = wienerFilter2(R,Smix);

m = length(y_out{1});
x2 = x(1:m);
n2 = n(1:m);
mix2 = mix(1:m);


Parms =  BSS_EVAL(x2, n2, y_out{1}, y_out{2}, mix2);
Parms


%% coding NMF

param_nmf.pos = 1;
param_nmf.lambda = 0.01;
param_nmf.lambda2 = 0;
param_nmf.iter = 1000;

H =  full(mexLasso(Pmix,[D1,D2],param_nmf));

Z1in = H(1:K,:);
Z2in = H(K+1:end,:);

R = {};
R{1} = D1* Z1in;
R{2} = D2* Z2in;

y_out = wienerFilter2(R,Smix);

m = length(y_out{1});
x2 = x(1:m);
n2 = n(1:m);
mix2 = mix(1:m);


Parms =  BSS_EVAL(x2, n2, y_out{1}, y_out{2}, mix2);
Parms


%% coding reweighted


param.lambda=0.15;
param.beta=2e-2;
param.groupsize=4;
param.time_groupsize=4;
param.nu=0.2;
param.lambdagn=0.1;
param.itersout=500;

param_aux = param;
param_aux.lambda = 0;%param.lambda/100;


[Z1out, Z1gnout,Z2out, Z2gnout,fp1,fp2] = twolevellasso_reweighted_demix(10*Pmix(:,1:end-1), D1, D1gn,D2, D2gn,10*Z1in(:,1:end-1),10*Z2in(:,1:end-1), param_aux);

R = {};
R{1} = D1* Z1out;
R{2} = D2* Z2out;

y_out = wienerFilter2(R,Smix(:,1:end-1));

m = length(y_out{1});
x2 = x(1:m);
n2 = n(1:m);
mix2 = mix(1:m);


Parms =  BSS_EVAL(x2, n2, y_out{1}, y_out{2}, mix2);
Parms


break

%%


if ~exist('X1','var')
    load '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s31.mat'
    X1 = Xc;
    clear Xc;
    
    epsilon = 1;
    
    load '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s14.mat'
    X2 = Xc;
    
    X = [X1 X2];
    
end

load('/misc/vlgscratch3/LecunGroup/bruna/grid_data/pooled_dictionaries_speaker31.mat');
D1 = double(D);
D1gn=double(Dbis);
K = size(D1,2);

load('/misc/vlgscratch3/LecunGroup/bruna/grid_data/pooled_dictionaries_speaker14.mat');
D2 = double(D);
D2gn=double(Dbis);

clear D Dbis


%renormalize data: whiten each frequency component.
eps=4e-1;
stds = std(X,0,2) + eps;
avenorm = mean(sqrt(sum(X.^2)));
clear X

X1 = X1./repmat(stds,1,size(X1,2));
X1 = X1/avenorm;
X2 = X2./repmat(stds,1,size(X2,2));
X2 = X2/avenorm;

gpud=gpuDevice(2);

param.nmf=1;
param.lambda=0.15;
param.beta=2e-2;
param.groupsize=4;
param.time_groupsize=4;
param.nu=0.2;
param.lambdagn=0.1;
param.itersout=500;

reset(gpud);

N = 500;
Xs1 = X1(:,1:N);
Xs2 = X2(:,1:N);
Xs = Xs1 + Xs2;

param_nmf.pos = 1;
param_nmf.lambda = param.lambda;
param_nmf.lambda2 = param.beta;
param_nmf.iter = 1000;


%%

H =  full(mexLasso(Xs,[D1,D2],param_nmf));

Z1in = H(1:K,:);
Z2in = H(K+1:end,:);

%%

figure(1)
subplot(221)
imagesc(log(Xs1+0.001))

subplot(222)
imagesc(log(Xs2+0.001))

subplot(223)
imagesc(log(D1*Z1in+0.001))

subplot(224)
imagesc(log(D2*Z2in+0.001))
drawnow

%%
param_aux = param;
param_aux.lambda = param.lambda/2;

[Z1out, Z1gnout,Z2out, Z2gnout] = twolevellasso_reweighted_demix(Xs, D1, D1gn,D2, D2gn,Z1in,Z2in, param_aux);

%[Z, Zgn,Poole] = twolevellasso_gpu(Xs, D, Dcut, param);


%%

rec1 = Xs1 - D1*Z1in;
a1 = 0.5*sum(rec1(:).^2);

rec2 = Xs2 - D2*Z2in;
b1 = 0.5*sum(rec2(:).^2);


rec1 = Xs1 - D1*Z1out;
a2 = 0.5*sum(rec1(:).^2);

rec2 = Xs2 - D2*Z2out;
b2 = 0.5*sum(rec2(:).^2);

[a1,b1,a2,b2]


%%

rec = Xs - D1*Z1out - D2*Z2out;
0.5*sum(rec(:).^2)

D1Z1 = (D1*Z1out).^2;
D2Z2 = (D2*Z2out).^2;

V_ap = D1Z1 + D1Z1 + eps;

SPEECH1 = (D1Z1./V_ap).*Xs;
SPEECH2 = (D2Z2./V_ap).*Xs;


rec1 = Xs1 - SPEECH1;
0.5*sum(rec1(:).^2)

rec2 = Xs2 - SPEECH2;
0.5*sum(rec2(:).^2)


%%

D1Z1 = (D1*Z1in).^2;
D2Z2 = (D2*Z2in).^2;

V_ap = D1Z1 + D1Z1 + eps;

SPEECH1 = (D1Z1./V_ap).*Xs;
SPEECH2 = (D2Z2./V_ap).*Xs;


rec1 = Xs1 - SPEECH1;
0.5*sum(rec1(:).^2)

rec2 = Xs2 - SPEECH2;
0.5*sum(rec2(:).^2)


%%

figure(2)
subplot(221)
imagesc(log(Xs1+0.001))

subplot(222)
imagesc(log(Xs2+0.001))

subplot(223)
imagesc(log(D1*Z1out+0.001))

subplot(224)
imagesc(log(D2*Z2out+0.001))
drawnow


