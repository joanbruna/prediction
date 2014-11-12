addpath ./stft/

gpud=gpuDevice(1);

tol = 1e-3;
n_iter_max = 1000;
beta = 1;

l_win = 1024;
NFFT = l_win;
overlap = l_win/2;
Fs = 16000;

lambda_ast = 0;
lambda = 0;


KK = [200];
KKgn = [40];
GG = [10];
LL = [0.1];


for hh = 1:length(KK)


K = KK(hh);
Kgn = KKgn(hh);

param0.K = K;
param0.posAlpha = 1;
param0.posD = 1;
param0.pos = 1;
param0.lambda = LL(1);
param0.lambda2 = 0;
param0.iter = 200;



p = 1;

folderv = '../external/deeplearningsourceseparation-master/codes/timit/Data_with_dev/';
train_file1 = 'female_train.wav';
train_file2 = 'male_train.wav';

test_file1 = 'female_dev.wav';
test_file2 = 'male_dev.wav';

%% Data
%F = 50;
%N = 100;
%V = abs(randn(F,N));


[x, fs] = audioread([folderv train_file1]);
x = resample(x,Fs,fs);
fs = Fs;
x = x'; T = length(x);

%Xt1 = cf_stft(x,l_win,overlap);
Xt1 = compute_spectrum(x,NFFT,overlap);

param.epsilon = 0.001;
epsilon = param.epsilon;
Xt1 = softNormalize(abs(Xt1),epsilon);

V1 = abs(Xt1).^p;
[F,N] = size(V1);

%W1 = rnmf(V1, beta, n_iter_max, tol, W_ini, H_ini, E_ini, lambda_ast,lambda,1);


Dnmf1 = mexTrainDL(Xt1,param0);
%alpha1= mexLasso(Xt1,Dnmf1,param0);
%Dnmf1s = sortDZ(Dnmf1,full(alpha1)');

%%

param.nmf=1;
param.lambda=LL(1)/4;
param.beta=1e-2;
param.overlapping=1;
param.groupsize=GG(hh);
param.time_groupsize=2;
param.nu=0;
param.lambdagn=1e-2;
param.betagn=0;
param.itersout=200;
param.K=K;
param.Kgn=Kgn;
param.epochs=20;
param.batchsize=size(Xt1,2)-1;
param.plotstuff=1;

reset(gpud);

param.initD = mexNormalize(rand(size(Dnmf1))+0.1);
[D1i, Dgn1] = twolevelDL_gpu(Xt1, param);


param.nu = 5;
param.initD = D1i;
[D1, Dgn1] = twolevelDL_gpu(Xt1, param);


%% Model 2

[x, fs] = audioread([folderv train_file2]);
x = resample(x,Fs,fs);
x = x'; T = length(x);

%Xt2 = cf_stft(x,l_win,overlap);
Xt2 = compute_spectrum(x,NFFT,overlap);


Xt2 = softNormalize(abs(Xt2),epsilon);

V2 = abs(Xt2).^p;
[F,N] = size(V2);


%W2 = rnmf(V2, beta, n_iter_max, tol, W_ini, H_ini, E_ini, lambda_ast,lambda,1);
%W2 = mexTrainDL(V2, param0);

Dnmf2 = mexTrainDL(Xt2,param0);
%alpha2= mexLasso(Xt2,Dnmf1,param0);
%Dnmf2s = sortDZ(Dnmf2,full(alpha1)');

%%

reset(gpud);
param.nu = 0;
param.batchsize=size(Xt2,2)-1;
param.initD = mexNormalize(rand(size(Dnmf2))+0.1);
[D2i, Dgn2] = twolevelDL_gpu(Xt2, param);

param.nu = 5;
param.initD = D2i;
[D2, Dgn2] = twolevelDL_gpu(Xt2, param);

%% Algo

[x, fs] = audioread([folderv test_file1]);
x = resample(x,Fs,fs);
x1 = x'; T1 = length(x1);

[x, fs] = audioread([folderv test_file2]);
x = resample(x,Fs,fs);
fs = Fs;
x2 = x'; T2 = length(x2);

T = min(T1,T2);

x1 = x1(1:T);
x2 = x2(1:T);

 x1 = x1/norm(x1); 
x2 = x2/norm(x2);

% X1 = cf_stft(x,l_win,overlap);
% V1 = abs(X1).^p;
% X2 = cf_stft(x2,l_win,overlap);
% V2 = abs(X2).^p;
X1 = compute_spectrum(x1,NFFT,overlap);
X2 = compute_spectrum(x2,NFFT,overlap);


mix = x1+x2;

%X = cf_stft(mix,l_win,overlap);
X = compute_spectrum(mix,NFFT,overlap);

[V,norms] = softNormalize(abs(X),epsilon);
V = abs(X).^p;
[F,N] = size(V);


%%

[Z1dm, Z1gn1dm, Z2dm, Zgn2dm] = twolevellasso_gpu_demix(V, D1, Dgn1, D2, Dgn2, param);

W1H1 = D1*Z1dm;
W2H2 = D2*Z2dm;

eps_1 = 1e-6;
V_ap = W1H1.^2 +W2H2.^2 + eps_1;

% wiener filter

SPEECH1 = ((W1H1.^2)./V_ap).*X;
SPEECH2 = ((W2H2.^2)./V_ap).*X;
speech1 = invert_spectrum(SPEECH1,NFFT,overlap,T);
speech2 = invert_spectrum(SPEECH2,NFFT,overlap,T);


%x1 = x1(overlap+1:overlap+T);
%x2 = x2(overlap+1:overlap+T);

Parms =  BSS_EVAL(x1', x2', speech1', speech2', mix');

Parms

outputs{hh}.r = Parms;


%%

H =  full(mexLasso(V,[Dnmf1,Dnmf2],param0));

W1H1 = Dnmf1*H(1:K,:);
W2H2 = Dnmf2*H(K+1:end,:);


%% Reconstruct sources
eps_1 = 1e-6;
V_ap = W1H1.^2 +W2H2.^2 + eps_1;

% wiener filter

SPEECH1 = ((W1H1.^2)./V_ap).*X;
SPEECH2 = ((W2H2.^2)./V_ap).*X;
speech1 = invert_spectrum(SPEECH1,NFFT,overlap,T);
speech2 = invert_spectrum(SPEECH2,NFFT,overlap,T);


%x1 = x1(overlap+1:overlap+T);
%x2 = x2(overlap+1:overlap+T);

Parms =  BSS_EVAL(x1', x2', speech1', speech2', mix');

Parms

outputs_nmf{hh}.r = Parms;

DD{hh}.D1 = D1;
DD{hh}.D2 = D2;


end

save results_demo outputs outputs_nmf DD


