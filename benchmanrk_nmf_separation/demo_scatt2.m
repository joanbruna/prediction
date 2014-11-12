clear all;

eps=2e-3;
Npad = 2^17;
options.N = Npad;
<<<<<<< HEAD
options.T = 2048;
options.Q = 64;
=======
options.T = 1024;%2048;
options.Q = 24;
options.Jhaar=1;
options.dohaar=0;
options.J_2 = 1;
>>>>>>> 05069e25206e5cf9f8723be9bcd1c558ac4419f7
options.fs = 16000;
filts = create_scattfilters(options);



folderv = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/Data_with_dev/';
train_file1 = 'female_train.wav';
train_file2 = 'male_train.wav';

test_file1 = 'female_test.wav';
test_file2 = 'male_test.wav';

%% Load Signal 1
Fs = 16000;
[x, fs] = audioread([folderv train_file1]);
x = resample(x,Fs,fs);
x = x';

TT=min(length(x),options.N);
x=x(1:TT);

[S2, S1] = audioscatt_fwd_haar(pad_mirror(x,Npad), filts, options);

data1.X1 = S1;
data1.X2 = S2;

%% Load Signal 2

[x, fs] = audioread([folderv train_file2]);
x = resample(x,Fs,fs);
x = x'; T = length(x);

TT=min(length(x),options.N);
x=x(1:TT);

[S2, S1] = audioscatt_fwd_haar(pad_mirror(x,Npad), filts, options);

data2.X1 = S1;
data2.X2 = S2;



%%
epsf = 1e-3;
Xtmp=[abs(data1.X1) abs(data2.X1)];
stds1 = std(Xtmp,0,2) + eps;
stds1 = ones(size(stds1));
data1.X1 = renorm_spect_data(data1.X1, stds1 , epsf);
data2.X1 = renorm_spect_data(data2.X1, stds1 , epsf);

eps=1e-3;
Xtmp=[abs(data1.X2) abs(data2.X2)];
stds2 = std(Xtmp,0,2) + eps;
stds2 = ones(size(stds2));
data1.X2 = renorm_spect_data(data1.X2, stds2 , epsf);
data2.X2 = renorm_spect_data(data2.X2, stds2, epsf);

%% Train models


KK1 = [64];
LL1 = [0.0];
param1.K = KK1;
param1.posAlpha = 1;
param1.posD = 1;
param1.pos = 1;
param1.lambda = LL1;
param1.iter = 1000;
param1.numThreads=16;
param1.batchsize=512;

if 0

Dnmf11 = mexTrainDL(abs(data1.X1),param1);
Dnmf12 = mexTrainDL(abs(data2.X1),param1);

else
Dnmf11 = softNormalize(abs(data1.X1), 1e-3);
Dnmf12 = softNormalize(abs(data2.X1), 1e-3);

end

%%

<<<<<<< HEAD
KK2 = [500];
LL2 = [0.1];
=======
KK2 = [256];
LL2 = [0.3];
>>>>>>> 05069e25206e5cf9f8723be9bcd1c558ac4419f7
param2.K = KK2;
param2.posAlpha = 1;
param2.posD = 1;
param2.pos = 1;
param2.lambda = LL2;
param2.iter = 1000;
param2.numThreads=16;
param2.batchsize=512;

if 0 
Dnmf21 = mexTrainDL(abs(data1.X2),param2);
Dnmf22 = mexTrainDL(abs(data2.X2),param2);
else
Dnmf21 = softNormalize(abs(data1.X2), 1e-3);
Dnmf22 = softNormalize(abs(data2.X2), 1e-3);

end

%% RUN TEST

Npad_2 = 2^16;
options.N = Npad_2;
filts = create_scattfilters(options);


[x, fs] = audioread([folderv test_file1]);
x = resample(x,Fs,fs);
x1 = x'; T1 = length(x1);

[x, fs] = audioread([folderv test_file2]);
x = resample(x,Fs,fs);
fs = Fs;
x2 = x'; T2 = length(x2);

T = min([Npad_2,T1,T2]);

x1 = x1(1:T);
x2 = x2(1:T);

x1 = x1/norm(x1); 
x2 = x2/norm(x2);

mix = x1+x2;

Dnmf11f=softNormalize(data1.X1,1e-3);
Dnmf12f=softNormalize(data2.X1,1e-3);
Dnmf21f=softNormalize(data1.X2,1e-3);
Dnmf22f=softNormalize(data2.X2,1e-3);
param1.lambda=0.00;
param2.lambda=0.2;

[speech1, speech2, s1ord1, s2ord1] = demix_scatt2top(mix, Dnmf11f, Dnmf12f, Dnmf21f, Dnmf22f, stds1, stds2, epsf, filts, options, param1, param2, Npad_2);

%[S2s, S1s] = audioscatt_fwd_haar(pad_mirror(speech1,Npad_2), filts, options);
%[S2o, S1o] = audioscatt_fwd_haar(pad_mirror(x1,Npad_2), filts, options);

Parms =  BSS_EVAL_RNN(x1', x2', speech1(1:T)', speech2(1:T)', mix');

Parms

Parms =  BSS_EVAL_RNN(x1', x2', s1ord1(1:T)', s2ord1(1:T)', mix');

Parms







