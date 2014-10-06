eps=2e-3;
Npad = 2^17;
options.N = Npad;
options.T = 2048;
options.Q = 32;
options.fs = 16000;
filts = create_scattfilters(options);



folderv = '../external/deeplearningsourceseparation-master/codes/timit/Data_with_dev/';
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
epsf = 1;
Xtmp=[abs(data1.X1) abs(data2.X1)];
stds1 = std(Xtmp,0,2) + eps;
data1.X1 = renorm_spect_data(data1.X1, stds1 , epsf);
data2.X1 = renorm_spect_data(data2.X1, stds1 , epsf);

eps=1e-3;
Xtmp=[abs(data1.X2) abs(data2.X2)];
stds2 = std(Xtmp,0,2) + eps;
data1.X2 = renorm_spect_data(data1.X2, stds2 , epsf);
data2.X2 = renorm_spect_data(data2.X2, stds2, epsf);

%% Train models


KK1 = [200];
LL1 = [0.1];
param1.K = KK1;
param1.posAlpha = 1;
param1.posD = 1;
param1.pos = 1;
param1.lambda = LL1;
param1.iter = 400;
param1.numThreads=16;
param1.batchsize=512;

Dnmf11 = mexTrainDL(abs(data1.X1),param1);
Dnmf12 = mexTrainDL(abs(data2.X1),param1);

%%

KK2 = [400];
LL2 = [0.1];
param2.K = KK2;
param2.posAlpha = 1;
param2.posD = 1;
param2.pos = 1;
param2.lambda = LL2;
param2.iter = 400;
param2.numThreads=16;
param2.batchsize=512;

Dnmf21 = mexTrainDL(abs(data1.X2),param2);
Dnmf22 = mexTrainDL(abs(data2.X2),param2);


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

[speech1, speech2] = demix_scatt2top(mix, Dnmf11, Dnmf12, Dnmf21, Dnmf22, stds1, stds2, epsf, filts, options, param1, param2, Npad);


Parms =  BSS_EVAL_RNN(x1', x2', speech1(1:T)', speech2(1:T)', mix');

Parms








