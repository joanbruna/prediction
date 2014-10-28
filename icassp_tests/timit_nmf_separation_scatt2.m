clear all;
% train model

representation = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/scatt2_fs16_NFFT2048/TRAIN/';

load(sprintf('%sfemale',representation));
data1 = data;
clear data

load(sprintf('%smale',representation));
data2 = data;
clear data

%renormalize data: whiten each frequency component.
eps  = 1e-3;
epsf = 1;%1e-3;
Xtmp=[abs(data1.X1) abs(data2.X1)];
stds1 = std(Xtmp,0,2) + eps;
%stds1 = ones(size(stds1));
data1.X1 = renorm_spect_data(data1.X1, stds1, epsf);
data2.X1 = renorm_spect_data(data2.X1, stds1, epsf);

eps=5e-4;
Xtmp=[abs(data1.X2) abs(data2.X2)];
stds2 = std(Xtmp,0,2) + eps;
%stds2 = ones(size(stds2));
data1.X2 = renorm_spect_data(data1.X2, stds2, epsf);
data2.X2 = renorm_spect_data(data2.X2, stds2, epsf);

KK1 = [200];
LL1 = [0.06];
param1.K = KK1;
param1.posAlpha = 1;
param1.posD = 1;
param1.pos = 1;
param1.lambda = LL1;
param1.iter = 2000;
param1.numThreads=16;
param1.batchsize=512;

Dnmf11 = mexTrainDL(abs(data1.X1),param1);
Dnmf12 = mexTrainDL(abs(data2.X1),param1);

KK2 = [1000];
LL2 = [0.1];
param2.K = KK2;
param2.posAlpha = 1;
param2.posD = 1;
param2.pos = 1;
param2.lambda = LL2;
param2.iter = 2000;
param2.numThreads=16;
param2.batchsize=512;

Dnmf21 = mexTrainDL(abs(data1.X2),param2);
Dnmf22 = mexTrainDL(abs(data2.X2),param2);

keyboard;

if ~exist('test_male','var')
load /misc/vlgscratch3/LecunGroup/pablo/TIMIT/TEST/male_audios_short.mat
end

if ~exist('test_female','var')
load /misc/vlgscratch3/LecunGroup/pablo/TIMIT/TEST/female_audios_short.mat
end

%%

idf = [1,1,1;...
    1,1,6;...
    1,2,9;...
    1,2,8;...
    1,4,10;...
    1,4,2;...
    2,2,8;...
    2,2,3;...
    2,7,4;...
    2,7,8;...
    2,8,3;...
    2,8,5];

idm = [1,2,7;...
    1,2,1;...
    1,3,1;...
    1,3,3;...
    1,7,10;...
    1,7,9;...
    2,1,3;...
    2,1,5;...
    2,8,2;...
    2,8,1;...
    2,16,9;...
    2,16,10];



%% Do the testing
clear options

options.id1 = idf;
options.id2 = idm;

%options.epsilon = epsilon;
options.fs = data1.fs;
%options.NFFT = data1.NFFT;
%options.hop = data1.hop;

%options.model_params = param;

Npad = 2^16;
T=2048;
options.N = Npad;
options.T = T;
options.Q = 32;
filts = create_scattfilters(options);

%testFun    = @(Xn) nmf_demix(Xn,D1,D2,param);
%[speech1, speech2, xest1, xest2] = demix_scatt2top(mix, Dnmf11, Dnmf12, Dnmf21, Dnmf22, stds1, stds2, data1.filts, data1.scparam, param1, param2, Npad);
testFun    = @(mix) demix_scatt2top(mix, Dnmf11, Dnmf12, Dnmf21, Dnmf22, stds1, stds2, epsf, filts, options, param1, param2, Npad);

options.SNR_dB = 0;
output = separation_test_joan(testFun,test_female,test_male,options);



