
% train model
representation = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/spect_fs16_NFFT1024_hop512/TRAIN/';


load(sprintf('%sfemale',representation));
data1 = data;
clear data

load(sprintf('%smale',representation));
data2 = data;
clear data


% epsilon = 1;
epsilon = 0.0001;
data1.X = softNormalize(abs(data1.X),epsilon);
data2.X = softNormalize(abs(data2.X),epsilon);

%%
clear param
param.K = 400;
param.posAlpha = 1;
param.posD = 1;
param.pos = 1;
param.lambda = 0.1;
param.lambda2 = 0;
param.iter = 4000;


D1 = mexTrainDL(abs(data1.X), param);

D2 = mexTrainDL(abs(data2.X), param);


%%

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

options.epsilon = epsilon;
options.fs = data1.fs;
options.NFFT = data1.NFFT;
options.hop = data1.hop;

%options.model_params = param;

testFun    = @(Xn) nmf_demix(Xn,D1,D2,param);

options.SNR_dB = 0;
output = separation_test(testFun,test_female,test_male,options);


