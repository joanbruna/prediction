


% train model

representation = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/cqt_fs16_NFFT2048_hop1024/TRAIN/';


load(sprintf('%sfemale',representation));


% epsilon = 1;
epsilon = 0.001;
data.X = softNormalize(abs(data.X),epsilon);


param.K = 200;
param.posAlpha = 1;
param.posD = 1;
param.pos = 1;
param.lambda = 0.1;
param.lambda2 = 0;
param.iter = 1000;


D = mexTrainDL(abs(data.X), param);


%%

noise_files = '/misc/vlgscratch3/LecunGroup/pablo/noise_texture/noise_texture_audios.mat';
if ~exist('noise','var')
load(noise_files)
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


% compute decomposition
nparam = param;
nparam.Kn=2; %
nparam.iter=100;
nparam.niter=10;
nparam.pos=1;
nparam.tau = 0.1;
nparam.verbose = 1;



%% Do the testing

options.idf = idf;

options.epsilon = epsilon;
options.fs = data.fs;%data.fs;
options.NFFT = data.NFFT;%data.NFFT;
options.hop = data.hop;%data.hop;
options.scparam = data.scparam;

options.is_stft = 0;

options.model_params = nparam;

testFun    = @(Xn) denoising_nmf(Xn,D,nparam);


options.SNR_dB = [0, 5];

results_cqt = denoising_test(testFun,test_female,noise,options);

save results_cqt results_cqt options
