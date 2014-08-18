


%

if 0
%load denoising/dics/NMF_dic_k500_l01
%load denoising/dics/NMF_dic_k100_l01
%load denoising/dics/NMF_s1_dic_k100_l01
%load denoising/dics/NMF_s1_dic_k642_l01
load dictionary_s4_NMF_k100

testFun    = @(Pmix,param) denoising_nmf(Pmix,D,param);
nparam = param;

end


if 1

load dictionary_s4_sort
D = DD;

param_nmf.K=100; % learns a dictionary with 100 elements 
param_nmf.lambda=0.1; 
%param.numThreads=12;	%	number	of	threads 
param_nmf.batchsize =1000;
param_nmf.iter=200; % let us see what happens after 1000 iterations .
param_nmf.posD=1;
param_nmf.posAlpha=1;
param_nmf.pos=1;

testFun_nmf    = @(Pmix,param,Px,Pn,obj) denoising_nmf(Pmix,D,param,Px,Pn);


param_group = options;
param_group.iter = 10;
param_group.semisup = 1;

testFun_group  = @(Pmix,param,Px,Pn) denoising_group_pooling(D,Pmix,param,Px,Pn);

end


%

speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s4/lrak4s.wav'; % same as training
%speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s18/sram2s.wav'; % different woman;
%speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s1/lrbr4n.wav';% man

% Noise
noise = '../../../../misc/vlgscratch3/LecunGroup/bruna/noise_data/train/noise_sample_08.wav'; % easy
%noise = '../../../../misc/vlgscratch3/LecunGroup/bruna/noise_data/babble/noise_sample_08.wav'; % hard


SNR_dB = 5;


%% use the same initial noise dictionary
Kn = 2;
Wo = mexNormalize(max(0.1+rand(size(D,1),Kn),0));


param_nmf.W = Wo;
param_nmf.iter = 60;
output_nmf = testFile(speech,noise,testFun_nmf,D,param_nmf,SNR_dB);


%%

param_grpup.W = Wo;
param_group.iter = 60;
output_group = testFile(speech,noise,testFun_group,D,param_group,SNR_dB);



