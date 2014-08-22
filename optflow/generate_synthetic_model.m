%this script generates synthetic dynamical systems 
%with various levels of smoothness

close all
clear all

addpath('../video_prediction')
addpath('nmf_linear_dynamics')
addpath('grouplasso')


options.N=256;
options.L=2^15;
options.Ksmooth=32;
options.ntemplates=2;

[X, temps, phaschange] = generate_jitter_data(options);

X=abs(X);
X=X./repmat(sqrt(sum(X.^2)),size(X,1),1);


if 0
%%%NMF with linear dynamics%%%%%%%%

options.lambda = 0.1;
options.mu = 1;
options.batchsize = 512;
options.K=options.N;
options.epochs=1;
options.p=2;

[D,W, verbo] = nmf_linear_dynamic(X, options);


%%%%% NMF plain %%%%%

param.K=options.K; % learns a dictionary with 100 elements 
param.lambda=0.1; 
param.numThreads=12;	%	number	of	threads 
param.batchsize =256;
param.iter=4000; % let us see what happens after 1000 iterations .
param.posD=1;
param.posAlpha=1;
param.pos=1;
param.verbose=false

Dspams=TrainDL_wrapper(X, param);
alphaspams = mexLasso(X, Dspams, param);

end;


%%%%% spatio-temporal pooling %%%%%%5
 
options.K=2*options.N;
options.epochs=2;
options.overlapping=1;
options.time_groupsize=1;
options.groupsize=2;
options.nmf = 1;
options.alpha_iters=200;
options.batchsize=512;
mu = mean(X,2);
Xc = X - repmat(mu,1,size(X,2));

[DD] = group_pooling_st(X, options);










