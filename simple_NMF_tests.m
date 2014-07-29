close all
clear all 


%this script tries simple NMF factorizations

%I: try first with a single speaker

tmp1 = load('/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_3.mat');
tmp2 = load('/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_6.mat');
X = [tmp1.chunk tmp2.chunk];

X = X ./ repmat(sqrt(sum(X.^2)),size(X,1),1);

param.K=128; % learns a dictionary with 100 elements 
param.lambda=0.05; 
param.numThreads=12;	%	number	of	threads 
param.batchsize =128;
param.iter=5000; % let us see what happens after 1000 iterations .
param.posD=1;
param.posAlpha=1;
param.pos=1;

D=mexTrainDL(X, param);


alpha = mexLasso(X, D, param);
rec = D * alpha;


