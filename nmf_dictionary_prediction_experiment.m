close all
clear all 

addpath utils
addpath stft
addpath grouplasso
addpath('../video_prediction')

tmp = load('/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/joint.mat');

epsilon = 1;

X = tmp.X ./ repmat(sqrt(epsilon^2+sum(tmp.X.^2)),size(tmp.X,1),1) ;
Xt_same = tmp.Xt_same ./ repmat(sqrt(epsilon^2+sum(tmp.Xt_same.^2)),size(tmp.X,1),1) ;
Xt_different = tmp.Xt_different ./ repmat(sqrt(epsilon^2+sum(tmp.Xt_different.^2)),size(tmp.X,1),1) ;

param.K=512; % learns a dictionary with 100 elements 
param.lambda=0.05; 
param.numThreads=12;	%	number	of	threads 
param.batchsize =256;
param.iter=15000; % let us see what happens after 1000 iterations .
param.posD=1;
param.posAlpha=1;
param.pos=1;

D=mexTrainDL(X, param);
alpha = mexLasso(X, D, param);


[lin_pred_err, A , Xpred] = linear_prediction(X, [3:size(tmp.X,2)], 1);
[lin_pred_err_alpha, Abis , alphapred] = linear_prediction(alpha, [3:size(tmp.X,2)], 1);


