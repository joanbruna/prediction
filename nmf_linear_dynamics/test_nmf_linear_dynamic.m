

% param.K=128; % learns a dictionary with 100 elements 
% param.lambda=0.05; 
% param.numThreads=12;	%	number	of	threads 
% param.batchsize =128;
% param.iter=500; % let us see what happens after 1000 iterations .
% param.posD=1;
% param.posAlpha=1;
% param.pos=1;
% 
% Dini=mexTrainDL(X, param);

addpath nmf_linear_dynamics/
addpath utils
addpath ../spams-matlab/build/

X = mexNormalize(X);

clear param

param = struct;
%param.D = Dini;
param.K = 512;
param.lambda = 0.05;
param.mu = 0.5;
param.epochs = 2;
param.batchsize = 2048;


[D,W,verbo] = nmf_linear_dynamic(X, param);


