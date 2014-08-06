close all
clear all 


%this script tries simple NMF factorizations

%I: try first with a single speaker
if ~exist('X','var') && ~exist('Xc','var')
    
    load ../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s26.mat
    X = Xc;
    clear Xc;
    
    epsilon = 1;
    X = X ./ repmat(sqrt(epsilon^2+sum(X.^2)),size(X,1),1) ;
    
end


param.K=50; % learns a dictionary with 100 elements 
param.lambda=0.1; 
%param.numThreads=12;	%	number	of	threads 
param.batchsize =1000;
param.iter=100; % let us see what happens after 1000 iterations .
param.posD=1;
param.posAlpha=1;
param.pos=1;

D=mexTrainDL(X, param);





