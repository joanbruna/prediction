clear all;
close all;


if ~exist('X1','var')
    % use single speaker for training
    load '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s31.mat'
    X1 = Xc;
    clear Xc;
    
    epsilon = 1;
    X1 = softNormalize(X1,epsilon);
    
    load '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s14.mat'
    X2 = Xc;

    X2 = softNormalize(X2,epsilon);
    
    
    X = [X1 X2];
    clear X1 X2
    
end


param.K=256; % learns a dictionary with 100 elements 
param.lambda=0.1; 
param.numThreads=12;	%	number	of	threads 
param.batchsize =512;
param.iter=1000; % let us see what happens after 1000 iterations .
param.posD=1;
param.posAlpha=1;
param.pos=1;


D=mexTrainDL(X, param);

z=mexLasso(X, D, param);

w=kernelization(z);
[valos,aux]=sort(w,2,'descend');

MM=128;
NN=128;
MAXiter = 1000; % Maximum iteration for KMeans Algorithm
REPlic = 10; % Replication for KMeans Algorithm
pp=50;
sigma = (mean(valos(:,pp)));
S = exp(-w.^2/(2*sigma^2));
DD = diag(sum(S));
%DDbis = DD.^(-1/2);
%L = eye(size(S,1)) - DDbis * S * DDbis;
L = DD - S;
[ee,ev]=eig(L);
W0=ee(:,1:MM);

%[idx,W1] = kmeans(W0,NN,'start','sample','maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
%W1=W1';

clval=0.1;
figure;scatter3(max(-clval,min(clval,W0(:,2))),max(-clval,min(clval,W0(:,3))),max(-clval,min(clval,W0(:,4))))



