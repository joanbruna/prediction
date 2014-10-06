clear all;
close all;


if ~exist('X1','var')
    load '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s31.mat'
    X1 = Xc;
    clear Xc;
    
    epsilon = 1;
    
    load '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s14.mat'
    X2 = Xc;
    
    X = [X1 X2];
    
end


%renormalize data: whiten each frequency component.
eps=4e-1;
stds = std(X,0,2) + eps;
X = X./repmat(stds,1,size(X,2));
avenorm = mean(sqrt(sum(X.^2)));
X = X/avenorm;

param.nmf=1;
param.lambda=0.15;
param.epochs=4;
param.batchsize=1024;
param.K=256;
param.produce_synthesis=1;
param.groupsize=2;
param.time_groupsize=2;
reset(gpuDevice);


% load dictionaries
[D,Z] = group_pooling_st_gpu(X, param);

[Dgn,Zgn] = group_pooling_st_gpu(Zp, param);

