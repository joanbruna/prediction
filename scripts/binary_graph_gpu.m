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

gpud=gpuDevice(4);

param.nmf=1;
param.lambda=0.15;
param.epochs=4;
param.batchsize=1024;
param.K=128;
param.produce_synthesis=0;
param.groupsize=4;
param.time_groupsize=4;
reset(gpud);

[D] = group_pooling_st_gpu(X, param);

reset(gpud);
param.alpha_itersout=200;
param.lambda = 0.05;
Z = infergrouplasso(D, X, param);


box=ones(param.groupsize,param.time_groupsize);

Zp = sqrt(conv2(Z.^2,box,'same'));
Zp=Zp(1:2:end,1:2:end);

keyboard;

%renormalize data: whiten each frequency component.
eps=4e-2;
stds = std(Zp,0,2) + eps;
Zp = Zp./repmat(stds,1,size(Zp,2));
avenorm = mean(sqrt(sum(Zp.^2)));
Zp = Zp/avenorm;

if 0
param.lambda=0.15;
param.K=64;
earam.groupsize=1;
param.time_groupsize=2;
reset(gpud);
[Dgn,Zgn] = group_pooling_st_gpu(Zp, param);
end

param.lambda=0.15;
param.K=64;
param.numThreads=16;
param.iter=1000;
param.posD=1;
param.posAlpha=1;
param.pos=1;
[Dbis]=mexTrainDL(Zp, param);







