clear all;
close all;


if ~exist('X1','var')
    load '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s31.mat'
    X1 = Xc;
    clear Xc;
    
    epsilon = 1;
    
    load '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s14.mat'
    X2 = Xc;
    
    X = [X1];
    
end

load('/misc/vlgscratch3/LecunGroup/bruna/grid_data/pooled_dictionaries_speaker31.mat');

%cut the last coordinate of Dbis for pooling without circular topology.
Dcut=Dbis(1:end-1,:);

%renormalize data: whiten each frequency component.
eps=4e-1;
stds = std(X,0,2) + eps;
X = X./repmat(stds,1,size(X,2));
avenorm = mean(sqrt(sum(X.^2)));
X = X/avenorm;

gpud=gpuDevice(4);

param.nmf=1;
param.lambda=0.15;
param.beta=2e-2;
param.groupsize=4;
param.time_groupsize=4;
param.nu=0.2;
param.lambdagn=0.1;
param.itersout=500;

reset(gpud);

[Z, Zgn] = twolevellasso_gpu(X, D, Dcut, param);




