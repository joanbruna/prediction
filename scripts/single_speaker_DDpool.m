clear all;
close all;


if ~exist('X1','var')
    load '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s31.mat'
    X = Xc;
    clear Xc;
    
end

%renormalize data: whiten each frequency component.
eps=4e-1;
stds = std(X,0,2) + eps;
X = X./repmat(stds,1,size(X,2));
avenorm = mean(sqrt(sum(X.^2)));
X = X/avenorm;

param.nmf=1;
param.lambda=0.2;
param.epochs=1;
param.batchsize=1024;
param.K=256;
reset(gpuDevice);

%[D, T, S] = binary_graph_dlearn(X, param); 
[D,zout] = group_pooling_st_gpu(X, param);



