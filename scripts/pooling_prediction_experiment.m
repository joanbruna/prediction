close all
clear all 

addpath utils
addpath stft
addpath grouplasso
addpath('../video_prediction')

tmp = load('/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/joint.mat');

X = tmp.X ./ repmat(sqrt(sum(tmp.X.^2)),size(tmp.X,1),1) ;
Xt_same = tmp.Xt_same ./ repmat(sqrt(sum(tmp.Xt_same.^2)),size(tmp.X,1),1) ;
Xt_different = tmp.Xt_different ./ repmat(sqrt(sum(tmp.Xt_different.^2)),size(tmp.X,1),1) ;


[N,L]=size(X);

options.epochs=4;
options.proxloops=1;
options.k=2;
options.lr=3e-6;
options.wavelet_init=2;
options.batchsize=256;

[W, A, alpha, W0, P, erro] = pred_layer(X, options);


