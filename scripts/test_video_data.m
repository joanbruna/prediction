%this script learns the optical flow model in charles cadieu's data. 


close all
clear all


L=300000;
t1=load('/misc/vlgscratch3/LecunGroup/bruna/charles_data_bord16.mat');
X=t1.Xout(:,1:L);
X=mexNormalize(X);

[N,L]=size(X);

if 0

options.K=256;
options.epochs=2;
options.batchsize=256;
options.sort_dict = 1;
options.plot_dict = 0;
options.plot_dict2d = 1;
options.deWhiten = t1.dewhitenMatrix;
options.lambda = 0.1;
options.mu = 0.5;
options.fista_iters=100;
options.flow_iters=10;

%% Train initial dictionary only with slowness and NMF initialization
options.init_nmf = 0;
options.use_flow = 1;
options.nmf = 0;

keyboard;

[Dslow,Dnmf] = train_nmf_optflow(X, options);

else

options.v=[[0 0];[0 1];[1 0];[1 1]]
options.groupsize=2;
options.lambda = 0.05;
options.K=300;
options.time_groupsize=2;
options.epochs=2;
options.batchsize=128;
options.iters=150;
options.plot_viddict=1;
options.dewhitenMatrix = t1.dewhitenMatrix;

D=group_pooling_st(X, options);

end



