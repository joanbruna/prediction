%this script generates synthetic dynamical systems 
%with various levels of smoothness

%close all
%clear all

%addpath('nmf_linear_dynamics')
%addpath('grouplasso')


options.N=256;
options.L=2^15;
options.Ksmooth=32;
options.ntemplates=2;

[X, temps, phaschange] = generate_jitter_data(options);

X=abs(X);
X=X./repmat(sqrt(sum(X.^2)),size(X,1),1);



options.K=70;
options.epochs=0.5;
options.nmf = 1;
options.alpha_iters=80;
options.batchsize=256;
options.sort_dict = 1;
options.plot_dict = 0;
options.lambda = 0.1;
options.mu = 0.5;


% Train initial dictionary only with slowness and NMF initialization
options.init_nmf = 1;
options.init_rand = 0;



% Slowness or flow
options.use_flow = 1;
options.iter_theta = 5;

%options.initdictionary = Dslow;
options.init_nmf = 1;
options.init_rand = 0;

options.epochs=0.5;

D = train_nmf_optflow(X, options);


if 0

options.K=80;
options.epochs=0.5;
options.nmf = 1;
options.alpha_iters=80;
options.batchsize=256;
options.sort_dict = 1;
options.plot_dict = 0;
options.lambda = 0.1;
options.mu = 0.5;


% Train initial dictionary only with slowness and NMF initialization
options.init_nmf = 1;
options.use_flow = 0;


[Dslow,Dnmf] = train_nmf_optflow(X, options);


% Train Dictionary with flow using slow dictionary as input


% Slowness or flow
options.use_flow = 1;
options.iter_theta = 5;

options.initdictionary = Dslow;
options.init_nmf = 0;

options.epochs=0.5;

D = train_nmf_optflow(X, options);

end

if 0
%%%NMF with linear dynamics%%%%%%%%

options.lambda = 0.1;
options.mu = 1;
options.batchsize = 512;
options.K=options.N;
options.epochs=1;
options.p=2;

[D,W, verbo] = nmf_linear_dynamic(X, options);


%%%%% NMF plain %%%%%

param.K=options.K; % learns a dictionary with 100 elements 
param.lambda=0.1; 
param.numThreads=12;	%	number	of	threads 
param.batchsize =256;
param.iter=4000; % let us see what happens after 1000 iterations .
param.posD=1;
param.posAlpha=1;
param.pos=1;
param.verbose=false

Dspams=TrainDL_wrapper(X, param);
alphaspams = mexLasso(X, Dspams, param);




%%%%% spatio-temporal pooling %%%%%%5
 
options.K=2*options.N;
options.epochs=2;
options.overlapping=1;
options.time_groupsize=1;
options.groupsize=2;
options.nmf = 1;
options.alpha_iters=200;
options.batchsize=512;
mu = mean(X,2);
Xc = X - repmat(mu,1,size(X,2));

[DD] = group_pooling_st(X, options);


end







