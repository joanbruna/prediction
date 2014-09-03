%%%this script studies the 2D synthetic model using optical flow and NMF


options.null=0;

X = generate_jitter_data_2d(options);
X = mexNormalize(X);

if 1
options.K=256;
options.epochs=2;
options.batchsize=256;
options.sort_dict = 1;
options.plot_dict = 0;
options.plot_dict2d = 1;
options.deWhiten = eye(options.K);
options.lambda = 0.1;
options.mu = 0.5;
options.fista_iters=100;
options.flow_iters=10;



%% Train initial dictionary only with slowness and NMF initialization
options.init_nmf = 1;
options.use_flow = 1;
options.nmf = 1;

keyboard;

[Dslow,Dnmf] = train_nmf_optflow(X, options);
else


%% Train spatio-temporal pooling (without nmf)
clear options
options.K=256;
options.epochs=2;
options.batchsize=256;
options.v=[[0 0];[0 1];[1 0];[1 1]]
options.groupsize=2;
options.lambda = 0.1;
options.K=300;
options.time_groupsize=2;
options.epochs=2;
options.batchsize=128;
options.iters=100;
options.plot_viddict=1;
options.dewhitenMatrix = eye(256);

D=group_pooling_st(X, options);


end






