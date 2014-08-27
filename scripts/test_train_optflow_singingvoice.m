

options.K=100;
options.epochs=2;
options.nmf = 1;
options.alpha_iters=80;
options.batchsize=256;
options.sort_dict = 1;
options.plot_dict = 0;
options.lambda = 0.1;
options.mu = 0.5;


% Train initial dictionary only with slowness and NMF initialization
options.init_nmf = 1;
options.use_flow = 1;


[Dslow,Dnmf] = train_nmf_optflow(XX_voice, options);


