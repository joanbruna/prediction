if ~exist('X1','var')
    % use single speaker for training
    load '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s31.mat'
    X1 = Xc;
    clear Xc;
    
    epsilon = 1;
    X1 = softNormalize(X1,epsilon);
    
    load ../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s14.mat
    X2 = Xc;

    X2 = softNormalize(X2,epsilon);
    
    
    X = [X1 X2];
    clear X1 X2
    
end



options.K=100;
options.epochs=2;
options.nmf = 1;
options.alpha_iters=80;
options.batchsize=256;
options.sort_dict = 1;
options.plot_dict = 0;
options.lambda = 0.1;
options.mu = 0.5;


%% Train initial dictionary only with slowness and NMF initialization
options.init_nmf = 1;
options.use_flow = 0;


[Dslow,Dnmf] = train_nmf_optflow(X, options);


%% Train Dictionary with flow using slow dictionary as input


% Slowness or flow
options.use_flow = 1;
options.iter_theta = 5;

options.initdictionary = Dslow;
options.init_nmf = 0;

D = train_nmf_optflow(X, options);
