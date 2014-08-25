if ~exist('X','var')
    % use single speaker for training
    load ../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s4.mat
    X = Xc;
    clear Xc;
end

%epsilon = 0.1;
%X = X ./ repmat(sqrt(epsilon^2+sum(X.^2)),size(X,1),1) ;

if 0
X = mexNormalize(X);
else
epsilon = 0.5;
norms_1 = sqrt(sum(X.^2));
idx = find(norms_1>epsilon);
X = X ./ repmat(sqrt(epsilon^2+sum(X.^2)),size(X,1),1) ;
% Xt_same = Xt_same ./ repmat(sqrt(epsilon^2+sum(Xt_same.^2)),size(X,1),1) ;
% Xt_different = Xt_different ./ repmat(sqrt(epsilon^2+sum(Xt_different.^2)),size(X,1),1) ;
end
norms_2 = sqrt(sum(X.^2));

%X = X(:,idx);

options.K=100;
options.epochs=2;
options.nmf = 1;
options.alpha_iters=80;
options.batchsize=256;
options.sort_dict = 1;
options.plot_dict = 0;
options.lambda = 0.1;
options.mu = 0.5;

% Slowness or flow
options.use_flow = 1;
options.iter_theta = 5;

options.initdictionary = DD;
options.init_nmf = 0;

[DD] = train_nmf_optflow(X, options);