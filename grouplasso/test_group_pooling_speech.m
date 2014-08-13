if 1
    % use single speaker for training
    load ../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s4.mat
    X = Xc;
    clear Xc;
else
    % use joint training set
    if ~exist('X','var')
        load ../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/joint.mat
    end
end

%epsilon = 0.1;
%X = X ./ repmat(sqrt(epsilon^2+sum(X.^2)),size(X,1),1) ;
X = mexNormalize(X);


options.K=100;
options.epochs=4;
options.overlapping=1;
options.time_groupsize=2;
options.groupsize=2;
options.nmf = 1;
options.alpha_iters=80;
options.batchsize=512;
options.v = [0,0;1,0;0,1;1,1];

[DD] = group_pooling_st(X, options);