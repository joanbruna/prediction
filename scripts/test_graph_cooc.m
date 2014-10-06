clear all;
close all;


if ~exist('X1','var')
    % use single speaker for training
    load '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s31.mat'
    X1 = Xc;
    clear Xc;
    
    epsilon = 1;
%    X1 = softNormalize(X1,epsilon);
    
    load '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s14.mat'
    X2 = Xc;

%    X2 = softNormalize(X2,epsilon);
    
    
    X = [X1 X2];
    clear X1 X2
    
end

X0=X;

eps=1e-2;
stds = std(X,0,2) + eps;
X = X./repmat(stds,1,size(X,2));

%epsilon=8;
%X = softNormalize(X,epsilon);
avenorm = mean(sqrt(sum(X.^2)));
X = X/avenorm;

param.K=400; % learns a dictionary 
param.lambda=0.03; 
param.numThreads=16;	%	number	of	threads 
param.batchsize =512;
param.iter=1000; % let us see what happens after 1000 iterations .
param.posD=1;
param.posAlpha=1;
param.pos=1;


D=mexTrainDL(X, param);
D = sortD(D);

z=mexLasso(X, D, param);

keyboard;
%temporal pooling 
Tpool=8;
h=hanning(Tpool)';
zpool=conv2(full(z),h,'same');

options.num_neighbors = 40;
[S, L, V, spect] = graphlaplacian(zpool', options);


MAXiter = 1000; % Maximum iteration for KMeans Algorithm
REPlic = 10; % Replication for KMeans Algorithm
W0=V(:,1:5);
NN=40;
[idx,W1] = kmeans(W0,NN,'start','sample','maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
[~,Idict]=sort(idx);



