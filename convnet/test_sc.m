

addpath ../../spams-matlab/build/

% Load MNIST data
if ~exist('imdb','var')
imdb = load('data/mnist-baseline/imdb');
imdb.images.data = bsxfun(@minus, imdb.images.data, mean(imdb.images.data,4)) ;
end
%
% Load the network
load('data/mnist-baseline/net-epoch-100');


%ii = find(imdb.images.set == 1);
%n = length(ii);
n = length(imdb.images.set);
ii = 1:n;

X = zeros(500,n);

batch_sz = 500;

imdb_post = imdb;
imdb_post.images.data = zeros(1,1,500,n,'single');

for i= 1:batch_sz:n
    
    % test gradient
    batch =i:min(i+batch_sz-1, n);
    
    im = imdb.images.data(:,:,:,ii(batch));
    labels = imdb.images.labels(1,ii(batch));
    
    net.layers{end}.class = labels;
    
    res = [];
    res = vl_simplenn(net, im, [], res, ...
        'disableDropout', true, ...
        'conserveMemory', 1, ...
        'sync', 1) ;
    
    X(:,batch) = reshape(res(7).x,[500,batch_sz]);
    
    imdb_post.images.data(:,:,:,ii(batch)) = res(7).x;
    i
end

% normalize and reshape data
X = mexNormalize(X);
data = reshape(X,[1,1,size(X,1),size(X,2)]);
%
imdb_sc = imdb;
imdb_sc.images.data = data;

%% Dictionary learning

param.K=200; % learns a dictionary with 100 elements 
param.lambda=0.1; 
%param.numThreads=12;	%	number	of	threads 
param.batchsize =1000;
param.iter=1000; % let us see what happens after 1000 iterations .
% param.posD=1;
% param.posAlpha=1;
% param.pos=1;

D=mexTrainDL(X(:,imdb.images.set==1), param);


%%

% evaluate test set

% Load the network
load('data/mnist-baseline/net-epoch-100');

info = cnn_test_mnist(net, imdb);

%%


%load('data/mnist-baseline/net-epoch-100');

% Define a network similar to LeNet
f=1/100 ;
net_post.layers = {} ;
net_post.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(1,1,500,10, 'single'),...
                           'biases', zeros(1,10,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net_post.layers{end+1} = struct('type', 'softmaxloss') ;


net_post.layers{1}.filters = net.layers{end-1}.filters;

info = cnn_test_mnist(net_post, imdb_post);


%%

%opts.expDir = 'data/mnist-post' ;
opts.train.batchSize = 100 ;
opts.train.numEpochs = 10 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.learningRate = 0.001 ;
%opts.train.expDir = opts.expDir ;


% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------


f=1/100 ;
net_post.layers = {} ;
net_post.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(1,1,500,10, 'single'),...
                           'biases', zeros(1,10,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net_post.layers{end+1} = struct('type', 'softmaxloss') ;

%
[net_post,info_post] = cnn_train(net_post, imdb_post, @getBatch, ...
    opts.train, ...
    'val', find(imdb_post.images.set == 3)) ;



break

%% ------------


Z = mexLasso(X,D,param);

imdb_sc_init = imdb;
imdb_sc_init.images.data =reshape(single(full(Z)),[1,1,size(Z,1),size(Z,2)]);


opts.expDir = 'data/mnist-debug/mnist-sc-init' ;
opts.train.batchSize = 100 ;
opts.train.numEpochs = 100 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.learningRate = 0.001 ;
opts.train.expDir = opts.expDir ;


net_sc_init.layers = {} ;
% net_sc.layers{end+1} = struct('type', 'sc', ...
%                            'dict', D,...
%                            'lambda', param.lambda, ...
%                            'stride', 1, ...
%                            'pad', 0) ;
net_sc_init.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(1,1,size(D,2),10, 'single'),...
                           'biases', zeros(1,10,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net_sc_init.layers{end+1} = struct('type', 'softmaxloss') ;


[net_sc_init,info_sc_init] = cnn_train(net_sc_init, imdb_sc_init, @getBatch, ...
    opts.train, ...
    'val', find(imdb_sc_init.images.set == 3)) ;

%% ------------


%Z = mexLasso(X,D,param);

imdb_sc = imdb;
imdb_sc.images.data = single(data);

opts.expDir = 'data/mnist-debug/mnist-sc' ;
opts.train.batchSize = 100 ;
opts.train.numEpochs = 100 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.learningRate = 0.001 ;
opts.train.expDir = opts.expDir ;


net_sc.layers = {} ;
net_sc.layers{end+1} = struct('type', 'sc', ...
                           'dict', single(D),...
                           'lambda', param.lambda, ...
                           'pos', 1, ...
                           'stride', 1, ...
                           'pad', 0) ;
net_sc.layers{end+1} = net_sc_init.layers{1}; %use pre trained layer
net_sc.layers{end+1} = struct('type', 'softmaxloss') ;




[net_sc,info_scin] = cnn_train(net_sc, imdb_sc, @getBatch, ...
    opts.train, ...
    'val', find(imdb_sc.images.set == 3)) ;



%% Train all the system, including convolutions

clear all

% load initial network
load('data/mnist-baseline/net-epoch-100');
net_cnn = net;

% Load MNIST data
if ~exist('imdb','var')
imdb = load('data/mnist-baseline/imdb');
imdb.images.data = bsxfun(@minus, imdb.images.data, mean(imdb.images.data,4)) ;
end

load('data/mnist-debug/mnist-sc-new/net-epoch-32');

net_top = net;
%

net_combined.layers = {} ;
% copy first layers from LeNet
for i=1:6
    net_combined.layers{end+1} = net_cnn.layers{i};
end

% need to add a normalization layer
net_combined.layers{end+1} = struct('type', 'normalize', ...
                           'param', [1000 0.0001 1 0.5]) ;

for i=1:length(net_top.layers)
    net_combined.layers{end+1} = net_top.layers{i};
end

%%

im = imdb.images.data(:,:,:,1:100) ;
labels = imdb.images.labels(1,1:100) ;

res = [];
res_a = vl_simplenn(net_combined, im, [], res, ...
    'disableDropout', true, ...
    'conserveMemory', 0, ...
    'sync', 1) ;

%%

% test that it runs alright
info = cnn_test_mnist(net_combined, imdb);


%% 

% train the whole machine
clear opts
opts.expDir = 'data/mnist-debug/mnist-sc-cnn' ;
opts.train.batchSize = 100 ;
opts.train.numEpochs = 50 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.learningRate = 0.001 ;
opts.train.expDir = opts.expDir ;


[net_sc_init,info_sc_init] = cnn_train(net_combined, imdb, @getBatch, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;


%% Pure Sparse coding in the image domain


if ~exist('imdb','var')
imdb = load('data/mnist-baseline/imdb');
imdb.images.data = bsxfun(@minus, imdb.images.data, mean(imdb.images.data,4)) ;
end

imdb_sc_pixel = imdb;


n = length(imdb.images.set);

X = reshape(imdb_sc_pixel.images.data,[28*28,n]);
X = mexNormalize(X);


param.K=200; % learns a dictionary with 100 elements 
param.lambda=0.15; 
%param.numThreads=12;	%	number	of	threads 
param.batchsize =1000;
param.iter=1000; % let us see what happens after 1000 iterations .
% param.posD=1;
% param.posAlpha=1;
% param.pos=1;

D=mexTrainDL(X(:,imdb.images.set==1), param);


%%
imdb_sc_pixel.images.data = single(reshape(X,[1,1,28*28,n]));


%

Z = single(full(mexLasso(X,D,param)));

imdb_sc_pixel_init = imdb_sc_pixel;
imdb_sc_pixel_init.images.data = single(reshape(Z,[1,1,size(D,2),n]));

%

%opts.expDir = 'data/mnist-debug/mnist-sc-init' ;
opts.expDir = 'data/mnist-debug/mnist-sc-pixel-init' ;
opts.train.batchSize = 100 ;
opts.train.numEpochs = 50 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.learningRate = 0.001 ;
opts.train.expDir = opts.expDir ;


f = 1/100;
net_sc_init.layers = {} ;
% net_sc.layers{end+1} = struct('type', 'sc', ...
%                            'dict', D,...
%                            'lambda', param.lambda, ...
%                            'stride', 1, ...
%                            'pad', 0) ;
net_sc_init.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(1,1,size(D,2),10, 'single'),...
                           'biases', zeros(1,10,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net_sc_init.layers{end+1} = struct('type', 'softmaxloss') ;


[net_sc_init,info_sc_init] = cnn_train(net_sc_init, imdb_sc_pixel_init, @getBatch, ...
    opts.train, ...
    'val', find(imdb_sc_pixel_init.images.set == 3)) ;


%

clear opts

opts.expDir = 'data/mnist-debug/mnist-sc-pixel' ;
%opts.expDir = 'data/mnist-debug/mnist-sc' ;
opts.train.batchSize = 100 ;
opts.train.numEpochs = 100 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.learningRate = 0.001 ;
opts.train.expDir = opts.expDir ;


net_sc.layers = {} ;
net_sc.layers{end+1} = struct('type', 'sc', ...
                           'dict', single(D),...
                           'lambda', param.lambda, ...
                           'stride', 1, ...
                           'pad', 0) ;
net_sc.layers{end+1} = net_sc_init.layers{1}; %use pre trained layer
net_sc.layers{end+1} = struct('type', 'softmaxloss') ;


[net_sc,info_scin] = cnn_train(net_sc, imdb_sc_pixel, @getBatch, ...
    opts.train, ...
    'val', find(imdb_sc_pixel.images.set == 3)) ;




opts.expDir = 'data/mnist-debug/mnist-sc' ;


