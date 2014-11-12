

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

load('data/mnist-debug/mnist-sc-new2/net-epoch-100');

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
opts.expDir = 'data/mnist-debug/mnist-sc-cnn-new2' ;
opts.train.batchSize = 100 ;
opts.train.numEpochs = 100 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.learningRate = [0.01*ones(1,10), 0.001*ones(1,10),0.0001] ;
opts.train.expDir = opts.expDir ;


[net_sc_init,info_sc_init] = cnn_train(net_combined, imdb, @getBatch, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;



break
%%
load data_sc_temp

opts.expDir = 'data/mnist-debug/mnist-sc-new2' ;
opts.train.batchSize = 100 ;
opts.train.numEpochs = 100 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.learningRate = [0.01*ones(1,10), 0.001] ;
opts.train.expDir = opts.expDir ;


net_sc.layers = {} ;
net_sc.layers{end+1} = struct('type', 'sc', ...
                           'dict', single(D),...
                           'lambda', param.lambda, ...
                           'output', 'Y', ...
                           'stride', 1, ...
                           'pad', 0) ;
net_sc.layers{end+1} = net_sc_init.layers{1}; %use pre trained layer
net_sc.layers{end+1} = struct('type', 'softmaxloss') ;



%
[net_sc,info_scin] = cnn_train(net_sc, imdb_sc, @getBatch, ...
    opts.train, ...
    'val', find(imdb_sc.images.set == 3)) ;