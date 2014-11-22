

if ~exist('imdb','var')
imdb = load('data/mnist-baseline/imdb');
imdb.images.data = bsxfun(@minus, imdb.images.data, mean(imdb.images.data,4)) ;
end

imdb_sc_pixel = imdb;


n = length(imdb.images.set);

X = reshape(imdb_sc_pixel.images.data,[28*28,n]);
X = mexNormalize(X);

imdb_sc_pixel.images.data = single(reshape(X,[1,1,28*28,n]));

L = 0;

for i=1:length(L)
    
param.K=20; % learns a dictionary with 100 elements 
param.lambda=0.15 + L(i)*0.025; 
param.lambda2 = 0;
%param.numThreads=12;	%	number	of	threads 
param.batchsize =200;
param.iter=1000; % let us see what happens after 1000 iterations .
% param.posD=1;
% param.posAlpha=1;
% param.pos=1;

D = [];
for j=1:10
ii = imdb_sc_pixel.images.labels == j & imdb_sc_pixel.images.set == 1;
Daux =mexTrainDL(X(:,ii), param);
D = [D, Daux];
end

%%



%

Z = single(full(mexLasso(X,D,param)));

imdb_sc_pixel_init = imdb_sc_pixel;
imdb_sc_pixel_init.images.data = single(reshape(Z,[1,1,size(D,2),n]));

%

%opts.expDir = 'data/mnist-debug/mnist-sc-init' ;
opts.expDir = ['data/mnist-debug/mnist-sc-pixel-init-' num2str(i) '-new2'];
opts.train.batchSize = 200 ;
opts.train.numEpochs = 100 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.learningRate = [0.1*ones(1,10), 0.01];
opts.train.expDir = opts.expDir ;
opts.weightDecay = 0.01 ;
%opts.momentum = 0.99 ;

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


end

%

break

clear opts

opts.expDir = 'data/mnist-debug/mnist-sc-pixel' ;
%opts.expDir = 'data/mnist-debug/mnist-sc' ;
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
                           'stride', 1, ...
                           'pad', 0) ;
net_sc.layers{end+1} = net_sc_init.layers{1}; %use pre trained layer
net_sc.layers{end+1} = struct('type', 'softmaxloss') ;


[net_sc,info_scin] = cnn_train(net_sc, imdb_sc_pixel, @getBatch, ...
    opts.train, ...
    'val', find(imdb_sc_pixel.images.set == 3)) ;


