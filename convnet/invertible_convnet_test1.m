

representation = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/spect_fs16_NFFT1024_hop512/TRAIN/';


if ~exist('imdb_f','var')
load([representation 'imdb_female.mat']);
imdb_f = imdb;
imdb_f.images.data = single(imdb_f.images.data); 
imdb_f.images.set = ones(1,size(imdb_f.images.data,4));

load([representation 'imdb_male.mat']) ;
imdb_m = imdb;
imdb_m.images.data = single(imdb_m.images.data); 
imdb_m.images.set = ones(1,size(imdb_m.images.data,4));
clear imdb;
end


%%

opts.expDir = 'matconvnet/data/inv-convnet-test' ;
opts.train.batchSize = 3 ;
opts.train.numEpochs = 10 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.learningRate = 0.001;
opts.train.expDir = opts.expDir ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------


% Define a network similar to LeNet
f=1/100 ;
net.layers = {} ;

% filter layer
filter_num = 16;
filter_sz = 5;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(filter_sz,filter_sz,1,filter_num, 'single'), ...
                           'biases', zeros(1, filter_num, 'single'), ...
                           'stride', 1, ...
                           'pad',floor((filter_sz-1)/2)) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
filter_num_2 = 8;
filter_sz_2 = 5;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(filter_sz_2,filter_sz_2, 2 ,filter_num_2, 'single'), ...
                           'biases', zeros(1, filter_num_2, 'single'), ...
                           'stride', 1, ...
                           'pad',floor((filter_sz_2-1)/2)) ;



% set validation set
imdb_m.images.set(end-2*opts.train.batchSize-1:end) = 2;
imdb_f.images.set(end-2*opts.train.batchSize-1:end) = 2;

%%

epsilon = 0.001;
gB    = @(imdb1, imdb2, batch,batch2) getBatch_nmf(imdb1, imdb2, batch,batch2,epsilon);




[net_nmf,info_sc_init] = inv_cnn_train(net, imdb_f, imdb_m, gB, opts.train) ;
