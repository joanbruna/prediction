

representation = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/spect_fs16_NFFT1024_hop512/TRAIN/';


epsilon = 0.0001;

%%

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

net.layers = {};

f = 1/100;
filter_num = 32;
filter_sz = 9;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(filter_sz,filter_sz,1,filter_num, 'single'), ...
                           'biases', zeros(1, filter_num, 'single'), ...
                           'stride', 1, ...
                           'pad',floor((filter_sz-1)/2)) ;
                       
net.layers{end+1} = struct('type', 'relu') ;

filter_num2 = 16;
filter_sz = 9;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(filter_sz,filter_sz,filter_num,filter_num2, 'single'), ...
                           'biases', zeros(1, filter_num2, 'single'), ...
                           'stride', 1, ...
                           'pad',floor((filter_sz-1)/2)) ;

net.layers{end+1} = struct('type', 'relu') ;

filter_num3 = 2;
filter_sz = 5;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(filter_sz,filter_sz,filter_num2/2,filter_num3, 'single'), ...
                           'biases', zeros(1, filter_num3, 'single'), ...
                           'stride', 1, ...
                           'pad',floor((filter_sz-1)/2)) ;

net.layers{end+1} = struct('type', 'relu');
                       
filter_num4 = 2;
filter_sz = 3;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(filter_sz,filter_sz,1,filter_num4, 'single'), ...
                           'biases', zeros(1, filter_num4, 'single'), ...
                           'stride', 1, ...
                           'pad',floor((filter_sz-1)/2)) ;
                       

net.layers{end+1} = struct('type', 'filtermask', ...
                           'p',2) ;

net.layers{end+1} = struct('type', 'fitting', ...
                           'loss', 'L2') ;

                       


opts.expDir = 'matconvnet/data/timit-cnn-test-2' ;
opts.train.batchSize = 3 ;
opts.train.numEpochs = 100;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.learningRate = 0.001;
opts.train.expDir = opts.expDir ;


% set validation set
imdb_m.images.set(end-2*opts.train.batchSize-1:end) = 2;
imdb_f.images.set(end-2*opts.train.batchSize-1:end) = 2;

gB    = @(imdb1, imdb2, batch,batch2) getBatch_nmf(imdb1, imdb2, batch,batch2,epsilon);


[net,info_sc_init] = nmf_train(net, imdb_f, imdb_m, gB,opts.train) ;

                       
