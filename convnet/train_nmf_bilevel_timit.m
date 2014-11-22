

representation = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/spect_fs16_NFFT1024_hop512/TRAIN/';

clear param
param.K = 200;
param.posAlpha = 1;
param.posD = 1;
param.pos = 1;
param.lambda = 0.1;
param.lambda2 = 0;
param.iter = 1000;


if ~exist('D1','var')
load([representation 'female.mat']);

% epsilon = 1;
epsilon = 0.0001;
data.X = softNormalize(abs(data.X),epsilon);
D1 = mexTrainDL(abs(data.X), param);

clear data

load([representation 'male.mat']);

% epsilon = 1;
data.X = softNormalize(abs(data.X),epsilon);
D2 = mexTrainDL(abs(data.X), param);
end
clear data;
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

net_nmf.layers = {} ;
net_nmf.layers{end+1} = struct('type', 'nmf', ...
                           'D1', single(D1), ...
                           'D2', single(D2), ...
                           'lambda',param.lambda ,...
                           'stride', 1, ...
                           'pad', 0) ;
                       
net_nmf.layers{end+1} = struct('type', 'filtermask', ...
                           'p',2) ;

net_nmf.layers{end+1} = struct('type', 'fitting', ...
                           'loss', 'L2') ;
%%

opts.expDir = 'matconvnet/data/timit-nmf-test-200' ;
opts.train.batchSize = 3 ;
opts.train.numEpochs = 6 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.learningRate = 0.001;
opts.train.expDir = opts.expDir ;


% set validation set
imdb_m.images.set(end-2*opts.train.batchSize-1:end) = 2;
imdb_f.images.set(end-2*opts.train.batchSize-1:end) = 2;

gB    = @(imdb1, imdb2, batch,batch2) getBatch_nmf(imdb1, imdb2, batch,batch2,epsilon);


[net_nmf,info_sc_init] = nmf_train(net_nmf, imdb_f, imdb_m, gB,opts.train) ;

                       