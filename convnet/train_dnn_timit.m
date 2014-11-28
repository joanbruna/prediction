

C = 1;
use_single = 1;

representation = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/spect_fs16_NFFT1024_hop512/TRAIN/';

load([representation 'female.mat']);
name = 'female';
imdb_f = prepareData_matconvnet(data,C,name,use_single);
clear data

load([representation 'male.mat']);
name = 'male';
imdb_m = prepareData_matconvnet(data,C,name,use_single);
clear data


%%

NFFT = size(imdb_f.images.data,1);

net.layers = {};

f = 1;
%filter_num = 1024;
filter_num = NFFT;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(NFFT,C,1,filter_num, 'single'), ...
                           'biases', zeros(1, filter_num, 'single'), ...
                           'stride', 1, ...
                           'pad',0) ;

net.layers{end+1} = struct('type', 'relu') ;


net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(1,1,filter_num,2*C*NFFT, 'single'), ...
                           'biases', zeros(1, 2*C*NFFT, 'single'), ...
                           'stride', 1, ...
                           'pad',0) ;

net.layers{end+1} = struct('type', 'reshape_dnn', ...
                           'N',NFFT,...
                           'C', C) ;
                       
net.layers{end+1} = struct('type', 'filtermask', ...
                           'p',2) ;

net.layers{end+1} = struct('type', 'fitting', ...
                           'loss', 'L2') ;

                       



%%

%opts.expDir = '/misc/vlgscratch3/LecunGroup/pablo/models/dnn/timit-dnn-test-context1-512H/' ;
opts.expDir = '/tmp/pablo/timit-dnn-test/';
opts.train.batchSize = 100 ;
opts.train.numEpochs = 50;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.learningRate = [0.01*ones(1,10), 0.01*ones(1,20), 0.001];
opts.train.expDir = opts.expDir ;


% set validation set
V = 100;
imdb_m.images.set(end-V*opts.train.batchSize+1:end) = 2;
imdb_f.images.set(end-V*opts.train.batchSize+1:end) = 2;

epsilon = 0.001;
%gB    = @(imdb1, imdb2, batch,batch2) getBatch_nmf_single(imdb1, imdb2, batch, batch2,epsilon);
gB    = @(imdb1, imdb2, batch,batch2) getBatch_nmf(imdb1, imdb2, batch, batch2,epsilon);

%

[net,info_sc_init] = nmf_train(net, imdb_f, imdb_m, gB,opts.train) ;

