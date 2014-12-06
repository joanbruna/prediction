
if ~exist('imdb_m','var')
close all
gpud=gpuDevice(4);
reset(gpud)

addpath('../utils/')
addpath('../icassp_tests/')
addpath('../bss_eval/')
addpath('../spect/')


%run('/home/bruna/matlab/matconvnet/matlab/vl_setupnn.m') ;
run('../matconvnet/matlab/vl_setupnn.m') ;

C = 2;
use_single = 1;

%representation = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/spect_fs16_NFFT1024_hop512/TRAIN/';
representation = '/tmp/';

load([representation 'female.mat']);
name = 'female';
imdb_f = prepareData_matconvnet(data,C,name,use_single);
clear data

load([representation 'male.mat']);
name = 'male';
imdb_m = prepareData_matconvnet(data,C,name,use_single);
clear data
fprintf('data ready \n')
end

%% Create validation set
if ~exist('test_male','var')
load /misc/vlgscratch3/LecunGroup/pablo/TIMIT/TEST/male_audios_short.mat
end

if ~exist('test_female','var')
load /misc/vlgscratch3/LecunGroup/pablo/TIMIT/TEST/female_audios_short.mat
end


idf = [1,3,3;...
    1,3,7;...
    2,5,10;...
    2,6,2];

% USED FOR TESTING
% idf = [1,1,1;...
%     1,1,6;...
%     1,2,9;...
%     1,2,8;...
%     1,4,10;...
%     1,4,2;...
%     2,2,8;...
%     2,2,3;...
%     2,7,4;...
%     2,7,8;...
%     2,8,3;...
%     2,8,5];

idm = [1,1,8;...
    1,6,6;...
    2,9,5;...
    2,13,4];

% USED FOR TESTING
% idm = [1,2,7;...
%     1,2,1;...
%     1,3,1;...
%     1,3,3;...
%     1,7,10;...
%     1,7,9;...
%     2,1,3;...
%     2,1,5;...
%     2,8,2;...
%     2,8,1;...
%     2,16,9;...
%     2,16,10];


clear options

options.id1 = idf;
options.id2 = idm;

epsilon = 0.001;
options.epsilon = epsilon;
options.fs = 16000;
options.NFFT = 1024;
options.hop = options.NFFT/2;
options.Npad = 2^15;
options.verbose = 0;


options.SNR_dB = 0;
valid_fun    = @(net) separation_test_net(@(net) cnn_demix(Xn,net),test_female,test_male,options);

%%

NFFT = size(imdb_f.images.data,3);
net.layers = {};
filter_num = 512;

%f1 = 1/sqrt(1*temp_context*NFFT);
f1 = 1;
temp_context = 1;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f1*randn(1, temp_context, NFFT,filter_num, 'single'), ...
                           'biases', zeros(1, filter_num, 'single'), ...
                           'stride', 1, ...
                           'pad',0);%[0 0 floor(temp_context/2) floor(temp_context/2)]) ;

net.layers{end+1} = struct('type', 'relu') ;

f1 = 1;
temp_context = 1;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f1*randn(1, temp_context, filter_num,filter_num, 'single'), ...
                           'biases', zeros(1, filter_num, 'single'), ...
                           'stride', 1, ...
                           'pad',0);%[0 0 floor(temp_context/2) floor(temp_context/2)]) ;

net.layers{end+1} = struct('type', 'relu') ;

%f2 = 1/sqrt(1*filter_num);
f2 = 1;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f2*randn(1,1,filter_num,2*NFFT, 'single'), ...
                           'biases', zeros(1, 2*NFFT, 'single'), ...
                           'stride', 1, ...
                           'pad',0) ;
                       
%net.layers{end+1} = struct('type', 'relu') ;

% net.layers{end+1} = struct('type', 'normalize', ...
%                            'param', [2 1e-5 1 0.5]) ;

net.layers{end+1} = struct('type', 'normalize_audio', ...
                           'param', [2 1e-5 1 0.5]) ;
                       
net.layers{end+1} = struct('type', 'fitting', ...
                           'loss', 'L2') ;

%opts.expDir = '/misc/vlgscratch3/LecunGroup/pablo/models/cnn/timit-cnn-512-2layer-lr/';
%opts.expDir = '/tmp/pablo/timit-cnn-test-lr/';
opts.expDir = '/misc/vlgscratch3/LecunGroup/bruna/speech/cnn_experiments/'
opts.train.batchSize = 500/C ;
opts.train.numEpochs = 600;
opts.train.continue = false ;
opts.train.useGpu = true ;

lrbase = [0.1 0.01 0.001 0.0001]/C;%/sqrt(C);


opts.train.learningRate = [lrbase(1)*ones(1,10) lrbase(2)*ones(1,30) lrbase(3)*ones(1,100) lrbase(4)];
opts.train.expDir = opts.expDir ;

% set validation set
epsilon = 1e-2;

gB    = @(imdb1, imdb2, batch,batch2) getBatch_nmf(imdb1, imdb2, batch, batch2,epsilon);

getValid    = @(net) separation_test_net(@(Xn) cnn_demix(Xn,net),test_female,test_male,options);

%
[net,info_sc_init] = cnn_train_audio(net, imdb_f, imdb_m, gB, getValid, opts.train) ;

