


if ~exist('filts','var')

representation = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/cqt_phase_fs16_NFFT2048_hop1024_old/TRAIN/';
%representation = '/tmp/';

load([representation 'female.mat']);
scparam = data.scparam;
filts = data.filts;
clear data
end


%%

if ~exist('test_male','var')
load /misc/vlgscratch3/LecunGroup/pablo/TIMIT/TEST/male_audios_short.mat
end

if ~exist('test_female','var')
load /misc/vlgscratch3/LecunGroup/pablo/TIMIT/TEST/female_audios_short.mat
end

%%

idf = [1,1,1;...
    1,1,6;...
    1,2,9;...
    1,2,8;...
    1,4,10;...
    1,4,2;...
    2,2,8;...
    2,2,3;...
    2,7,4;...
    2,7,8;...
    2,8,3;...
    2,8,5];

idm = [1,2,7;...
    1,2,1;...
    1,3,1;...
    1,3,3;...
    1,7,10;...
    1,7,9;...
    2,1,3;...
    2,1,5;...
    2,8,2;...
    2,8,1;...
    2,16,9;...
    2,16,10];



% Do the testing
clear options

options.id1 = idf;
options.id2 = idm;

epsilon = 1e-8;
options.epsilon = epsilon;
options.fs = 16000;
options.NFFT = 1024;

options.filts = filts;
options.scparam = scparam;

options.Npad = 2^15;
options.verbose = 1;

options.is_stft = 0;


%model = 'matconvnet/data/timit-dnn-test-/';
%model = '/misc/vlgscratch3/LecunGroup/pablo/models/dnn/timit-dnn-test/';
%model = '/misc/vlgscratch3/LecunGroup/pablo/models/dnn/timit-dnn-test-context1-512H/';
%model = '/misc/vlgscratch3/LecunGroup/pablo/models/dnn/timit-cnn-test/' ;
%model = '/tmp/pablo/timit-cnn-test/';
%model = 'result/';
%model =  '/misc/vlgscratch3/LecunGroup/bruna/audio_bss/cnn/timit-cnn/' ;
%model = '/misc/vlgscratch3/LecunGroup/pablo/models/cnn/timit-cnn-cqt/';
%model =  '/misc/vlgscratch3/LecunGroup/pablo/models/cnn/timit-cnn-cqt-2nd-comp/';
%model =  '/misc/vlgscratch3/LecunGroup/pablo/models/cnn/timit-cnn-cqt-debug/';


%%

% single frame DNN

%options.model_params = param;
load /misc/vlgscratch3/LecunGroup/pablo/models/cnn/timit-cnn-cqt-2nd-comp/net-epoch-570.mat

net_cnn.layers = net.layers(1:end-1) ;

% No ground_truth 
% net_nmf.layers{end+1} = struct('type', 'fitting', ...
%                            'loss', 'L2') ;

testFun    = @(Xn) cnn_demix(Xn,net_cnn);

options.SNR_dB = 0;
output_net = separation_test_net(testFun,test_female,test_male,options);

%% --

%options.model_params = param;
load /misc/vlgscratch3/LecunGroup/pablo/models/cnn/timit-cnn-cqt-cnn2/net-epoch-240.mat

net_cnn_2.layers = net.layers(1:end-1) ;

temp_context = size(net_cnn_2.layers{1}.filters,2);
net_cnn_2.layers{1}.pad = [0 0 floor(temp_context/2) floor(temp_context/2)];
temp_context2 = size(net_cnn_2.layers{3}.filters,2);
net_cnn_2.layers{3}.pad = [0 0 floor(temp_context2/2) floor(temp_context2/2)];

% No ground_truth 
% net_nmf.layers{end+1} = struct('type', 'fitting', ...
%                            'loss', 'L2') ;

testFun    = @(Xn) cnn_demix(Xn,net_cnn_2);

options.SNR_dB = 0;
%output_net_2 = separation_test_net(testFun,test_female,test_male,options);

%%


net_multi{1}.layers = net_cnn.layers(1:2);
net_multi{2}.layers = net_cnn.layers(3:end);
net_multi{3}.layers = net_cnn_2.layers(3:end);
net_multi{4}.layers = [];

testFun    = @(Xn) cnn_ensemble_demix(Xn,net_multi);

options.SNR_dB = 0;
output_net_3 = separation_test_net(testFun,test_female,test_male,options);


