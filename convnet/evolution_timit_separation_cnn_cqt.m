
addpath('../utils/')
addpath('../scatt/')
addpath('../icassp_tests/')
addpath('../bss_eval/')
run('../matconvnet/matlab/vl_setupnn.m') ;
if ~exist('filts','var')

representation = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/cqt_phase_fs16_NFFT2048_hop1024/TRAIN/';
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

%model = '/scratch/joan/cnn/batch_depth_1_12_17_12_16_40_J5/';
model = '/scratch/joan/cnn/batch_depth_1_12_18_15_23_7_J5/';
%model = '/scratch/joan/cnn/batch_depth_1_12_16_21_16_19/';

d = dir([model 'net-epoch-*']);

epsilon = 1e-8;
J=5;

for i=250

%options.model_params = param;
load([model 'net-epoch-' num2str(i) '.mat'])

net_cnn.layers = {} ;
for j =1:length(net.layers)-1
net_cnn.layers{end+1} = net.layers{j};
end
% No ground_truth 
% net_nmf.layers{end+1} = struct('type', 'fitting', ...
%                            'loss', 'L2') ;


%testFun    = @(Xn) cnn_ensemble_demix_haar(Xn,net_cnn,J,epsilon);
testFun    = @(Xn) cnn_ensemble_demix(Xn,net_cnn,J,epsilon);

options.SNR_dB = 0;
output_net{i+1} = separation_test_net(testFun,test_female,test_male,options);

%--

NSDR_net(i+1) = output_net{i+1}.stat.mean_NSDR;


figure(1)
plot(NSDR_net,'k.-')
drawnow

end


