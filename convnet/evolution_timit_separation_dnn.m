

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

epsilon = 0.0001;
options.epsilon = epsilon;
options.fs = 16000;
options.NFFT = 1024;
options.hop = options.NFFT/2;

clear param
param.K = 400;
param.posAlpha = 1;
param.posD = 1;
param.pos = 1;
param.lambda = 0.1;
param.lambda2 = 0;
param.iter = 40;

%model = 'matconvnet/data/timit-dnn-test-/';
%model = '/misc/vlgscratch3/LecunGroup/pablo/models/dnn/timit-dnn-test/';
model = '/misc/vlgscratch3/LecunGroup/pablo/models/dnn/timit-dnn-test-context1/';

d = dir([model 'net-epoch-*']);

for i=1:length(d)-1

%options.model_params = param;
load([model 'net-epoch-' num2str(i) '.mat'])

net_dnn.layers = {} ;
net_dnn.layers{end+1} = net.layers{1};
net_dnn.layers{end+1} = net.layers{2};
net_dnn.layers{end+1} = net.layers{3};
net_dnn.layers{end+1} = net.layers{4};
net_dnn.layers{end+1} = net.layers{5};

% No ground_truth 
% net_nmf.layers{end+1} = struct('type', 'fitting', ...
%                            'loss', 'L2') ;


testFun    = @(Xn) dnn_demix(Xn,net_dnn);

options.SNR_dB = 0;
output_net{i+1} = separation_test_net(testFun,test_female,test_male,options);

%--

NSDR_net(i+1) = output_net{i+1}.stat.mean_NSDR;


figure(1)
plot(NSDR_net,'k.-')
drawnow

end



