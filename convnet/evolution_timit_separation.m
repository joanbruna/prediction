

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


for i=0:10

%options.model_params = param;
load(['matconvnet/data/timit-nmf-test-200/net-epoch-' num2str(i) '.mat'])

net.layers{1}.D1 = double(net.layers{1}.D1);
net.layers{1}.D2 = double(net.layers{1}.D2);

testFun    = @(Xn) nmf_demix(Xn,net.layers{1}.D1,net.layers{1}.D2,param);

options.SNR_dB = 0;
output{i+1} = separation_test(testFun,test_female,test_male,options);

% Evaluate net implementation


net_nmf.layers = {} ;
net_nmf.layers{end+1} = net.layers{1};
net_nmf.layers{end+1} = net.layers{2};


% No ground_truth 
% net_nmf.layers{end+1} = struct('type', 'fitting', ...
%                            'loss', 'L2') ;


testFun    = @(Xn) nmf_demix_net(Xn,net_nmf);

options.SNR_dB = 0;
output_net{i+1} = separation_test_net(testFun,test_female,test_male,options);

%--

NSDR(i+1) = output{i+1}.stat.mean_NSDR;
NSDR_net(i+1) = output_net{i+1}.stat.mean_NSDR;


figure(1)
plot(NSDR,'r.-')
hold on
plot(NSDR_net,'k.-')
drawnow

end



