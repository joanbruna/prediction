

%% load data

representation = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/scatt_fs16_NFFT2048_hop1024/';

id_1 = 18;
id_2 = 19;

% another man!
%id_2 = 14;


load(sprintf('%ss%d',representation,id_1));
data1 = data;
clear data


load(sprintf('%ss%d',representation,id_2));
data2 = data;
clear data

% epsilon = 1;
epsilon = 1e-8;
param.epsilon = epsilon;
data1.X = softNormalize(abs(data1.X),epsilon);
data2.X = softNormalize(abs(data2.X),epsilon);

param.renorm=0;
param.save_files = 0;

if param.renorm
%renormalize data: whiten each frequency component.
eps=4e-1;
Xtmp=[abs(data1.X) abs(data2.X)];
stds = std(Xtmp,0,2) + eps;

data1.X = renorm_spect_data(data1.X, stds);
data2.X = renorm_spect_data(data2.X, stds);
end


%% train models

model = 'NMF-L2-softnorm';

KK = [200];
LL = [0.1];



param.K = KK(1);
param.posAlpha = 1;
param.posD = 1;
param.pos = 1;
param.lambda = LL(1);
param.lambda2 = 0;
param.iter = 100;


D1 = mexTrainDL(abs(data1.X), param);

D2 = mexTrainDL(abs(data2.X), param);

%%

representation = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/scatt_fs16_NFFT2048_hop1024/';

speaker_f = 18;
speaker_m = 19;

% another man!
%id_2 = 14;


load(sprintf('%ss%d',representation,speaker_f));
data1 = data;
clear data


load(sprintf('%ss%d',representation,speaker_m));
data2 = data;
clear data

%%

idf = [1,1,1];

idm = [1,1,1];

fs = 16000;
[x1, Fs] = audioread(sprintf('%s%s',data1.folder,data1.d(data1.testing_idx(2) ).name) );
x1 = resample(x1,fs,Fs);
test_f{1}{1}{1}.x = x1';
test_f{1}{1}{1}.fs = fs;
test_f{1}{1}{1}.speaker = ['s' num2str(speaker_f)];

[x2, Fs] = audioread(sprintf('%s%s',data2.folder,data2.d(data2.testing_idx(2) ).name) );
x2 = resample(x2,fs,Fs);
test_m{1}{1}{1}.x = x2';
test_m{1}{1}{1}.fs = fs;
test_m{1}{1}{1}.speaker =  ['s' num2str(speaker_m)];

%%

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
model =  '/misc/vlgscratch3/LecunGroup/pablo/models/cnn/timit-cnn-cqt-2nd-comp/';



%options.model_params = param;
load([model 'net-epoch-500.mat'])

net_cnn.layers = {} ;
for j =1:length(net.layers)-1
net_cnn.layers{end+1} = net.layers{j};
end
% No ground_truth 
% net_nmf.layers{end+1} = struct('type', 'fitting', ...
%                            'loss', 'L2') ;


%%

testFun    = @(Xn) cnn_demix(Xn,net_cnn);

options.SNR_dB = 0;
output_net = separation_test_net(testFun,test_f,test_m,options);

%--

NSDR_net = output_net.stat.mean_NSDR;


testFun    = @(Xn) nmf_demix(Xn,D1,D2,param);

output = separation_test_net(testFun,test_f,test_m,options);

