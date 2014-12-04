
if ~exist('filts','var')

representation = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/cqt_phase_fs16_NFFT2048_hop1024/TRAIN/';
%representation = '/tmp/';

load([representation 'female.mat']);
scparam = data.scparam;
filts = data.filts;
clear data
end


%%

representation = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_fs16_NFFT1024_hop512/';

speaker_f = 4;
speaker_m = 1;

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
test_m{1}{1}{1}.speaker =  ['s' num2str(speaker_f)];

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

d = dir([model 'net-epoch-*']);

for i=500

%options.model_params = param;
load([model 'net-epoch-' num2str(i) '.mat'])

net_cnn.layers = {} ;
for j =1:length(net.layers)-1
net_cnn.layers{end+1} = net.layers{j};
end
% No ground_truth 
% net_nmf.layers{end+1} = struct('type', 'fitting', ...
%                            'loss', 'L2') ;


testFun    = @(Xn) cnn_demix(Xn,net_cnn);

options.SNR_dB = 0;
output_net{i+1} = separation_test_net(testFun,test_f,test_m,options);

%--

NSDR_net(i+1) = output_net{i+1}.stat.mean_NSDR;


figure(1)
plot(NSDR_net,'k.-')
drawnow

end

