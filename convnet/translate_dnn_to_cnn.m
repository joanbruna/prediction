
clear NSDR*

% load dnn
%model_dnn = '/misc/vlgscratch3/LecunGroup/pablo/models/dnn/timit-dnn-test-context1-512H/';
model_dnn = '/misc/vlgscratch3/LecunGroup/pablo/models/dnn/timit-dnn-test-context1/';
%epoch = 'net-epoch-47.mat';

d = dir([model_dnn 'net-epoch-*']);

% load cnn
%model_cnn = '/tmp/pablo/timit-cnn-test/';

%model_cnn =  '/misc/vlgscratch3/LecunGroup/bruna/audio_bss/cnn/timit-cnn/' ;
%model_cnn = '/misc/vlgscratch3/LecunGroup/bruna/audio_bss/cnn/timit-cnn-init/';
%model_cnn = '/misc/vlgscratch3/LecunGroup/pablo/models/cnn/timit-cnn-init/';
%model_cnn = '/misc/vlgscratch3/LecunGroup/pablo/models/cnn/timit-cnn-batch-new/';
%model_cnn = '/misc/vlgscratch3/LecunGroup/pablo/models/cnn/timit-cnn-batch-large-lr/';
%model_cnn = '/misc/vlgscratch3/LecunGroup/pablo/models/cnn/timit-cnn-batch-relu-long/';
model_cnn = '/misc/vlgscratch3/LecunGroup/pablo/models/cnn/timit-cnn-512-2layer/';


d_cnn = dir([model_cnn 'net-epoch-*']);

%% Run comparision

if ~exist('test_male','var')
load /misc/vlgscratch3/LecunGroup/pablo/TIMIT/TEST/male_audios_short.mat
end

if ~exist('test_female','var')
load /misc/vlgscratch3/LecunGroup/pablo/TIMIT/TEST/female_audios_short.mat
end



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


%% ------------


%for i=[1,5:5:5*(length(d)-1)
for i=135:5:200   

load([model_cnn 'net-epoch-' num2str(i) '.mat'])
net_cnn = net;

net_cnn_orig.layers = {} ;
for j =1:length(net_cnn.layers)-1
net_cnn_orig.layers{end+1} = net_cnn.layers{j};
end


testFun    = @(Xn) cnn_demix(Xn,net_cnn_orig);

options.SNR_dB = 0;
output_cnn_old = separation_test_net(testFun,test_female,test_male,options);

NSDR_3(i) = output_cnn_old.stat.mean_NSDR;


figure(2)
%plot(NSDR_1,'g')
hold on
plot(NSDR_3,'g')
hold off
drawnow
end

figure(3)
n = length(NSDR_3);
plot([1,5:5:n],NSDR_3([1,5:5:end]),'r.-')


for i=1:length(d)

load([model_dnn d(i).name])
net_dnn = net;
clear net


%% replicate exact same net

net.layers = {};
net.layers{1} = net_cnn.layers{1};
net.layers{1}.filters = gpuArray( permute( net_dnn.layers{1}.filters, [3, 2, 1, 4]) );
net.layers{1}.biases = gpuArray( net_dnn.layers{1}.biases );

net.layers{2} = net_cnn.layers{2};

net.layers{3} = net_cnn.layers{3};

A = net_dnn.layers{3}.filters;
B = 0*A;
B(1,1,:,1:2:end) = A(1,1,:,1:end/2);
B(1,1,:,2:2:end) = A(1,1,:,(end/2+1):end);
net.layers{3}.filters = gpuArray(B);

a = net_dnn.layers{3}.biases;
b = 0*a;
b(1,1:2:end) = a(1,1:end/2);
b(1,2:2:end) = a(1,(end/2+1):end);
net.layers{3}.biases = gpuArray(b);

net.layers{4} = net_cnn.layers{4};


% for evaluation
net_dnn2.layers = {} ;
net_dnn2.layers{end+1} = net_dnn.layers{1};
net_dnn2.layers{end+1} = net_dnn.layers{2};
net_dnn2.layers{end+1} = net_dnn.layers{3};
net_dnn2.layers{end+1} = net_dnn.layers{4};
net_dnn2.layers{end+1} = net_dnn.layers{5};

%------------
if 0
testFun    = @(Xn) dnn_demix(Xn,net_dnn2);

options.SNR_dB = 0;
output_dnn = separation_test_net(testFun,test_female,test_male,options);

NSDR_1(count) = output_dnn.stat.mean_NSDR;

%gives exact same result as cnn 2
end

%% ------------


testFun    = @(Xn) cnn_demix(Xn,net);

options.SNR_dB = 0;
output_cnn_new = separation_test_net(testFun,test_female,test_male,options);

NSDR_2(i) = output_cnn_new.stat.mean_NSDR;

figure(2)
%plot(NSDR_1,'g')
hold on
plot(NSDR_2,'b')
hold off

end
