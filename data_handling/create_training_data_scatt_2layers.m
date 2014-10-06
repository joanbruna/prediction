clear all;
close all;


root = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/';

fs = 16000;
Npad = 2^15;
T = 2048;

options.N = Npad;
options.T = T;
options.Q = 32;

filts = create_scattfilters(options);

idx = randperm(1000);

Ntrain = 500;
Ntest = 500;

% files per-speaker are different, for simplicity use the same indexes for training
training_idx = idx(1:Ntrain);
testing_idx = idx(Ntrain+1:end);

label = 'scatt2';

save_folder = sprintf('%s%s_fs%d_NFFT%d/',root,label,fs/1000,T);
mkdir(save_folder)

for i = 1:11
    
    folder = sprintf('%s%s%d/',root,'s',i);
    fprintf('%s\n',folder)
    d = dir(sprintf('%s%s',folder,'*.wav'));
    
    Xo = zeros(Npad,Ntrain);
    
    for j=1:Ntrain
        
        [x,Fs] = audioread(sprintf('%s%s',folder,d(training_idx(j) ).name));
        x = resample(x,fs,Fs);
	Xo(:,j)=pad_mirror(x,Npad);        
        
    end

	[X2, X1] = audioscatt_fwd_haar(Xo,filts, options);    

    data.X1 = X1(:,:);
    data.X2 = X2(:,:);
    data.T = T;
    data.fs = fs;
    data.training_idx = training_idx;
    data.testing_idx = testing_idx;
    data.d = d;
    data.folder = folder;
    data.filts = filts;
    data.scparam = options;
    save_file = sprintf('%s%s%d.mat',save_folder,'s',i);
    save(save_file,'data', '-v7.3')
    unix(sprintf('chmod 777 %s ',save_file));
	fprintf('done %d \n', i)
    
end



