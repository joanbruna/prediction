
root = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/';


% sampling parameters
fs = 16000;
NFFT = 2048;
hop = NFFT/2;

Npad = 2^15;

param.T = NFFT;
param.os = 1;%NFFT/hop;
param.Q = 32;
param.N=Npad;

filts = cqt_prepare(param);

idx = randperm(1000);

Ntrain = 500;
Ntest = 500;

% files per-speaker are different, for simplicity use the same indexes for training
training_idx = idx(1:Ntrain);
testing_idx = idx(Ntrain+1:end);

label = 'scatt';

save_folder = sprintf('%s%s_fs%d_NFFT%d_hop%d/',root,label,fs/1000,NFFT,hop);
mkdir(save_folder)

for i = 2:34
    
    folder = sprintf('%s%s%d/',root,'s',i);
    fprintf('%s\n',folder)
    d = dir(sprintf('%s%s',folder,'*.wav'));
    
    Xo = zeros(Npad,Ntrain);
    
    for j=1:Ntrain
        
        [x,Fs] = audioread(sprintf('%s%s',folder,d(training_idx(j) ).name));
        x = resample(x,fs,Fs);
	Xo(:,j)=pad_mirror(x,Npad);        
        
    end

	keyboard;
	X = batchscatt(Xo,filts, param);    

    data.X = X(:,:);
    data.NFFT = NFFT;
    data.hop = hop;
    data.fs = fs;
    data.training_idx = training_idx;
    data.testing_idx = testing_idx;
    data.d = d;
    data.folder = folder;
    data.filts = filts;
    data.scparam = param;
    save_file = sprintf('%s%s%d.mat',save_folder,'s',i);
    save(save_file,'data')
    unix(sprintf('chmod 777 %s ',save_file));
	fprintf('done %d \n', i)
    
end



