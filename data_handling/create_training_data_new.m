


root = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/';


% sampling parameters
fs = 16000;
NFFT = 1024;
hop = NFFT/2;


idx = randperm(1000);

Ntrain = 500;
Ntest = 500;

% files per-speaker are different, for simplicity use the same indexes for training
training_idx = idx(1:Ntrain);
testing_idx = idx(Ntrain+1:end);

label = 'spect';

save_folder = sprintf('%s%s_fs%d_NFFT%d_hop%d/',root,label,fs/1000,NFFT,hop);

%unix(sprintf('mkdir %s',save_folder));
%unix(sprintf('chmod 777 %s ',save_folder));

for i = 2:34
    
    folder = sprintf('%s%s%d/',root,'s',i);
    fprintf('%s\n',folder)
    d = dir(sprintf('%s%s',folder,'*.wav'));
    
    X = [];
    
    for j=1:Ntrain
        
        [x,Fs] = audioread(sprintf('%s%s',folder,d(training_idx(j) ).name));
        x = resample(x,fs,Fs);
        x = x(:)';
        
        Xt = compute_spectrum(x,NFFT,hop);
        
        X = [X,Xt];
        
    end
    
    save_file = sprintf('%s%s%d.mat',save_folder,'s',i);
    save(save_file,'X','fs','NFFT','hop','training_idx','testing_idx')
    unix(sprintf('chmod 777 %s ',save_file));
    
end



