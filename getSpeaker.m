
function data = getSpeaker(folder,N,Nv,params)


if ~exist('params','var')
    params = audio_config();
end

files = dir([folder '/*.wav']);


idx = randperm(length(files));

Strain = [];
% create training samples
sp = zeros(1,N);
sp(1) = 1;
for j=0:N-1
    
    [x,Fs] = audioread([folder '/' files(idx(j+1)).name]);
    if Fs~= params.fs
        x = resample(x,Fs,params.fs);
        x = x(:);
    end
    
    
    S = params.scf * stft(x, params.NFFT , params.winsize, params.hop);
    
    % starting point of the next training file
    if j>0
        sp(j+1) = sp(j) + size(S,2);
    else
        sp(j+1) = size(S,2)+1;
    end
    
    Strain = [Strain S];
end


% Save traing data
idx_train = idx(1:N);
idx_valid = idx((N+1):(N+Nv));
idx_test = idx((N+Nv+1):end);


data.list = files;
data.train_list = files(idx_train);
data.valid_list = files(idx_valid);
data.test_list = files(idx_test);
data.idx_train = idx_train;
data.idx_test = idx_test;
data.idx_valid = idx_valid;


data.N = N;

data.params = params;
data.S = Strain;


% fprintf('done\n')



