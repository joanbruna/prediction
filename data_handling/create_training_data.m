
addpath utils


addpath utils
addpath stft

numspeakers=20;
numclips=300;
numtclips=100;
numtdclips=200;

totspeakers=34;


for i=1:totspeakers
    
    fprintf('Processing speaker %d.',s)
    % voice
    folderv = load(sprintf('/misc/vlgscratch3/LecunGroup/bruna/grid_data/s%d/',i));
    
    dv = dir([folderv '*.wav']);
    
    idx = randperm(length(dv));
    
    
    Strain = [];
    % create training samples
    for j=1:N
        [vaux,fs_orig] = audioread([folderv dv(idx(j)).name]);
        
        
        mean(vaux)
        
        vaux =  resample(vaux,fs,fs_orig);
        
        

    end
    
    
    % Save traing data
    idx_train = idx(1:N);
    idx_valid = idx((N+1):(N+Nv));
    idx_test = idx((N+Nv+1):end);
    
end



fprintf('done\n')