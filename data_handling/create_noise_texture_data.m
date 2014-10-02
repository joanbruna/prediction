
root = '/misc/vlgscratch3/LecunGroup/pablo/noise_texture/';

% sampling parameters
fs = 16000;
NFFT = 1024;
hop = NFFT/2;



%label = 'female';
label = 'noise_texture';


files = dir(sprintf('%s*.wav',root));

for k = 1:length(files)
    
    
    fprintf('%s\n',files(k).name)
    
    [x,Fs] = audioread(sprintf('%s%s',root,files(k).name));
    x = resample(x,fs,Fs);
    x = x(:)';
    
    a.file = files(k).name;
    a.x = x;
    a.fs = fs;
    
    noise{k} = a;
    
end

save_file = sprintf('%s/%s_audios.mat',root,label);
save(save_file,'noise')
unix(sprintf('chmod 777 %s ',save_file));
