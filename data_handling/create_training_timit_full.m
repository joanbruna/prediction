%root = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/spect_fs16_NFFT1024_hop512/TRAIN/';
root = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/cqt_fs16_NFFT2048_hop1024/TRAIN/';

d = dir(sprintf('%s*_F*',root));

MM = floor(3*length(d)/4);
Xc = cell(MM,1);

label = 'female';
for j = 1:MM
    
    
    file = d(j).name;
    file_speaker = sprintf('%s%s',root,file);

    load(file_speaker)
    
    fprintf('%s\n',file)
    
    Xc{j} = data.X;
    
    n(j) = size(data.X,2);
    
    M = size(data.X,1);
    
    clear data

end

load(file_speaker)
    
NFFT = data.NFFT;
hop = data.hop;
fs = data.fs;
scparam = data.scparam;
clear data

%%

X = zeros(M,sum(n));

count = 1;
for j=1:length(Xc)
    disp(j)
    X(:,count:count-1+n(j)) = Xc{j};
    count = count+n(j);
end


data.X = X;
data.NFFT = NFFT;
data.hop = hop;
data.fs = fs;
data.gender = label;
data.scparam = scparam;

save_file = sprintf('%s%s.mat',root,label);
data.file = save_file;
save(save_file,'data')
unix(sprintf('chmod 777 %s ',save_file));

