clear all;

%root = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/scatt2_fs16_NFFT2048/TRAIN/';
root = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/cqt_phase_fs16_NFFT2048_hop1024/TRAIN/';

d = dir(sprintf('%s*_M*',root));
%d = dir(sprintf('%s*_F*',root));

MM = floor(3*length(d)/4);
Xc1 = cell(MM,1);

%label = 'female';
label = 'male';

for j = 1:MM
    
    
    file = d(j).name;
    file_speaker = sprintf('%s%s',root,file);

    load(file_speaker)
    
    fprintf('%s\n',file)
    
    Xc1{j} = data.X;
    
    n1(j) = size(data.X,2);
    
    M1 = size(data.X,1);
    
    clear data

end

load(file_speaker)
    
fs = data.fs;
scparam = data.scparam;

%%

% sampling parameters
fs = 16000;
NFFT = 2048;
hop = NFFT/2;

Npad = 2^15;

scparam.T = NFFT;
scparam.os = 1;%NFFT/hop;
scparam.Q = 32;
scparam.N=Npad;

filts = cqt_prepare(scparam);

T=NFFT;
clear data

%%

X1 = zeros(M1,sum(n1));


count1 = 1;
count2 = 1;
for j=1:length(Xc1)
    disp(j)
    X1(:,count1:count1-1+n1(j)) = Xc1{j};
    count1 = count1+n1(j);
end

data.X = X1;
data.T = T;
data.fs = fs;
data.filts = filts;
data.gender = label;
data.scparam = scparam;

save_file = sprintf('%s%s.mat',root,label);
data.file = save_file;
save(save_file,'data', '-v7.3')
unix(sprintf('chmod 777 %s ',save_file));

