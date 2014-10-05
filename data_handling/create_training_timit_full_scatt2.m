clear all;

root = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/scatt2_fs16_NFFT2048/TRAIN/';

d = dir(sprintf('%s*_M*',root));

MM = floor(3*length(d)/4);
Xc1 = cell(MM,1);
Xc2 = cell(MM,1);

label = 'male';

keyboard

for j = 1:MM
    
    
    file = d(j).name;
    file_speaker = sprintf('%s%s',root,file);

    load(file_speaker)
    
    fprintf('%s\n',file)
    
    Xc1{j} = data.X1;
    Xc2{j} = data.X2;
    
    n1(j) = size(data.X1,2);
    n2(j) = size(data.X2,2);
    
    M1 = size(data.X1,1);
    M2 = size(data.X2,1);
    
    clear data

end

load(file_speaker)
    
fs = data.fs;
scparam = data.scparam;
filts = data.filts;
T=data.T;
clear data

%%

X1 = zeros(M1,sum(n1));
X2 = zeros(M2,sum(n2));

count1 = 1;
count2 = 1;
for j=1:length(Xc1)
    disp(j)
    X1(:,count1:count1-1+n1(j)) = Xc1{j};
    count1 = count1+n1(j);
    X2(:,count2:count2-1+n2(j)) = Xc2{j};
    count2 = count2+n2(j);
end

data.X1 = X1;
data.X2 = X2;
data.T = T;
data.fs = fs;
data.filts = filts;
data.gender = label;
data.scparam = scparam;

save_file = sprintf('%s%s.mat',root,label);
data.file = save_file;
save(save_file,'data', '-v7.3')
unix(sprintf('chmod 777 %s ',save_file));

