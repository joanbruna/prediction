
representation = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_fs16_NFFT1024_hop512/';
% representation is only used for the index of testing files
representation2 = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/scatt2_fs16_NFFT2048/';

load('/misc/vlgscratch3/LecunGroup/bruna/grid_data/icassp_grid_exp/renorm_params')

id_f = [4,7,11];
id_m = [1,2,3];

results = cell(length(id_f),length(id_m));

for j = 3:length(id_f)
    
load (['/misc/vlgscratch3/LecunGroup/bruna/grid_data/icassp_grid_exp/dictionaries_NMF_spect_s' num2str( id_f(j) ) '.mat'])
D1 = Dnmf;
clear Dnmf

load(sprintf('%ss%d',representation,id_f(j)) );
data1 = data;
clear data

load(sprintf('%ss%d',representation2,id_f(j)) );
data1.testing_idx = data.testing_idx;
data1.d = data.d;
clear data

for k = 3:length(id_m)

load(sprintf('%ss%d',representation,id_m(k)) );
data2 = data;
clear data

load(sprintf('%ss%d',representation2,id_m(k)) );
data2.testing_idx = data.testing_idx;
data2.d = data.d;
clear data

    
load (['/misc/vlgscratch3/LecunGroup/bruna/grid_data/icassp_grid_exp/dictionaries_NMF_spect_s' num2str( id_m(k) ) '.mat'])
D2 = Dnmf;
clear Dnmf

fs = data1.fs;
NFFT = data1.NFFT;
hop = data1.hop;
N_test = 200;

for i = 1:N_test
    
    [x1, Fs] = audioread(sprintf('%s%s',data1.folder,data1.d(data1.testing_idx(i) ).name) );
    x1 = resample(x1,fs,Fs);
    x1 = x1(:)'; T1 = length(x1);
    
    
    [x2, Fs] = audioread(sprintf('%s%s',data2.folder,data2.d(data2.testing_idx(i) ).name) );
    x2 = resample(x2,fs,Fs);
    x2 = x2(:)'; T2 = length(x2);
    
    T = min([T1,T2]);
    
    x1 = x1(1:T);
    x2 = x2(1:T);
    
    mix = (x1+x2);
    
    X = compute_spectrum(mix,NFFT,hop);
    
    epsilon = 0.5;
    Xn = softNormalize(abs(X),epsilon);
    
    % compute decomposition
    H =  full(mexLasso(Xn,[D1,D2],param));
    
    W1H1 = D1*H(1:size(D1,2),:);
    W2H2 = D2*H(size(D1,2)+1:end,:);
    
    eps_1 = 1e-6;%eps_1=0;
    V_ap = W1H1.^2 +W2H2.^2 + eps_1;
    
    % wiener filter
    
    SPEECH1 = ((W1H1.^2)./V_ap).*X;
    SPEECH2 = ((W2H2.^2)./V_ap).*X;
    speech1 = invert_spectrum(SPEECH1,NFFT,hop,T);
    speech2 = invert_spectrum(SPEECH2,NFFT,hop,T);
    
    
    
    Parms =  BSS_EVAL(x1', x2', speech1(1:T)', speech2(1:T)', mix');
    Parms
    
    
    output0{i} = Parms;
    
    SDR(i) = mean(Parms.SDR);
    NSDR(i) = mean(Parms.NSDR);
    SAR(i) = mean(Parms.SAR);
    SIR(i) = mean(Parms.SIR);

        
end



result{j,k}.SDR = SDR;
result{j,k}.NSDR = NSDR;
result{j,k}.SIR = SIR;
result{j,k}.SAR = SAR;

result{j,k}.output0 = output0;

% globl variables

stat.mean_SDR = mean(SDR(:));
stat.std_SDR = std(SDR(:));

stat.mean_SIR = mean(SIR(:));
stat.std_SIR = std(SIR(:));

stat.mean_SAR = mean(SAR(:));
stat.std_SAR = std(SAR(:));

stat.mean_NSDR = mean(NSDR(:));
stat.std_NSDR = std(NSDR(:));


result{j,k}.stat = stat;


end
end