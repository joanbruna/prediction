
representation = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/scatt2_fs16_NFFT2048/';

load('/misc/vlgscratch3/LecunGroup/bruna/grid_data/icassp_grid_exp/renorm_params')

id_f = [4,7,11];
id_m = [1,2,3];

results = cell(length(id_f),length(id_m));

for j = 1:length(id_f)
    
load (['/misc/vlgscratch3/LecunGroup/bruna/grid_data/icassp_grid_exp/dictionaries_NMF_scatt2_s' num2str( id_f(j) ) '.mat'])
Dnmf11 = Dnmf1;
Dnmf21 = Dnmf2;
clear Dnmf1 Dnmf2

load(sprintf('%ss%d',representation,id_f(j)) );
data1 = data;
clear data

for k = 1:length(id_m)

load(sprintf('%ss%d',representation,id_m(k)) );
data2 = data;
clear data
    
load (['/misc/vlgscratch3/LecunGroup/bruna/grid_data/icassp_grid_exp/dictionaries_NMF_scatt2_s' num2str( id_m(k) ) '.mat'])
    
Dnmf12 = Dnmf1;
Dnmf22 = Dnmf2;

clear Dnmf1 Dnmf2

fs = data1.fs;
N_test = 200;

Npad = data1.scparam.N;

for i = 1:N_test
    
    [x1, Fs] = audioread(sprintf('%s%s',data1.folder,data1.d(data1.testing_idx(i) ).name) );
    x1 = resample(x1,fs,Fs);
    x1 = x1(:)'; T1 = length(x1);
    
    
    [x2, Fs] = audioread(sprintf('%s%s',data2.folder,data2.d(data2.testing_idx(i) ).name) );
    x2 = resample(x2,fs,Fs);
    x2 = x2(:)'; T2 = length(x2);
    
    T = min([T1,T2,Npad]);
    
    x1 = x1(1:T);
    x2 = x2(1:T);
    
    mix = (x1+x2);
    
    %[X, phmix] = batchscatt(pad_mirror(mix',Npad),data1.filts, data1.scparam);
    
    
    %%% demixing second order scatt.
    %%% min || | W1 xi | - D1i z1i || + || |W2 | W1 xi | | - D2i z2i || st x=x1+x2
    
    %[speech1, speech2, xest1, xest2] = demix_scatt2(mix, Dnmf11, Dnmf12, Dnmf21, Dnmf22, stds1, stds2, data1.filts, data1.scparam, param1, param2, Npad);
    %[speech1, speech2, xest1, xest2] = demix_scatt2top(mix, Dnmf11, Dnmf12, Dnmf21, Dnmf22, stds1, stds2, eps, data1.filts, data1.scparam, param1, param2, Npad);
    [speech1, speech2, xest1, xest2] = demix_oracle_scatt2top(mix, x1,x2,stds1, stds2, eps, data1.filts, data1.scparam, param1, param2, Npad);

    %[speech1b, speech2b] = demix_scatt2(mix, Dnmf11, Dnmf12, Dnmf21, Dnmf22, stds1, stds2, data1.filts, data1.scparam, param1, param2, Npad);
    
    
    Parms =  BSS_EVAL(x1', x2', speech1(1:T)', speech2(1:T)', mix');
    %Parmsb =  BSS_EVAL(x1', x2', speech1b(1:T), speech2b(1:T), mix');
    Parms1 =  BSS_EVAL(x1', x2', xest1(1:T)', xest2(1:T)', mix');
    
    
    Parms
    Parms1
    %Parmsb
    
    output0{i} = Parms;
    %outputb{i} = Parmsb;
    output1{i} = Parms1;
    
    
    SDR(i) = mean(Parms.SDR);
    NSDR(i) = mean(Parms.NSDR);
    SAR(i) = mean(Parms.SAR);
    SIR(i) = mean(Parms.SIR);
    
    SDR1(i) = mean(Parms1.SDR);
    NSDR1(i) = mean(Parms1.NSDR);
    SAR1(i) = mean(Parms1.SAR);
    SIR1(i) = mean(Parms1.SIR);
    
    
end



result{j,k}.SDR = SDR;
result{j,k}.NSDR = NSDR;
result{j,k}.SIR = SIR;
result{j,k}.SAR = SAR;

result{j,k}.SDR1 = SDR1;
result{j,k}.NSDR1 = NSDR1;
result{j,k}.SIR1 = SIR1;
result{j,k}.SAR1 = SAR1;

result{j,k}.output0 = output0;
result{j,k}.output1 = output1;

% globl variables

stat.mean_SDR = mean(SDR(:));
stat.std_SDR = std(SDR(:));

stat.mean_SIR = mean(SIR(:));
stat.std_SIR = std(SIR(:));

stat.mean_SAR = mean(SAR(:));
stat.std_SAR = std(SAR(:));

stat.mean_NSDR = mean(NSDR(:));
stat.std_NSDR = std(NSDR(:));

%---

stat.mean_SDR1 = mean(SDR1(:));
stat.std_SDR1 = std(SDR1(:));

stat.mean_SIR1 = mean(SIR1(:));
stat.std_SIR1 = std(SIR1(:));

stat.mean_SAR1 = mean(SAR1(:));
stat.std_SAR1 = std(SAR1(:));

stat.mean_NSDR1 = mean(NSDR1(:));
stat.std_NSDR1 = std(NSDR1(:));

result{j,k}.stat = stat;


end
end