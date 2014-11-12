if ~exist('Dnmf11','var')

<<<<<<< HEAD
    
    %% load data
    
    representation = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/scatt2_fs16_NFFT2048/';
    
    id_1 = 1;
    id_2 = 7;
    
    % another man!
    %id_2 = 14;
    
    load(sprintf('%ss%d',representation,id_1));
    data1 = data;
    clear data
    
    load(sprintf('%ss%d',representation,id_2));
    data2 = data;
    clear data
    
    Npad = 2^15;
    
    options.renorm=1;
    if options.renorm
        %renormalize data: whiten each frequency component.
        eps=2e-3;
        Xtmp=[abs(data1.X1) abs(data2.X1)];
        stds1 = std(Xtmp,0,2);
        data1.X1 = renorm_spect_data(data1.X1, stds1, eps);
        data2.X1 = renorm_spect_data(data2.X1, stds1, eps);
=======
%% load data

representation = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/scatt2_fs16_NFFT2048/';

id_1 = 8;
id_2 = 7;

% another man!
%id_2 = 14;

load(sprintf('%ss%d',representation,id_1));
data1 = data;
clear data

load(sprintf('%ss%d',representation,id_2));
data2 = data;
clear data

Npad = 2^15;

options.renorm=1;
if options.renorm
%renormalize data: whiten each frequency component.
eps=2e-3;
epsf=1;
Xtmp=[abs(data1.X1) abs(data2.X1)];
stds1 = std(Xtmp,0,2) + eps;
data1.X1 = renorm_spect_data(data1.X1, stds1, epsf);
data2.X1 = renorm_spect_data(data2.X1, stds1, epsf);

eps=1e-3;
Xtmp=[abs(data1.X2) abs(data2.X2)];
stds2 = std(Xtmp,0,2) + eps;
data1.X2 = renorm_spect_data(data1.X2, stds2, epsf);
data2.X2 = renorm_spect_data(data2.X2, stds2, epsf);
end


%% train models

model = 'NMF-scatt2';


%%%%Plain NMF%%%%%%%
KK1 = [160];
LL1 = [0.04];
param1.K = KK1;
param1.posAlpha = 1;
param1.posD = 1;
param1.pos = 1;
param1.lambda = LL1;
param1.iter = 4000;
param1.numThreads=16;
param1.batchsize=512;

Dnmf11 = mexTrainDL(abs(data1.X1),param1);
Dnmf12 = mexTrainDL(abs(data2.X1),param1);

KK2 = [768];
LL2 = [0.1];
param2.K = KK2;
param2.posAlpha = 1;
param2.posD = 1;
param2.pos = 1;
param2.lambda = LL2;
param2.iter = 4000;
param2.numThreads=16;
param2.batchsize=512;

Dnmf21 = mexTrainDL(abs(data1.X2),param2);
Dnmf22 = mexTrainDL(abs(data2.X2),param2);

keyboard;

    model_name = sprintf('%s-K1%d-K2%d',model,KK1,KK2);
    save_folder = sprintf('/misc/vlgscratch3/LecunGroup/bruna/speech/%s-s%d-s%d-%s/',model_name,id_1,id_2,date());

    try
        unix(sprintf('mkdir %s',save_folder));
        unix(sprintf('chmod 777 %s ',save_folder));
    catch
    end


end


    fs = data1.fs;
    N_test = 200;
    SDR = 0;
    NSDR = 0;
    SIR = 0;
    
    for i = 1:N_test

        [x1, Fs] = audioread(sprintf('%s%s',data1.folder,data1.d(data1.testing_idx(i) ).name) );
        x1 = resample(x1,fs,Fs);
        x1 = x1(:)'; T1 = length(x1);


        [x2, Fs] = audioread(sprintf('%s%s',data2.folder,data2.d(data2.testing_idx(i) ).name) );
        x2 = resample(x2,fs,Fs);
        x2 = x2(:)'; T2 = length(x2);

        T = min(T1,T2);

        x1 = x1(1:T);
        x2 = x2(1:T);

        mix = (x1+x2);

	%[X, phmix] = batchscatt(pad_mirror(mix',Npad),data1.filts, data1.scparam);


	%%% demixing second order scatt. 
	%%% min || | W1 xi | - D1i z1i || + || |W2 | W1 xi | | - D2i z2i || st x=x1+x2

	%[speech1, speech2, xest1, xest2] = demix_scatt2(mix, Dnmf11, Dnmf12, Dnmf21, Dnmf22, stds1, stds2, data1.filts, data1.scparam, param1, param2, Npad);
	[speech1, speech2, xest1, xest2] = demix_scatt2top(mix, Dnmf11, Dnmf12, Dnmf21, Dnmf22, stds1, stds2, epsf, data1.filts, data1.scparam, param1, param2, Npad);
	%[speech1b, speech2b] = demix_scatt2(mix, Dnmf11, Dnmf12, Dnmf21, Dnmf22, stds1, stds2, data1.filts, data1.scparam, param1, param2, Npad);


        Parms =  BSS_EVAL(x1', x2', speech1(1:T)', speech2(1:T)', mix');
        %Parmsb =  BSS_EVAL(x1', x2', speech1b(1:T), speech2b(1:T), mix');
        Parms1 =  BSS_EVAL(x1', x2', xest1(1:T)', xest2(1:T)', mix');
        
        SDR = SDR+mean(Parms.SDR)/N_test;
        NSDR = NSDR+mean(Parms.NSDR)/N_test;
        SIR = SIR+mean(Parms.SIR)/N_test;
>>>>>>> 05069e25206e5cf9f8723be9bcd1c558ac4419f7
        
        eps=1e-3;
        Xtmp=[abs(data1.X2) abs(data2.X2)];
        stds2 = std(Xtmp,0,2);
        data1.X2 = renorm_spect_data(data1.X2, stds2, eps);
        data2.X2 = renorm_spect_data(data2.X2, stds2, eps);
    end
    
    
    %% train models
    
    model = 'NMF-scatt2';
    
    
    %%%%Plain NMF%%%%%%%
    KK1 = [160];
    LL1 = [0.1];
    param1.K = KK1;
    param1.posAlpha = 1;
    param1.posD = 1;
    param1.pos = 1;
    param1.lambda = LL1;
    param1.iter = 10;
    param1.numThreads=16;
    param1.batchsize=512;
    
    Dnmf11 = mexTrainDL(abs(data1.X1),param1);
    Dnmf12 = mexTrainDL(abs(data2.X1),param1);
    
    KK2 = [768];
    LL2 = [0.1];
    param2.K = KK2;
    param2.posAlpha = 1;
    param2.posD = 1;
    param2.pos = 1;
    param2.lambda = LL2;
    param2.iter = 10;
    param2.numThreads=16;
    param2.batchsize=512;
    
    Dnmf21 = mexTrainDL(abs(data1.X2),param2);
    Dnmf22 = mexTrainDL(abs(data2.X2),param2);
    
%    keyboard;
    
    
    
%     model_name = sprintf('%s-K1%d-K2%d',model,KK1,KK2);
%     save_folder = sprintf('/misc/vlgscratch3/LecunGroup/bruna/speech/%s-s%d-s%d-%s/',model_name,id_1,id_2,date());
%     
%     try
%         unix(sprintf('mkdir %s',save_folder));
%         unix(sprintf('chmod 777 %s ',save_folder));
%     catch
%     end
    
    
end


fs = data1.fs;
N_test = 200;
SDR = 0;
NSDR = 0;
SIR = 0;

for i = 1:N_test
    
    [x1, Fs] = audioread(sprintf('%s%s',data1.folder,data1.d(data1.testing_idx(i) ).name) );
    x1 = resample(x1,fs,Fs);
    x1 = x1(:)'; T1 = length(x1);
    
    
    [x2, Fs] = audioread(sprintf('%s%s',data2.folder,data2.d(data2.testing_idx(i) ).name) );
    x2 = resample(x2,fs,Fs);
    x2 = x2(:)'; T2 = length(x2);
    
    T = min(T1,T2);
    
    x1 = x1(1:T);
    x2 = x2(1:T);
    
    mix = (x1+x2);
    
    %[X, phmix] = batchscatt(pad_mirror(mix',Npad),data1.filts, data1.scparam);
    
    
    %%% demixing second order scatt.
    %%% min || | W1 xi | - D1i z1i || + || |W2 | W1 xi | | - D2i z2i || st x=x1+x2
    
    %[speech1, speech2, xest1, xest2] = demix_scatt2(mix, Dnmf11, Dnmf12, Dnmf21, Dnmf22, stds1, stds2, data1.filts, data1.scparam, param1, param2, Npad);
    [speech1, speech2, xest1, xest2] = demix_scatt2top(mix, Dnmf11, Dnmf12, Dnmf21, Dnmf22, stds1, stds2, eps, data1.filts, data1.scparam, param1, param2, Npad);
    %[speech1b, speech2b] = demix_scatt2(mix, Dnmf11, Dnmf12, Dnmf21, Dnmf22, stds1, stds2, data1.filts, data1.scparam, param1, param2, Npad);
    
    
    Parms =  BSS_EVAL(x1', x2', speech1(1:T)', speech2(1:T)', mix');
    %Parmsb =  BSS_EVAL(x1', x2', speech1b(1:T), speech2b(1:T), mix');
    Parms1 =  BSS_EVAL(x1', x2', xest1(1:T)', xest2(1:T)', mix');
    
    SDR = SDR+mean(Parms.SDR)/N_test;
    NSDR = NSDR+mean(Parms.NSDR)/N_test;
    SIR = SIR+mean(Parms.SIR)/N_test;
    
    Parms
    Parms1
    %Parmsb
    
    output0{i} = Parms;
    %outputb{i} = Parmsb;
    output1{i} = Parms1;
    
    
end

save_file = sprintf('%sresults.mat',save_folder,'s');
save(save_file,'output','D1','D2','param','SDR','NSDR','SIR')
unix(sprintf('chmod 777 %s ',save_file));
AA{ii,jj}.res = output;
clear output


