close all;
clear all;

%% load data

%representation = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_fs16_NFFT1024_hop512/';
representation = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/scatt_fs16_NFFT2048_hop1024/';

id_1 = 2;
id_2 = 11;

% another man!
%id_2 = 14;


load(sprintf('%ss%d',representation,id_1));
data1 = data;
clear data


load(sprintf('%ss%d',representation,id_2));
data2 = data;
clear data


Npad = 2^15;

param.renorm=1;
if param.renorm
%renormalize data: whiten each frequency component.
eps=2e-3;
Xtmp=[abs(data1.X) abs(data2.X)];
stds = std(Xtmp,0,2) + eps;

data1.X = renorm_spect_data(data1.X, stds);
data2.X = renorm_spect_data(data2.X, stds);
end


%% train models


model = 'NMF-scatt';
spectrum = 0;

G = 1;

KK = [160];
KKgn = [48];
LL = [0.1];

for ii = 1:length(KK)
for jj = 1:length(LL)

%%%%Plain NMF%%%%%%%
param0.K = KK(ii);
param0.posAlpha = 1;
param0.posD = 1;
param0.pos = 1;
param0.lambda = LL(jj);
param0.iter = 4000;
param0.numThreads=16;
param0.batchsize=512;

[NN,LL]=size(data1.X);
Lbis = G*floor(LL/G);
X = reshape(data1.X(:,1:Lbis),G*size(data1.X,1),Lbis/G);

Dnmf1 = mexTrainDL(X,param0);

[NN,LL]=size(data2.X);
Lbis = G*floor(LL/G);
X = reshape(data2.X(:,1:Lbis),G*size(data2.X,1),Lbis/G);
Dnmf2 = mexTrainDL(X,param0);

%alpha1= mexLasso(abs(data1.X),Dnmf1,param0);
%alpha2= mexLasso(abs(data2.X),Dnmf2,param0);

%Dnmf1s = sortDZ(Dnmf1,full(alpha1)');
%Dnmf2s = sortDZ(Dnmf2,full(alpha2)');

%gpud=gpuDevice(4);

%param.nmf=1;
param.lambda=LL(jj)/4;
param.beta=1e-2;
%param.overlapping=1;
%param.groupsize=2;
%param.time_groupsize=2;
%param.nu=0.5;
%param.lambdagn=1e-2;
%param.betagn=0;
%param.itersout=200;
param.K=KK(ii);
%param.Kgn=KKgn(ii);
%param.epochs=3;
%param.batchsize=4096;
%param.plotstuff=1;
%
%reset(gpud);

%param.initD = Dnmf1s;
%[D1, Dgn1] = twolevelDL_gpu(abs(data1.X), param);

%reset(gpud);

%param.initD = Dnmf2s;
%[D2, Dgn2] = twolevelDL_gpu(abs(data2.X), param);

%reset(gpud);

    model_name = sprintf('%s-K%d-lambda%d-beta%d',model,param.K,round(100*param.lambda),round(100*param.beta));
    save_folder = sprintf('/misc/vlgscratch3/LecunGroup/bruna/speech/%s-s%d-s%d-%s/',model_name,id_1,id_2,date());

    try
        unix(sprintf('mkdir %s',save_folder));
        unix(sprintf('chmod 777 %s ',save_folder));
    catch
    end

    NFFT = data1.NFFT;
    fs = data1.fs;
    hop = data1.hop;
    N_test = 200;
    SDR = 0;
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

        %X = compute_spectrum(mix,NFFT,hop);
	[X, phmix] = batchscatt(pad_mirror(mix',Npad),data1.filts, data1.scparam);

	if param.renorm
	Xr = renorm_spect_data(X, stds);
	end
  
	 % compute decomposition
	Lbis = G*floor(size(Xr,2)/G);
	X = reshape(abs(Xr(:,1:Lbis)),G*size(Xr,1), Lbis/G);
        H =  full(mexLasso(X,[Dnmf1,Dnmf2],param0));
        W1H1 = Dnmf1*H(1:size(Dnmf1,2),:);
        W2H2 = Dnmf2*H(size(Dnmf1,2)+1:end,:);
	W1H1 = reshape(W1H1,size(Xr,1), Lbis);
	W2H2 = reshape(W2H2,size(Xr,1), Lbis);

	eps = 1e-6;
        V_ap = W1H1.^2 +W2H2.^2 + eps;
        SPEECH1 = ((W1H1.^2)./(V_ap)).*Xr(:,1:size(V_ap,2));
        SPEECH2 = ((W2H2.^2)./(V_ap)).*Xr(:,1:size(V_ap,2));
	
	if spectrum
        	speech1 = invert_spectrum(SPEECH1,NFFT,hop,T);
        	speech2 = invert_spectrum(SPEECH2,NFFT,hop,T);
	else
		
		[speech1] = audioreconstruct(SPEECH1, data1, phmix);
		[speech2] = audioreconstruct(SPEECH2, data2, phmix);
	end

        Parms =  BSS_EVAL(x1', x2', speech1(1:T), speech2(1:T), mix');
        
        SDR = SDR+mean(Parms.NSDR)/N_test;
        SIR = SIR+mean(Parms.SIR)/N_test;
        
        Parms
        output{i} = Parms;

        file1 = sprintf('%s%dspeech-1.wav',save_folder,i);
        audiowrite(file1,speech1,fs);
        unix(sprintf('chmod 777 %s',file1));

        file2 = sprintf('%s%dspeech-2.wav',save_folder,i);
        audiowrite(file2,speech2,fs);
        unix(sprintf('chmod 777 %s',file2));

        filemix = sprintf('%s%dmix.wav',save_folder,i);
        audiowrite(filemix,mix,fs);
        unix(sprintf('chmod 777 %s',filemix));

    end
    save_file = sprintf('%sresults.mat',save_folder,'s');
    save(save_file,'output','D1','D2','param','NSDR','SIR')
    unix(sprintf('chmod 777 %s ',save_file));
    AA{ii,jj}.res = output;
    clear output

end
end

