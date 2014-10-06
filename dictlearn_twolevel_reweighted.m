
%% load data

representation = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_fs16_NFFT1024_hop512/';

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


param.renorm=1;

if param.renorm
%renormalize data: whiten each frequency component.
eps=4e-1;
Xtmp=[abs(data1.X) abs(data2.X)];
stds = std(Xtmp,0,2) + eps;

data1.X = renorm_spect_data(data1.X, stds);
data2.X = renorm_spect_data(data2.X, stds);
end


%% train models

model = 'NMF-pooling';

KK = [200];
KKgn = [80];
LL = [0.1];
ii = 1;
jj = 1;



%%%%Plain NMF%%%%%%%
% param0.K = KK(ii);
% param0.posAlpha = 1;
% param0.posD = 1;
% param0.pos = 1;
% param0.lambda = LL(jj);
% param0.iter = 1000;
% param0.numThreads=16;
% param0.batchsize=512;
% 
% Dnmf1 = mexTrainDL(abs(data1.X),param0);
% Dnmf2 = mexTrainDL(abs(data2.X),param0);
% 
% alpha1= mexLasso(abs(data1.X),Dnmf1,param0);
% alpha2= mexLasso(abs(data2.X),Dnmf2,param0);
% 
% Dnmf1s = sortDZ(Dnmf1,full(alpha1)');
% Dnmf2s = sortDZ(Dnmf2,full(alpha2)');


data1.X = data1.X(:,1:10000);


gpud=gpuDevice(1);

param.nmf=1;
param.lambda=LL(jj)/4;
param.beta=1e-2;
param.overlapping=1;
param.groupsize=2;
param.time_groupsize=2;
param.nu=0.5;
param.lambdagn=0;
param.betagn=0;
param.itersout=200;
param.K=KK(ii);
param.Kgn=KKgn(ii);
param.epochs=4;

param.batchsize=min(4096,size(data1.X,2));
param.plotstuff=0;

reset(gpud);



[Dout, Dgnout,Din,Dgnin] = twolevelDL_reweighted(abs(data1.X), param);
