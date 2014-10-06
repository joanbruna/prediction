

gpud=gpuDevice(1);

%representation = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_fs16_NFFT1024_hop512/';
representation = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/spect_fs16_NFFT1024_hop512/TRAIN/';



load(sprintf('%sfemale',representation));
data1 = data;
clear data



% epsilon = 1;
param.epsilon = 0.001;
epsilon = param.epsilon;
data1.X = softNormalize(abs(data1.X),epsilon);




%% train models

model = 'NMF-L2-softnorm';

KK = [128 128 500 1000];
KKgn = [64 32 64 128];


GG = [4 8 8 10];

LL = [0.1 0.1 0.1 0.1];



for ii = 1:length(KK)

    param.nmf=1;
    param.lambda=LL(ii)/4;
    param.beta=1e-2;
    param.overlapping=1;
    param.groupsize=GG(ii);
    param.time_groupsize=2;
    param.lambdagn=1e-1;
    param.betagn=0;
    param.itersout=200;
    param.K=KK(ii);
    param.Kgn=KKgn(ii);
    param.batchsize=4096;
    param.plotstuff=1;
    
    reset(gpud);
    
    param.nu=0;
    param.epochs=1;
    param.initD = mexNormalize(rand(size(data1.X,1),param.K)+0.1);
    [D1i, Dgn1] = twolevelDL_gpu(data1.X, param);
    
    
    reset(gpud);
    
    param.nu = 0.5;
    param.epochs=3;
    param.initD = D1i;
    [D1, Dgn1] = twolevelDL_gpu(data1.X, param);
    

    AA{ii}.D = D1;
    AA{ii}.Dgn = Dgn1;
    AA{ii}.param = param;

end


%    model_name = sprintf('%s-K%d-lambda%d-lambda2%d',model,param.K,round(10*param.lambda),round(10*param.lambda2));

% %%
% ii = 1;
% LL(ii) = 0.1;
% KK(ii) = 200;
% KKgn(ii) = 20; 
% 
% param.nmf=1;
% param.lambda = LL(ii);
% param.beta=1e-2;
% param.overlapping=0;
% param.groupsize=4;
% param.time_groupsize=2;
% param.nu=0;
% param.lambdagn=1e-2;
% param.betagn=0;
% param.itersout=200;
% param.K=KK(ii);
% param.Kgn=KKgn(ii);
% param.epochs=3;
% param.batchsize=4096;
% param.plotstuff=1;
% 
% if param.overlapping
%     param.lambda=param.lambda/4;
% end
% 
% reset(gpud);
% 
% param.initD = mexNormalize(rand(size(data.X,1),KK(ii))+0.1);
% [D3, Dgn] = twolevelDL_gpu(abs(data.X), param);


