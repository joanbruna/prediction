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

% reduce data
data1.X = data1.X(:,1:1000);

%% train models

model = 'NMF-pooling';

KK = [200];
KKgn = [80];
LL = [0.1];

ii = 1; jj= 1;


param0.K = KK(ii);
param0.posAlpha = 1;
param0.posD = 1;
param0.pos = 1;
param0.lambda = LL(jj);
param0.iter = 20;
param0.numThreads=16;
param0.batchsize=512;

Dnmf1 = mexTrainDL(abs(data1.X),param0);
Dnmf2 = mexTrainDL(abs(data2.X),param0);

alpha1= mexLasso(abs(data1.X),Dnmf1,param0);
alpha2= mexLasso(abs(data2.X),Dnmf2,param0);

Dnmf1s = sortDZ(Dnmf1,full(alpha1)');
Dnmf2s = sortDZ(Dnmf2,full(alpha2)');


%%

gpud=gpuDevice(1);

param.nmf=1;
param.lambda=LL(jj)/4;
param.beta=1e-2;
param.overlapping=1;
param.groupsize=2;
param.time_groupsize=2;
param.nu=0;
param.lambdagn=0;
param.betagn=0;
param.itersout=200;
param.K=KK(ii);
param.Kgn=KKgn(ii);
param.epochs=1;
param.batchsize=4096;
param.plotstuff=0;

%%



reset(gpud);


param.initD = Dnmf1s;
[D1, Dgn1] = twolevelDL_gpu(abs(data1.X), param);

break
%%

eps = 1e-8;
%reset(gpud);

param.nu=0;

X = abs(data1.X(:,1:50));
param.itersout=20000;
param.eps=1e-2;

%D1 = double(gather(D1));
%Dgn1 = double(gather(Dgn1));
% 
%X = rand(size(X))+0.01;
%D1 = rand(size(D1))+0.01;
D1 = Dnmf1s;
Dgn1 = rand(size(Dgn1))+0.01;

for j=1:10
    

    %alpha = zeros(size(alpha));
    
    %f = betadiv(V,D*lassoRes,beta);
    [Zout, Zgnout,Poole] = twolevellasso_cpu(X, D1, Dgn1, param);
    %Zout(Zout(:)<eps/100) = 0;
    
    a = 0.3*ones(size(Zout));

    f = 0.5*sum((a(:)-Zout(:)).^2);
    G = (Zout-a);
    
    [dZout,dZgnout] = twolevellasso_grads(X,D1,Dgn1,Zout,Zgnout,G, param,Poole);
    
    dvar = eps*randn(size(D1));
    D1_ = D1 + dvar;

    [Zout_, Zgnout_,Poole_] = twolevellasso_cpu(X, D1_, Dgn1, param);
   % Zout_(Zout_(:)<eps/100) = 0;

    f_ = 0.5*sum((a(:)-Zout_(:)).^2);
    [f_-f dZout(:)'*dvar(:)]/eps
    
    break
    


    f_ = measure_bilevel_cost(Z_, D_, Dgn_, V, lambda1,lambda2, lambda1gn, lambda2gn, groupsize,time_groupsize);
    
    disp([f_ - f,df(:)'*dvar(:)]/eps)
    
end

