clear all;
close all;

if ~exist('X1','var')
    load '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s31.mat'
    X1 = Xc;
    clear Xc;
    
    epsilon = 1;
    
    load '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s14.mat'
    X2 = Xc;
    
    X = [X1 X2];
    clear X1 X2
    
end

X0=X;

%renormalize data: whiten each frequency component.
eps=4e-1;
stds = std(X,0,2) + eps;
X = X./repmat(stds,1,size(X,2));
avenorm = mean(sqrt(sum(X.^2)));
X = X/avenorm;

%clear X

param.stds = stds;
param.avenorm = avenorm;

%%init phase: D is initialized with NMF. 
%% S is initialized by looking at temporally smoothed activations
param.K=400; % learns a dictionary 
param.lambda=0.01; 
param.numThreads=16;	%	number	of	threads 
param.batchsize =512;
param.iter=1000; % let us see what happens after 1000 iterations .
param.posD=1;
param.posAlpha=1;
param.pos=1;
param.nmf=1;
param.lambda=0.01;
param.epochs=1;
param.time_groupsize=2;
param.Jmax = 4;

if ~exist('D','var')
    load('/home/bruna/matlab/prediction/scripts/dicttree1new.mat')
param.time_groupsize=2;
param.Jmax = 4;
end

%speech1 ='/misc/vlgscratch3/LecunGroup/bruna/grid_data/s1/lrbr4n.wav';
speech2 = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/s14/prin8s.wav';
speech1 ='/misc/vlgscratch3/LecunGroup/bruna/grid_data/s31/pwag9a.wav';

SNR_dB = 0;
params_aux = audio_config();

fs = params_aux.fs;
NFFT = params_aux.NFFT;
hop = params_aux.hop;

[x,Fs] = audioread(speech1);
x = resample(x,fs,Fs);
x = x(:);
[n,Fs] = audioread(speech2);
n = resample(n,fs,Fs);
n = n(:);
% adjust the size
m = min(length(x),length(n));
x = x(1:m);
n = n(1:m);
% adjust SNR
x = x/sqrt(sum(power(x,2)));
if sum(power(n,2))>0
    n = n/sqrt( sum(power(n,2)));
    n = n*power(10,(-SNR_dB)/20);
end

% compute noisy signal
mix = x + n;


Smix = compute_spectrum(mix,NFFT, hop);
Vmix = abs(Smix);
Pmix = Vmix;%softNormalize(Vmix,epsilon);

Sx = compute_spectrum(x,NFFT, hop);
Vx = abs(Sx);
Px  = Vx;%softNormalize(Vx,epsilon);

Sn = compute_spectrum(n,NFFT, hop);
Vn = abs(Sn);
Pn  = Vn;% softNormalize(Vn,epsilon);

%%

Pmix = Pmix./repmat(param.stds,1,size(Pmix,2));
Pmix = Pmix/param.avenorm;
Px = Px./repmat(param.stds,1,size(Pmix,2));
Px = Px/param.avenorm;

Pn = Pn./repmat(param.stds,1,size(Pmix,2));
Pn = Pn/param.avenorm;

if mod(size(Pmix,2),2)>0
Pmix = Pmix(:,1:end-1);
Px = Px(:,1:end-1);
Pn = Pn(:,1:end-1);
end


	[indexes,indexes_inv] = getTreeIndexes(size(D,2),size(Pmix,2),T,param.time_groupsize, param.Jmax);
	param.indexes = indexes;
	param.indexes_inv = indexes_inv;
        t0 = getoptions(param,'alpha_step',0.25);
        t0 = t0 * (1/max(svd(D))^2);

    [alphad1,alphad2,cost_aux] = group_pooling_graph_demix( D, T, Pmix, param,t0);
    [alpha1,cost_aux] = group_pooling_graph( D, T, Px, param,t0);
    [alpha2,cost_aux] = group_pooling_graph( D, T, Pn, param,t0);

%	T0=param.initT;
%	clear T0;
%	T0{1} = [1:size(D,2)]';
%	D0=param.initD;
%	[indexes,indexes_inv] = getTreeIndexes(size(D,2),size(Pmix,2),T0,param.time_groupsize, param.Jmax);
%	param.indexes = indexes;
%	param.indexes_inv = indexes_inv;
%        t0 = getoptions(param,'alpha_step',0.25);
%        t0 = t0 * (1/max(svd(D0))^2);
%
%    [ialpha_d1, ialpha_d2,cost_aux] = group_pooling_graph_demix( D0, T0, Pmix, param,t0);
%    [ialpha1,cost_aux] = group_pooling_graph( D0, T0, Px, param,t0);
%    [ialpha2,cost_aux] = group_pooling_graph( D0, T0, Pn, param,t0);
%
%


   % keyboard;
   % chunk=X(:,1:2000);
   %[alphaX] = group_pooling_graph( D, T, chunk, param,t0);

   S_t = estimate_similarity_from_trees(T);

if 0
for t=1:size(alpha,2)
t
Ibis = find(alpha(:,t)>0);
Sbis = S_t .* sqrt(alpha(:,t)*(alpha(:,t)'));
Sbis = Sbis(Ibis,Ibis);
Dbis = diag(sum(Sbis).^(-1/2));
Lbis = eye(size(Sbis,1)) - Dbis * Sbis * Dbis;
[Vbis,ev_t]=eig(Lbis);
Vtmp = Vbis(:,1:2);
mask(Ibis,t) =  kmeans(Vtmp,2,'start','sample','maxiter',200,'replicates',4,'EmptyAction','singleton');
end

%[rec1, rec2] = spectralcluster(alpha, T);


figure
subplot(231);imagesc(Pmix);
subplot(232);imagesc(Px);
subplot(233);imagesc(Pn);
subplot(234);imagesc(alpha);
subplot(235);imagesc(alpha1);
subplot(236);imagesc(alpha2);

figure
subplot(231);imagesc(Pmix);
subplot(232);imagesc(Px);
subplot(233);imagesc(Pn);
subplot(234);imagesc(ialpha);
subplot(235);imagesc(ialpha1);
subplot(236);imagesc(ialpha2);

if 0
figure
subplot(131);imagesc(Vmix);
subplot(132);imagesc(Vx);
subplot(133);imagesc(Vn);
end
   
end


%%%%%simple experiments with exemplar based

P2=Pmix;
P2(:,2:end)=Pmix(:,1:end-1);
P3=Pmix;
P3(:,1:end-1)=Pmix(:,2:end);
Pj = [Pmix ; P2; P3];
clear P2; 
clear P3;


X2=X;
X2(:,2:end)=X(:,1:end-1);
X3=X;
X3(:,1:end-1)=X(:,2:end);
Xj=[X ; X2; X3];
Xn = mexNormalize(Xj);
clear Xj
clear X2
clear X3

param.lambda=0.5;
param.pos=1;
param.posD=1;
param.posAlpha=1;

brutus = mexLasso(Pj, Xn, param);
[coefis,posis] = sort(brutus, 1, 'descend');
posis=posis(1:2,:);


for p=1:size(posis,2)
%ref(:,1) = X(:,posis(1,p));
%ref(:,2) = X(:,posis(2,p));
%alphas=pinv(ref)*Pmix(:,p);
%Prec1(:,p)= alphas(1)*ref(:,1);
%Prec2(:,p)= alphas(2)*ref(:,2);
Prec1(:,p) = coefis(1,p)*X(:,posis(1,p));
Prec2(:,p) = coefis(2,p)*X(:,posis(2,p));

end






