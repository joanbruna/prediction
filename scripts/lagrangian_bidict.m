

if ~exist('X1','var')
    load '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s31.mat'
    X1 = Xc;
    clear Xc;
    
    tt=size(X1,2);

    
    load '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s14.mat'
    X2 = Xc;
    
    X = [X1 X2];
    clear Xc;
    clear X1; 
    clear X2;   
 
end

%renormalize data: whiten each frequency component.
eps=4e-1;
stds = std(X,0,2) + eps;
X = X./repmat(stds,1,size(X,2));
norms = sqrt(sum(X.^2)) + 1;
%avenorm = mean(sqrt(sum(X.^2)));
%X = X/avenorm;
X = X./repmat(norms,size(X,1),1);

%%%%Plain NMF%%%%%%%
param0.K = 192;
param0.posAlpha = 1;
param0.posD = 1;
param0.pos = 1;
param0.lambda = 0.1;
param0.iter = 1000;
param0.numThreads=16;
param0.batchsize=512;

Dnmf1 = mexTrainDL(X(:,1:tt),param0);
Dnmf2 = mexTrainDL(X(:,tt+1:end),param0);


alpha1= mexLasso(X(:,1:tt),Dnmf1,param0);
alpha2= mexLasso(X(:,tt+1:end),Dnmf2,param0);

Dnmf1s = sortDZ(Dnmf1,full(alpha1)');
Dnmf2s = sortDZ(Dnmf2,full(alpha2)');

gpud=gpuDevice(2);

param.nmf=1;
param.lambda=2e-2;
param.beta=1e-2;
param.overlapping=1;
param.groupsize=2;
param.time_groupsize=2;
param.nu=0.5;
param.lambdagn=0;
param.betagn=0;
param.itersout=200;
param.K=192;
param.Kgn=64;
param.epochs=4;
param.batchsize=4096;
param.plotstuff=1;

reset(gpud);

param.initD = Dnmf1s;
[D1, Dgn1] = twolevelDL_gpu(X(:,1:tt), param);

reset(gpud);

%[Z1, Zgn1] = twolevellasso_gpu(X(:,1:tt), D1, Dgn1, param);



param.initD = Dnmf2s;
[D2, Dgn2] = twolevelDL_gpu(X(:,tt+1:end), param);

reset(gpud);

%[Z2, Zgn2] = twolevellasso_gpu(X(:,tt+1:end), D2, Dgn2, param);
end


%evaluate on a mix
dir1 = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/s31/';
dir2 = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/s14/';
dd1=dir(dir1);
dd2=dir(dir2);
Ntest=10;
I1 = randperm(size(dd1,1)-2);
I2 = randperm(size(dd2,1)-2);
for n=1:Ntest
%pick two random examples
s1 =fullfile(dir1,dd1(2+I1(n)).name);
s2 =fullfile(dir2,dd2(2+I2(n)).name);
params_aux = audio_config();
fs = params_aux.fs;
NFFT = params_aux.NFFT;
hop = params_aux.hop;
[x1,Fs] = audioread(s1);
x1 = resample(x1,fs,Fs);
x1 = x1(:);
[x2,Fs] = audioread(s2);
x2 = resample(x2,fs,Fs);
x2 = x2(:);
% adjust the size
m = min(length(x1),length(x2));
x1 = x1(1:m);
x2 = x2(1:m);
mix = x1+x2;

Smix = compute_spectrum(mix,NFFT, hop);
Vmix = abs(Smix);
Vmix = Vmix./repmat(stds,1,size(Vmix,2));
norms = sqrt(sum(Vmix.^2)) + 1;
Vmix = Vmix./repmat(norms,size(Vmix,1),1);

X1= compute_spectrum(x1,NFFT,hop);
V1 = abs(X1);
V1 = V1./repmat(stds,1,size(V1,2));
norms = sqrt(sum(V1.^2)) + 1;
V1 = V1./repmat(norms,size(V1,1),1);
X2= compute_spectrum(x2,NFFT,hop);
V2 = abs(X2);
V2 = V2./repmat(stds,1,size(V2,2));
norms = sqrt(sum(V2.^2)) + 1;
V2 = V2./repmat(norms,size(V2,1),1);

reset(gpud);
if isfield(param,'Z1in')
param=rmfield(param,'Z1in');
param=rmfield(param,'Z2in');
end
oldnu = param.nu;
param.nu=0;
param.alpha_step=1;
param.gradient_descent=0;
param.itersout=800;
[Z1dm, Z1gn1dm, Z2dm, Zgn2dm] = twolevellasso_gpu_demix(Vmix, D1, Dgn1, D2, Dgn2, param);
param.Z1in=Z1dm;
param.Z2in=Z2dm;
reset(gpud);
param.nu=oldnu;
param.gradient_descent=0;
param.alpha_step=0.5;
[Z1dm, Z1gn1dm, Z2dm, Zgn2dm] = twolevellasso_gpu_demix(Vmix, D1, Dgn1, D2, Dgn2, param);

R = {};
R{1} = D1* Z1dm;
R{2} = D2* Z2dm;
Ro = {};
Ro{1} = D1* param.Z1in;
Ro{2} = D2* param.Z2in;
Smix = Smix(:,1:size(R{1},2));

y_out = wienerFilter2(R,Smix);
y_outo = wienerFilter2(Ro,Smix);
m = length(y_out{1});

Parms{n} =  BSS_EVAL(x1(1:m), x2(1:m), y_out{1}, y_out{2}, mix(1:m));
Parmso{n} =  BSS_EVAL(x1(1:m), x2(1:m), y_outo{1}, y_outo{2}, mix(1:m));

Smix = compute_spectrum(mix,NFFT, hop);
Vmix = abs(Smix);
Vmix = Vmix./repmat(stds,1,size(Vmix,2));
norms = sqrt(sum(Vmix.^2)) + 1;
Vmix = Vmix./repmat(norms,size(Vmix,1),1);
alphas = mexLasso(Vmix,[Dnmf1, Dnmf2],param0);
Rnmf={};
Rnmf{1} = Dnmf1 * alphas(1:size(Dnmf1,2),:);
Rnmf{2} = Dnmf2 * alphas(1+size(Dnmf1,2):end,:);
y_outnmf = wienerFilter2(Rnmf,Smix);
Parmsnmf{n} =  BSS_EVAL(x1(1:m), x2(1:m), y_outnmf{1}, y_outnmf{2}, mix(1:m));

fprintf('done ex %d \n', n)

end

for n=1:Ntest
SDR_pool(n)=Parms{n}.SDR;
SDR_nmf(n)=Parmsnmf{n}.SDR;
NSDR_pool(n)=Parms{n}.NSDR;
NSDR_nmf(n)=Parmsnmf{n}.NSDR;
SAR_pool(n)=Parms{n}.SAR;
SAR_nmf(n)=Parmsnmf{n}.SAR;
end
[Z, Zgn] = twolevellasso_gpu(X, D, Dgn, param);




%figure;
%subplot(411);imagesc(V1);
%subplot(412);imagesc(Rnmf{1});
%subplot(413);imagesc(Ro{1});
%subplot(414);imagesc(R{1});


