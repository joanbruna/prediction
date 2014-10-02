
noise_files = '/misc/vlgscratch3/LecunGroup/pablo/noise_texture/noise_texture_audios.mat';
if ~exist('noise','var')
load(noise_files)
end

if ~exist('data1','var')
load /misc/vlgscratch3/LecunGroup/pablo/TIMIT/TEST/female_audios_short.mat
end

idf = [1,1,1;...
    1,1,6;...
    1,2,9;...
    1,2,8;...
    1,4,10;...
    1,4,2;...
    2,2,8;...
    2,2,3;...
    2,7,4;...
    2,7,8;...
    2,8,3;...
    2,8,5];

fs = 16000;
NFFT = 1024;
hop = NFFT/2;

% load models
%load timit/results_pooling_5.mat
load models_1

D = AA{4}.D;
Dgn = AA{4}.Dgn;
param = AA{4}.param;
param.lambda = param.lambda*2;
param.beta = param.lambda*2;
param.nu = param.lambda*2;


% D = D1;
% Dgn = D1gn;

representation = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/spect_fs16_NFFT1024_hop512/TRAIN/';
load(sprintf('%sfemale',representation));
data1 = data;
clear data

% epsilon = 1;
epsilon = param.epsilon;
data1.X = softNormalize(abs(data1.X),epsilon);


param_nmf.K = 200;
param_nmf.posAlpha = 1;
param_nmf.posD = 1;
param_nmf.pos = 1;
param_nmf.lambda = 0.1;
param_nmf.lambda2 = 0;
param_nmf.iter = 1000;


D_nmf = mexTrainDL(abs(data1.X), param_nmf);



for j=1:length(noise)
for i=1:size(idf,1)
    
    %%
    
    x1 = test_female{idf(i,1)}{idf(i,2)}{idf(i,3)}.x;
    
    x2 = noise{1}.x;
    
    T = min(length(x1),length(x2));
    
    x1 = x1(1:T);
    x2 = x2(1:T);
    
    x11 = x1;
    x22 = x2;
    
    x1 = x1/norm(x1);
    x2 = x2/norm(x2);
    
    mix = (x1+x2);
    
    X = compute_spectrum(mix,NFFT,hop);
    Xn = softNormalize(abs(X),param.epsilon);
    
    % compute decomposition
    param.semisup = 1;
    param.Kn = 5;
    param.tau = 0;
    [Z1dm,Z2dm,W] = denoising_twolevel(Xn,D,Dgn, param);
    
    W1H1 = D*Z1dm;
    W2H2 = W*Z2dm;
    
    eps_1 = 1e-6;%eps_1=0;
    V_ap = W1H1.^2 +W2H2.^2 + eps_1;
    
    % wiener filter
    
    SPEECH1 = ((W1H1.^2)./V_ap).*X;
    SPEECH2 = ((W2H2.^2)./V_ap).*X;
    speech1 = invert_spectrum(SPEECH1,NFFT,hop,T);
    speech2 = invert_spectrum(SPEECH2,NFFT,hop,T);
    
    Parms =  BSS_EVAL(x1', x2', speech1', speech2', mix');
    
    Parms
    output{j,i} = Parms;
    
    % NMF
    % compute decomposition
    nparam = param_nmf;
    nparam.Kn=5; %
    nparam.iter=1000;
    nparam.pos=1;
    nparam.verbose = 1;
    nparam.lambda_ast = 0.1;
    
    %Pmix = Vmix ./ repmat(sqrt(epsilon^2+sum(Vmix.^2)),size(Vmix,1),1) ;
    [H,Wn] = nmf_beta(Xn,D_nmf,nparam);
    K_nmf = size(D_nmf,2);
    
    W1H1 = D_nmf*H(1:K_nmf,:);
    W2H2 = Wn*H((K_nmf+1):end,:);
    
    
    eps_1 = 1e-6;%eps_1=0;
    V_ap = W1H1.^2 +W2H2.^2 + eps_1;
    
    % wiener filter
    
    SPEECH1 = ((W1H1.^2)./V_ap).*X;
    SPEECH2 = ((W2H2.^2)./V_ap).*X;
    speech1 = invert_spectrum(SPEECH1,NFFT,hop,T);
    speech2 = invert_spectrum(SPEECH2,NFFT,hop,T);
    
    Parms2 =  BSS_EVAL(x1', x2', speech1', speech2', mix');
    Parms2
    
    output2{j,i} = Parms2;

end
end
