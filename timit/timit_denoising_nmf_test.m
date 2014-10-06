
% train model

representation = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/spect_fs16_NFFT1024_hop512/TRAIN/';



load(sprintf('%sfemale',representation));
data1 = data;
clear data

% epsilon = 1;
param.epsilon = 0;
epsilon = param.epsilon;
data1.X = softNormalize(abs(data1.X),epsilon);


param.K = 200;
param.posAlpha = 1;
param.posD = 1;
param.pos = 1;
param.lambda = 0.1;
param.lambda2 = 0;
param.iter = 1000;


D = mexTrainDL(abs(data1.X), param);


%%

noise_files = '/misc/vlgscratch3/LecunGroup/pablo/noise_texture/noise_texture_audios.mat';
if ~exist('noise','var')
load(noise_files)
end

if ~exist('test_female','var')
load /misc/vlgscratch3/LecunGroup/pablo/TIMIT/TEST/female_audios_short.mat
end

%%

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

%LL = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5];
%SS = [0 5 10];

LL = [0.1];
SS = [0];


for m=1:length(SS)
    

     SNR_dB = SS(m);

    
    
for k=1:length(LL)
LL(k)
    param.lambda= LL(k);
    
for j=1:length(noise)
for i=1:size(idf,1)
    
    x1 = test_female{idf(i,1)}{idf(i,2)}{idf(i,3)}.x;

    x2 = noise{j}.x;
    
    T = min(length(x1),length(x2));

    x1 = x1(1:T)/norm(x1);
    x2 = x2(1:T);
    
    n1 = sqrt(mean(power(x1,2)));
    n2 = sqrt(mean(power(x2,2)));
    
    x2 = x2/norm(x2)*power(10,(-SNR_dB)/20);
    mix = x1 + x2;
    
    ss = snr(x1,x2);

    %mix = (x1/norm(x1)+x2/norm(x2));
    
    X = compute_spectrum(mix,NFFT,hop);
    X1 = compute_spectrum(x1,NFFT,hop);
    X2 = compute_spectrum(x2,NFFT,hop);

    Xn = softNormalize(abs(X),param.epsilon);
    
    % compute decomposition
    nparam = param;
    nparam.Kn=2; %
    nparam.iter=100;
    nparam.niter=10;
    nparam.pos=1;
    nparam.tau = 0;
 %   nparam.lambda_ast = 1;
    nparam.verbose = 1;
    
    %Pmix = Vmix ./ repmat(sqrt(epsilon^2+sum(Vmix.^2)),size(Vmix,1),1) ;
%     [H,Wn] = nmf_beta(Xn,D,nparam);
%     K = size(D,2);
%     W1H1 = D*H(1:K,:);
%     W2H2 = Wn*H((K+1):end,:);
    
    [W1H1,W2H2] = denoising_nmf(Xn,D,nparam);
%    K = size(D,2);
    %W1H1 = D*Hs;
    %W2H2 = Wn*Hn;
    
%    norm(W2H2)
    
    eps_1 = 1e-6;%eps_1=0;
    V_ap = W1H1.^2 +W2H2.^2 + eps_1;
    
    % wiener filter
    
    SPEECH1 = ((W1H1.^2)./V_ap).*X;
    SPEECH2 = ((W2H2.^2)./V_ap).*X;
    speech1 = invert_spectrum(SPEECH1,NFFT,hop,T);
    speech2 = invert_spectrum(SPEECH2,NFFT,hop,T);


    Parms =  BSS_EVAL(x1', x2', speech1', speech2', mix');

    Parms
    
    Parms2.snr_in = snr(x1,x2);
    Parms2.snr_out = snr(x1,x1-speech1);
  %  output{j,i} = Parms;
    keyboard
    
end
end

AA{m,k} = output;

end
end


%save nmf_runs AA