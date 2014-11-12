
if ~exist('D1','var')
representation = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/spect_fs16_NFFT1024_hop512/TRAIN/';


load(sprintf('%sfemale',representation));
data1 = data;
clear data

load(sprintf('%smale',representation));
data2 = data;
clear data


% epsilon = 1;
epsilon = 0.0001;
data1.X = softNormalize(abs(data1.X),epsilon);
data2.X = softNormalize(abs(data2.X),epsilon);


clear param
param.K = 200;
param.posAlpha = 1;
param.posD = 1;
param.pos = 1;
param.lambda = 0.1;
param.lambda2 = 0;
param.iter = 500;


D1 = mexTrainDL(abs(data1.X), param);

D2 = mexTrainDL(abs(data2.X), param);


end

%%

if ~exist('test_male','var')
load /misc/vlgscratch3/LecunGroup/pablo/TIMIT/TEST/male_audios_short.mat
end

if ~exist('test_female','var')
load /misc/vlgscratch3/LecunGroup/pablo/TIMIT/TEST/female_audios_short.mat
end

%%

fs = data1.fs;
NFFT = data1.NFFT;
hop = data1.hop;

lambda = param.lambda;

x1 = test_female{2}{3}{1}.x;
x2 = test_male{2}{1}{9}.x;

T = min(length(x1),length(x2));

x1 = x1(1:T)/norm(x1);
x2 = x2(1:T)/norm(x2);
mix = x1 + x2;

X = compute_spectrum(mix,NFFT,hop);
X1_orig = compute_spectrum(x1,NFFT,hop);
X2_orig = compute_spectrum(x2,NFFT,hop);
%X = softNormalize(X,epsilon);

Xn = abs(X);

[W1H1, W2H2,Z1_init,Z2_init] = nmf_demix(Xn,D1,D2,param);

eps_1 = 1e-6;


SPEECH1 = W1H1.*exp(1i*angle(X));
SPEECH2 = W2H2.*exp(1i*angle(X));

speech1 = invert_spectrum(SPEECH1,NFFT,hop,T);
speech2 = invert_spectrum(SPEECH2,NFFT,hop,T);

Parms_no_mask =  BSS_EVAL(x1', x2', speech1', speech2', mix');



%%
V_ap = W1H1.^2 +W2H2.^2 + eps_1;

SPEECH1 = ((W1H1.^2)./V_ap).*X;
SPEECH2 = ((W2H2.^2)./V_ap).*X;

speech1 = invert_spectrum(SPEECH1,NFFT,hop,T);
speech2 = invert_spectrum(SPEECH2,NFFT,hop,T);

Parms_init =  BSS_EVAL(x1', x2', speech1', speech2', mix');

X1_init = compute_spectrum(speech1,NFFT,hop);
X2_init = compute_spectrum(speech2,NFFT,hop);


%%

Z1 = Z1_init;
Z2 = Z2_init;

fprop    = @(Xn) abs(Xn);
bprop    = @(X,DZ1,DZ2,X1) stft_bprop(X,DZ1,DZ2,X1,1,1);

obj_init = measure_cost(D1,D2,Z1_init,Z2_init,X1_init,X2_init,fprop,lambda);


%% GRADIENT DESCENT --------

X1 = X1_init;
X2 = X2_init;

Z1 = Z1_init;
Z2 = Z2_init;

rho = 0.01;
count = 1;
min_ = 10000;



for j=1:200
    
    % gradient step on X1
    dfX1 = bprop(X,D1*Z1,D2*Z2,X1);
    X1_aux = X1 - rho*dfX1;
    
    % enforce signal to be audio
    At = invert_spectrum(X1_aux,NFFT,hop);
    X1 = compute_spectrum(At,NFFT,hop);
    
    obj(count) = measure_cost(D1,D2,Z1,Z2,X1,X-X1,fprop,lambda,1,X1);
    
    % minimize over Z1 and Z2
    Z1 = mexLasso(fprop(X1),D1,param);  
    Z2 = mexLasso(fprop(X-X1),D2,param);
    
    count = count +1;
    obj(count) = measure_cost(D1,D2,Z1,Z2,X1,X-X1,fprop,lambda,10);
    
    % save best result
    if min_ > obj(count)
        X1_end = X1;
        X2_end = X- X1;
        Z1_end = Z1;
        Z2_end = Z2;
        min_ = obj(count);
        disp(obj(count))
    end
    
    count = count +1;

end

%%

% X1_end = X1;
% X2_end = X- X1_end;
% 
% Z1_end = Z1;
% Z2_end = Z2;

obj_end = measure_cost(D1,D2,Z1_end,Z2_end,X1_end,X2_end,fprop,lambda);



%%

speech1_ = invert_spectrum(X1_end,NFFT,hop,T);
speech2_ = invert_spectrum(X-X1_end,NFFT,hop,T);


Parms_ =  BSS_EVAL(x1', x2', speech1_', speech2_', mix');

X1_ = compute_spectrum(speech1_,NFFT,hop);
X2_ = compute_spectrum(speech2_,NFFT,hop);

obj_ = measure_cost(D1,D2,Z1,Z2,X1_,X2_,fprop,lambda);

%%

W1H1 = abs(X1_end);
W2H2 = abs(X2_end);
V_ap = W1H1.^2 +W2H2.^2 + eps_1;

SPEECH1g = ((W1H1.^2)./V_ap).*X;
SPEECH2g = ((W2H2.^2)./V_ap).*X;

speech1g = invert_spectrum(SPEECH1g,NFFT,hop,T);
speech2g = invert_spectrum(SPEECH2g,NFFT,hop,T);

disp('a')
Parmsg =  BSS_EVAL(x1', x2', speech1g', speech2g', mix');


%%


W1H1 = abs(X1_end);
W2H2 = abs(X2_end);
V_ap = W1H1.^2 +W2H2.^2 + eps_1;

SPEECH1g = ((W1H1.^2)./V_ap).*X;
SPEECH2g = ((W2H2.^2)./V_ap).*X;

speech1g = invert_spectrum(SPEECH1g,NFFT,hop,T);
speech2g = invert_spectrum(SPEECH2g,NFFT,hop,T);

disp('a')
Parmsg =  BSS_EVAL(x1', x2', speech1g', speech2g', mix');


%%
W1H1 = D1*Z1;
W2H2 = D2*Z2;
V_ap = W1H1.^2 +W2H2.^2 + eps_1;

SPEECH1f = ((W1H1.^2)./V_ap).*X;
SPEECH2f = ((W2H2.^2)./V_ap).*X;

speech1f = invert_spectrum(SPEECH1f,NFFT,hop,T);
speech2f = invert_spectrum(SPEECH2f,NFFT,hop,T);

Parmsf =  BSS_EVAL(x1', x2', speech1f', speech2f', mix');



%%

 
Z1_opt = mexLasso(fprop(X1_orig),D1,param);
Z2_opt = mexLasso(fprop(X2_orig),D2,param);

obj_opt = measure_cost(D1,D2,Z1_opt,Z2_opt,X1_orig,X2_orig,fprop,lambda);




