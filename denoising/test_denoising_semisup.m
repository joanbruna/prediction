

addpath nmf_linear_dynamics/
addpath bss_eval_3/
addpath spect/
addpath ../spams-matlab/build/
addpath utils
addpath denoising/

%%

% Load data for single speaker

%load class_s4.mat
if 1
    % use single speaker for training
    load ../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s4.mat
    X = Xc;
    clear Xc;
else
    % use joint training set
    if ~exist('X','var')
        load ../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/joint.mat
    end
end

%epsilon = 0.1;
%X = X ./ repmat(sqrt(epsilon^2+sum(X.^2)),size(X,1),1) ;
X = mexNormalize(X);


%%

% Train dictionary for single speaker


param.K=100; % learns a dictionary with 100 elements 
param.lambda=0.1; 
%param.numThreads=12;	%	number	of	threads 
param.batchsize =1000;
param.iter=200; % let us see what happens after 1000 iterations .
param.posD=1;
param.posAlpha=1;
param.pos=1;


D=mexTrainDL(X, param);


%keyboard

% a=mexLasso(X,D, param);

%%

% Train linear temporal dynamics


% ld_param = struct;
% %param.D = Dini;
% ld_param.K = 500;
% ld_param.lambda = 0.1;
% ld_param.mu = 10;
% ld_param.epochs = 1;
% ld_param.batchsize = 100;
% ld_param.renorm_input = 0;
% 
% 
% 
% [D_ld,A_ld] = nmf_linear_dynamic(X, ld_param);
% 
% keyboard

%% 

speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s4/lrak4s.wav'; % same as training
%speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s18/sram2s.wav'; % different woman;
%speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s1/lrbr4n.wav';% man

% Noise
noise = '../../../../misc/vlgscratch3/LecunGroup/bruna/noise_data/train/noise_sample_08.wav'; % easy
%noise = '../../../../misc/vlgscratch3/LecunGroup/bruna/noise_data/babble/noise_sample_08.wav'; % hard


params = audio_config();

SNR_dB = 0;

[x,Fs] = audioread(speech);
x = resample(x,params.fs,Fs);
x = x(:);


[n,Fs] = audioread(noise);
n = resample(n,params.fs,Fs);
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

Smix = compute_spectrum(mix,params.NFFT, params.hop);
Vmix = abs(Smix);


[N,K] = size(D);
Pmix = mexNormalize(Vmix);

%% 

nparam = struct;
rep = 10;
rates = zeros(rep,3);
obj = zeros(rep,1);


for i=1:rep

nparam.Kn=2; %
nparam.iter=1000; 
nparam.pos=1;
nparam.lambda = param.lambda;
nparam.verbose = 0;



%Pmix = Vmix ./ repmat(sqrt(epsilon^2+sum(Vmix.^2)),size(Vmix,1),1) ;
[H,Wn,obj(i)] = nmf_beta(Pmix,D,nparam);


Hs = H(1:K,:);
Hn = H((K+1):end,:);


R = {};
R{1} = D* Hs;
R{2} = Wn* Hn;

y_out = wienerFilter2(R,Smix);



[SDR,SIR,SAR,perm] = bss_eval_sources( [y_out{1}(1:m);y_out{2}(1:m)],[x,n]');


rates(i,:) = [SDR(1) SIR(1) SAR(1)];
obj2(i) = compute_obj(Pmix,[Hs;Hn],D,Wn,nparam);

end

disp(mean(rates))
disp(max(rates))
disp(min(rates))
rates2 = rates;

%%

rep = 10;
rates = zeros(rep,3);
obj = zeros(rep,1);

nparam = param;

nparam.Kn=2; %
nparam.iter=50; 
nparam.pos=1;

for i=1:rep


%Pmix = mexNormalize(Vmix);
%Pmix = Vmix ./ repmat(sqrt(epsilon^2+sum(Vmix.^2)),size(Vmix,1),1) ;
%[H,Wn,obj(i)] = nmf_beta(Pmix,D,nparam);
[Hs,Hn,Wn] = denoising_nmf(Pmix,D,nparam);


R = {};
R{1} = D* Hs;
R{2} = Wn* Hn;

y_out = wienerFilter2(R,Smix);


% m = length(y_out{1});
% x2 = x;
% n2 = n(1:m);

[SDR,SIR,SAR,perm] = bss_eval_sources( [y_out{1}(1:m);y_out{2}(1:m)],[x,n]');

if isnan(SDR(1))
    keyboard
end

rates(i,:) = [SDR(1) SIR(1) SAR(1)];

obj(i) = compute_obj(Pmix,[Hs;Hn],D,Wn,nparam);
i
end

% [disp(mean(rates)),disp(max(rates)),disp(min(rates))]

break
%%

ld_nparam = ld_param;

rep = 10;
rates = zeros(rep,3);
obj = zeros(rep,1);
ld_nparam.iter = 100;

ld_nparam.Kn = 2;

for i=1:rep


Pmix = mexNormalize(Vmix);
%Pmix = Vmix ./ repmat(sqrt(epsilon^2+sum(Vmix.^2)),size(Vmix,1),1) ;
%[H,Wn,obj(i)] = nmf_beta(Pmix,D,nparam);
[Hs,Hn,Wn] = denoising_nmf(Pmix,D_ld,ld_nparam,A_ld);


R = {};
R{1} = D_ld* Hs;
R{2} = Wn* Hn;

y_out = wienerFilter2(R,Smix);


m = length(y_out{1});
x2 = x(1:m);
n2 = n(1:m);

[SDR,SIR,SAR,perm] = bss_eval_sources( [y_out{1},y_out{2}]',[x2,n2]');

if isnan(SDR(1))
    keyboard
end

rates(i,:) = [SDR(1) SIR(1) SAR(1)];

obj(i) = compute_obj(Pmix,[Hs;Hn],D_ld,Wn,ld_nparam);

end



