addpath ./stft/

tol = 1e-3;
n_iter_max = 1000;
beta = 1;

l_win = 1024;
overlap = l_win/2;
Fs = 16000;

lambda_ast = 0.2;
lambda = 0.01;


K = 100;

param0.K = K;
param0.posAlpha = 1;
param0.posD = 1;
param0.pos = 1;
param0.lambda = lambda;
param0.lambda2 = lambda_ast;
param0.iter = 1000;



p = 1;

folderv = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/Data_with_dev/';
train_file1 = 'female_train.wav';
train_file2 = 'male_train.wav';

test_file1 = 'female_dev.wav';
test_file2 = 'male_dev.wav';

%% Data
%F = 50;
%N = 100;
%V = abs(randn(F,N));


[x, fs] = audioread([folderv train_file1]);
x = resample(x,Fs,fs);
fs = Fs;
x = x'/norm(x); T = length(x);

Xt1 = cf_stft(x,l_win,overlap);
V1 = abs(Xt1).^p;
[F,N] = size(V1);

%W1 = rnmf(V1, beta, n_iter_max, tol, W_ini, H_ini, E_ini, lambda_ast,lambda,1);
W1 = mexTrainDL(V1, param0);



%% Model 2

[x, fs] = audioread([folderv train_file2]);
x = resample(x,Fs,fs);
x = x'/norm(x); T = length(x);

Xt2 = cf_stft(x,l_win,overlap);
V2 = abs(Xt2).^p;
[F,N] = size(V2);


%W2 = rnmf(V2, beta, n_iter_max, tol, W_ini, H_ini, E_ini, lambda_ast,lambda,1);
W2 = mexTrainDL(V2, param0);


%% Algo

[x, fs] = audioread([folderv test_file1]);
x = resample(x,Fs,fs);
x1 = x'; T1 = length(x1);

[x, fs] = audioread([folderv test_file2]);
x = resample(x,Fs,fs);
fs = Fs;
x2 = x'; T2 = length(x2);

T = min(T1,T2);

x1 = x1(1:T);
x2 = x2(1:T);

x1 = x1/norm(x1); 
x2 = x2/norm(x2);

X1 = cf_stft(x,l_win,overlap);
V1 = abs(X1).^p;
X2 = cf_stft(x2,l_win,overlap);
V2 = abs(X2).^p;

mix = x1+x2;

X = cf_stft(mix,l_win,overlap);
V = abs(X).^p;
[F,N] = size(V);

H =  full(mexLasso(V,[W1,W2],param0));

W1H1 = W1*H(1:K,:);
W2H2 = W2*H(K+1:end,:);

%% Display results
eps = 1e-6;
V_ap = W1H1 +W2H2 + eps;
im1 = log10(V+eps);
im2 = log10(V_ap+eps);
im3 = log10(W1H1+eps);
im4 = log10(W2H2+eps);

% Unify colormaps
ma = max([im1(:); im2(:); im3(:); im4(:)]);
mi  = min([im1(:); im2(:); im3(:); im4(:)]);
im1 = (im1 - mi)/(ma - mi) * 64;
im2 = (im2 - mi)/(ma - mi) * 64;
im3 = (im3 - mi)/(ma - mi) * 64;
im4 = (im4 - mi)/(ma - mi) * 64;

figure(2);
subplot(221); 
image(im1); 
title('V'); axis xy; 

subplot(222); 
image(im2); 
title('WH '); axis xy; 

subplot(223); 
image(im3); 
title('W1H1'); axis xy; 

subplot(224); 
image(im4); 
title('W2H2'); axis xy;
drawnow

%% Reconstruct sources
SPEECH1 = ((W1H1)./V_ap).*X;
SPEECH2 = ((W2H2)./V_ap).*X;
speech1 = cf_istft(SPEECH1,l_win,overlap);
speech1 = speech1(overlap+1:overlap+T);
speech2 = cf_istft(SPEECH2,l_win,overlap);
speech2 = speech2(overlap+1:overlap+T);


%x1 = x1(overlap+1:overlap+T);
%x2 = x2(overlap+1:overlap+T);

Parms =  BSS_EVAL(x1', x2', speech1', speech2', mix');

Parms



%% Write audio

audiowrite('../../public_html/speech/mix.wav',mix,fs);
unix(['chmod 777 * ../../public_html/speech/mix.wav']);

audiowrite('../../public_html/speech/speech1.wav',speech1,fs);
unix(['chmod 777 * ../../public_html/speech/speech1.wav']);
audiowrite('../../public_html/speech/speech2.wav',speech2,fs);
unix(['chmod 777 * ../../public_html/speech/speech2.wav']);




