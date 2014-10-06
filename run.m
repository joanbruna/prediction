

W1H1 = abs(X1_end);
W2H2 = abs(X2_end);
V_ap = W1H1.^2 +W2H2.^2 +  1e-10;

SPEECH1f = ((W1H1.^2)./V_ap).*X;
SPEECH2f = ((W2H2.^2)./V_ap).*X;

speech1f = invert_spectrum(SPEECH1f,NFFT,hop,T);
speech2f = invert_spectrum(SPEECH2f,NFFT,hop,T);

Parmsf2 =  BSS_EVAL(x1', x2', speech1f', speech2f', mix');

break


%% in spectrum

NFFT = 1024;
hop = 512;

nf = 30;
% 
% epsilon = 0.001;
% 
% x1 = randn(1,nf*NFFT);
% x = randn(1,nf*NFFT);
% X1 = compute_spectrum(x1,NFFT,hop);
% %X1n = softNormalize(X1,epsilon);
% X = compute_spectrum(x,NFFT,hop);
% %Xn = softNormalize(X,epsilon);


lambda = 1;
alpha = 1;

X1 = randn(NFFT,nf) + 1i*randn(NFFT,nf);
X = randn(NFFT,nf) + 1i*randn(NFFT,nf);

P1 = abs(X1);
P2 = abs(X-X1);

Dz1 = rand(size(P1)) + 0.001;
Dz2 = rand(size(P1)) + 0.001;


f = alpha*0.5*norm(P1-Dz1,'fro')^2 + lambda*0.5*norm( P2 - Dz2 , 'fro').^2;

eps = 1e-8;
dX1 = eps*(randn(size(X1))+1i*randn(size(X1)));%exp(2i*pi*rand(size(X1n)));
X1_ = X1 + dX1;

P1_ = abs(X1_);
P2_ = abs(X-X1_);

f_ = alpha*0.5*norm(P1_-Dz1,'fro')^2 + lambda*0.5*norm( P2_ - Dz2 , 'fro').^2;

dfX1 = stft_bprop(X,Dz1,Dz2,X1,alpha,lambda);
[f_-f,real(dfX1(:)'*dX1(:))]



break

%% in time

alpha = 1;
lambda = 1;
x1 = randn(1024,1);
x = randn(size(x1));

Dz = rand(size(x1));


P = abs(fft(x1));

f = alpha*0.5*norm(x-x1,'fro')^2 + lambda*0.5*norm( P - Dz , 'fro').^2;


eps = 1e-8;
dx1 = eps*randn(size(x1));
x1_ = x1 + dx1;

P_ = abs(fft(x1_));
f_ = alpha*0.5*norm(x-x1_,'fro')^2 + lambda*0.5*norm( P_ - Dz , 'fro').^2;


dfx1 = stft_bprop(x,Dz,x1,alpha,lambda);

[f_-f,dfx1(:)'*dx1(:)]







