function dP = stft_bprop(X,Dz1,Dz2,X1,alpha,lambda)


%NFFT = length(x1);
eps = 1e-10;

P1 = abs(X1+eps);
P2 = abs(X-X1+eps);

G1 = P1 - Dz1;
G2 = P2 - Dz2;


C1 = G1.*X1./P1;
C2 = G2.*(X1-X)./P2;
dP = alpha*C1 + lambda*C2;
