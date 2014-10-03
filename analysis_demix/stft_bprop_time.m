function dP = stft_bprop_time(x,Dz,x1,alpha,lambda)


NFFT = length(x1);

P = abs(fft(x1));

G2 = P - Dz;
G1 = x1 - x;

C = G2.*fft(x1)./P;
dP = alpha*G1 + lambda*NFFT*real(ifft(C));
