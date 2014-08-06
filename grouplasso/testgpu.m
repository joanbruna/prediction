clear all
close all

defo=gpuDevice;
reset(defo);
%gpuDevice(1);


N=256;
K=512;
L=2^19;
chsz=2^16;

nch=L/chsz;

X=randn(N,L);
D=randn(N,K);
X=single(X);
D=single(D);

options.lambda = 0.1;
options.iters=200;

aviam=zeros(K,L);

for n=1:nch

[~,~,aviam(:,1+(n-1)*chsz:n*chsz)]=time_coeffs_update22(D, X(:,1+(n-1)*chsz:n*chsz), options);
fprintf('done %d examples \n', n*chsz)

end



