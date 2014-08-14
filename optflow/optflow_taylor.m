function theta = optflow_taylor(z, options)
%this computes optical flow using simple taylor expansion

[N, L] = size(z);
h=zeros(N,1);
h(1)=1;
h(end)=-1;

hf=fft(h,[],1);
zf=fft(z,[],1);

gradz=ifft(repmat(hf,1,L).*zf);

zbis=0*z;
zbis(:,2:end)=z(:,1:end-1);

lambda=getoptions(options,'lambda',1);
G=eye(N);
G=G - circshift(G,[1 0]);
Gg=G'*G;

for l=1:L
theta(:,l) = (diag(gradz(:,l).^2) + lambda *Gg)\(gradz(:,l).*zbis(:,l));
end




