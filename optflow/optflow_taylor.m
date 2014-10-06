function [theta,estim] = optflow_taylor(z, options)
%this computes optical flow using simple taylor expansion

[N, L] = size(z);
h=zeros(N,1);
h(1)=1;
h(end)=-1;

lambda=getoptions(options,'lambda',1);
lambdar=getoptions(options,'lambdar',0);
sigma=getoptions(options,'sigma',1);

hh=gabor(N,sigma);

hf=fft(h,[],1);
zf=fft(z,[],1);
zb = real(ifft(zf.*repmat(hh,1,L)));

gradz=real(ifft(repmat(hh,1,L).*repmat(hf,1,L).*zf));

zbis=0*zb;
zbis(:,2:end)=zb(:,1:end-1);
zdif = zb-zbis;

G=eye(N);
G=G - circshift(G,[0 1]);
G=G(1:end-1,:);
%G(1,end)=0;
Gg=G'*G;

for l=1:L
theta(:,l) = (diag(gradz(:,l).^2) + lambda *Gg + lambdar * eye(N))\(gradz(:,l).*zdif(:,l));
end

%gradz=real(ifft(repmat(hf,1,L).*zf));
estim = zb - gradz.*theta;

end


function f = gabor(N,sigma)
extent = 1;         % extent of periodization - the higher, the better

f = zeros(N,1);

% Calculate the 2*pi-periodization of the filter over 0 to 2*pi*(N-1)/N
for k = -extent:1+extent
    f = f+exp(-(((0:N-1).'-k*N)/N*2*pi).^2./(2*sigma^2));
end
end


