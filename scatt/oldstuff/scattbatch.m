function out=scattbatch(X, options)
%1d scattering in batch

[N,L]=size(X);

J=getoptions(options,'J',3);
border=getoptions(options,'border',2^J*8);
Nbis=N-2*border;

FX = fft(X);

out=zeros(Nbis*(J+1),L);

tmp = ifft(FX.*repmat(options.filters.phi{1},1,L));
out(1:Nbis,:)=tmp(border+1:end-border,:);
for j=1:J
	tmp=abs(ifft(FX.*repmat(options.filters.psi{1}{j}{1},1,L)));
	out(1+j*Nbis:(j+1)*Nbis,:)=tmp(border+1:end-border,:);
end


